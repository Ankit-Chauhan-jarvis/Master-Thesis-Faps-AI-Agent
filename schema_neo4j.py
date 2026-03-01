from neo4j import GraphDatabase
import sys
import json
from collections import defaultdict 


URI = "" 
USERNAME = ""  
PASSWORD = ""  
DATABASE = ""  
OUTPUT_FILENAME = f"schema_{DATABASE}_full_labels_v3.json" 


class Neo4jSchemaExtractor:
    def __init__(self, uri, user, password, database):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            self.database = database
            print(f"✅ Successfully connected to Neo4j at {uri} (Database: '{self.database}')...")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}", file=sys.stderr)
            print("   Please check your URI, username, password, and that the database is running.", file=sys.stderr)
            sys.exit(1)

    def close(self):
        if self.driver:
            self.driver.close()
            print("✅ Connection to Neo4j closed.")

    def get_node_labels(self):
        """Fetches all unique node labels in the database."""
        query = "CALL db.labels() YIELD label RETURN label"
        print("🔬 Fetching node labels...")
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)
                labels = sorted([record["label"] for record in result])
                print(f"✅ Found {len(labels)} labels.")
                return labels
        except Exception as e:
            print(f"❌ Error fetching node labels: {e}", file=sys.stderr)
            return []

    def get_node_properties(self):
        """
        Fetches all node properties and their data types using the
        db.schema.nodeTypeProperties() procedure. Returns a dictionary
        mapping label -> {propertyName: propertyType}.
        """
        query = "CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName, propertyTypes RETURN *"
        print("🔬 Fetching node properties and types...")
        properties_map = defaultdict(dict)
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)
                records_processed = 0
                for record in result:
                    records_processed += 1
                    prop_name = record["propertyName"]
                    prop_type = (record["propertyTypes"][0] if record["propertyTypes"] else "UNKNOWN").upper()
                    for label in record["nodeLabels"]:
                        if prop_name not in properties_map[label]:
                            properties_map[label][prop_name] = prop_type
                
                if records_processed == 0:
                     print("   ⚠️ Warning: `db.schema.nodeTypeProperties` returned no results. Schema might be empty.")
                else:
                    print(f"✅ Found properties for {len(properties_map)} labels.")
                return dict(properties_map)

        except Exception as e:
            if "Unknown procedure" in str(e) or "no procedure with the name" in str(e):
                 print(f"❌ Error: Procedure 'db.schema.nodeTypeProperties' not found.", file=sys.stderr)
                 print("   This procedure is needed for accurate type information.", file=sys.stderr)
            else:
                print(f"❌ Error fetching node properties: {e}", file=sys.stderr)
            return {}

    def get_relationships_for_label(self, label):
        """
        Fetches outgoing relationships for a specific label by querying the graph.
        Returns a dictionary mapping relationshipType -> List of TargetLabels.
        """
        escaped_label = f"`{label.replace('`', '``')}`"
        
        query = f"""
        MATCH (n:{escaped_label})-[r]->(m)
        WHERE n IS NOT NULL AND m IS NOT NULL AND type(r) IS NOT NULL
        WITH DISTINCT type(r) AS relType, labels(m) AS targetLabels
        WHERE size(targetLabels) > 0 // Ensure target node has labels
        UNWIND targetLabels as targetLabel // Unwind the list of labels
        RETURN relType, targetLabel // Return each relationship type + target label pair
        LIMIT 100000 // Add a limit to prevent extremely long queries
        """
        
        relationships = defaultdict(set)
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)
                for record in result:
                    rel_type = record["relType"]
                    target_label = record["targetLabel"]
                    relationships[rel_type].add(target_label)

            final_relationships = {}
            for rel_type, targets in relationships.items():
                final_relationships[rel_type] = sorted(list(targets))
                
            return final_relationships
            
        except Exception as e:
            print(f"   ❌ Error fetching relationships for label '{label}': {e}", file=sys.stderr)
            return {}


    def get_full_schema(self):
        """Extracts and formats the full schema."""
        print(f"\n🔬 Extracting full schema for database: '{self.database}' ...")

        all_labels = self.get_node_labels()
        if not all_labels:
            print("No node labels found. Cannot extract schema.")
            return []

        all_properties = self.get_node_properties()
        if not all_properties:
             print("   ⚠️ Warning: Could not fetch property types. Attributes section will be empty.")


        final_schema = []
        print(f"\n🔬 Fetching relationships for {len(all_labels)} labels (this may take a while)...")
        count = 0
        total_labels = len(all_labels)
        for label in all_labels:
            count += 1
            progress = int((count / total_labels) * 50) # 50 chars wide bar
            print(f"  [{'=' * progress}{' ' * (50 - progress)}] ({count}/{total_labels}) Processing label: {label}", end='\r')

            label_relationships = self.get_relationships_for_label(label)
            schema_entry = {
                "label": label,
                "attributes": all_properties.get(label, {}),
                "relationships": label_relationships
            }
            final_schema.append(schema_entry)

        print("\n✅ Schema extraction complete.") 
        return final_schema


if __name__ == "__main__":
    print("--- Neo4j Schema Extractor (v3 - Full Labels) ---")
    if PASSWORD == "Ankujarvis@1094": 
        print("="*60)
        print("ℹ️  Using password from script. Running extraction...")
        print("="*60)
    
    extractor = None 
    full_schema = []
    try:
        extractor = Neo4jSchemaExtractor(URI, USERNAME, PASSWORD, DATABASE)
        full_schema = extractor.get_full_schema()
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during execution: {e}", file=sys.stderr)
    finally:
        if extractor:
            extractor.close()

    if full_schema:
        try:
            with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
                json.dump(full_schema, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Schema successfully saved to '{OUTPUT_FILENAME}'")
        except Exception as e:
            print(f"\n❌ Error saving schema to JSON: {e}", file=sys.stderr)
    else:
        print("\n⚠️ Schema extraction did not produce results. No JSON file saved.")

    print("--- Script finished ---")