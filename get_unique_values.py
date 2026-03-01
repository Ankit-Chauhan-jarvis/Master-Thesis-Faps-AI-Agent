import json

INPUT_FILE = 'neo4j_schema.json'
# --- New constant for the output JSON file ---
OUTPUT_FILE = 'unique_schema_items.json'

def extract_unique_items(schema_file):
    """
    Parses the apoc.meta.schema output and extracts unique sets of
    node labels, relationship types, node properties, and relationship properties.
    
    Saves the results to a JSON file.
    """
    node_labels = set()
    relationship_types = set()
    node_properties = set()
    relationship_properties = set()

    try:
        with open(schema_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'data_schema' not in data:
            print(f"Error: 'data_schema' key not found in {schema_file}.")
            print("Please ensure you are using the correct JSON schema file.")
            return

        data_schema = data['data_schema']

        for name, details in data_schema.items():
            entry_type = details.get('type')
            
            if entry_type == 'node':
                # This is a Node Label
                node_labels.add(name)
                
                # Add all its properties to node_properties
                for prop_name in details.get('properties', {}):
                    node_properties.add(prop_name)
                
                # Add all its relationships to relationship_types
                for rel_name in details.get('relationships', {}):
                    relationship_types.add(rel_name)

            elif entry_type == 'relationship':
                # This is a Relationship Type
                relationship_types.add(name)
                
                # Add all its properties to relationship_properties
                for prop_name in details.get('properties', {}):
                    relationship_properties.add(prop_name)

        # --- Prepare the output data ---
        output_data = {
            "node_labels": sorted(list(node_labels)),
            "relationship_types": sorted(list(relationship_types)),
            "node_properties": sorted(list(node_properties)),
            "relationship_properties": sorted(list(relationship_properties))
        }

        # --- Write the results to a JSON file ---
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        
        print(f"✅ Successfully extracted unique items and saved to '{OUTPUT_FILE}'")
        print(f"  - Node Labels: {len(output_data['node_labels'])}")
        print(f"  - Relationship Types: {len(output_data['relationship_types'])}")
        print(f"  - Node Properties: {len(output_data['node_properties'])}")
        print(f"  - Relationship Properties: {len(output_data['relationship_properties'])}")


    except FileNotFoundError:
        print(f"Error: The file '{schema_file}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{schema_file}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    extract_unique_items(INPUT_FILE)

