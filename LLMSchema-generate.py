import json
import os

INPUT_FILE = 'neo4j_schema.json'
OUTPUT_FILE = 'schema_for_llm.json' # Will overwrite with the new, better format

def generate_structured_schema(schema_file, output_file):
    """
    Parses the full apoc.meta.schema output (neo4j_schema.json) and 
    generates a comprehensive JSON file for an LLM, including:
    - Node labels and their properties (with types)
    - Relationship types and their properties (with types)
    - Graph structure (connection patterns)
    - Unique constraints
    - Full-text indexes
    """
    
    node_definitions = {}  # Stores: { "Label": {"prop1": "TYPE", ...} }
    rel_definitions = {}   # Stores: { "REL_TYPE": {"prop1": "TYPE", ...} }
    graph_structure = set() # Stores tuples: ("StartLabel", "REL_TYPE", "EndLabel")
    
    # --- New containers for constraints and indexes ---
    unique_constraints = []
    fulltext_indexes = []

    try:
        with open(schema_file, 'r', encoding='utf-8') as f:
            full_schema_data = json.load(f)
        
        # --- 1. Process Data Schema (Nodes, Rels, Properties, Structure) ---
        if 'data_schema' not in full_schema_data or not full_schema_data['data_schema']:
            print("Error: 'data_schema' key is missing or empty.")
            return

        data_schema = full_schema_data['data_schema']

        # Pass 1: Get all node/rel definitions and their properties with types
        for name, details in data_schema.items():
            entry_type = details.get('type')
            properties_with_types = {}
            for prop_name, prop_details in details.get('properties', {}).items():
                prop_type = prop_details.get('type', 'UNKNOWN')
                if prop_details.get('array') is True:
                    prop_type = f"LIST<{prop_type}>"
                properties_with_types[prop_name] = prop_type

            if entry_type == 'node':
                node_definitions[name] = properties_with_types
            elif entry_type == 'relationship':
                rel_definitions[name] = properties_with_types

        # Pass 2: Determine the graph structure
        for node_name, node_details in data_schema.items():
            if node_details.get('type') != 'node': continue
            start_label = node_name
            if start_label not in node_definitions: node_definitions[start_label] = {}

            for rel_name, rel_details_list in node_details.get('relationships', {}).items():
                if not isinstance(rel_details_list, list): rel_details_list = [rel_details_list]

                for rel_details in rel_details_list:
                    direction = rel_details.get('direction', 'out')
                    target_labels = rel_details.get('labels', [])
                    
                    if rel_name not in rel_definitions:
                        rel_props = {}
                        for p_name, p_details in rel_details.get('properties', {}).items():
                            p_type = p_details.get('type', 'UNKNOWN')
                            if p_details.get('array') is True: p_type = f"LIST<{p_type}>"
                            rel_props[p_name] = p_type
                        rel_definitions[rel_name] = rel_props
                    
                    for end_label in target_labels:
                        if end_label not in node_definitions:
                             node_definitions[end_label] = {}
                        
                        if direction == 'out':
                            graph_structure.add((start_label, rel_name, end_label))
                        elif direction == 'in':
                            graph_structure.add((end_label, rel_name, start_label))

        # --- 2. Process Constraints ---
        if 'constraints' in full_schema_data:
            for constraint in full_schema_data['constraints']:
                if constraint.get('type') == 'UNIQUENESS':
                    label = constraint.get('labelsOrTypes', [])[0]
                    prop = constraint.get('properties', [])[0]
                    unique_constraints.append(f"{label}.{prop}")

        # --- 3. Process Indexes ---
        if 'indexes' in full_schema_data:
            for index in full_schema_data['indexes']:
                if index.get('type') == 'FULLTEXT':
                    label = index.get('labelsOrTypes', [])[0]
                    props = index.get('properties', [])
                    index_name = index.get('name', 'fulltext_index') # Get the index name
                    fulltext_indexes.append({
                        "index_name": index_name,
                        "label": label,
                        "properties": props
                    })

        # --- 4. Prepare and Save Final JSON Output ---
        node_labels_json = {label: dict(sorted(props.items())) for label, props in sorted(node_definitions.items())}
        rel_types_json = {rel_type: dict(sorted(props.items())) for rel_type, props in sorted(rel_definitions.items())}
        graph_structure_json = sorted([f"({start})-[:{rel}]->({end})" for (start, rel, end) in graph_structure])

        output_data = {
            "node_labels": node_labels_json,
            "relationship_types": rel_types_json,
            "graph_structure": graph_structure_json,
            "unique_constraints": sorted(unique_constraints),
            "fulltext_indexes": fulltext_indexes
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        
        print(f"✅ Successfully generated comprehensive LLM schema at '{output_file}'")
        print(f"   (Includes property types, constraints, and full-text indexes)")

    except FileNotFoundError:
        print(f"Error: The file '{schema_file}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{schema_file}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        print("Please make sure the neo4j_schema.json file is in the same directory.")
    else:
        generate_structured_schema(INPUT_FILE, OUTPUT_FILE)