import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file.")

client = Groq(api_key=GROQ_API_KEY)

def get_relevant_nodes(question: str, labels_file: str, knowledge_string: str) -> list[str]:
    
    try:
        with open(labels_file, 'r') as f:
            data = json.load(f)
        labels = data.get("labels", [])

        if not labels:
            print("Warning: 'labels' key not found or is empty in the JSON file.")
            return []

        labels_string = ", ".join(labels)

        prompt = (
            f"You are a database schema expert. A user has a question about a database.\n"
            f"Use the provided 'Schema Knowledge' to understand the meaning of labels and relationships. "
            f"Based on the user's question and this knowledge, identify the most relevant database labels (nodes) "
            f"from the 'Available Labels' list. Return only the list of labels comma-separated, "
            f"with no other text.\n\n"
            f"--- Schema Knowledge ---\n"
            f"{knowledge_string}\n"
            f"------------------------\n\n"
            f"Question: '{question}'\n\n"
            f"Available Labels: {labels_string}"
        )
        
        print("Sending request to Groq API (for relevant nodes)...")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="groq/compound-mini",
            temperature=0.0,
        )
        
        response_text = chat_completion.choices[0].message.content.strip()
        
        relevant_nodes = [node.strip() for node in response_text.split(',')]
        
        valid_nodes = [node for node in relevant_nodes if node in labels]
        
        return valid_nodes

    except Exception as e:
        print(f"An error occurred in get_relevant_nodes: {e}")
        return []

def get_sub_schema(relevant_nodes: list[str], full_schema_file: str) -> dict:
    
    try:
        with open(full_schema_file, 'r', encoding='utf-8') as f:
            full_schema_list = json.load(f)
        
        sub_schema = {}
        for node in relevant_nodes:
            for schema_item in full_schema_list:
                if schema_item.get("label") == node:
                    sub_schema[node] = schema_item
                    break
        
        return sub_schema
        
    except FileNotFoundError:
        print(f"Error: Full schema file not found at '{full_schema_file}'")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{full_schema_file}'")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred while processing the schema: {e}")
        return {}

def get_cypher_query(question: str, sub_schema: dict, knowledge_string: str) -> str:
    
    try:
        sub_schema_string = json.dumps(sub_schema, indent=2)

        prompt = (
            f"You are a Neo4j Cypher query expert. A user has a question about a database.\n"
            f"Use the 'Sub-schema' to understand the node properties and relationships, "
            f"and use the 'Schema Knowledge' to understand the real-world meaning of the schema. "
            f"Based on both, generate a single, valid Cypher query that answers the user's question. "
            f"Do not include any extra text, comments, or explanations. Only provide the query.\n\n"
            f"--- Sub-schema ---\n"
            f"{sub_schema_string}\n"
            f"--------------------\n\n"
            f"--- Schema Knowledge ---\n"
            f"{knowledge_string}\n"
            f"------------------------\n\n"
            f"Question: '{question}'"
        )
        
        print("Sending request to Groq API (for Cypher query)...")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="groq/compound-mini",
            temperature=0.0,
        )
        
        cypher_query = chat_completion.choices[0].message.content.strip()
        
        if cypher_query.startswith("```cypher"):
            cypher_query = cypher_query[9:]
        if cypher_query.startswith("```"):
            cypher_query = cypher_query[3:]
        if cypher_query.endswith("```"):
            cypher_query = cypher_query[:-3]
            
        return cypher_query.strip()

    except Exception as e:
        print(f"An error occurred while generating the Cypher query: {e}")
        return ""


def main():
    
    labels_json_file = "unique-1.json"
    full_schema_json_file = "schema_neo4j_full_labels_v3.json" 
    
    user_question = input("Enter your question: ")
    
    print("\n--- Enter your additional schema knowledge below ---")
    print("(Press ENTER on an empty line when you are finished)")
    
    knowledge_lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        knowledge_lines.append(line)

    knowledge_base = "\n".join(knowledge_lines)
    
    if not knowledge_base:
        print("\nNo additional knowledge provided, proceeding...")
        knowledge_base = "No additional knowledge provided."
    else:
         print("\n--- Knowledge base captured successfully ---")

    
    print(f"\nProcessing question: '{user_question}'")
    
    nodes = get_relevant_nodes(user_question, labels_json_file, knowledge_base)
    print(f"Relevant nodes: {nodes}")
    
    print("\nAttempting to generate sub-schema...")
    if nodes:
        sub_schema = get_sub_schema(nodes, full_schema_json_file)
        print("\nGenerated sub-schema:")
        print(json.dumps(sub_schema, indent=2))
        
        cypher_query = get_cypher_query(user_question, sub_schema, knowledge_base)
        print("\nGenerated Cypher Query:")
        print(cypher_query)
        
    else:
        print("No relevant nodes found to generate a sub-schema.")

if __name__ == "__main__":
    main()