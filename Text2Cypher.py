import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file.")

client = Groq(api_key=GROQ_API_KEY)

def get_relevant_nodes(question: str, labels_file: str) -> list[str]:
    
    try:
        with open(labels_file, 'r') as f:
            data = json.load(f)
        labels = data.get("labels", [])

        if not labels:
            print("Warning: 'labels' key not found or is empty in the JSON file.")
            return []

        labels_string = ", ".join(labels)

        prompt = (
            f"You are a database schema expert. A user has a question about a database. "
            f"Based on their question, identify the most relevant database labels (nodes) which can be used "
            f"from the following list. Return only the list of labels comma-separated, "
            f"with no other text.\n\n"
            f"Question: '{question}'\n\n"
            f"Available Labels: {labels_string}"
        )
        
        print("Sending request to Groq API...")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            #model="openai/gpt-oss-20b",
            model="groq/compound-mini",
            temperature=0.0,
            #model="llama-3.3-70b-versatile",
        )
        
        response_text = chat_completion.choices[0].message.content.strip()
        
        relevant_nodes = [node.strip() for node in response_text.split(',')]
        
        valid_nodes = [node for node in relevant_nodes if node in labels]
        
        return valid_nodes

    except Exception as e:
        print(f"An error occurred: {e}")
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

def get_cypher_query(question: str, sub_schema: dict) -> str:
    
    try:
        sub_schema_string = json.dumps(sub_schema, indent=2)

        prompt = (
            f"You are a Neo4j Cypher query expert. A user has a question about a database. "
            f"Based on the provided sub-schema and the user's question, "
            f"generate a single, valid Cypher query that answers the question. "
            f"Do not include any extra text, comments, or explanations. Only provide the query.\n\n"
            f"Sub-schema: \n{sub_schema_string}\n\n"
            f"Question: '{question}'"
        )
        
        print("Sending request to Groq API for Cypher query...")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="groq/compound-mini",
            temperature=0.0,
            #model="openai/gpt-oss-20b",
        )
        
        cypher_query = chat_completion.choices[0].message.content.strip()
        return cypher_query

    except Exception as e:
        print(f"An error occurred while generating the Cypher query: {e}")
        return ""


def main():
    
    labels_json_file = "unique-1.json"
    full_schema_json_file = "schema_neo4j_full_labels_v3.json" 
    
    user_question = input("Enter your question: ")
    print(f"\nProcessing question: '{user_question}'")
    nodes = get_relevant_nodes(user_question, labels_json_file)
    print(f"Relevant nodes: {nodes}")
    
    print("\nAttempting to generate sub-schema...")
    if nodes:
        sub_schema = get_sub_schema(nodes, full_schema_json_file)
        print("\nGenerated sub-schema:")
        print(json.dumps(sub_schema, indent=2))
        
        cypher_query = get_cypher_query(user_question, sub_schema)
        print("\nGenerated Cypher Query:")
        print(cypher_query)
        
    else:
        print("No relevant nodes found to generate a sub-schema.")

if __name__ == "__main__":
    main()