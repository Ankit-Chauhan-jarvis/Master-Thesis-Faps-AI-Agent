import json
import re
import networkx as nx
import community as louvain
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

def parse_schema(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Clean content and extract JSON
            content = re.sub(r'---.*?---', '', content, flags=re.DOTALL)
            content = content.strip()
            
            # Handle trailing commas
            content = re.sub(r',\s*\]', ']', content)
            content = re.sub(r',\s*\}', '}', content)
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Fallback to line-by-line parsing
                print("JSON parse failed. Attempting line-by-line parsing...")
                data = []
                for line in content.splitlines():
                    line = line.strip()
                    if line and line[0] in ('{', '['):
                        try:
                            # Handle array lines
                            if line.startswith('['):
                                array_data = json.loads(line)
                                if isinstance(array_data, list):
                                    data.extend(array_data)
                            # Handle object lines
                            elif line.startswith('{'):
                                data.append(json.loads(line))
                        except:
                            continue
                return data
    except Exception as e:
        print(f"Critical error reading file: {str(e)}")
        return []

def build_schema_graph(data):
    G = nx.Graph()
    node_definitions = {}
    
    for node_def in data:
        if not isinstance(node_def, dict):
            continue
            
        label = node_def.get("label")
        if not label:
            continue
            
        # Store node attributes for later use
        attributes = node_def.get("attributes", {})
        relationships = node_def.get("relationships", {})
        node_definitions[label] = {
            "attributes": attributes,
            "relationships": relationships
        }
        
        # Add node with attributes
        G.add_node(label, **attributes)
        
        # Add relationships
        for rel_type, target_label in relationships.items():
            if target_label and isinstance(target_label, str):
                G.add_edge(label, target_label, type=rel_type)
    
    return G, node_definitions

def analyze_communities(G):
    # Detect communities with a higher resolution
    partition = louvain.best_partition(G, resolution=1.5, randomize=True)
    
    # Organize communities
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)
    
    # Sort by size
    sorted_communities = sorted(communities.values(), key=len, reverse=True)
    
    # Create mapping from node to sorted community index
    sorted_partition = {}
    for idx, comm in enumerate(sorted_communities):
        for node in comm:
            sorted_partition[node] = idx
    
    return sorted_communities, partition, sorted_partition

def analyze_bridge_nodes(G, partition, min_connections=2, min_degree=5):
    
    bridges = defaultdict(lambda: {'communities': set(), 'degree': 0})
    
    for node, comm_id in partition.items():
        node_degree = G.degree(node)
        
        if node_degree >= min_degree:
            external_communities = set()
            for neighbor in G.neighbors(node):
                neighbor_comm_id = partition.get(neighbor)
                if neighbor_comm_id is not None and neighbor_comm_id != comm_id:
                    external_communities.add(neighbor_comm_id)
            
            if len(external_communities) >= min_connections:
                bridges[node] = {
                    'communities': external_communities,
                    'degree': node_degree
                }
                
    return bridges

def calculate_community_metrics(G, sorted_communities, sorted_partition):
    community_metrics = []
    
    for i, comm in enumerate(sorted_communities):
        comm_id = i + 1
        num_nodes = len(comm)
        
        subgraph = G.subgraph(comm)
        internal_edges = subgraph.number_of_edges()
        
        possible_edges = num_nodes * (num_nodes - 1) / 2
        
        if possible_edges > 0:
            density = min(1.0, internal_edges / possible_edges)
        else:
            density = 0.0
        
        external_connections = defaultdict(int)
        for node in comm:
            for neighbor in G.neighbors(node):
                if neighbor not in comm:
                    neighbor_comm = sorted_partition.get(neighbor, -1)
                    if neighbor_comm != -1:
                        external_connections[neighbor_comm] += 1
        
        sorted_connections = sorted(external_connections.items(), key=lambda item: item[1], reverse=True)
        ext_conn_str = ", ".join(
            [f"C{id+1}:{count}" for id, count in sorted_connections]
        )
        
        community_metrics.append({
            "id": comm_id,
            "nodes": comm,
            "size": num_nodes,
            "internal_edges": internal_edges,
            "density": density,
            "external_connections": dict(external_connections),
            "ext_conn_str": ext_conn_str
        })
    
    return community_metrics

def identify_bridge_relationships(community_metrics, threshold=6):
    bridge_relationships = []
    
    for metrics in community_metrics:
        comm_id = metrics["id"]
        for target_comm, count in metrics["external_connections"].items():
            if count >= threshold:
                
                if not any(br for br in bridge_relationships 
                          if br[0] == comm_id and br[1] == target_comm+1):
                    bridge_relationships.append((comm_id, target_comm+1, count))
    
    
    bridge_relationships.sort(key=lambda x: x[2], reverse=True)
    return bridge_relationships

def visualize_communities(G, partition, output_file, title_suffix="", bridge_nodes=None):
    
    plt.figure(figsize=(20, 15))
    
    if bridge_nodes is None:
        bridge_nodes = set()
    else:
        bridge_nodes = set(bridge_nodes)
        
    communities = set(partition.values())
    
    cmap = plt.get_cmap('viridis', len(communities))
    
    node_colors = []
    for n in G.nodes():
        if n in bridge_nodes:
            node_colors.append('red')
        else:
            node_colors.append(cmap(partition[n]))
    
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    nx.draw_networkx_nodes(
        G, pos, node_size=800,
        node_color=node_colors,
        alpha=0.8
    )
    
    nx.draw_networkx_edges(
        G, pos, width=1.0, alpha=0.2,
        edge_color='gray'
    )
    
    for node, (x, y) in pos.items():
        plt.text(x, y, node, 
                 fontsize=9, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.7, 
                          edgecolor='none', boxstyle='round,pad=0.2'))
    
    if bridge_nodes:
        plt.scatter([], [], c='red', s=50, label='Bridge Nodes')
        plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Legend')
        
    plt.title(f"Neo4j Schema Communities {title_suffix}", fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def generate_report(community_metrics, bridge_relationships, output_file, title_prefix=""):
    with open(output_file, 'w', encoding='utf-8') as f:
        # Community listing
        f.write(f"Detected Sub-Schemas: {title_prefix}\n")
        f.write("=" * 60 + "\n\n")
        
        for metrics in community_metrics:
            f.write(f"Community #{metrics['id']} ({metrics['size']} nodes):\n")
            f.write("-" * 60 + "\n")
            
            # List nodes in groups of 10
            nodes = metrics['nodes']
            for i in range(0, len(nodes), 10):
                f.write(", ".join(nodes[i:i+10]) + "\n")
            f.write("\n")
        
        # Cohesion analysis
        f.write("\nCommunity Cohesion Analysis:\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Community':<12} | {'Nodes':<6} | {'Internal Edges':<15} | {'Density':<8} | {'Connections to Other Communities'}\n")
        f.write("-" * 80 + "\n")
        
        for metrics in community_metrics:
            f.write(f"{'C'+str(metrics['id']):<12} | {metrics['size']:<6} | {metrics['internal_edges']:<15} | {metrics['density']:.4f}    | {metrics['ext_conn_str']}\n")
        
        # Bridge relationships
        f.write("\n\nKey Integration Points (Bridge Relationships):\n")
        f.write("=" * 60 + "\n")
        if bridge_relationships:
            f.write(f"\nTop Bridge Relationships (>=10 connections):\n")
            f.write("-" * 60 + "\n")
            for source, target, count in bridge_relationships:
                f.write(f"Community #{source} ↔ Community #{target}: {count} connections\n")
        else:
            f.write("\nNo significant bridge relationships found\n")
        
        # Community overview
        f.write("\n\nCommunity Structure Overview:\n")
        f.write("=" * 60 + "\n")
        for metrics in community_metrics[:10]:  # Top 10 communities
            f.write(f"\nCommunity #{metrics['id']} (Size: {metrics['size']}):\n")
            f.write(", ".join(metrics['nodes'][:10]) + "\n")
            if metrics['size'] > 10:
                f.write(f"+ {metrics['size']-10} more nodes\n")


def recursive_refine_communities(G, community_nodes, name_prefix, output_dir, threshold, prev_size):
    """
    Recursively refines a community by running Louvain on its subgraph.
    """
    subgraph = G.subgraph(community_nodes)
    
    if subgraph.number_of_nodes() <= 1 or subgraph.number_of_edges() == 0:
        return
        
    print(f"Refining community: {name_prefix} (size: {subgraph.number_of_nodes()})...")
    
    sub_communities, sub_partition, _ = analyze_communities(subgraph)

    sub_community_metrics = calculate_community_metrics(subgraph, sub_communities, {node: i for i, sub_comm in enumerate(sub_communities) for node in sub_comm})
    sub_bridge_relationships = identify_bridge_relationships(sub_community_metrics, threshold=1)
    
    sub_bridge_nodes = analyze_bridge_nodes(subgraph, sub_partition, min_connections=2, min_degree=2)

    report_file = os.path.join(output_dir, f"{name_prefix}_report.txt")
    image_file = os.path.join(output_dir, f"{name_prefix}_visualization.png")
    
    generate_report(sub_community_metrics, sub_bridge_relationships, report_file, title_prefix=f"(Refined from {name_prefix})")
    visualize_communities(subgraph, sub_partition, image_file, title_suffix=f"(Refined from {name_prefix})", bridge_nodes=sub_bridge_nodes)
    
    for i, sub_comm in enumerate(sub_communities):
        sub_comm_size = len(sub_comm)
        if sub_comm_size > threshold:
            # Check for stagnation in community size
            if sub_comm_size == prev_size:
                print(f"Skipping further refinement of {name_prefix}_sub{i+1} as its size has not changed ({sub_comm_size} nodes). It may be a highly cohesive, inseparable community.")
                continue

            new_name_prefix = f"{name_prefix}_sub{i+1}"
            recursive_refine_communities(subgraph, sub_comm, new_name_prefix, output_dir, threshold, sub_comm_size)


def main():
    OUTPUT_DIR = "output_deep_recursive"
    LARGE_COMMUNITY_THRESHOLD = 3 # You can adjust this threshold for different levels of granularity

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading and parsing schema...")
    data = parse_schema('schema.txt')
    if not data:
        print("No valid data found. Exiting.")
        return
    print(f"Parsed {len(data)} node definitions")
    
    # Build graph
    print("Building schema graph...")
    G, node_definitions = build_schema_graph(data)
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    print("\nStarting initial community detection...")
    initial_communities, initial_partition, sorted_partition = analyze_communities(G)

    print("Generating initial full graph report...")
    initial_community_metrics = calculate_community_metrics(G, initial_communities, sorted_partition)
    initial_bridge_relationships = identify_bridge_relationships(initial_community_metrics)
    generate_report(initial_community_metrics, initial_bridge_relationships, os.path.join(OUTPUT_DIR, "initial_communities_report.txt"), title_prefix="(Initial Full Graph)")
    
    bridge_nodes = analyze_bridge_nodes(G, initial_partition, min_connections=2, min_degree=5)
    
    # Create a separate report for the bridge nodes
    with open(os.path.join(OUTPUT_DIR, "bridge_nodes_report.txt"), "w") as f:
        f.write("=== Top Bridge Node Analysis ===\n")
        f.write("A bridge node is a single node connecting multiple communities. "
                "These are crucial for understanding the relationships between sub-schemas.\n\n")
        f.write("Node Label | Degree | Communities Connected\n")
        f.write("----------------------------------------------\n")
        for node, data in sorted(bridge_nodes.items(), key=lambda item: item[1]['degree'], reverse=True):
            community_labels = ", ".join([f"C{c+1}" for c in data['communities']])
            f.write(f"{node:<10} | {data['degree']:<6} | {community_labels}\n")

    print("\nTop bridge nodes report generated.")
    
    # Visualize initial graph with bridge nodes highlighted
    visualize_communities(G, initial_partition, os.path.join(OUTPUT_DIR, "initial_communities.png"), title_suffix="(Initial Full Graph)", bridge_nodes=bridge_nodes.keys())

    print("\nStarting deep recursive refinement for all large communities...")
    
    for i, comm in enumerate(initial_communities):
        comm_id = i + 1
        num_nodes = len(comm)

        if num_nodes > LARGE_COMMUNITY_THRESHOLD:
            name_prefix = f"comm{comm_id}_{num_nodes}"
            recursive_refine_communities(G, comm, name_prefix, OUTPUT_DIR, LARGE_COMMUNITY_THRESHOLD, num_nodes)
        else:
            print(f"Skipping initial community #{comm_id} (size: {num_nodes}) as it is below the threshold.")
            
    print("\n=== Recursive Refinement Complete ===")

if __name__ == "__main__":
    main()