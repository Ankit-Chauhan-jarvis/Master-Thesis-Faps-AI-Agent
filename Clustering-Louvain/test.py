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
    """Robust schema parser with enhanced error handling"""
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
    """Build schema graph from parsed data"""
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
    """Perform community detection and analysis"""
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
    """
    Identifies and analyzes nodes that act as bridges between communities.
    
    Args:
        G (nx.Graph): The graph.
        partition (dict): The community partition.
        min_connections (int): Minimum number of communities a node must connect to.
        min_degree (int): Minimum degree of a node to be considered a bridge.

    Returns:
        dict: A dictionary of bridge nodes and their connections.
    """
    bridges = defaultdict(lambda: {'communities': set(), 'degree': 0})
    
    for node, comm_id in partition.items():
        node_degree = G.degree(node)
        
        # Only consider nodes with a high enough degree
        if node_degree >= min_degree:
            external_communities = set()
            for neighbor in G.neighbors(node):
                neighbor_comm_id = partition.get(neighbor)
                if neighbor_comm_id is not None and neighbor_comm_id != comm_id:
                    external_communities.add(neighbor_comm_id)
            
            # If the node connects to multiple communities, it's a potential bridge
            if len(external_communities) >= min_connections:
                bridges[node] = {
                    'communities': external_communities,
                    'degree': node_degree
                }
                
    return bridges

def calculate_community_metrics(G, sorted_communities, sorted_partition):
    """Calculate cohesion and connection metrics"""
    community_metrics = []
    
    for i, comm in enumerate(sorted_communities):
        comm_id = i + 1
        num_nodes = len(comm)
        
        # Create subgraph
        subgraph = G.subgraph(comm)
        
        # Internal metrics
        internal_edges = subgraph.number_of_edges()
        
        # Calculate possible edges, handling communities with 1 node
        possible_edges = num_nodes * (num_nodes - 1) / 2
        
        # Corrected density calculation to cap at 1.0
        # This handles cases where internal_edges > possible_edges, as in a multigraph
        if possible_edges > 0:
            density = min(1.0, internal_edges / possible_edges)
        else:
            density = 0.0
        
        # External connections
        external_connections = defaultdict(int)
        for node in comm:
            for neighbor in G.neighbors(node):
                if neighbor not in comm:
                    neighbor_comm = sorted_partition.get(neighbor, -1)
                    if neighbor_comm != -1:
                        external_connections[neighbor_comm] += 1
        
        # Sort connections by count in descending order for better intuition
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
    """Identify significant connections between communities"""
    bridge_relationships = []
    
    for metrics in community_metrics:
        comm_id = metrics["id"]
        for target_comm, count in metrics["external_connections"].items():
            if count >= threshold:
                # Avoid duplicates
                if not any(br for br in bridge_relationships 
                          if br[0] == comm_id and br[1] == target_comm+1):
                    bridge_relationships.append((comm_id, target_comm+1, count))
    
    # Sort by strength
    bridge_relationships.sort(key=lambda x: x[2], reverse=True)
    return bridge_relationships

def visualize_communities(G, partition, output_file, title_suffix="", bridge_nodes=None):
    """Visualize communities with different colors, highlighting bridge nodes in red."""
    plt.figure(figsize=(32, 24))  # Further increased figure size
    
    if bridge_nodes is None:
        bridge_nodes = set()
    else:
        bridge_nodes = set(bridge_nodes)
        
    # Get unique community IDs
    communities = set(partition.values())
    
    # Generate color map with higher contrast
    cmap = plt.get_cmap('tab20', len(communities))
    
    # Assign colors to nodes, with red for bridge nodes
    node_colors = []
    node_sizes = []
    for n in G.nodes():
        if n in bridge_nodes:
            node_colors.append('red')
            node_sizes.append(5000)  # Even larger for bridge nodes
        else:
            node_colors.append(cmap(partition[n]))
            node_sizes.append(4000)  # Larger node size
    
    # Layout with more spacing
    pos = nx.spring_layout(G, k=2.0, iterations=150, seed=42)  # Further increased k and iterations
    
    # Draw edges with significantly increased width and contrast
    nx.draw_networkx_edges(
        G, pos, width=6.0, alpha=0.7,  # Further increased width and alpha
        edge_color='darkblue'  # Changed to darker color for better contrast
    )
    
    # Draw nodes with border for better visibility - much larger size
    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes,  # Use customized node sizes
        node_color=node_colors,
        alpha=0.95,
        edgecolors='black',
        linewidths=4.0  # Thicker borders
    )
    
    # Draw labels with significantly increased font size and better contrast
    for node, (x, y) in pos.items():
        plt.text(x, y, node, 
                 fontsize=30, ha='center', va='center',  # Font size increased to 30
                 bbox=dict(facecolor='white', alpha=0.95, 
                          edgecolor='black', boxstyle='round,pad=1.0', linewidth=2.0))  # More padding and thicker border
    
    # Add a more prominent legend for the bridge nodes
    if bridge_nodes:
        plt.scatter([], [], c='red', s=300, label='Bridge Nodes')
        plt.legend(scatterpoints=1, frameon=True, labelspacing=2.0, 
                  title='Legend', fontsize=24, title_fontsize=26,
                  framealpha=0.9, edgecolor='black')
        
    plt.title(f"Neo4j Schema Communities {title_suffix}", fontsize=32, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')  # Higher DPI and white background
    plt.close()

def create_community_size_chart(community_metrics, output_file):
    """Create a bar chart showing the size of each community"""
    plt.figure(figsize=(16, 10))
    
    # Extract community IDs and sizes
    comm_ids = [f"C{metrics['id']}" for metrics in community_metrics]
    sizes = [metrics['size'] for metrics in community_metrics]
    
    # Create bar chart
    bars = plt.bar(comm_ids, sizes, color=plt.cm.tab20(np.linspace(0, 1, len(comm_ids))))
    
    # Add value labels on top of each bar with larger font
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontsize=16)
    
    plt.xlabel('Community ID', fontsize=20)
    plt.ylabel('Number of Nodes', fontsize=20)
    plt.title('Community Sizes', fontsize=24)
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

def generate_report(community_metrics, bridge_relationships, output_file, title_prefix=""):
    """Generate comprehensive report in text format"""
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
    
    # Only proceed if the subgraph has more than 1 node and some edges
    if subgraph.number_of_nodes() <= 1 or subgraph.number_of_edges() == 0:
        return
        
    print(f"Refining community: {name_prefix} (size: {subgraph.number_of_nodes()})...")
    
    sub_communities, sub_partition, _ = analyze_communities(subgraph)

    # Calculate metrics for the sub-communities
    sub_community_metrics = calculate_community_metrics(subgraph, sub_communities, {node: i for i, sub_comm in enumerate(sub_communities) for node in sub_comm})
    sub_bridge_relationships = identify_bridge_relationships(sub_community_metrics, threshold=1)
    
    # Identify bridge nodes for the refined subgraph
    sub_bridge_nodes = analyze_bridge_nodes(subgraph, sub_partition, min_connections=2, min_degree=2)

    # Generate reports and visualizations with concise names
    report_file = os.path.join(output_dir, f"{name_prefix}_report.txt")
    image_file = os.path.join(output_dir, f"{name_prefix}_visualization.png")
    
    generate_report(sub_community_metrics, sub_bridge_relationships, report_file, title_prefix=f"(Refined from {name_prefix})")
    visualize_communities(subgraph, sub_partition, image_file, title_suffix=f"(Refined from {name_prefix})", bridge_nodes=sub_bridge_nodes)
    
    # Recursively call for large sub-communities
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
    # Define constants
    OUTPUT_DIR = "output_deep_recursive"
    LARGE_COMMUNITY_THRESHOLD = 3 # You can adjust this threshold for different levels of granularity

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and parse schema
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
    
    # -------------------------------------------------------------------------
    # Initial Community Detection
    # -------------------------------------------------------------------------
    print("\nStarting initial community detection...")
    initial_communities, initial_partition, sorted_partition = analyze_communities(G)

    # Generate an initial report for the full graph
    print("Generating initial full graph report...")
    initial_community_metrics = calculate_community_metrics(G, initial_communities, sorted_partition)
    initial_bridge_relationships = identify_bridge_relationships(initial_community_metrics)
    generate_report(initial_community_metrics, initial_bridge_relationships, os.path.join(OUTPUT_DIR, "initial_communities_report.txt"), title_prefix="(Initial Full Graph)")
    
    # Create community size chart
    create_community_size_chart(initial_community_metrics, os.path.join(OUTPUT_DIR, "community_sizes.png"))
    
    # New: Analyze bridges for the full graph
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
            f.write(f"{node:<15} | {data['degree']:<6} | {community_labels}\n")
    
    # Visualize the full graph with bridge nodes highlighted
    print("Generating full graph visualization...")
    visualize_communities(G, initial_partition, os.path.join(OUTPUT_DIR, "full_graph_communities.png"), title_suffix="(Full Graph)", bridge_nodes=bridge_nodes.keys())
    
    # -------------------------------------------------------------------------
    # Recursive Refinement
    # -------------------------------------------------------------------------
    print("\nStarting recursive refinement...")
    for i, comm in enumerate(initial_communities):
        comm_size = len(comm)
        if comm_size > LARGE_COMMUNITY_THRESHOLD:
            print(f"Refining community #{i+1} (size: {comm_size})...")
            recursive_refine_communities(G, comm, f"comm_{i+1}", OUTPUT_DIR, LARGE_COMMUNITY_THRESHOLD, comm_size)
        else:
            print(f"Skipping refinement of community #{i+1} (size: {comm_size} - below threshold {LARGE_COMMUNITY_THRESHOLD})")
    
    print(f"\nAnalysis complete! Results saved to '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()