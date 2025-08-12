import numpy as np 
import networkx as nx 
from collections import Counter
import random
import matplotlib.pyplot as plt
from scipy.stats import norm 
from datetime import datetime, timedelta
import os

def validate_and_clamp_ranges(ranges: list, max_possible_value: int):
    """
    Validates and clamps ranges to ensure they don't exceed the maximum possible value.
    
    Args:
        ranges (list): A list of dictionaries with 'start', 'end', and 'prob' keys
        max_possible_value (int): The maximum value that ranges can have
        
    Returns:
        list: A list of validated ranges with clamped start/end values
    """
    validated_ranges = []
    for r in ranges:
        clamped_start = min(r['start'], max_possible_value)
        clamped_end = min(r['end'], max_possible_value)
        
        # Only include range if start is valid
        if clamped_start <= max_possible_value:
            validated_ranges.append({
                'start': clamped_start,
                'end': clamped_end,
                'prob': r['prob']
            })
    
    return validated_ranges

def generate_values_from_ranges_and_normal(max_num_values: int, seed: int, ranges: list, normal_mean: float, normal_std_dev: float):
    """
    Generates a list of values based on user-defined ranges and normal distribution.
    This is a general-purpose function that can be used for any type of value generation.
    
    Args:
        max_num_values (int): The maximum number of values to pick from.
        seed (int): A seed for the random number generator to ensure reproducibility.
        ranges (list): A list of dictionaries, where each dictionary defines a value
                       range and its probability, start and end inclusive. E.g.,
                       [{'start': 0, 'end': 5, 'prob': 0.3}, {'start': 10, 'end': 15, 'prob': 0.2}]
        normal_mean (float): The mean for the normal distribution.
        normal_std_dev (float): The standard deviation for the normal distribution
                                used for values not covered by a user-defined range.
    Returns:
        list: A list of generated values.
    """
    # Seed the random number generators for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Calculate maximum possible value (for degrees, this is max_num_values - 1)
    max_possible_value = max_num_values - 1
    
    # Validate and clamp ranges to max_possible_value
    validated_ranges = validate_and_clamp_ranges(ranges, max_possible_value)
    
    # If no valid ranges, use normal distribution for all values
    if not validated_ranges:
        values_normal = np.random.normal(loc=normal_mean, scale=normal_std_dev, size=max_num_values)
        values_normal = np.round(values_normal).astype(int)
        values_normal = np.clip(values_normal, 0, max_possible_value)
        return values_normal.tolist()

    # Calculate the number of values to be assigned from each method
    num_from_ranges = 0
    for r in validated_ranges:
        num_from_ranges += int(r['prob'] * max_num_values)

    num_from_normal = max_num_values - num_from_ranges

    # Generate the value sequence based on the specified rules
    value_sequence = []

    # A. Generate values from user-defined validated ranges
    for r in validated_ranges:
        num_values_in_range = int(r['prob'] * max_num_values)
        # Use random.randint to select a value within the range for each item
        for _ in range(num_values_in_range):
            if(r['start'] == r['end']):
                value = r['start']
            else:
                value = random.randint(r['start'], r['end'])
            value_sequence.append(value)

    # B. Generate values from a normal distribution for the remaining items
    if num_from_normal > 0:
        # Use numpy.random.normal to generate values, then round and clip to valid range
        values_normal = np.random.normal(loc=normal_mean, scale=normal_std_dev, size=num_from_normal)
        values_normal = np.round(values_normal).astype(int)
        
        # Clip the values to be within the valid range
        values_normal = np.clip(values_normal, 0, max_possible_value)
        
        value_sequence.extend(values_normal.tolist())

    # Ensure we have exactly max_num_values in our sequence
    if len(value_sequence) < max_num_values:
        # If there's a small discrepancy due to rounding, pad with random values
        remaining = max_num_values - len(value_sequence)
        for _ in range(remaining):
            value_sequence.append(random.randint(0, max_possible_value))
    elif len(value_sequence) > max_num_values:
        # If there's a small discrepancy, trim the list
        value_sequence = value_sequence[:max_num_values]

    return value_sequence

def generate_random_bans(my_graph: object, node: int, ban_ranges: list):
    """
    Generates a list of random bans based on range dictionaries with probabilities.
    Args:
        my_graph (object): The graph object from which to generate bans.
        node (int): The node for which to generate bans.
        ban_ranges (list): A list of dictionaries, where each dictionary defines a range
                          and its probability, start and end inclusive. E.g.,
                          [{'start': 0, 'end': 0, 'prob': 0.4}, {'start': 1, 'end': 2, 'prob': 0.3}]
    """

    # Calculate maximum possible bans (all nodes except self and neighbors)
    total_nodes = my_graph.number_of_nodes()
    neighbors_count = len(list(my_graph.neighbors(node)))
    max_possible_bans = total_nodes - neighbors_count - 1  # Exclude self and neighbors
    max_possible_bans = max(0, max_possible_bans)  # Ensure non-negative
    
    # Validate and clamp ranges to max_possible_bans
    validated_ranges = validate_and_clamp_ranges(ban_ranges, max_possible_bans)
    
    # If no valid ranges, return empty list
    if not validated_ranges:
        return []
    
    # Calculate total probability from validated ranges
    total_prob_from_ranges = sum(r['prob'] for r in validated_ranges)
    remaining_prob = 1.0 - total_prob_from_ranges
    
    # Get all values covered by the validated ranges
    covered_values = set()
    for r in validated_ranges:
        for val in range(r['start'], r['end'] + 1):
            covered_values.add(val)
    
    # Find uncovered values up to max_possible_bans
    all_possible_values = set(range(0, max_possible_bans + 1))
    uncovered_values = all_possible_values - covered_values
    
    # Generate random value to select ban count
    rand_val = random.random()
    
    # First, check if it falls within the validated ranges
    cumulative_prob = 0
    for r in validated_ranges:
        cumulative_prob += r['prob']
        if rand_val <= cumulative_prob:
            # Selected this range
            if r['start'] == r['end']:
                num_bans = r['start']
            else:
                num_bans = random.randint(r['start'], r['end'])
            break
    else:
        # If no range was selected, it falls in the remaining probability
        if remaining_prob > 0 and uncovered_values:
            # Evenly distribute remaining probability among uncovered values
            uncovered_list = sorted(list(uncovered_values))
            prob_per_uncovered = remaining_prob / len(uncovered_list)
            
            # Adjust rand_val to be within the remaining probability space
            adjusted_rand = (rand_val - total_prob_from_ranges) / remaining_prob
            
            # Select from uncovered values
            selected_index = int(adjusted_rand * len(uncovered_list))
            selected_index = min(selected_index, len(uncovered_list) - 1)  # Ensure valid index
            num_bans = uncovered_list[selected_index]
        else:
            # Fallback: use the first range or 0 if no ranges
            if validated_ranges:
                first_range = validated_ranges[0]
                if first_range['start'] == first_range['end']:
                    num_bans = first_range['start']
                else:
                    num_bans = random.randint(first_range['start'], first_range['end'])
            else:
                num_bans = 0
    
    # Ensure num_bans doesn't exceed maximum possible (double check)
    num_bans = min(num_bans, max_possible_bans)
    
    # Now generate the actual list of banned node IDs
    if num_bans == 0:
        return []
    
    # Get all possible nodes to ban (excluding self and neighbors)
    all_nodes = list(my_graph.nodes())
    neighbors = set(my_graph.neighbors(node))
    possible_bans = [n for n in all_nodes if n != node and n not in neighbors]
    
    # Randomly select nodes to ban
    banned_nodes = random.sample(possible_bans, min(num_bans, len(possible_bans)))
    
    return banned_nodes

def generate_graph_from_distribution(num_nodes: int, seed: int, ranges: list, default_std_dev: float = 1.0):
    """
    Generates a simple graph with a degree sequence determined by user-defined ranges
    and a normal distribution for the remaining nodes.

    Args:
        num_nodes (int): The number of nodes in the graph.
        seed (int): A seed for the random number generator to ensure reproducibility.
        ranges (list): A list of dictionaries, where each dictionary defines a degree
                       range and its probability,start and end inclusive. E.g.,
                       [{'start': 0, 'end': 5, 'prob': 0.3}, {'start': 10, 'end': 15, 'prob': 0.2}]
        default_std_dev (float): The standard deviation for the normal distribution
                                 used for nodes not covered by a user-defined range.

    Returns:
        A networkx.Graph object if a valid graph can be created, otherwise None.
    """
    # Calculate the mean degree for the normal distribution
    mean_degree = (num_nodes - 1) / 2
    
    # Generate the initial degree sequence using the general function
    degree_sequence = generate_values_from_ranges_and_normal(
        max_num_values=num_nodes,
        seed=seed,
        ranges=ranges,
        normal_mean=mean_degree,
        normal_std_dev=default_std_dev,
    )
    
    # --- Degree-specific validation and adjustment ---
    d_sum = sum(degree_sequence)
    msg = ""
    if d_sum % 2 == 0:
       msg = ", sum is even (satisfies the Handshaking Lemma)"

    print(f"Initial degree sequence, sum: {d_sum}{msg}")
    
    # The Handshaking Lemma states the sum of degrees must be even.
    # We check for this and make a minimal adjustment if needed.
    if sum(degree_sequence) % 2 != 0:
        print("Sum of degrees is odd. Adjusting one random degree, To Satisfy The Handshaking Lemma")
        # Pick a random node and adjust its degree by +/- 1
        idx = random.randint(0, len(degree_sequence) - 1)
        current_degree = degree_sequence[idx]
        
        # Choose to increment or decrement, ensuring the degree stays within valid bounds
        if current_degree < num_nodes - 1:
            degree_sequence[idx] += 1
        elif current_degree > 0:
            degree_sequence[idx] -= 1
        else:
            # If the degree is 0, we can only increment
            degree_sequence[idx] += 1
        
        print(f"Adjusted degree sequence (sum: {sum(degree_sequence)})")

    # --- Construct the graph using the Configuration Model ---
    # The Configuration Model is a common method to build a graph from a degree sequence.
    # It first creates a multigraph, which may have self-loops and multi-edges.
    try:
        graph = nx.configuration_model(degree_sequence)
        
        # We need to convert the multigraph to a simple graph
        # by removing self-loops and parallel edges.
        graph = nx.Graph(graph)  # This automatically removes multi-edges
        graph.remove_edges_from(nx.selfloop_edges(graph)) # This removes self-loops
        
        return graph

    except nx.NetworkXUnfeasible as e:
        print(f"Could not generate a graph from the degree sequence. {e}")
        return None

def select_weight_from_probabilities(weight_probabilities: dict):
    """
    Selects a weight based on probability distribution.
    
    Args:
        weight_probabilities (dict): A dictionary where keys are weights and values are probabilities
                                   E.g., {1: 0.4, 2: 0.35, 3: 0.25}
    
    Returns:
        int: Selected weight based on the probability distribution
    """
    # Validate weight probabilities
    valid_weights = [1, 2, 3]
    for weight in weight_probabilities.keys():
        if weight not in valid_weights:
            raise ValueError(f"Weight {weight} is not valid. Weights must be between 1 and 3 inclusive.")
    
    # Normalize probabilities if they don't sum to 1
    total_prob = sum(weight_probabilities.values())
    if total_prob != 1.0:
        normalized_probs = {w: p/total_prob for w, p in weight_probabilities.items()}
    else:
        normalized_probs = weight_probabilities
    
    # Create cumulative probability distribution for weight selection
    weights = list(normalized_probs.keys())
    probabilities = list(normalized_probs.values())
    cumulative_probs = []
    cumsum = 0
    for prob in probabilities:
        cumsum += prob
        cumulative_probs.append(cumsum)
    
    # Select weight based on probabilities
    rand_val = random.random()
    for i, cum_prob in enumerate(cumulative_probs):
        if rand_val <= cum_prob:
            return weights[i]
    return weights[-1]  # Fallback to last weight

def is_node_available(availability_prob=0.8):
    return random.random() < availability_prob

def generate_node_data(graph, seed=42):
    """
    Generates dummy data for each node in the graph.
    This function populates the graph with attributes like connections, bans, group IDs, etc.
    
    Args:
        graph: The NetworkX graph object to add node data to
        seed: Random seed for reproducible results (optional)
    """
    random.seed(seed)
    types = ["All", "Half+", "One+"]
    
    # Define ban ranges using list of dictionaries (probabilities don't need to sum up to 1)
    ban_ranges = [
        {'start': 0, 'end': 0, 'prob': 0.4},    # 40% get exactly 0 bans
        {'start': 1, 'end': 2, 'prob': 0.3},    # 30% get 1-2 bans
        {'start': 3, 'end': 10, 'prob': 0.1}    # 10% get 3-10 bans
        # Remaining 20% will be evenly distributed among uncovered values
    ]

    # Affinity is the same as weight. (preferably sum up probabilities to 1)
    min_affinity_constraint_probabilities = select_weight_from_probabilities({
            1: 0,
            2: 1,
            3: 0
        })
    
    for node in graph.nodes():
        graph.nodes[node]["waiting_since"] = datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        graph.nodes[node]["available"] = is_node_available(availability_prob=0.8)  
        graph.nodes[node]["mac"] = min_affinity_constraint_probabilities
        graph.nodes[node]["irl_rated_nodes"] = {}  # Node ID : Rating (-2 to 2) inclusive
        graph.nodes[node]["groupIDs"] = []
        # Testing Only (The resulting groupIDs are unrealistic)
        #graph.nodes[node]["groupIDs"] = random.sample(range(1, 10), random.randint(0, 5))  # Up to 5 arbitrary group IDs
        graph.nodes[node]["bans"] = generate_random_bans(graph, node, ban_ranges)
        graph.nodes[node]["type"] = random.choice(types)

def generate_edge_data(graph, weight_probabilities: dict):
    """
    Generates weight data for each edge in the graph.
    Each edge gets a dictionary with node IDs as keys and their assigned weights as values.
    
    Args:
        graph: The NetworkX graph object to add edge data to
        weight_probabilities (dict): A dictionary where keys are weights (1-3) and values are probabilities
                                   E.g., {1: 0.4, 2: 0.35, 3: 0.25}
    """
    # Generate edge data for each edge
    for edge in graph.edges():
        u, v = edge
        
        # Assign weights to both nodes of the edge using the reusable function
        w1 = select_weight_from_probabilities(weight_probabilities)
        w2 = select_weight_from_probabilities(weight_probabilities)
        
        # Create edge data dictionary
        edge_data = {
            u: w1,
            v: w2
        }
        
        # Add the data to the edge
        graph.edges[edge]["weight"] = edge_data

def display_graph_stats(my_graph):
    print("\n--- Graph Generation Stats ---")
    print(f"Number of nodes: {my_graph.number_of_nodes()}")
    print(f"Number of edges: {my_graph.number_of_edges()}")

def display_degree_stats(my_graph):
    print("\n--- Degree Distribution Stats ---")

    # Calculate mean degree
    degrees = [d for n, d in my_graph.degree()]
    mean_degree = sum(degrees) / len(degrees)
    num_nodes = my_graph.number_of_nodes()
    print(f"Mean degree: {mean_degree:.2f} ({mean_degree / (num_nodes - 1):.2%})")

    # Calculate median degree
    median_degree = sorted(degrees)[len(degrees) // 2]
    print(f"Median degree: {median_degree} ({median_degree / (num_nodes - 1):.2%})")

    # Display the counts of each degree to see the distribution
    degree_counts = Counter([d for n, d in my_graph.degree()])
    print("Degree distribution:")
    for degree, count in sorted(degree_counts.items()):
        print(f"  - Degree {degree}: {count} nodes")


def display_ban_stats(my_graph):
    print("\n--- Ban Distribution Stats ---")
    
    # Get the number of bans for all nodes
    ban_counts = [len(my_graph.nodes[node]["bans"]) for node in my_graph.nodes()]
    
    # Calculate mean number of bans
    mean_bans = sum(ban_counts) / len(ban_counts)
    print(f"Mean bans per node: {mean_bans:.2f}")
    
    # Calculate median number of bans
    median_bans = sorted(ban_counts)[len(ban_counts) // 2]
    print(f"Median bans per node: {median_bans}")
    
    # Calculate min and max bans
    min_bans = min(ban_counts)
    max_bans = max(ban_counts)
    print(f"Min bans: {min_bans}, Max bans: {max_bans}")
    
    # Display the counts of each ban number to see the distribution
    ban_distribution = Counter(ban_counts)
    print("Ban count distribution:")
    for ban_count, node_count in sorted(ban_distribution.items()):
        print(f"  - {ban_count} bans: {node_count} nodes")

def display_edge_weight_stats(my_graph):
    print("\n--- Edge Weight Distribution Stats ---")
    
    if my_graph.number_of_edges() == 0:
        print("No edges in the graph.")
        return
    
    # Collect all weights from all edges
    all_weights = []
    for edge in my_graph.edges():
        if "weight" in my_graph.edges[edge]:
            edge_data = my_graph.edges[edge]["weight"]
            for node, weight in edge_data.items():
                all_weights.append(weight)
    
    if not all_weights:
        print("No edge weight data found.")
        return
    
    # Calculate statistics
    mean_weight = sum(all_weights) / len(all_weights)
    print(f"Mean edge weight: {mean_weight:.2f}")
    
    median_weight = sorted(all_weights)[len(all_weights) // 2]
    print(f"Median edge weight: {median_weight}")
    
    min_weight = min(all_weights)
    max_weight = max(all_weights)
    print(f"Min weight: {min_weight}, Max weight: {max_weight}")
    
    # Display weight distribution
    weight_distribution = Counter(all_weights)
    print("Weight distribution:")
    for weight, count in sorted(weight_distribution.items()):
        percentage = (count / len(all_weights)) * 100
        print(f"  - Weight {weight}: {count} assignments ({percentage:.1f}%)")
    
    # Display edge-level statistics
    edge_weight_sums = []
    for edge in my_graph.edges():
        if "weight" in my_graph.edges[edge]:
            edge_data = my_graph.edges[edge]["weight"]
            edge_sum = sum(edge_data.values())
            edge_weight_sums.append(edge_sum)
    
    if edge_weight_sums:
        mean_edge_sum = sum(edge_weight_sums) / len(edge_weight_sums)
        print(f"\nMean total weight per edge: {mean_edge_sum:.2f}")
        
        edge_sum_distribution = Counter(edge_weight_sums)
        print("Edge total weight distribution:")
        for total, count in sorted(edge_sum_distribution.items()):
            percentage = (count / len(edge_weight_sums)) * 100
            print(f"  - Total {total}: {count} edges ({percentage:.1f}%)")

def plot_degree_distribution(my_graph):
    # Get the degrees of all nodes
    degrees = [d for n, d in my_graph.degree()]

    # Ensure output directory exists
    os.makedirs("outputs/generation_analysis", exist_ok=True)
    # Create a histogram of the degree distribution
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=np.arange(min(degrees), max(degrees) + 1, 1), density=True, alpha=0.6, color='skyblue', label='Degree Distribution')

    # Fit a normal distribution to the degree data only if there's variation
    if len(set(degrees)) > 1:  # Check if there's more than one unique value
        mu, std = norm.fit(degrees)
        
        # Only plot the fitted curve if std > 0
        if std > 0:
            # Plot the PDF of the fitted normal distribution
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2, label=f'Fitted Normal Distribution (μ={mu:.2f}, σ={std:.2f})')
        else:
            plt.axvline(x=mu, color='k', linewidth=2, label=f'All values equal: {mu:.2f}')
    else:
        # All values are the same
        unique_value = degrees[0]
        plt.axvline(x=unique_value, color='k', linewidth=2, label=f'All values equal: {unique_value}')

    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Degree Distribution with Fitted Normal Curve")
    plt.legend()
    plt.grid(True)

    # Save the plot as an image file
    plt.savefig("outputs/generation_analysis/degree_distribution.png")
    plt.close()

def plot_ban_distribution(my_graph):
    # Get the number of bans for all nodes
    ban_counts = [len(my_graph.nodes[node]["bans"]) for node in my_graph.nodes()]

    # Ensure output directory exists
    os.makedirs("outputs/generation_analysis", exist_ok=True)
    # Create a histogram of the ban distribution
    plt.figure(figsize=(10, 6))
    plt.hist(ban_counts, bins=np.arange(min(ban_counts), max(ban_counts) + 1, 1), density=True, alpha=0.6, color='red', label='Ban Distribution')

    # Fit a normal distribution to the ban data only if there's variation
    if len(set(ban_counts)) > 1:  # Check if there's more than one unique value
        mu, std = norm.fit(ban_counts)
        
        # Only plot the fitted curve if std > 0
        if std > 0:
            # Plot the PDF of the fitted normal distribution
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2, label=f'Fitted Normal Distribution (μ={mu:.2f}, σ={std:.2f})')
        else:
            plt.axvline(x=mu, color='k', linewidth=2, label=f'All values equal: {mu:.2f}')
    else:
        # All values are the same
        unique_value = ban_counts[0]
        plt.axvline(x=unique_value, color='k', linewidth=2, label=f'All values equal: {unique_value}')

    plt.xlabel("Number of Bans")
    plt.ylabel("Frequency")
    plt.title("Ban Distribution with Fitted Normal Curve")
    plt.legend()
    plt.grid(True)

    # Save the plot as an image file
    plt.savefig("outputs/generation_analysis/ban_distribution.png")
    plt.close()

# Visualize the graph
def visualize_graph(my_graph):
    # Ensure output directory exists
    os.makedirs("outputs/generation_analysis", exist_ok=True)
    plt.figure(figsize=(12, 8))
    
    # Create a temporary graph without edge attributes for layout calculation
    temp_graph = nx.Graph()
    temp_graph.add_nodes_from(my_graph.nodes())
    temp_graph.add_edges_from(my_graph.edges())
    
    # Create layout for consistent node positioning
    np.random.seed(42)
    pos = nx.spring_layout(temp_graph, k=2, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(my_graph, pos, node_color='skyblue', node_size=300)
    
    # Draw node labels
    nx.draw_networkx_labels(my_graph, pos, font_size=8)
    
    # Draw edges
    nx.draw_networkx_edges(my_graph, pos, edge_color='k', width=1)
    
    # Create edge labels with weight sums
    edge_labels = {}
    for edge in my_graph.edges():
        if "weight" in my_graph.edges[edge]:
            weight_data = my_graph.edges[edge]["weight"]
            weight_sum = sum(weight_data.values())
            edge_labels[edge] = str(weight_sum)
        else:
            edge_labels[edge] = "0"
    
    # Draw edge labels
    nx.draw_networkx_edge_labels(my_graph, pos, edge_labels, font_size=6, font_color='darkblue')
    
    plt.title("Generated Graph Visualization with Edge Weight Sums")
    plt.axis('off')  # Turn off axis for cleaner look

    # Save the plot as an image file
    plt.savefig("outputs/generation_analysis/graph_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()


# --- Example Usage ---

def generate_graph(graph_size=50, seed=42, st_dev=10):
    """
    Generates a graph with a specified size, random seed, degree ranges, and edge weight probabilities.
    Returns:
        A networkx.Graph object if the graph is generated successfully, otherwise None.
    """

    # Define parameters for the graph generation
    GRAPH_SIZE = graph_size
    RANDOM_SEED = seed
    DEFAULT_STDEV = st_dev
    DEGREE_RANGES = [
        {'start': 0, 'end': 1, 'prob': 0.25},
        {'start': GRAPH_SIZE - 2, 'end': GRAPH_SIZE - 1, 'prob': 0.1},
        # The remaining % of nodes will be assigned a degree from the normal distribution
    ]
    # Edge weight probabilities try to make them sum up to 1 (not mandatory but preferred)
    # Edge weight : probability mapping
    EDGE_WEIGHT_PROBABILITIES = {
        1: 0.4, 
        2: 0.35, 
        3: 0.25   
    }

    # Generate the graph
    print("--- Generating Graph ---")

    my_graph = generate_graph_from_distribution(
        num_nodes=GRAPH_SIZE,
        seed=RANDOM_SEED,
        ranges=DEGREE_RANGES,
        default_std_dev=DEFAULT_STDEV
    )

    generate_node_data(my_graph, seed=RANDOM_SEED)
    generate_edge_data(my_graph, EDGE_WEIGHT_PROBABILITIES)

    print("Graph generated successfully.")
    return my_graph

def generate_graph_statistics(my_graph):
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    display_graph_stats(my_graph)
    display_degree_stats(my_graph)
    display_ban_stats(my_graph)
    display_edge_weight_stats(my_graph)
    plot_degree_distribution(my_graph)
    plot_ban_distribution(my_graph)
    visualize_graph(my_graph)

if __name__ == "__main__":
    my_graph = generate_graph()

    if my_graph:
        generate_graph_statistics(my_graph)
    else:
        print("Graph generation failed.")