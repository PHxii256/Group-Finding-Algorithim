import math 
import logging
import random
import os
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
from algoGraphGenerator import generate_graph

#------------------------ Constants And Setup ------------------------

# The size of the group we are trying to form
GROUP_SIZE = 4 

# The maximum number of premade nodes allowed per group
MAX_PREMADE_NODES_PER_GROUP = 2 

# The maximum number of groups allowed per node for the current iteration / run
MAX_GROUPS_PER_NODE = 3

# Seed for random number generation
SEED = 42

# Generate a graph for testing
my_graph = generate_graph(seed=SEED, graph_size=50)

# Set of nodes that cannot be grouped for current iteration
# Unused potential optimization to speed but not needed for now
ungroupable_nodes = set()

# Dictionary to track failure reasons
failure_reasons = {
    1: "Max groups per node exceeded",
    2: "Dissimilar groups constraint violated", 
    3: "Mutual positive IRL ratings required",
    4: "Type requirements not met",
    5: "Path not feasible",
    6: "No valid MAC picks"
}

# Counter for each failure type
failure_counts = Counter()

# Set up logging to file
log_filename = f"outputs/algo_analysis/group_finding_output_log_(seed_{SEED}).txt"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Log the start of the algorithm
logger.info(f"🔎 Starting group finding algorithm (Seed: {SEED})")
# Track start time
start_time = datetime.now()

#------------------------ Utility Functions ------------------------

# Remove nodes that are not available wheather self imposed or done automatically, 
# when it was added to the max no. of groups for a cooldown period
def remove_unavailable_nodes(my_graph):
    removed_count = 0
    # Get all nodes that are not available
    nonAvailable_nodes = [node for node, data in my_graph.nodes(data=True) if not data.get('available', True)]

    for node in nonAvailable_nodes:
        my_graph.remove_node(node)
        removed_count += 1
    logger.info(f"• Removed {removed_count} unavailable nodes out of {len(my_graph.nodes()) + removed_count}.")
    return removed_count

# Calculate the sum of weights between two nodes
def edge_total_weight(node_id, frontier_id):
    return sum(my_graph.get_edge_data(frontier_id, node_id)['weight'].values())

# Calculate the sum of weights between all of path's nodes and the given node
def path_total_weight(node_id, path):
    weight_sum = 0
    for path_node_id in path:
        if my_graph.has_edge(path_node_id, node_id):
            weight_sum += sum(my_graph.get_edge_data(path_node_id, node_id)['weight'].values())
    return weight_sum

def get_path_bans(path):
    bans_set = set()
    for node_id in path:
        bans_set.update(my_graph.nodes[node_id]['bans'])
    return bans_set

# Utility function to get all nodes belonging to a specific group ID
def get_nodes_by_group_id(group_id):
    return [node for node, data in my_graph.nodes(data=True) if group_id in data.get('groupIDs', [])]

# Temporary function to generate a unique group ID
def generate_unique_group_id():
    # Collect all used groupIDs in the graph
    used_ids = set()
    for _, data in my_graph.nodes(data=True):
        used_ids.update(data.get('groupIDs', []))
    # Try random IDs under 10000 until a unique one is found
    attempts = 0
    while attempts < 1000:  # avoid infinite loop
        candidate = random.randint(1, 9999)
        if candidate not in used_ids:
            return candidate
        attempts += 1
    raise RuntimeError("Could not generate a unique group ID under 10000.")

def get_type_connection_count(node_type):
    type_connection_count = 0 
    if node_type == 'One+':
        type_connection_count = 1
    elif node_type == 'Half+':
        type_connection_count = math.ceil((GROUP_SIZE - 1)/ 2)
    elif node_type == 'All':
        type_connection_count = GROUP_SIZE - 1
    if type_connection_count == 0:
        raise ValueError(f"Type connection count is 0 (unexpected), node type: {node_type}.")
    return type_connection_count

def get_node_conn_count_with_path(node_id, path):
    conn_count_with_path = len(set(my_graph.neighbors(node_id)) & set(path))
    return conn_count_with_path

def get_node_picks(node_id, path, bans = None, include_path=False):
    if bans is None:
        bans = get_path_bans(path)
    if include_path:
        return set(my_graph.neighbors(node_id)) - get_path_bans(path)
    return set(my_graph.neighbors(node_id)) - set(path) - get_path_bans(path)


def is_path_feasible(path):
    # Check if the remaining connections for the path are sufficient
    # GROUP_SIZE - len(path) is the number of connections left to fill for this path
    
    for node_id in path:
        remaining_connections = get_node_remaining_conn_count(node_id, path, print_debug=True)

        if GROUP_SIZE - len(path) < remaining_connections:
            return False
    
    # Additional check: If we're at full group size, validate type requirements
    if len(path) == GROUP_SIZE:
        return validate_group_type_requirements(path)
    
    return True

def get_node_remaining_conn_count(node_id, path, print_debug=False):
    type_connection_count = get_type_connection_count(my_graph.nodes[node_id]['type'])
    connection_count_with_path = get_node_conn_count_with_path(node_id, path)
    remaining_connections =  max(type_connection_count - connection_count_with_path, 0)
    if print_debug:
        logger.info(f"Node {node_id} of type {my_graph.nodes[node_id]['type']} that needs {type_connection_count} has {remaining_connections} r_conns with path: {path}.")
    return remaining_connections

def pass_group_constraints(node_id, original_path):
    path = original_path.copy()  # Create a copy of the path to avoid modifying the original
    path.append(node_id)
    path_groups_list = []
    path_group_count_dict = {}
    violating_nodes = set()

    # Max groups per node check
    for node in path:
        if len(my_graph.nodes[node]["groupIDs"]) >= MAX_GROUPS_PER_NODE:
            failure_counts[1] += 1
            logger.info(f"❌ group constraint violated: node {node} is already in {len(my_graph.nodes[node]['groupIDs'])} groups (max allowed: {MAX_GROUPS_PER_NODE}) per iteration")
            return False
        path_groups_list.extend(my_graph.nodes[node]["groupIDs"])

    # Groups dissimilarity constraint: if a group ID appears more than MAX_PREMADE_NODES_PER_GROUP times in the path, it violates the constraint
    for group_id in path_groups_list:
        # Count how many times this group ID appears in the path
        path_group_count_dict[group_id] = path_group_count_dict.get(group_id, 0) + 1
        if path_group_count_dict[group_id] > MAX_PREMADE_NODES_PER_GROUP:
            violating_nodes = [n for n in path if group_id in my_graph.nodes[n]["groupIDs"]]
            failure_counts[2] += 1
            logger.info(f"❌ group constraint violated with node {node_id} (dissimilar groups constraint). Violating Group: {get_nodes_by_group_id(group_id)} (ID: {group_id}), nodes: {violating_nodes}")
            return False  # Node fails group constraints

    # (+ve) group rating constraint: if two nodes in the path share a group, both must have a +ve irl rating towards each other
    for i, node_a in enumerate(path):
        for node_b in path[i+1:]:
            shared_groups = set(my_graph.nodes[node_a]["groupIDs"]) & set(my_graph.nodes[node_b]["groupIDs"])
            if shared_groups:
                irl_a_to_b = my_graph.nodes[node_a]["irl_rated_nodes"].get(node_b, 0)
                irl_b_to_a = my_graph.nodes[node_b]["irl_rated_nodes"].get(node_a, 0)
                if irl_a_to_b <= 0 or irl_b_to_a <= 0:
                    failure_counts[3] += 1
                    logger.info(f"❌ group constraint violated: nodes {node_a} and {node_b} share group(s) {shared_groups} but do not have mutual +ve irl ratings (A→B: {irl_a_to_b}, B→A: {irl_b_to_a})")
                    return False

    return True  # Node passes group constraints

# Get list of path's nodes which are mandatory to have their picks included in the next selection
def get_mandatory_nodes_picks(path, path_bans = set()):
    m_nodes_picks = set()

    for node_id in path:
        r_conn_count = get_node_remaining_conn_count(node_id, path)
        if r_conn_count == GROUP_SIZE - len(path):
            m_nodes_picks.update(set(my_graph.neighbors(node_id)))

        elif r_conn_count > GROUP_SIZE - len(path):
            raise ValueError(f"ERROR: Node {node_id}'s remaining connections > group_size - len(path).")

    return m_nodes_picks - path_bans - set(path)

def get_frontier_picks(path, path_bans= None, frontier=None):
    if frontier is None:
        frontier = path[-1]
    if path_bans is None:    
        path_bans = get_path_bans(path)
    frontier_picks = get_node_picks(frontier, path, bans=path_bans)
    m_picks = get_mandatory_nodes_picks(path, path_bans)  # Get mandatory nodes picks
   
    # If there are mandatory picks, filter frontier picks to only include common / intersecting picks
    return (frontier_picks & m_picks) - set(path) 

def pass_mac(frontier_id, pick_id):
    edge_data = my_graph.get_edge_data(frontier_id, pick_id)
    if edge_data is None:
        # print(f"Edge data for {frontier_id} to {pick_id} is None, skipping MAC check.")
        return False
    if edge_data['weight'][pick_id] >= my_graph.nodes[frontier_id]['mac']:
        return True
    else:  
        return False

# Get valid MAC picks for the current frontier node
def get_valid_sorted_mac_pics(frontier_id, path):
    valid_mac_pics = []
    # mac picks here should be outside of the path
    for pick_id in get_frontier_picks(path, frontier=path[-1]):
        if pass_mac(frontier_id, pick_id):
            if pass_group_constraints(pick_id, path):
                valid_mac_pics.append(pick_id)
    return sorted(valid_mac_pics, key=lambda x: edge_total_weight(x, frontier_id), reverse=False)

# Get valid picks for all of current path (No MAC check)
def get_valid_sorted_path_pics(path):
    path_pics = set()
    valid_pics = []
    bans = get_path_bans(path)

    for node_id in path:
        path_pics.update(get_frontier_picks(path, path_bans=bans, frontier=node_id))

    for pick_id in path_pics:
        if pass_group_constraints(pick_id, path):
            valid_pics.append(pick_id)
    return sorted(valid_pics, key=lambda x: path_total_weight(x, path), reverse=False)

def validate_group_mac_requirements(group):
    """
    Validates that all nodes in the group meet their MAC requirements.
    Returns True if all nodes satisfy their MAC requirements, False otherwise.
    """
    logger.info(f"  Validating MAC requirements for group: {group}")
    for node in group:
        node_type = my_graph.nodes[node]['type']
        type_requirement = get_type_connection_count(node_type)
        
        # Count actual connections to other group members
        actual_connections = 0
        for other_node in group:
            if other_node != node and my_graph.has_edge(node, other_node):
                actual_connections += 1
        
        if actual_connections < type_requirement:
            logger.info(f"   ❌ Node {node} (Type: {node_type}) needs {type_requirement} connections but only has {actual_connections}")
            return False
        else:
            logger.info(f"   ✅ Node {node} (Type: {node_type}) needs {type_requirement} connections and has {actual_connections}")

    return True

def validate_group_type_requirements(group):
    """
    Validates that all nodes in the group meet their type requirements.
    Returns True if all nodes satisfy their requirements, False otherwise.
    """
    logger.info(f"  Validating Type requirements for group: {group}")
    for node in group:
        node_type = my_graph.nodes[node]['type']
        type_requirement = get_type_connection_count(node_type)
        
        # Count actual connections to other group members
        actual_connections = 0
        for other_node in group:
            if other_node != node and my_graph.has_edge(node, other_node):
                actual_connections += 1
        
        if actual_connections < type_requirement:
            logger.info(f"   ❌ Node {node} (Type: {node_type}) needs {type_requirement} connections but only has {actual_connections}")
            return False
        else:
            logger.info(f"   ✅ Node {node} (Type: {node_type}) needs {type_requirement} connections and has {actual_connections}")
    
    return True

def dfs_iterative(start_node):
    loop_count = 0
    visited = set()
    # Stack stores (current_node, path_to_reach_node, remaining_mac_picks_for_this_level)
    stack = [(start_node, [], [])]  # path as list to maintain order
    logger.info(f"=== Starting DFS with seed node: {start_node} ===")

    while stack:
        loop_count += 1
        if loop_count > 10000:  # Prevent accidental infinite loops
            logger.warning("Loop count exceeded 10,000, breaking out of DFS.")
            break
       
        logger.info(f"--- Loop {loop_count} ---")
        current_node, current_path, remaining_picks = stack.pop()
        logger.info(f"POPPED: node={current_node}, path={current_path}, remaining_picks={remaining_picks}")
        
        # If we have remaining picks for this level, try the next one
        if remaining_picks:
            next_pick = remaining_picks.pop(0)  # Take the best remaining pick
            logger.info(f"Trying next pick from current level: {next_pick}")
            # Put the current state back on stack with remaining picks
            stack.append((current_node, current_path, remaining_picks))
            # Add the next pick to explore
            stack.append((next_pick, current_path + [current_node], []))
            continue
        
        if current_node in visited:
            logger.info(f"Node {current_node} already visited, backtracking...")
            continue
            
        visited.add(current_node)
        new_path = current_path + [current_node]
        logger.info(f"Exploring node {current_node}, new path: {new_path}")
        
        # Check if we've reached the target group size
        if len(new_path) == GROUP_SIZE:
            # Validate that all nodes in the group meet their type requirements
            if validate_group_type_requirements(new_path):
                logger.info(f"🎉 FOUND COMPLETE GROUP: {new_path}")
                return new_path  # Return the complete group
            else:
                failure_counts[4] += 1
                logger.info(f"❌ Group {new_path} rejected - not all members meet type requirements")
                continue  # This path doesn't form a valid group, backtrack
        
        # Check if path is still feasible
        if not is_path_feasible(new_path):
            failure_counts[5] += 1
            logger.info(f"Path {new_path} is not feasible, backtracking...")
            continue
        
        # Get valid MAC picks for the current node
        mac_picks = get_valid_sorted_mac_pics(current_node, new_path)
        logger.info(f"Valid MAC picks for {current_node}: {mac_picks}")
        
        if mac_picks:
            # Take the best pick and put the rest as remaining picks
            best_pick = mac_picks[0]
            remaining_mac_picks = mac_picks[1:]
            
            logger.info(f"Best pick: {best_pick}, remaining: {remaining_mac_picks}")
            
            # If there are remaining picks, put them on stack for backtracking
            if remaining_mac_picks:
                stack.append((current_node, new_path[:-1], remaining_mac_picks))
                logger.info(f"Added backtrack option: node={current_node}, remaining_picks={remaining_mac_picks}")
            
            # Explore the best pick
            stack.append((best_pick, new_path, []))
            logger.info(f"Exploring best pick: {best_pick} with path: {new_path}")
        else:
            failure_counts[6] += 1
            logger.info(f"No valid MAC picks for {current_node}, backtracking...")
            # This path is exhausted, will backtrack automatically
        
        logger.info(f"Stack state: {[(node, path, picks) for node, path, picks in stack]}")

    logger.info(f"DFS complete in {loop_count} iterations. No complete group found.")
    return None

# Detailed analysis of each group
def analyze_group_details(group, group_number):
    logger.info(f"{'='*60}")
    logger.info(f"📊 DETAILED ANALYSIS FOR GROUP {group_number}: {group}")
    # Calculate connectivity score: sum of all edge weights between group members
    connectivity_score = 0
    for i, node1 in enumerate(group):
        for node2 in group[i+1:]:
            if my_graph.has_edge(node1, node2):
                # Sum both directions 
                edge_data = my_graph.get_edge_data(node1, node2)
                # Defensive: check for 'weight' dict and both nodes
                if edge_data and 'weight' in edge_data:
                    w1 = edge_data['weight'].get(node2, 0)
                    w2 = edge_data['weight'].get(node1, 0)
                    connectivity_score += w1 + w2
    logger.info(f"🔗 Connectivity Score (sum of all edge weights in group): {connectivity_score}")
    logger.info(f"{'='*60}")
    
    for node in group:
        node_data = my_graph.nodes[node]
        node_type = node_data['type']
        node_mac = node_data['mac']

        # New: number of bans and number of connections with all nodes in the graph
        num_bans = len(node_data.get('bans', []))
        num_connections = len(list(my_graph.neighbors(node)))

        logger.info(f"🔹 NODE {node} (Type: {node_type}, MAC: {node_mac})")
        logger.info(f"    • General Stats: No. Connections to graph: {num_connections} | Bans: {num_bans}")
        logger.info(f"    {'-' * 56}")
        logger.info(f"    • Connections to other group members:")

        # Find connections to other nodes in the group
        connected_to = []
        not_connected_to = []

        for other_node in group:
            if other_node != node:
                if my_graph.has_edge(node, other_node):
                    # Get the weight/rating for this connection
                    edge_data = my_graph.get_edge_data(node, other_node)
                    weight_to_other = edge_data['weight'][other_node]
                    weight_from_other = edge_data['weight'][node]

                    connected_to.append({
                        'node': other_node,
                        'weight_to': weight_to_other,
                        'weight_from': weight_from_other,
                        'type': my_graph.nodes[other_node]['type'],
                        'mac': my_graph.nodes[other_node]['mac']
                    })
                else:
                    not_connected_to.append({
                        'node': other_node,
                        'type': my_graph.nodes[other_node]['type'],
                        'mac': my_graph.nodes[other_node]['mac']
                    })

        # Print connections
        if connected_to:
            for conn in connected_to:
                logger.info(f"     ✓ Connected to Node {conn['node']} (Type: {conn['type']}, MAC: {conn['mac']})")
                logger.info(f"       • {node} → {conn['node']}: weight {conn['weight_to']}")
                logger.info(f"       • {conn['node']} → {node}: weight {conn['weight_from']}")

        # Print missing connections
        if not_connected_to:
            logger.info(f"     ❌ NOT connected to:")
            for no_conn in not_connected_to:
                logger.info(f"       • Node {no_conn['node']} (Type: {no_conn['type']}, MAC: {no_conn['mac']})")

        # Calculate connection statistics for this node
        total_possible_connections = len(group) - 1
        actual_connections = len(connected_to)
        connection_percentage = (actual_connections / total_possible_connections) * 100 if total_possible_connections > 0 else 0

        logger.info(f"     📈 Connection Stats: {actual_connections}/{total_possible_connections} ({connection_percentage:.1f}%)")

        # Check if node meets its type requirements
        type_requirement = get_type_connection_count(node_type)
        meets_requirement = actual_connections >= type_requirement
        status = "✅ MEETS" if meets_requirement else "❌ FAILS"
        logger.info(f"     🎯 Type Requirement: Needs {type_requirement}, Has {actual_connections} ({status})")


def analyze_failure_reasons():
    """
    Creates a bar chart showing failure causes as percentage of total failures.
    """
    # Calculate total failures
    total_failures = sum(failure_counts.values())
    
    if total_failures == 0:
        logger.info("📊 No failures found.")
        return
        
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs/algo_analysis", exist_ok=True)

    # Prepare data for plotting
    failure_ids = []
    failure_percentages = []
    
    for failure_id, count in failure_counts.items():
        if count > 0:
            percentage = (count / total_failures) * 100
            failure_ids.append(failure_id)
            failure_percentages.append(percentage)
            
    # Create the bar chart
    plt.figure(figsize=(16, 10))
    bars = plt.bar(range(len(failure_ids)), failure_percentages, color='indianred', alpha=1, zorder=2)

    # Calculate max bar height for ylim padding
    max_height = max(failure_percentages) if failure_percentages else 1
    ylim_top = max_height + 10  # Add 10% extra space above the tallest bar
    plt.ylim(0, ylim_top)

    # Customize the plot
    plt.xlabel('Failure Reason ID', fontsize=20, labelpad=20)
    plt.ylabel('Percentage of Total Failures (%)', fontsize=20, labelpad=20)
    plt.title(f'Group Finding Algorithm - Failure Reasons Analysis - (Total Failures: {total_failures})', fontsize=24, pad=30)
    plt.xticks(range(len(failure_ids)), [f"ID {fid}" for fid in failure_ids], fontsize=16)
    plt.yticks(fontsize=16)

    # Add percentage labels on top of bars, with extra padding above the bar
    for i, (bar, percentage) in enumerate(zip(bars, failure_percentages)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (ylim_top * 0.02), 
                f'{percentage:.1f}%', ha='center', va='bottom', fontsize=18)

    # Add count labels inside bars
    for i, (bar, failure_id) in enumerate(zip(bars, failure_ids)):
        count = failure_counts[failure_id]
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                f'{count}', ha='center', va='center', color='white', fontsize=18)
    
    # Create legend showing failure reason descriptions
    legend_text = []
    for fid in failure_ids:
        legend_text.append(f"ID {fid}: {failure_reasons[fid]}")
    
    plt.legend(bars, legend_text, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=18)
    
    plt.grid(axis='y', alpha=1, zorder=0)
    plt.tight_layout()
    
    # Save the plot
    chart_path = "outputs/algo_analysis/failure_reasons_analysis.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', pad_inches=0.66)
    plt.close()
    
    # Log the results
    logger.info(f"📊 FAILURE REASONS ANALYSIS")
    logger.info(f"Total failures detected: {total_failures}")
    logger.info(f"Failure breakdown:")
    
    for failure_id in sorted(failure_counts.keys()):
        count = failure_counts[failure_id]
        if count > 0:
            percentage = (count / total_failures) * 100
            logger.info(f"  ID {failure_id}: {count} occurrences ({percentage:.1f}%) - {failure_reasons[failure_id]}")
            
    logger.info(f"📈 Failure analysis chart saved to: {chart_path}")


def main():
    print("--- Starting group finding algorithm ---")
    remove_unavailable_nodes(my_graph)

    # Seeds are the potential starting nodes for a group
    # We sort the nodes by their 'waiting_since' attribute longest waiting time first
    # Waiting_since must not be None, it's the time since the node was last added to a group
    # but if it was never added, it's value will be the time since added to the graph 
    # i.e showing interest in the activity this graph represents
    sorted_seeds = sorted(my_graph.nodes.data("waiting_since"), key=lambda x: x[1])

    groups_found = []
    group_ids_found = []
    for node in sorted_seeds:
        node_id = node[0]
        result = dfs_iterative(node_id)
        if result:
            # Generate a unique group ID
            new_group_id = generate_unique_group_id()
            logger.info(f"✅ Group Created (ID {new_group_id})")
            # Assign this group ID to each member's groupIDs
            for member in result:
                group_ids = my_graph.nodes[member].get('groupIDs', [])
                if new_group_id not in group_ids:
                    group_ids.append(new_group_id)
                    my_graph.nodes[member]['groupIDs'] = group_ids
            groups_found.append(result)
            group_ids_found.append(new_group_id)
            # Remove only the seed node from available nodes for next iteration
            sorted_seeds = [s for s in sorted_seeds if s[0] != node_id]
            logger.info(f"Assigned group ID {new_group_id} to group: {result}")
            logger.info(f"Remaining seeds: {[s[0] for s in sorted_seeds]}")
        else:
            logger.info(f"❌ No group found starting from {node_id}")
        # Optional: stop after finding first group or continue to find more
        # break  # Uncomment to stop after first group

    logger.info(f"🎯 Total groups found: {len(groups_found)}")
    for i, (group, gid) in enumerate(zip(groups_found, group_ids_found), 1):
        logger.info(f"Group {i} (ID {gid}): {group}")

    # Log the number of unique nodes that found a group out of the number of available nodes in the graph
    grouped_nodes = set(node for group in groups_found for node in group)
    total_nodes = len(my_graph.nodes())
    logger.info(f"⭐ Unique nodes in groups: {len(grouped_nodes)} out of {total_nodes} available nodes in the graph.")

    # Analyze each group in detail
    for i, group in enumerate(groups_found, 1):
        analyze_group_details(group, i)

    # Track end time and print total runtime
    end_time = datetime.now()
    elapsed = end_time - start_time
    elapsed_seconds = elapsed.total_seconds()
    print(f"Analysis complete in {elapsed_seconds:.2f} seconds. Full log saved to: {log_filename}")

    # Analyze failure reasons and create chart
    analyze_failure_reasons()


if __name__ == "__main__":
    main()