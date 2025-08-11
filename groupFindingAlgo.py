import math 
import logging
from datetime import datetime
from algoGraphGenerator import generate_graph

# Set up logging to file
log_filename = f"group_finding_output_log.txt"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# Track start time
start_time = datetime.now()

logger.info(f"Starting group finding algorithm - Log file: {log_filename}")

my_graph = generate_graph()
GROUP_SIZE = 4 # The size of the group we are trying to form
MAX_PREMADE_NODES_PER_GROUP = 2 # The maximum number of premade nodes allowed per group

failed_nodes = set() # Set of nodes that have failed in the current path
ungroupable_nodes = set() # Set of nodes that cannot be grouped for current itteration

# -------------------- V2 ------------------------

# --- Remove non 'Available' nodes ---
# Remove nodes that are not available wheather self imposed or done automatically, 
# when it was added to the max no. of groups for a cooldown period
def remove_unavailable_nodes(my_graph):
    removed_count = 0
    # Get all nodes that are not available
    nonAvailable_nodes = [node for node, data in my_graph.nodes(data=True) if not data.get('available', True)]

    for node in nonAvailable_nodes:
        my_graph.remove_node(node)
        removed_count += 1
    logger.info(f"Removed {removed_count} unavailable nodes out of {len(my_graph.nodes()) + removed_count}.")
    return removed_count

def total_edge_weight(node_id, frontier_id):
    return sum(my_graph.get_edge_data(frontier_id, node_id)['weight'].values())

def get_path_bans(path):
    bans_set = set()
    for node_id in path:
        bans_set.update(my_graph.nodes[node_id]['bans'])
    return bans_set

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
    
    for node in path:
        path_groups_list.extend(my_graph.nodes[node]["groupIDs"])

    for group_id in path_groups_list:
        path_group_count_dict[group_id] = path_group_count_dict.get(group_id, 1) + 1
        if path_group_count_dict[group_id] > MAX_PREMADE_NODES_PER_GROUP:
            return False  # Node fails group constraints
        
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

def get_frontier_picks(path):
    path_bans = get_path_bans(path)

    frontier_picks = get_node_picks(path[-1], path, bans=path_bans)
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

def frontier_pass_mac_using_path(frontier_id, path):
    for node_id in path:
        if my_graph.has_edge(frontier_id, node_id):
            edge_data = my_graph.get_edge_data(frontier_id, node_id)
            if edge_data['weight'][node_id] >= my_graph.nodes[frontier_id]['mac']:
                return True

def get_valid_sorted_mac_pics(frontier_id, path):
    valid_mac_pics = []
    # mac picks here should be outside of the path
    for pick_id in get_frontier_picks(path):
        if pass_mac(frontier_id, pick_id):
            if pass_group_constraints(pick_id, path):
                valid_mac_pics.append(pick_id)
    return sorted(valid_mac_pics, key=lambda x: total_edge_weight(x, frontier_id), reverse=False)

def validate_group_type_requirements(group):
    """
    Validates that all nodes in the group meet their type requirements.
    Returns True if all nodes satisfy their requirements, False otherwise.
    """
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
                logger.info(f"❌ Group {new_path} rejected - not all members meet type requirements")
                continue  # This path doesn't form a valid group, backtrack
        
        # Check if path is still feasible
        if not is_path_feasible(new_path):
            logger.info(f"Path {new_path} is not feasible, backtracking...")
            continue
        
        # Check if current node passes MAC with existing path
        if len(new_path) > 1 and frontier_pass_mac_using_path(current_node, new_path):
            logger.info(f"Node {current_node} passes MAC with path, continuing...")
        
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
            logger.info(f"No valid MAC picks for {current_node}, backtracking...")
            # This path is exhausted, will backtrack automatically
        
        logger.info(f"Stack state: {[(node, path, picks) for node, path, picks in stack]}")

    logger.info(f"DFS complete in {loop_count} iterations. No complete group found.")
    return None

# --- Main (V2) ---

print("--- Starting group finding algorithm ---")

remove_unavailable_nodes(my_graph)

# Seeds are the potential starting nodes for a group
# We sort the nodes by their 'waiting_since' attribute longest waiting time first
# Waiting_since must not be None, it's the time since the node was last added to a group
# but if it was never added, it's value will be the time since added to the graph 
# i.e showing interest in the activity this graph represents
sorted_seeds = sorted(my_graph.nodes.data("waiting_since"), key=lambda x: x[1])

groups_found = []
for node in sorted_seeds:
    node_id = node[0]
    logger.info(f">>> Starting DFS from seed node: {node_id}")
    
    result = dfs_iterative(node_id)
    if result:
        logger.info(f"✅ Found group: {result}")
        groups_found.append(result)
        
        # Remove only the seed node from available nodes for next iteration
        sorted_seeds = [s for s in sorted_seeds if s[0] != node_id]
        
        logger.info(f"Remaining seeds: {[s[0] for s in sorted_seeds]}")
    else:
        logger.info(f"❌ No group found starting from {node_id}")
    
    # Optional: stop after finding first group or continue to find more
    # break  # Uncomment to stop after first group

logger.info(f"🎯 Total groups found: {len(groups_found)}")
for i, group in enumerate(groups_found, 1):
    logger.info(f"Group {i}: {group}")

# Detailed analysis of each group
def analyze_group_details(group, group_number):
    logger.info(f"{'='*60}")
    logger.info(f"📊 DETAILED ANALYSIS FOR GROUP {group_number}: {group}")
    logger.info(f"{'='*60}")
    
    for node in group:
        node_data = my_graph.nodes[node]
        node_type = node_data['type']
        node_mac = node_data['mac']
        
        logger.info(f"🔹 NODE {node} (Type: {node_type}, MAC: {node_mac})")
        logger.info(f"   Connections to other group members:")
        
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
                logger.info(f"       • {node} rates {conn['node']}: {conn['weight_to']}")
                logger.info(f"       • {conn['node']} rates {node}: {conn['weight_from']}")
        
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

# Analyze each group in detail
for i, group in enumerate(groups_found, 1):
    analyze_group_details(group, i)


# Track end time and print total runtime
end_time = datetime.now()
elapsed = end_time - start_time
elapsed_seconds = elapsed.total_seconds()
print(f"Analysis complete in {elapsed_seconds:.2f} seconds. Full log saved to: {log_filename}")