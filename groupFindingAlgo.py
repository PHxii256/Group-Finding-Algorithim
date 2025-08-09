import math 
from algoGraphGenerator import generate_graph

my_graph = generate_graph()
GROUP_SIZE = 4 # The size of the group we are trying to form
MAX_PREMADE_NODES_PER_GROUP = 2 # The maximum number of premade nodes allowed per group

failed_nodes = set() # Set of nodes that have failed in the current path
ungroupable_nodes = set() # Set of nodes that cannot be grouped for current itteration
new_groups = {} # Dictionary to hold new groups formed during the process

# -------------------- V2 ------------------------

# --- Remove non 'Available' nodes ---
# Remove nodes that are not available wheather self imposed or done automatically, 
# when it was added to the max no. of groups for a cooldown period
def remove_unavailable_nodes(my_graph):
    nonAvailable_nodes = [node for node, data in my_graph.nodes(data=True) if not data.get('available', True)]

    for node in nonAvailable_nodes:
        my_graph.remove_node(node)

def total_edge_weight(node_id, frontier_id):
    return sum(my_graph.get_edge_data(frontier_id, node_id))['weight'].values()

def get_path_bans(path):
    bans_set = set()
    for node in path:
        bans_set.update(my_graph.nodes[node['id']]['bans'])
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
    path_node_ids = [node['id'] for node in path]
    conn_count_with_path = len(set(my_graph.neighbors(node_id)) & set(path_node_ids))
    return conn_count_with_path

def is_path_feasible(path):
    # Check if the remaining connections for the path are sufficient
    # GROUP_SIZE - len(path) is the number of connections left to fill for this path
    for node_id in path:
        remaining_connections = get_node_remaining_conn_count(node_id, path)

        if GROUP_SIZE - len(path) < remaining_connections:
            return False
    return True

def get_node_remaining_conn_count(node_id, path):
    type_connection_count = get_type_connection_count(my_graph.nodes[node_id]['data']['type'])
    connection_count_with_path = get_node_conn_count_with_path(node_id, path)
    remaining_connections =  max(type_connection_count - connection_count_with_path, 0)
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
def get_mandatory_nodes_picks(path):
    m_nodes_list = []
    m_nodes_picks = set()
    path_bans = get_path_bans(path)

    for node in path:
        r_conn_count = get_node_remaining_conn_count(node, path)
        if r_conn_count == GROUP_SIZE - len(path):
            m_nodes_list.append(node)

        elif r_conn_count > GROUP_SIZE - len(path):
            raise ValueError(f"ERROR: Node {node}'s remaining connections > group_size - len(path).")

    for node_id in m_nodes_list:
        m_nodes_picks.update(my_graph.neighbors(node_id) - path_bans)

    return m_nodes_picks - path_bans

def get_frontier_picks(path, excluded_nodes = set()):
    frontier_picks = set()
    for node_id in path:
        frontier_picks = set(my_graph.neighbors(node_id)) & get_mandatory_nodes_picks(path)
        frontier_picks -= get_path_bans(path) - excluded_nodes
    return frontier_picks

def pass_mac(frontier_id, pick_id):
    edge_data = my_graph.get_edge_data(frontier_id, pick_id)
    if edge_data['weight'][pick_id] >= my_graph.nodes[frontier_id]['data']['mac']:
        return True
    else:  
        return False

def frontier_pass_mac_with_path(frontier_id, path):
    for node in path:
        if my_graph.has_edge(frontier_id, node['id']):
            edge_data = my_graph.get_edge_data(frontier_id, node['id'])
            if edge_data['weight'][node['id']] >= my_graph.nodes[frontier_id]['data']['mac']:
                return True

def get_valid_sorted_mac_pics(frontier_id, path):
    valid_mac_pics = []
    # mac picks here should be outside of the path
    for pick_id in get_frontier_picks(path, excluded_nodes= path):
        if pass_mac(frontier_id, pick_id):
            if pass_group_constraints(pick_id, path):
                valid_mac_pics.append(pick_id)
    return sorted(valid_mac_pics, key=lambda x: total_edge_weight(x, frontier_id), reverse=True)

def dfs_iterative(start_node):
    visited = set()
    stack = [(start_node, set())]  # Store (node, path_to_reach_node) tuples

    while stack:
        current_node, path_so_far = stack.pop()

        if current_node not in visited:
            visited.add(current_node)
            current_path = path_so_far.copy()  # Create current path
            current_path.add(current_node)     # Add current node to path
            
            if not is_path_feasible(current_path):
                print(f"Path is not feasible: {current_path}")
                continue  # Skip this path if it's not feasible
            
            if frontier_pass_mac_with_path(current_node, current_path):
                print(f"Current node {current_node} passes MAC with path: {current_path}")
                # pick from the combined pool of picks for path the next node
                continue

            mac_list = get_valid_sorted_mac_pics(current_node, list(current_path))
            if mac_list:
                for mac_pick in reversed(mac_list): # Reverse to maintain typical DFS order
                    if mac_pick not in visited and mac_pick not in current_path:
                        # add more logic here if needed, e.g., checking for specific conditions
                        stack.append((mac_pick, current_path.copy()))
                    else:
                        print(f"{mac_pick} is already visited or in the current path.")


# --- Main (V2) ---

remove_unavailable_nodes(my_graph)

# Seeds are the potential starting nodes for a group
# We sort the nodes by their 'waiting_since' attribute longest waiting time first
# Waiting_since must not be None, it's the time since the node was last added to a group
# but if it was never added, it's value will be the time since added to the graph 
# i.e showing interest in the activity this graph represents
sorted_seeds = sorted(my_graph.nodes.data("waiting_since"), key=lambda x: x[1])

for node in sorted_seeds:
    node_id = node[0]
    node_data = my_graph.nodes[node_id]
    print(f"Starting DFS from seed node: {node_id} with data: {node_data}")
    dfs_iterative(node_id)