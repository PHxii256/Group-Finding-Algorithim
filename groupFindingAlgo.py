from algoGraphGenerator import generate_graph

my_graph = generate_graph()

GROUP_SIZE = 4 # The size of the group we are trying to form
MAX_PREMADE_NODES_PER_GROUP = 2 # The maximum number of premade nodes allowed per group

failed_nodes = set() # Set of nodes that have failed in the current path
ungroupable_nodes = set() # Set of nodes that cannot be grouped for current itteration

path = list() # List of nodes in the current path
path_bans_set = set()
path_nodes_valid_picks = {} # Set of picks for the current path, node id : it's valid picks
remaining_connections_dict = {} # Remaining connections for the current path, node id : remaining connections
path_group_count_dict = {} # Dictionary of premade node groups and their count in path, group id : number of nodes from that group in the path

current_seed = None # The current seed node being processed
current_frontier = None # The current frontier node being processed
current_frontier_picks = set() # Picks of the current frontier node

# to:do
# Remove nodes that are in the 'failed' set 

#-------------------------------------------------

# --- Remove non 'Available' nodes ---

# Remove nodes that are not available wheather self imposed or done automatically, 
# when it was added to the max no. of groups for a cooldown period
nonAvailable_nodes = [node for node, data in my_graph.nodes(data=True) if not data.get('available', True)]

for node in nonAvailable_nodes:
    my_graph.remove_node(node)

# Seeds are the potential starting nodes for a group
# We sort the nodes by their 'waiting_since' attribute longest waiting time first

# Waiting_since must not be None, it's the time since the node was last added to a group
# but if it was never added, it's value will be the time since added to the graph 
# i.e showing interest in the activity this graph represents

sorted_seeds = sorted(my_graph.nodes.data("waiting_since"), key=lambda x: x[1])

def push_to_path(node_id):
    global current_frontier
    global path_bans_set
    global path
    global path_group_count_dict

    node_data = my_graph.nodes[node_id]
    current_frontier = {"id": node_id, "data": node_data}
    path_bans_set.update(node_data.get('bans', set()))
    path.append(current_frontier)
    print(f"Pushing frontier node {current_frontier} on path stack.")
    
    for group_id in node_data["groupIDs"]:
        if group_id not in path_group_count_dict:
            path_group_count_dict[group_id] = 1
        else:
            path_group_count_dict[group_id] += 1

        if path_group_count_dict[group_id] > MAX_PREMADE_NODES_PER_GROUP:
            raise ValueError(f"Group {group_id} has exceeded the maximum number of premade nodes allowed in the path.")

# Assign the first available seed to current_seed, and exit if no seeds remaining
if not sorted_seeds:
    print("No available seeds found or remaining. Exiting algorithm.")
else:
    current_seed = {"id": sorted_seeds[0][0], "data": my_graph.nodes(data=True)[sorted_seeds[0][0]]}
    push_to_path(current_seed['id'])

# --- Update path's remaining connection count ---

def getPathConnectionCount(node_id):
    path_node_ids = [node['id'] for node in path]
    return len(set(my_graph.neighbors(node_id)) & set(path_node_ids))

for node in path:
    remaining_connections_dict[node['id']] = getPathConnectionCount(node['id'])

print(f"Remaining connections for current path: {remaining_connections_dict}")

# --- Get frontier picks not banned by path members ---

# Get picks of the current frontier node and cache them in path_nodes_valid_picks dict
current_frontier_picks = set(my_graph.neighbors(current_frontier['id'])) - path_bans_set
path_nodes_valid_picks[current_frontier['id']] = current_frontier_picks

def node_intersection_filter():
    """
    This function filters the current frontier picks based on mandatory nodes' picks.
    It ensures that the picks of mandatory nodes are included in the next selection.
    """

    global current_frontier_picks
    global path_nodes_valid_picks
    global remaining_connections_dict

    # --- Node Intersection Filter ---

    # Get list of path's nodes which are mandatory to have their picks included in the next selection
    m_nodes_list = []
    for node in remaining_connections_dict:
        if remaining_connections_dict[node] == GROUP_SIZE - len(path):
            m_nodes_list.append(node)

        elif remaining_connections_dict[node] > GROUP_SIZE - len(path):
            raise ValueError(f"ERROR: Node {node}'s remaining connections > group_size - len(path).")

    print(f"Mandatory nodes list: {m_nodes_list}")

    # Get the picks of mandatory nodes
    mandatory_picks = set()
    for node in m_nodes_list:
        mandatory_picks.update(path_nodes_valid_picks.get(node, set()))

    # filter current_frontier_picks based on mandatory nodes's picks
    for node in m_nodes_list:
        current_frontier_picks = current_frontier_picks & path_nodes_valid_picks.get(node, set())

    # Intersection filter = Mandatory nodes' picks Intersection with current frontier picks
    print(f"Valid Picks (After Intersection Filter): {current_frontier_picks}")

def mac_filter():
    """
    This function applies the MAC filter to the current frontier picks.
    It ensures that the picks are valid according to the MAC constraints.
    """
    # Get the MAC constraints for the current frontier node
    mac_constraints = my_graph.nodes[current_frontier['id']]["mac"]
    # pool of nodes that pass the min affinity constraint (MAC)
    mac_pool = []

    # Filter current_frontier_picks based on MAC constraints
    # We check the weight of the edge from current_frontier to each node in current_frontier_picks
    # and if it meets the MAC constraints, we add it to the mac_pool
    # We use the edge weight of the other node relative to the current frontier node's constraint
    for node in current_frontier_picks:
        if my_graph.get_edge_data(current_frontier['id'], node)['weight'][node] >= mac_constraints:
            mac_pool.append(node)

    if not mac_pool:
        print("MAC pool is empty. No valid picks available.")
        #
        #
        #        Pool == Empty 
        #
        #
        return
    
    print(f"MAC pool: {mac_pool}")

    # --- MAC Pool Group Filtering ---

    exclude_group_ids = set()

    # Check if the group has reached its maximum number of premade nodes in path and remove nodes from mac_pool
    for groupID, count in path_group_count_dict.items():
        if count == MAX_PREMADE_NODES_PER_GROUP:
            exclude_group_ids.add(groupID)

   
    for node in mac_pool:
        if any(group_id in exclude_group_ids for group_id in my_graph.nodes[node]["groupIDs"]):
            mac_pool.remove(node)

    if not mac_pool:
        print("MAC Group Filtered Pool is empty. No valid picks available.")
        #
        #
        #        Pool == Empty 
        #
        #
        return

    print(f"MAC Group Filtered Pool: {mac_pool}")

    if(len(path) == GROUP_SIZE):
        print(f"Path is complete. path: {path}")
        #
        #
        #        Path.len == Group Size 
        #
        #
        return
    
    return mac_pool

# If the path has more than one node, we can check for valid picks
# set >= for testing, should be = instead
if len(path) >= 1:
    # Apply the node intersection filter
    node_intersection_filter()
    mac_pool = mac_filter()
    if mac_pool:
        # --- Sort MAC Pool by sum of weights on the edges to the current frontier node ---
        mac_pool.sort(key=lambda node: sum(my_graph.get_edge_data(current_frontier['id'], node)['weight'].values()), reverse=True)
        print(f"Sorted MAC Pool: {mac_pool}")