from datetime import datetime

# --- NODE DATA EXAMPLE ---

# This example provides a template for node data in a graph.
# The node data structure follows following format, node id : node data
# to get node connections, get them using neighbors(node_id)
# to get edge weights, use the 'weight' attribute to get the weights for both nodes on the edge 
nodes_data_example = {
    1: {
        "Bans": [],
        "GroupIDs": [1, 2],
        "Waiting_Since" : datetime(2025, 7, 23, 10, 30, 45),
        "Type": "All",
        "Available": True
    },
    2: {
        "Bans": [],
        "GroupIDs": [1, 2],
        "Waiting_Since": datetime(2025, 7, 24, 10, 30, 45),
        "Type": "Half+",
        "Available": True
    },
    3: {
        "Bans": [1],
        "GroupIDs": [],
        "Waiting_Since": datetime(2025, 7, 25, 10, 30, 45),
        "Type": "One+",
        "Available": True
    },
}