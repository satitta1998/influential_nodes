import numpy as np
import networkx as nx
import gc # Garbage collection
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import psutil # To check memory usage
from datetime import datetime 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Params ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
MAX_DEPTH = 6       # Maximum recursion depth for effective distance calculation


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Functions  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
def get_prob(prob_matrix, start, end):
    """
    Retrieves the transition probability from node 'start' to node 'end' from a sparse probability matrix.
    
    :param prob_matrix: Sparse matrix representing transition probabilities between nodes
    :param start: Row index in the matrix representing the starting node
    :param end: Column index in the matrix representing the target node
    :return: The transition probability P_{start→end} if it exists; otherwise, 0
    """
    # Extract row for node 'start'
    row = prob_matrix.getrow(start)
    
    # If 'end' is a neighbor of 'start', return the transition probability
    # `row.indices` gives the column indices where the non-zero elements are
    if end in row.indices:
        prob = row.data[row.indices == end][0]  # Extract the actual value of the transition probability
        return prob
    else:
        return 0  # If there's no edge (probability is zero)


def find_min_effective_distance(start, end, prob_matrix, visited, memo, shortest_paths_dict, index_to_node_id, depth):
    """
    Recursive function to calculate the minimum effective distance from 'start' to 'end'.
    The effective distance D_{n|m} = 1 - log2(P_{nm}) is calculated only along valid paths, with pruning based on a maximum depth
    and the existence of a path (as defined by the shortest_paths_dict).
    
    :param start: Starting node index
    :param end: Target node index
    :param prob_matrix: Sparse probability matrix
    :param visited: A set to track visited nodes
    :param memo: A dictionary to cache previously calculated distances
    :param shortest_paths_dict: Dictionary of shortest paths lenghts between all nodes in graph
    :param index_to_node_id: Mapping from matrix indices to actual node IDs
    :param depth: The current recursion depth, used to limit how far the search can go.
    :return: Minimum effective distance from 'start' to 'end'
    """
    # Base Case 1: If the result is already computed, return it from memo
    if (start, end) in memo:
        return memo[(start,end)]
    
    # Base Case 2: Distance from a node to itself is 0
    if start == end:
        return 0
    
    # Base Case 3: Directly connected nodes
    prob = get_prob(prob_matrix, start, end)
    if prob > 0:
        return 1 - np.log2(prob)
    
    # Base Case 4: Stop if maximum depth exceeded
    if depth >= MAX_DEPTH:
        return np.inf

    
    # Mark current node as visited to prevent cycles
    visited.add(start)
    min_distance = np.inf
    
    
    # Recursive Case: Explore all intermediate nodes
    neighbors = prob_matrix.getrow(start).indices  # All non-zero neighbors
    
    for neighbor in neighbors:
        if neighbor not in visited:
            
            distance_to_neighbor = 1 - np.log2(get_prob(prob_matrix, start, neighbor))
            
            # Check if the result is already computed in the memoization cache
            if (neighbor, end) not in memo:
                
                neighbor_id = index_to_node_id[neighbor]  # Get the actual node ID for neighbor
                end_id = index_to_node_id[end]            # Get the actual node ID for end
                
                # Check if path exists between node neighbor and end within MAX_DEPTH
                if end_id in shortest_paths_dict.get(neighbor_id, {}): 
                    
                    # Recursive call to find effective distance from node neighbor to end - D end|neighbor
                    distance_via_neighbor = distance_to_neighbor + find_min_effective_distance(
                        neighbor, end, prob_matrix, visited, memo, shortest_paths_dict, index_to_node_id, depth+1)
                else:
                    distance_via_neighbor = np.float32(np.inf) # No path or exceeds MAX_DEPTH
            else:
                # Use cached result
                distance_via_neighbor = distance_to_neighbor + memo[(neighbor, end)]

            # Update the minimum distance if this path is shorter
            min_distance = min(min_distance, distance_via_neighbor)
    
    # Unmark the node for other paths
    visited.remove(start)
    
    # Store the result in the memorization cache
    memo[(start, end)] = np.float32(min_distance)
    
    return min_distance

def calculate_node_degrees(adj_matrix):
    """
    Calculate the degree (sum of edges) for each node in the graph.
    :param adj_matrix: Sparse adjacency matrix of the graph
    :return: Degrees of all nodes (numpy array)
    """
    
    degrees = np.ravel(adj_matrix.sum(axis=1))  # sum over rows (out-degree)
    return degrees

def calculate_transition_probabilities(adj_matrix, degrees):
    """
    Vectorized calculation of transition probabilities: P[n|m] = a[m,n] / k[m]
    :param adj_matrix: Sparse adjacency matrix
    :param degrees: Degree of each node (length N)
    :return: Sparse pProbability matrix 
    """
    # Prevent division by zero by setting 0 degrees to 1 temporarily (will fix after)
    safe_degrees = degrees.copy()
    safe_degrees[safe_degrees == 0] = 1
    
    # Create inverse degree diagonal matrix
    # D^(-1) with 1/k[m] on the diagonal
    inv_deg = sp.diags(1.0 / safe_degrees)
    
    # Compute transition probability matrix: P = D^(-1) * A
    # Scales each row of the adjacency matrix by the inverse of the degree of that row 
    # (divides each value of adjacency matrix by the number of outgoing edges)
    prob_matrix = inv_deg @ adj_matrix
    
    # Remove self-loops (set diagonal to 0)
    prob_matrix = prob_matrix.tolil() # Convert matrix to List of Lists format.
    prob_matrix.setdiag(0)
    prob_matrix = prob_matrix.tocsr()
        
    return prob_matrix

def calculate_effective_distance(prob_matrix, shortest_paths_dict, index_to_node_id, node_id_to_index):
    """
    Calculate effective distances based on transition probabilities.
    The effective distance D_{n|m} is computed only if a valid path from node m to node n exists  within the given shortest_paths_dict.
    :param prob_matrix: Sparse transition probability matrix (numpy array)
    :param shortest_paths_dict: Dictionary of shortest paths lenghts between all nodes in graph
    :param index_to_node_id: Mapping from matrix indices to actual node IDs
    :param node_id_to_index: Mapping from actual node ID to matrix index.
    :return: Effective distance matrix
    """

    num_nodes = prob_matrix.shape[0]
    effective_distance = np.full((num_nodes, num_nodes), np.float32(np.inf), dtype = np.float32) # Initialize all distances to infinity
    
    np.fill_diagonal(effective_distance, 0)  # Self-distance is zero
    
    # A memoization cache to store previously computed distances
    memo = {}
    
    # Loop over nodes and their reachable target nodes within MAX_DEPTH
    for m_id, reachable_nodes in shortest_paths_dict.items():
        if len(reachable_nodes) == 1 and m_id in reachable_nodes:
            # Node only reachable to itself, skip further computation (already zero)
            continue
        
        m = node_id_to_index[m_id]  # Convert source node ID to matrix index
        
        for n_id in reachable_nodes:
            if m_id == n_id:
                continue # Skip self-distance (already zero)
        
            n = node_id_to_index[n_id] # Convert target node ID to matrix index
            
            # Check if the result is already computed in the memoization cache
            if (m, n) not in memo:
                # Find effective distance from node m to n - D n|m
                effective_distance[m,n] = find_min_effective_distance(m, n, prob_matrix, set(), memo, shortest_paths_dict, index_to_node_id, depth=0)
            else:
                # Use the cached result
                effective_distance[m, n] = memo[(m, n)]

    return effective_distance


def calculate_interaction_scores(effective_distance, degrees):
    """
    Calculate interaction scores based on effective distances and node degrees.
    :param effective_distance: Effective distance matrix
    :param degrees: Degrees of all nodes
    :return: Sparse interaction scores matrix
    """
    # Create a mask for valid entries (finite and > 0, excluding diagonal)
    mask = np.isfinite(effective_distance) & (effective_distance > 0)
    np.fill_diagonal(mask, False)  # Exclude diagonal

    # Get indices where the mask is True
    row_idx, col_idx = np.nonzero(mask)

    # Vectorized computation of scores
    deg_i = degrees[row_idx]
    deg_j = degrees[col_idx]
    dist = effective_distance[row_idx, col_idx]
    scores = (deg_i * deg_j) / (dist * dist)

    # Build sparse matrix
    return coo_matrix((scores.astype(np.float32), (row_idx, col_idx)), shape=effective_distance.shape).tocsr()

def calculate_effg_centrality(interaction_scores):
    """
    Calculate EffG centrality scores for each node.
    :param interaction_scores: Sparse interaction scores matrix
    :return: EffG centrality scores 
    """
    # Row-wise summation over the sparse matrix. The result is a column vector Nx1.
    # Converting the Nx1 column array into a flat 1D array of length N.
    return np.array(interaction_scores.sum(axis=1)).flatten().astype(np.float32)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN PROCESS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
def main(G: nx.DiGraph):

    start_time = datetime.now()
    print(f"Main started at: {start_time.strftime('%H:%M')}")
    
    # Step 1. Build node list and mapping dictionaries
    node_list = list(G.nodes())
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_list)}
    index_to_node_id = {idx: node_id for node_id, idx in node_id_to_index.items()}
    
    
    # Step 2. Compute sparse adjacency matrix in correct order
    adj_matrix = nx.adjacency_matrix(G, nodelist=node_list, dtype=np.float32)
    
    
    # Step 3: Calculate degrees for each node
    degrees = calculate_node_degrees(adj_matrix)
    print("Node Degrees:")
    print(degrees)
    

    # Step 4: Calculate transition probabilities
    prob_matrix = calculate_transition_probabilities(adj_matrix, degrees)
    print("\nTransition Probability Matrix (P_n|m):")
    print(prob_matrix)
    
    
    # Memory cleanup: Delete large intermediate matrices that are no longer needed
    del adj_matrix
    gc.collect() # Force garbage collection to reclaim memory
    
    
    # Step 5. Precompute shortest paths dict using actual node IDs
    shortest_paths_dict = dict(nx.all_pairs_shortest_path_length(G, cutoff=MAX_DEPTH))
    
    # Step 6: Calculate effective distances
    eff_dist = calculate_effective_distance(prob_matrix, shortest_paths_dict, index_to_node_id, node_id_to_index)
    print("\nEffective Distance Matrix:")
    print(eff_dist)
    
    # Memory cleanup: Delete large intermediate matrices that are no longer needed
    del prob_matrix
    gc.collect() # Force garbage collection to reclaim memory
    
    
    # Step 7: Compute interaction scores
    interaction_scores = calculate_interaction_scores(eff_dist, degrees)
    print("\nInteraction Scores Matrix:")
    print(interaction_scores)
    
    print(f"Memory usage before cleaning degrees and eff_dist: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
    
    # Memory cleanup: Delete large intermediate matrices that are no longer needed
    del degrees
    del eff_dist
    gc.collect() # Force garbage collection to reclaim memory
    
    print(f"Memory usage after cleaning degrees and eff_dist: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
    
    # Step 8: Compute EffG centrality scores
    effg_scores = calculate_effg_centrality(interaction_scores)
    print("\nEffG Centrality Scores:")
    print(effg_scores)
        
    # Memory cleanup: Delete large intermediate matrices that are no longer needed
    del interaction_scores
    gc.collect() # Force garbage collection to reclaim memory
    
    print(f"Memory usage after cleaning interaction_scores: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")
    
    # Step 9: Convert to dictionary {node_id: gravity_score}
    gravity_scores = {node_id: effg_scores[idx] for idx, node_id in index_to_node_id.items()}
    
    end_time = datetime.now()
    print(f"Main ended at: {end_time.strftime('%H:%M')}")
    
    return gravity_scores
    