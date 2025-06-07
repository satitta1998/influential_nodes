import json
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import psutil, os
import numpy as np
from operator import itemgetter
from networkx.algorithms.centrality import out_degree_centrality
from EffG_Model import main # Import the main function from EffG model file

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Params ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
MIN_YEAR = 1990 # The smallest year which will be considered if exists in the data set (papers with smaller publish year will be ignored)
MAX_PAPERS_PER_YEAR = 10000 # if number of papers in data set will exceed this number for a certain year the following papers will be ignored
YEARS_OF_INTEREST = {1995}  # The final plot will show scores for papers published in the years given here.
PLOTTER = "pyplot" # choose which plotter toy use pyplot (static) or plotly (dynamic)
FILE_PATH = r'D:\Datasets\dblp.v10\dblp-ref\dblp-ref-0.json'
MODEL_TO_EXECUTE = "local_gravity"
SCALE_PR = False # WHen using PageRank_Gravity choose if scale
PR_SCALE_FACTOR = 1e5 # choose how much to scale PageRank_Gravity results
SIGNIFICANT_GROWTH_THRESHOLD = 3e-06  # This value is the minimum change required in order to be plotted
#SIGNIFICANT_GROWTH_THRESHOLD = 1e-12   # page rank with smaller values require smaller derivative threshold
SKIP_UNSIGNIFICANTS = False
EXECUTE_ANALYSIS = True
DATASET = "DBLP_V10"
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optimize ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
p = psutil.Process(os.getpid())
p.nice(psutil.HIGH_PRIORITY_CLASS)   # Set high priority on windows


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Functions  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
def get_papers_dict_DBLP_dataset(file_path):
    """ Extracts papers from dataset file. Will return a dictionary: keys - years, values - dictionaries representing papers (with keys: id, year, references list)"""
    # Default dict won't throw error when adding to non-existing key.
    papers_by_year = defaultdict(list)

    # Parse data set line by line and build papers dictionary.
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())

            publish_year = item.get('year')
            if not publish_year or publish_year < MIN_YEAR:
                continue  # Skip invalid years

            if len(papers_by_year[publish_year]) < MAX_PAPERS_PER_YEAR:
                papers_by_year[publish_year].append({
                    'id': item.get('id'),
                    'year': publish_year,
                    'references': item.get('references', []),
                })

    return dict(papers_by_year)  # Convert back to a normal dict to save memory

def get_papers_dict_CitHep_dataset(file_path):
    """ Extracts papers from dataset file. Returns a dictionary:
        keys - years, values - list of papers (each with keys: id, year, references).
    """
    papers_by_year = defaultdict(list)

    with open(file_path, 'r') as f:
        papers = json.load(f)  # load the entire array at once

        for item in papers:
            publish_year = item.get('year')
            if not publish_year or publish_year < MIN_YEAR:
                continue  # Skip invalid or too old papers

            if len(papers_by_year[publish_year]) < MAX_PAPERS_PER_YEAR:
                papers_by_year[publish_year].append({
                    'id': item.get('id'),
                    'year': publish_year,
                    'references': item.get('references', []),
                })

    return dict(papers_by_year)

# def get_papers_dict(file_path):
#     """ Extracts papers from dataset file. Will return a dictionary: keys - years, values - dictionaries representing papers (with keys: id, year, references list)"""
#     # Default dict won't throw error when adding to non-existing key.
#     papers_by_year = defaultdict(list)
#
#     # Parse data set line by line and build papers dictionary.
#     with open(file_path, 'r') as f:
#         for line in f:
#             item = json.loads(line.strip())
#
#             publish_year = item.get('year')
#             if not publish_year or publish_year < MIN_YEAR:
#                 continue  # Skip invalid years
#
#             if len(papers_by_year[publish_year]) < MAX_PAPERS_PER_YEAR:
#                 papers_by_year[publish_year].append({
#                     'id': item.get('id'),
#                     'year': publish_year,
#                     'references': item.get('references', []),
#                 })
#
#     return dict(papers_by_year)  # Convert back to a normal dict to save memory

def init_graph():
    """Initializes a directed graph."""
    return nx.DiGraph()

def add_papers_of_year(graph, papers):
    """Adds papers and their references to the graph."""
    for paper in papers:
        graph.add_node(paper['id'], year=paper['year'])

        for ref in paper['references']:
            if ref in graph:
                graph.add_edge(ref, paper['id'])  # add edge from ref to paper id. If B cites A then edge A -> B added

def calculate_importance(graph, model="in_degree"):
    """Calculates importance scores based on selected model (page_rank, degree_centrality, in_degree, out_degree) Defaults to out_degree"""
    if model == "page_rank":
        return nx.pagerank(graph.reverse(), alpha=0.7, max_iter=100) # In PageRank More incoming edges to a node => higher importance (higher PageRank). that is why we reverse the graph
    elif model == "out_degree_centrality":
        return nx_out_degree_centrality(graph)
    elif model == "in_degree":
        return nx.in_degree_centrality(graph)
    elif model == "local_gravity":
        return local_gravity_model(graph)  # based upon outdegree, if in your dataset the citing papaer is the source of the edge - reverse the graph.
    elif model == "eigenvector_centrality":
        return nx.eigenvector_centrality(graph, max_iter = 1000)  # Eigen Vector centrality requires stongly connected graphs - does not work for citation networks
    elif model == "page_rank_gravity":
        return compute_gravity_rank(graph.reverse())            # Use page rank and thus requires reverse as in pageRank.
    elif model == "effective_distance_gravity_model":
        return effective_distance_gravity_model(graph)
    return dict(graph.out_degree())  # Default: return in-degree


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Normalizations  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
def normalize_scores(scores):
    values = list(scores.values())
    mean = np.mean(values)
    std = np.std(values)

    if std == 0:
        # All scores are the same — return zeros
        return {k: 0.0 for k in scores}

    return {k: (v - mean) / std for k, v in scores.items()}

def divide_by_avg(scores):
    num_of_papers = 1
    if len(scores) > 0:
        num_of_papers = len(scores)
    avg = sum(scores.values()) / num_of_papers
    if avg == 0:
        avg = 1
    print(f"AVG: {avg}")
    return {k: v / avg for k, v in scores.items()}

def min_max_normalize_scores(scores):
    values = list(scores.values())
    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        return {k: 0.0 for k in scores}  # all values equal

    return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Models  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
def local_gravity_model(graph):
    """
    Calculates influence scores of all nodes in the graph using the local gravity model.
    Uses out-degree as the mass (k), and shortest path length as distance (d).
    Returns:
        dict: {node_id: gravity_score}
    """
    gravity_scores = {}
    degrees = dict(graph.out_degree())

    for i in graph.nodes:
        ki = degrees.get(i, 0)
        total_gravity = 0
        # Use the shortest paths from node i to all other nodes
        lengths = nx.single_source_shortest_path_length(graph, i)

        for j, dij in lengths.items():
            if i == j or dij == 0:
                continue  # skip self or zero distance

            kj = degrees.get(j, 0)
            total_gravity += (ki * kj) / (dij ** 2)

        gravity_scores[i] = total_gravity

    # Normalize - devide by avg
    gravity_scores = divide_by_avg(gravity_scores)
    print(f"AFTER {sorted(gravity_scores.values())[-10:]}")

    # Filter out zero values
    nonzero_scores = {node: score for node, score in gravity_scores.items() if score > 0}
    # Sort by score
    sorted_scores = sorted(nonzero_scores.items(), key=lambda x: x[1], reverse=True)

    top_10 = sorted_scores[:10]
    bottom_10 = sorted_scores[-10:]

    print(f"Graph Size: {graph.size()}")

    # Print as a matrix
    print(f"{'Top 10':<40}             | {'Bottom 10':<40}")
    print("-" * 85)
    for i in range(max(len(top_10), len(bottom_10))):
        top_str = f"{top_10[i][0]}: {top_10[i][1]:.6f}" if i < len(top_10) else ""
        bottom_str = f"{bottom_10[i][0]}: {bottom_10[i][1]:.6f}" if i < len(bottom_10) else ""
        print(f"{top_str:<40} | {bottom_str:<40}")

    #print("Done Local Gravity Calculations ")
    return gravity_scores

def compute_gravity_rank(G, alpha=0.8, k=2, cutoff=None):
    if G.number_of_edges() < 10:
        print("Almost no edges")
        return {n: 0.0 for n in G.nodes}

    pr = nx.pagerank(G, alpha=alpha, max_iter=100)

    if SCALE_PR:
        pr = {k: v * PR_SCALE_FACTOR for k, v in pr.items()}  # Scale PR scores if needed

    TOP_N = 10

    # Compute all shortest paths up to cutoff
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G, cutoff=cutoff))
    all_lengths = [d for dist in shortest_paths.values() for d in dist.values()]
    max_path_length = max(all_lengths)
    print(f"Maximum shortest path length: {max_path_length}")

    print("Graph Size:", G.size())

    gravity_rank = {}
    for i in G.nodes:
        gravity_score = 0.0
        for j, dij in shortest_paths.get(i, {}).items():
            if i != j and dij > 0:
                gravity_score += (pr[i] * pr[j]) / (dij ** k)
        gravity_rank[i] = gravity_score

    # Filter out scores that are too small or zero
    filtered_gravity = {node: score for node, score in gravity_rank.items() if score > 0}

    # Sort by gravity score
    sorted_gravity = sorted(filtered_gravity.items(), key=lambda x: x[1], reverse=True)

    # Avoid overlapping top and bottom if fewer than 20 nonzero values
    top_10 = sorted_gravity[:TOP_N]
    bottom_10 = sorted_gravity[-TOP_N:] if len(sorted_gravity) > TOP_N else []

    # Print matrix-style output
    print(f"\n{'Top 10 Gravity':<40} | {'Bottom 10 Gravity':<40}")
    print("-" * 85)
    for i in range(max(len(top_10), len(bottom_10))):
        top_str = f"{top_10[i][0]}: {top_10[i][1]:.6e}" if i < len(top_10) else ""
        bottom_str = f"{bottom_10[i][0]}: {bottom_10[i][1]:.6e}" if i < len(bottom_10) else ""
        print(f"{top_str:<40} | {bottom_str:<40}")

    # Compute and print average
    avg_score = sum(filtered_gravity.values()) / len(filtered_gravity) if filtered_gravity else 0.0
    print(f"\nAverage gravity score (non-zero): {avg_score:.6e}")
    print("Done Gravity")

    return gravity_rank


    # Sort the dictionary by rank in descending order and get the top 10
    # top_10_nodes = sorted(filtered_gravity.items(), key=lambda x: x[1], reverse=True)[:10]

    top_10 = sorted(filtered_gravity.items(), key=lambda x: x[1], reverse=True)[:10]
    bottom_10 = sorted(filtered_gravity.items(), key=lambda x: x[1], reverse=True)[-10:]

    # # Print the top 10 nodes
    # for node, rank in top_10_nodes:
    #     print(f"{node}: {rank:.6f}")



    for i in range(max(len(top_10), len(bottom_10))):
        top_str = f"{top_10[i][0]}: {top_10[i][1]:.6f}" if i < len(top_10) else ""
        bottom_str = f"{bottom_10[i][0]}: {bottom_10[i][1]:.6f}" if i < len(bottom_10) else ""
        print(f"{top_str:<40} | {bottom_str:<40}")

    print("Done Gravity")
    return gravity_rank

def nx_out_degree_centrality(G):
    #nx_out_degree = nx.out_degree_centrality()
    nx_out_degree = dict(G.out_degree())
    # Set how many top entries you want to see
    top_n = 30

    # Sort by PageRank in descending order and print
    #top_nx_out_degree = sorted(nx_out_degree.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_nx_out_degree = dict(sorted(nx_out_degree.items(), key=itemgetter(1), reverse=True)[:top_n])

    print(f"Top {top_n} nodes by nx-out-degree:")
    # for node, score in top_nx_out_degree:
    #     print(f"{node}: {score}")

    for node, score in top_nx_out_degree.items():
        print(f"{node}: {score}")

    return nx_out_degree

def nx_page_rank(G, alpha=0.85):
    pr = nx.pagerank(G, alpha=alpha)
    if SCALE_PR:
        pr = {k: v * PR_SCALE_FACTOR for k, v in
              pr.items()}  # Scale the pr since the page rank results are very samll in large graph

    return pr

def effective_distance_gravity_model(graph):
    gravity_scores = {}
    gravity_scores = main(graph)

    return gravity_scores
