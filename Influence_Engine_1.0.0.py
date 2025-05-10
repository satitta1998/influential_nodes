import json
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import psutil, os
import numpy as np
from operator import itemgetter
from networkx.algorithms.centrality import out_degree_centrality

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Params ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
MIN_YEAR = 1999 # The smallest year which will be considered if exists in the data set (papers with smaller publish year will be ignored)
MAX_PAPERS_PER_YEAR = 10000 # if number of papers in data set will exceed this number for a certain year the following papers will be ignored
YEARS_OF_INTEREST = {2000}  # The final plot will show scores for papers published in the years given here.
PLOTTER = "pyplot" # choose which plotter toy use pyplot (static) or plotly (dynamic)
FILE_PATH = r'D:\Datasets\dblp.v10\dblp-ref\dblp-ref-0.json'
MODEL_TO_EXECUTE = "page_rank_gravity"
SCALE_PR = False # WHen using PageRank_Gravity choose if scale
PR_SCALE_FACTOR = 1e5 # choose how much to scale PageRank_Gravity results
#SIGNIFICANT_GROWTH_THRESHOLD = 3e-06  # This value is the minimum change required in order to be plotted
SIGNIFICANT_GROWTH_THRESHOLD = 1e-12   # page rank with smaller values require smaller derivative threshold

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optimize ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
p = psutil.Process(os.getpid())
p.nice(psutil.HIGH_PRIORITY_CLASS)   # Set high priority on windows


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Functions  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
def get_papers_dict(file_path):
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
        return nx.pagerank(graph.reverse())
    elif model == "out_degree_centrality":
        return nx_out_degree_centrality(graph)
    elif model == "in_degree":
        return nx.in_degree_centrality(graph)
    elif model == "local_gravity":
        return local_gravity_model(graph)
    elif model == "eigenvector_centrality":
        return nx.eigenvector_centrality(graph.reverse(), max_iter = 1000)
    elif model == "page_rank_gravity":
        return compute_gravity_rank(graph)
    return dict(graph.out_degree())  # Default: return in-degree

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Models  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
def local_gravity_model(graph):
    """
    Calculates influence scores of all nodes in the graph using the local gravity model.
    Uses in-degree as the mass (k), and shortest path length as distance (d).
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

    return gravity_scores

def compute_gravity_rank(G, alpha=0.85, k=2, cutoff=None):
    if G.number_of_edges() < 10:
        print("almost no adges")
        return {n: 0.0 for n in G.nodes}

    pr = nx.pagerank(G, alpha=alpha)

    if SCALE_PR:
        pr = {k: v * PR_SCALE_FACTOR for k, v in pr.items()} # Scale the pr since the page rank results are very samll in large graph
    # Set how many top entries you want to see
    TOP_N = 10

    # Sort by PageRank in descending order and print
    top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:TOP_N]

    print(f"Top {TOP_N} nodes by PageRank:")
    for node, score in top_pr:
        print(f"{node}: {score}")

    shortest_paths = dict(nx.all_pairs_shortest_path_length(G, cutoff=cutoff))
    # Extract all shortest path lengths into a flat list
    all_lengths = []
    for target_dict in shortest_paths.values():
        all_lengths.extend(target_dict.values())

    # Find and print the maximum
    max_path_length = max(all_lengths)
    print(f"Maximum shortest path length: {max_path_length}")


    gravity_rank = {i: 0.0 for i in G.nodes}
    print("Graph Size: ",  G.size())
    print("Done PageRank")

    for i, targets in shortest_paths.items():
        for j, d in targets.items():
            if i != j and d > 0:
                gravity_rank[i] += (pr[i] * pr[j]) / (d ** k)

    filtered_gravity = {node: rank for node, rank in gravity_rank.items() if rank > 0.1}
    print(f"Filtered gravity_rank (only > 0):")
    # Sort the dictionary by rank in descending order and get the top 10
    top_10_nodes = sorted(filtered_gravity.items(), key=lambda x: x[1], reverse=True)[:10]

    # Print the top 10 nodes
    for node, rank in top_10_nodes:
        print(f"{node}: {rank:.6f}")

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN PROCESS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
papers_dict = get_papers_dict(FILE_PATH)                    # Extract data from dataset.
g = init_graph()                                            # Init empty graph
tracked_papers = defaultdict(list)                          # Init dict - stores {paper_id: [score_over_years]}
years_read_from_ds = []                                          # Init list - store years for plotting
for year in sorted(papers_dict.keys()):                     # Iterate through all years in chronological order
    print("YEAR - ", year)
    papers_of_year = papers_dict[year]                      # Get specific papers for given year
    add_papers_of_year(g, papers_of_year)                   # Add papers and citations to graph
    scores = calculate_importance(g, MODEL_TO_EXECUTE)    # Calculate scores with given model
    if year in YEARS_OF_INTEREST:                           # If this is an interesting year, start tracking these papers
        for paper in papers_of_year:
            tracked_papers[paper['id']] = []                # Initialize default dict
    for paper_id in tracked_papers:                         # Update scores for all tracked papers
        tracked_papers[paper_id].append(scores.get(paper_id, 0)) # Add score if paper in scores otherwise 0
    years_read_from_ds.append(year)                              # Add all years read from the dataset

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Analyze ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# assuming importance_scores is a dict: {paper_id: [score_t0, score_t1, ..., score_tN]}
derivatives = {
    paper: np.gradient(scores)  # or np.diff(scores) if you prefer
    for paper, scores in tracked_papers.items()
}

most_growth_paper = max(derivatives.items(), key=lambda item: item[1][-1])
print("Most rapidly growing paper:", most_growth_paper[0])

avg_growth = {
    paper: np.mean(deriv)
    for paper, deriv in derivatives.items()
}
most_consistent_riser = max(avg_growth.items(), key=lambda item: item[1])
print("Most consistent riser:", most_consistent_riser[0])


max_single_jump = {
    paper: max(deriv)
    for paper, deriv in derivatives.items()
}
biggest_spike_paper = max(max_single_jump.items(), key=lambda item: item[1])
print("Paper with biggest spike:", biggest_spike_paper[0])


significant_papers = set()

for paper_id, deriv in derivatives.items():
    mean_deriv = np.mean(deriv)
    print(f"Paper ID: {paper_id}, Mean Derivative: {mean_deriv:.10f}")
    if mean_deriv >= SIGNIFICANT_GROWTH_THRESHOLD:
        significant_papers.add(paper_id)

print(f"Number of significant papers: {len(significant_papers)}")


total_papers = len(derivatives)
num_significant = len(significant_papers)
print("=========================================")
print(f"Total number of papers: {total_papers}")
print(f"Number of significant papers: {num_significant}")
print(f"Percentage significant: {num_significant / total_papers * 100:.2f}%")
print("=========================================")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PLOT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

if PLOTTER == "pyplot":
    #plot most interesting
    plt.figure(figsize=(20, 10))
    for paper_id, score_history in tracked_papers.items():
        if paper_id not in significant_papers:
            continue  # skip uninteresting papers

        years_to_plot = years_read_from_ds[-len(score_history):]
        plt.plot(years_to_plot, score_history, marker='o', label=f'Paper {paper_id}')
        # Annotate at the last point
        if score_history:
            plt.text(
                years_to_plot[-1] + 0.2,  # a little to the right
                score_history[-1],  # last score
                str(paper_id)[-6:],  # shorten ID for clarity
                fontsize=8
            )

    plt.xlabel("Year")
    plt.ylabel("Importance Score")
    plt.title(f"Tracked Papers with Significant Growth (>{SIGNIFICANT_GROWTH_THRESHOLD})")
    plt.show()


    #Plot all
    # plt.figure(figsize=(20, 10))
    # for paper_id, score_history in tracked_papers.items():
    #     years_to_plot = years_read_from_ds[-len(score_history):]    # Take only years after chosen year
    #     plt.plot(years_to_plot, score_history, marker='o', label=f'Paper {paper_id}')
    # plt.xlabel("Year")
    # plt.ylabel("Importance Score")
    # plt.legend()
    # plt.show()

if PLOTTER == "plotly":
    # <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 Ploty <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3 <3
    import plotly.express as px
    import plotly.io as pio
    import pandas as pd

    pio.renderers.default = "browser"

    # Prepare the data for Plotly
    plot_data = []

    for paper_id, score_history in tracked_papers.items():
        if paper_id not in significant_papers:
            continue  # Skip uninteresting papers
        years_to_plot = years_read_from_ds[-len(score_history):]
        for year, score in zip(years_to_plot, score_history):
            plot_data.append({
                'Year': year,
                'Score': score,
                'Paper ID': str(paper_id),
            })

    df = pd.DataFrame(plot_data)

    # Create the plot
    fig = px.line(df, x='Year', y='Score', color='Paper ID', line_group='Paper ID')

    fig.update_layout(
        title=f"Importance Score Over Time (Significant Growth > {SIGNIFICANT_GROWTH_THRESHOLD})",
        xaxis_title="Year",
        yaxis=dict(
            range=[30, 1000],
            # tickformat=".2e",
            title="Importance Score"
        ),
        hovermode='closest',

    )

    fig.show()
