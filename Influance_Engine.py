import json
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Params ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
MIN_YEAR = 1900 # The smallest year which will be considered if exists in the data set (papers with smaller publish year will be ignored)
MAX_PAPERS_PER_YEAR = 10000 # if number of papers in data set will exceed this number for a certain year the following papers will be ignored
YEARS_OF_INTEREST = {1990}  # The final plot will show scores for papers published in the years given here.
FILE_PATH = r'D:\Datasets\dblp.v10\dblp-ref\dblp-ref-0.json'
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
                graph.add_edge(ref, paper['id'])  # Only add edge if the reference exists
def calculate_importance(graph, model="in_degree"):
    """Calculates importance scores based on selected model (page_rank, degree_centrality, in_degree, out_degree) Defaults to out_degree"""
    if model == "page_rank":
        return nx.pagerank(graph)
    elif model == "degree_centrality":
        return nx.in_degree_centrality(graph)
    elif model == "in_degree":
        return nx.in_degree_centrality(graph)
    return dict(graph.out_degree())  # Default: return in-degree
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN PROCESS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
papers_dict = get_papers_dict(FILE_PATH)                    # Extract data from dataset.
g = init_graph()                                            # Init empty graph
tracked_papers = defaultdict(list)                          # Init dict - stores {paper_id: [score_over_years]}
years_read_from_ds = []                                          # Init list - store years for plotting
for year in sorted(papers_dict.keys()):                     # Iterate through all years in chronological order
    papers_of_year = papers_dict[year]                      # Get specific papers for given year
    add_papers_of_year(g, papers_of_year)                   # Add papers and citations to graph
    scores = calculate_importance(g, "out_degree")    # Calculate scores with given model
    if year in YEARS_OF_INTEREST:                           # If this is an interesting year, start tracking these papers
        for paper in papers_of_year:
            tracked_papers[paper['id']] = []                # Initialize default dict
    for paper_id in tracked_papers:                         # Update scores for all tracked papers
        tracked_papers[paper_id].append(scores.get(paper_id, 0)) # Add score if paper in scores otherwise 0
    years_read_from_ds.append(year)                              # Add all years read from the dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PLOT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
plt.figure(figsize=(20, 10))
for paper_id, score_history in tracked_papers.items():
    years_to_plot = years_read_from_ds[-len(score_history):]    # Take only years after chosen year
    plt.plot(years_to_plot, score_history, marker='o', label=f'Paper {paper_id}')
plt.xlabel("Year")
plt.ylabel("Importance Score")
plt.legend()
plt.show()