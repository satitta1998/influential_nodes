import json
import matplotlib.pyplot as plt
import networkx as nx

MIN_YEAR = 1900
MAX_PAPERS_PER_YEAR = 3000
years_of_interest = [1980]  # Papers from these years will be tracked


def get_papers_dict():
    papers_by_year = {}
    with open(r'D:\Datasets\dblp.v10\dblp-ref\dblp-ref-0.json', 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            year = item.get('year')
            if year and year >= MIN_YEAR:
                if year not in papers_by_year:
                    papers_by_year[year] = []

                if len(papers_by_year[year]) < MAX_PAPERS_PER_YEAR:
                    papers_by_year[year].append({
                        'id': item.get('id'),
                        'year': year,
                        'references': item.get('references', []),
                    })
    return papers_by_year


def init_graph():
    return nx.DiGraph()


def add_papers_of_year(graph, papers):
    for paper in papers:
        graph.add_node(paper['id'], year=paper['year'])

    for paper in papers:
        for ref in paper['references']:
            if ref in graph:
                graph.add_edge(ref, paper['id'])


def calculate_importance(graph, model="in_degree"):
    if model == "page_rank":
        return nx.pagerank(graph)
    elif model == "degree_centrality":
        return nx.in_degree_centrality(graph)  # FIXED: Now returns importance scores
    else:
        return dict(graph.out_degree())


papers_dict = get_papers_dict()
g = init_graph()
tracked_papers = {}  # {paper_id: []} to store scores over years
years_tracked = []  # Store years for plotting

# Iterate through all years in chronological order
for year in sorted(papers_dict.keys()):
    papers_of_year = papers_dict[year]
    add_papers_of_year(g, papers_of_year)

    scores = calculate_importance(g, "in_degree")

    # If this is an interesting year, start tracking these papers
    if year in years_of_interest:
        for paper in papers_of_year:
            tracked_papers[paper['id']] = []  # Initialize an empty score history

    # Update scores for all tracked papers
    for paper_id in tracked_papers.keys():
        tracked_papers[paper_id].append(scores.get(paper_id, 0))  # Append new score, default 0 if not found

    years_tracked.append(year)  # Track the year

# Plot the score evolution for tracked papers
plt.figure(figsize=(20, 10))

for paper_id, score_history in tracked_papers.items():
    plt.plot(years_tracked[-len(score_history):], score_history, marker='o', label=f'Paper {paper_id}')

plt.xlabel("Year")
plt.ylabel("Importance Score")
plt.legend()
plt.show()
