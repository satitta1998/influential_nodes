import json
import matplotlib.pyplot as plt
import networkx as nx

MIN_YEAR = 1900
MAX_PAPERS_PER_YEAR = 3000

# Get the papers from the dataset
# Returns a dictionary: keys contain years and values a dictionaries which represents papers
# We only keep id, year, ids of references instead of all data from the dataset.
def get_papers_dict():
        papers_by_year = {}  # Dictionary to store papers by year
        with open(r'D:\Datasets\dblp.v10\dblp-ref\dblp-ref-3.json', 'r') as f:
            for line in f:
                item = json.loads(line.strip())  # Parse each line as a JSON object
                year = item.get('year')  # Use .get() to avoid KeyError
                if year and year >= MIN_YEAR:  # Ensure the year exists and meets the min_year requirement
                    if year not in papers_by_year:
                        papers_by_year[year] = []  # Initialize a list for this year if not already present

                    if len(papers_by_year[
                               year]) < MAX_PAPERS_PER_YEAR:  # Check if max papers for the year is not exceeded
                        local_paper = {
                            'id': item.get('id'),
                            'year': year,
                            'references': item.get('references', []),  # Use .get() to avoid KeyError
                        }
                        papers_by_year[year].append(local_paper)

        return papers_by_year

# Initialize an  empty graph
def init_graph():
    # Build the directed graph
    return  nx.DiGraph()

# Add papers given to given graph in given year.
def add_papers_of_year(graph, papers , year):
    for paper in papers:
        if paper['year'] == year:
            graph.add_node(paper['id'], year=paper['year'])
            print("Added node: " + paper['id'])
        else:
            print("Error - papers include paper not from year")


    for paper in papers:
        if paper['year'] == year:
            for ref in paper['references']:
                if ref in graph:
                    graph.add_edge(ref, paper['id'])  # Directed edge from cited paper to citing paper
                    print("Added edge: " + ref)
                else :
                    print("Error ref not in graph")


# # Function to calculate importance scores (PageRank as an example)
def calculate_importance(graph):
    return nx.pagerank(graph)

# Main Execution
papers_dict = get_papers_dict()
g = init_graph()
importance_scores_by_year = {}

# Sort the years in ascending order and iterate over them
for year in sorted(papers_dict.keys()):
    #print("YEAR: ", year)
    papers_of_year = papers_dict[year] # Extract the papers of the current year
    add_papers_of_year(g, papers_of_year, year) # Add papers to the grapgh
    scores = calculate_importance(g) # calculate scores of the graph.
    #print("SCORES:", scores)
    importance_scores_by_year[year] = scores
    #sleep(3)


plt.figure(figsize=(20, 10))

# Track importance scores of each paper across years
paper_scores = {}

# build dictionary: keys are paper IDs values are taples of (year, score)
for year, scores in importance_scores_by_year.items():
    for paper_id, score in scores.items():
        if paper_id not in paper_scores:
            paper_scores[paper_id] = []
        paper_scores[paper_id].append((year, score))

# Plot each paper's importance over time
for paper_id, data in paper_scores.items():
    data.sort()  # Ensure data is sorted by year
    print(paper_id,data)
    years, scores = zip(*data)  # Separate years and scores
    plt.plot(years, scores, label=f'Paper {paper_id}', alpha=0.5)

plt.xlabel('Year')
plt.ylabel('Importance Score')
plt.title('Paper Importance Scores Over the Years')
plt.legend([], title="Papers", loc="upper left")  # Hide excessive legend entries
plt.show()

plt.show()

