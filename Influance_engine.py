import json
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt


class InfluenceEngine:
    """
    Class for calculating influence scores for papers in citation networks.
    This design uses instance attributes to store the DataFrame, graph, and score dictionary.
    """

    def __init__(self, dataset_name: str, path: str):
        """
        Initialize the engine with a dataset name and file path.
        The engine will automatically load and preprocess the dataset.
        """
        self.dataset_name = dataset_name
        self.path = path
        self.df = None       # DataFrame to store paper data
        self.graph = None    # NetworkX graph representing the citation network
        self.scores = None   # Dictionary to store influence scores
        self.filtered_scores = None # Dictionary to keep scores based on years
        self._preprocess_dataset()

    def _preprocess_dataset(self):
        """
        Choose the right preprocessing method based on the dataset name.
        """
        if self.dataset_name == "dblp":
            self._preprocess_dblp()
        else:
            raise ValueError(f"Preprocessing for dataset '{self.dataset_name}' is not defined.")

    def _load_data_from_json_lines(self):
        """
        Reads a JSON lines file and extracts the relevant fields.
        :return: List of dictionaries containing 'id', 'year', and 'references'.
        """
        data = []
        with open(self.path, 'r', encoding="utf-8") as f:
            for line in f:
                paper = json.loads(line)
                data.append({
                    "id": paper["id"],
                    "year": paper["year"],
                    "references": paper.get("references", [])
                })
        return data

    def _preprocess_dblp(self):
        """
        Preprocess the DBLP dataset.
        Loads the data, creates a DataFrame, and builds the citation graph.
        """
        data = self._load_data_from_json_lines()
        self.df = pd.DataFrame(data)
        self.graph = self._generate_graph_from_dataframe(self.df)

    def _generate_graph_from_dataframe(self, df: pd.DataFrame):
        """
        Generates a directed graph using NetworkX from the given DataFrame.
        :param df: DataFrame with 'id', 'year', and 'references' columns.
        :return: A NetworkX DiGraph representing the citation network.
        """
        G = nx.DiGraph()
        for _, row in df.iterrows():
            paper_id = row["id"]
            references = row["references"]
            publish_year = row["year"]
            G.add_node(paper_id, publish_year=publish_year)
            for ref_id in references:
                G.add_edge(paper_id, ref_id)
        return G

    def calculate_scores(self, model_name: str = "pagerank"):
        """
        Calculate influence scores using the specified model and store the results.
        Currently implemented model: PageRank.
        :param model_name: Name of the scoring model.
        :return: A dictionary of scores.
        """
        if model_name == "pagerank":
            self.scores = nx.pagerank(self.graph)
        else:
            raise ValueError("Bad model name or model not implemented")
        return self.scores

    def filter_scores_by_year(self, years):
        """
        Filters the influence scores to only include papers published in the specified years,
        storing them in a nested dictionary structure { "paper_id" : { "year" : score } }.

        :param years: A list (or set) of years to filter by.
        :return: A dictionary with filtered scores.
        """
        # Ensure scores have been calculated
        if self.scores is None:
            raise ValueError("Scores have not been calculated. Run calculate_scores() first.")

        self.filtered_scores = {}

        for _, row in self.df.iterrows():
            paper_id = row["id"]
            year = row["year"]

            if year in years and paper_id in self.scores:
                if paper_id not in self.filtered_scores:
                    self.filtered_scores[paper_id] = {}

                self.filtered_scores[paper_id][year] = self.scores[paper_id]

        return self.filtered_scores

    def plot_filtered_scores(self):
        """
        Plots the filtered influence scores with each paper's score plotted against its publication year.
        """
        if self.filtered_scores is None:
            raise ValueError("Filtered scores have not been calculated. Run filter_scores_by_year() first.")

        # map paper id to year
        paper_years = {row["id"]: row["year"] for _, row in self.df.iterrows()}
        years = []
        scores = []

        for paper_id, score in self.filtered_scores.items():
            if paper_id in paper_years:
                years.append(paper_years[paper_id])
                scores.append(score)

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(years, scores, alpha=0.5)
        plt.title("Influence Scores of Papers Over the Years")
        plt.xlabel("Publication Year")
        plt.ylabel("Influence Score")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

# Example usage:
engine = InfluenceEngine("dblp", r'D:\Datasets\dblp.v10\dblp-ref\dblp-ref-0.json')
engine.calculate_scores()
engine.filter_scores_by_year([1990])  # Filter for specific years
engine.plot_filtered_scores()  # Plot filtered scores
