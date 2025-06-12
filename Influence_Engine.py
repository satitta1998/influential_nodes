import json
import networkx as nx
from collections import defaultdict
import psutil
import os
import numpy as np
from operator import itemgetter
from EffG_Model import main  # Import the main function from EffG model file


class CitationNetworkAnalyzer:
    """
    A class for analyzing citation networks using various centrality and influence models.

    Initialize the Citation Network Analyzer with configuration parameters.

    Args:
        min_year (int): The smallest year to consider in the dataset
        max_papers_per_year (int): Maximum papers per year to process
        years_of_interest (set): Years to analyze in the final plot
        plotter (str): Plotter type ("pyplot" or "plotly")
        file_path (str): Path to the dataset file
        model_to_execute (str): Model to use for analysis
        scale_pr (bool): Whether to scale PageRank results
        pr_scale_factor (float): Factor for scaling PageRank
        significant_growth_threshold (float): Minimum change threshold for plotting
        skip_insignificant (bool): Whether to skip insignificant results
        execute_analysis (bool): Whether to execute analysis
        dataset (str): Dataset identifier
    """

    def __init__(self,
                 min_year=1990,
                 max_papers_per_year=10000,
                 years_of_interest={1995},
                 plotter="pyplot",
                 file_path=r'D:\Datasets\dblp.v10\dblp-ref\dblp-ref-0.json',
                 model_to_execute="local_gravity",
                 scale_pr=False,
                 pr_scale_factor=1e5,
                 significant_growth_threshold=3e-06,
                 skip_insignificant=False,
                 execute_analysis=True,
                 dataset="DBLP_V10"):

        # Configuration parameters
        self.min_year = min_year
        self.max_papers_per_year = max_papers_per_year
        self.years_of_interest = years_of_interest
        self.plotter = plotter
        self.file_path = file_path
        self.model_to_execute = model_to_execute
        self.scale_pr = scale_pr
        self.pr_scale_factor = pr_scale_factor
        self.significant_growth_threshold = significant_growth_threshold
        self.skip_insignificants = skip_insignificant
        self.execute_analysis = execute_analysis
        self.dataset = dataset

        # Initialize process optimization
        self._optimize_process()

        # Initialize graph
        self.graph = None
        self.papers_by_year = None

    def _optimize_process(self):
        """Set high priority for the current process on Windows."""
        try:
            p = psutil.Process(os.getpid())
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        except Exception as e:
            print(f"Warning: Could not set process priority: {e}")

    # =============================== Data Loading Methods =============================== #

    def get_papers_dict_dblp_dataset(self, file_path=None):
        """
        Extracts papers from DBLP dataset file.

        Args:
            file_path (str, optional): Path to dataset file. Uses self.file_path if None.

        Returns:
            dict: Dictionary with years as keys and lists of paper dictionaries as values
        """
        if file_path is None:
            file_path = self.file_path

        papers_by_year = defaultdict(list)

        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())

                publish_year = item.get('year')
                if not publish_year or publish_year < self.min_year:
                    continue

                if len(papers_by_year[publish_year]) < self.max_papers_per_year:
                    papers_by_year[publish_year].append({
                        'id': item.get('id'),
                        'year': publish_year,
                        'references': item.get('references', []),
                    })

        self.papers_by_year = dict(papers_by_year)
        return self.papers_by_year

    def get_papers_dict_cithep_dataset(self, file_path=None):
        """
        Extracts papers from CitHep dataset file.

        Args:
            file_path (str, optional): Path to dataset file. Uses self.file_path if None.

        Returns:
            dict: Dictionary with years as keys and lists of paper dictionaries as values
        """
        

        if file_path is None:
            file_path = self.file_path

        papers_by_year = defaultdict(list)

        with open(file_path, 'r') as f:
            papers = json.load(f)
            for item in papers:
                publish_year = item.get('year')
                if not publish_year or publish_year < self.min_year:
                    continue
                if len(papers_by_year[publish_year]) < self.max_papers_per_year:
                    papers_by_year[publish_year].append({
                        'id': item.get('id'),
                        'year': publish_year,
                        'references': item.get('references', []),
                    })
        self.papers_by_year = dict(papers_by_year)
        return self.papers_by_year

    # =============================== Graph Methods =============================== #

    def init_graph(self):
        """Initializes a directed graph."""
        self.graph = nx.DiGraph()
        return self.graph

    def add_papers_of_year(self, papers, graph=None):
        """
        Adds papers and their references to the graph.

        Args:
            papers (list): List of paper dictionaries
            graph (nx.DiGraph, optional): Graph to add papers to. Uses self.graph if None.
        """
        if graph is None:
            if self.graph is None:
                self.init_graph()
            graph = self.graph

        for paper in papers:
            graph.add_node(paper['id'], year=paper['year'])

            for ref in paper['references']:
                if ref in graph:
                    graph.add_edge(ref, paper['id'])

    # =============================== Importance Calculation Methods =============================== #

    def calculate_importance(self, model=None, graph=None):
        """
        Calculates importance scores based on selected model.

        Args:
            model (str, optional): Model to use. Uses self.model_to_execute if None.
            graph (nx.DiGraph, optional): Graph to analyze. Uses self.graph if None.

        Returns:
            dict: Dictionary mapping node IDs to importance scores
        """
        if model is None:
            model = self.model_to_execute
        if graph is None:
            graph = self.graph

        if model == "page_rank":
            return nx.pagerank(graph.reverse(), alpha=0.7, max_iter=100)
        elif model == "out_degree_centrality":
            return self.nx_out_degree_centrality(graph)
        elif model == "in_degree":
            return nx.in_degree_centrality(graph)
        elif model == "local_gravity":
            return self.local_gravity_model(graph)
        elif model == "eigenvector_centrality":
            return nx.eigenvector_centrality(graph, max_iter=1000)
        elif model == "page_rank_gravity":
            return self.compute_gravity_rank(graph.reverse())
        elif model == "effective_distance_gravity_model":
            return self.effective_distance_gravity_model(graph)

        return dict(graph.out_degree())  # Default: return out-degree

    # =============================== Normalization Methods =============================== #

    def normalize_scores(self, scores):
        """Normalize scores using z-score normalization."""
        values = list(scores.values())
        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return {k: 0.0 for k in scores}

        return {k: (v - mean) / std for k, v in scores.items()}

    def divide_by_avg(self, scores):
        """Normalize scores by dividing by average."""
        num_of_papers = max(1, len(scores))
        avg = sum(scores.values()) / num_of_papers
        if avg == 0:
            avg = 1
        print(f"AVG: {avg}")
        return {k: v / avg for k, v in scores.items()}

    def min_max_normalize_scores(self, scores):
        """Normalize scores using min-max normalization."""
        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            return {k: 0.0 for k in scores}

        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}

    # =============================== Model Implementation Methods =============================== #

    def local_gravity_model(self, graph):
        """
        Calculates influence scores using the local gravity model.
        Uses out-degree as mass and shortest path length as distance.

        Args:
            graph (nx.DiGraph): The citation network graph

        Returns:
            dict: Dictionary mapping node IDs to gravity scores
        """
        gravity_scores = {}
        degrees = dict(graph.out_degree())

        for i in graph.nodes:
            ki = degrees.get(i, 0)
            total_gravity = 0
            lengths = nx.single_source_shortest_path_length(graph, i)

            for j, dij in lengths.items():
                if i == j or dij == 0:
                    continue

                kj = degrees.get(j, 0)
                total_gravity += (ki * kj) / (dij ** 2)

            gravity_scores[i] = total_gravity

        # Display results
        self._display_top_bottom_scores(gravity_scores, "Local Gravity")
        print(f"Graph Size: {graph.size()}")

        return gravity_scores

    def compute_gravity_rank(self, graph, alpha=0.8, k=2, cutoff=None):
        """
        Compute gravity rank combining PageRank with distance-based gravity.

        Args:
            graph (nx.DiGraph): The citation network graph
            alpha (float): PageRank damping parameter
            k (int): Distance exponent for gravity calculation
            cutoff (int, optional): Maximum path length to consider

        Returns:
            dict: Dictionary mapping node IDs to gravity rank scores
        """
        if graph.number_of_edges() < 10:
            print("Almost no edges")
            return {n: 0.0 for n in graph.nodes}

        pr = nx.pagerank(graph, alpha=alpha, max_iter=100)

        if self.scale_pr:
            pr = {k: v * self.pr_scale_factor for k, v in pr.items()}

        # Compute shortest paths
        shortest_paths = dict(nx.all_pairs_shortest_path_length(graph, cutoff=cutoff))
        all_lengths = [d for dist in shortest_paths.values() for d in dist.values()]
        max_path_length = max(all_lengths) if all_lengths else 0
        print(f"Maximum shortest path length: {max_path_length}")
        print("Graph Size:", graph.size())

        gravity_rank = {}
        for i in graph.nodes:
            gravity_score = 0.0
            for j, dij in shortest_paths.get(i, {}).items():
                if i != j and dij > 0:
                    gravity_score += (pr[i] * pr[j]) / (dij ** k)
            gravity_rank[i] = gravity_score

        # Display results
        self._display_top_bottom_scores(gravity_rank, "Gravity Rank", use_scientific=True)

        # Compute average
        filtered_gravity = {node: score for node, score in gravity_rank.items() if score > 0}
        avg_score = sum(filtered_gravity.values()) / len(filtered_gravity) if filtered_gravity else 0.0
        print(f"\nAverage gravity score (non-zero): {avg_score:.6e}")
        print("Done Gravity")

        return gravity_rank

    def nx_out_degree_centrality(self, graph):
        """Calculate and display out-degree centrality."""
        nx_out_degree = dict(graph.out_degree())
        top_n = 30

        top_nx_out_degree = dict(sorted(nx_out_degree.items(),
                                        key=itemgetter(1),
                                        reverse=True)[:top_n])

        print(f"Top {top_n} nodes by nx-out-degree:")
        for node, score in top_nx_out_degree.items():
            print(f"{node}: {score}")

        return nx_out_degree

    def nx_page_rank(self, graph, alpha=0.85):
        """Calculate PageRank with optional scaling."""
        pr = nx.pagerank(graph, alpha=alpha)
        if self.scale_pr:
            pr = {k: v * self.pr_scale_factor for k, v in pr.items()}
        return pr

    def effective_distance_gravity_model(self, graph):
        """Calculate effective distance gravity model using external implementation."""
        return main(graph)

    # =============================== Utility Methods =============================== #

    def _display_top_bottom_scores(self, scores, model_name, top_n=10, use_scientific=False):
        """
        Display top and bottom scores in a formatted table.

        Args:
            scores (dict): Dictionary of scores
            model_name (str): Name of the model for display
            top_n (int): Number of top/bottom scores to display
            use_scientific (bool): Whether to use scientific notation
        """
        # Filter out zero values
        nonzero_scores = {node: score for node, score in scores.items() if score > 0}

        if not nonzero_scores:
            print(f"No non-zero scores for {model_name}")
            return

        # Sort by score
        sorted_scores = sorted(nonzero_scores.items(), key=lambda x: x[1], reverse=True)

        top_scores = sorted_scores[:top_n]
        bottom_scores = sorted_scores[-top_n:] if len(sorted_scores) > top_n else []

        # Format numbers
        fmt = ".6e" if use_scientific else ".6f"

        # Print header
        print(f"\n{'Top ' + str(top_n) + ' ' + model_name:<40} | {'Bottom ' + str(top_n) + ' ' + model_name:<40}")
        print("-" * 85)

        # Print scores
        for i in range(max(len(top_scores), len(bottom_scores))):
            top_str = f"{top_scores[i][0]}: {top_scores[i][1]:{fmt}}" if i < len(top_scores) else ""
            bottom_str = f"{bottom_scores[i][0]}: {bottom_scores[i][1]:{fmt}}" if i < len(bottom_scores) else ""
            print(f"{top_str:<40} | {bottom_str:<40}")

    def load_dataset(self, dataset_type="dblp"):
        """
        Load dataset based on type.

        Args:
            dataset_type (str): Type of dataset ("dblp" or "cithep")

        Returns:
            dict: Papers organized by year
        """
        if dataset_type.lower() == "dblp":
            return self.get_papers_dict_dblp_dataset()
        elif dataset_type.lower() == "cithep":
            return self.get_papers_dict_cithep_dataset()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def run_analysis(self, dataset_type="dblp"):
        """
        Run complete analysis pipeline.

        Args:
            dataset_type (str): Type of dataset to analyze

        Returns:
            dict: Analysis results
        """
        if not self.execute_analysis:
            print("Analysis execution is disabled")
            return {}

        # Load data
        print("Loading dataset...")
        papers_by_year = self.load_dataset(dataset_type)

        # Initialize graph
        print("Initializing graph...")
        self.init_graph()

        # Add papers for years of interest
        print("Adding papers to graph...")
        for year in self.years_of_interest:
            if year in papers_by_year:
                self.add_papers_of_year(papers_by_year[year])
                print(f"Added {len(papers_by_year[year])} papers from year {year}")

        # Calculate importance scores
        print(f"Calculating importance using {self.model_to_execute}...")
        scores = self.calculate_importance()

        print("Analysis complete!")
        return scores


# Example usage:
if __name__ == "__main__":
    # Create analyzer instance with custom parameters
    analyzer = CitationNetworkAnalyzer(
        min_year=1990,
        years_of_interest={1995, 2000, 2005},
        model_to_execute="local_gravity"
    )

    # Run analysis
    results = analyzer.run_analysis("dblp")

    print(f"Analysis completed. Found {len(results)} scored papers.")