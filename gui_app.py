import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
import threading
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import networkx as nx
from Influence_Engine import *


class ConsoleRedirect:
    """Redirects stdout/stderr to the GUI console"""

    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()

    def flush(self):
        pass


class CitationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Citation Importance Analyzer")
        self.root.geometry("1200x800")

        # Initialize parameters with defaults from Influence_Engine
        self.file_path = FILE_PATH
        self.min_year = tk.IntVar(value=MIN_YEAR)
        self.max_papers_per_year = tk.IntVar(value=MAX_PAPERS_PER_YEAR)
        self.years_of_interest = tk.StringVar(value="1995")
        self.plotter_var = tk.StringVar(value=PLOTTER)
        self.model_var = tk.StringVar(value=MODEL_TO_EXECUTE)
        self.scale_pr = tk.BooleanVar(value=SCALE_PR)
        self.pr_scale_factor = tk.DoubleVar(value=PR_SCALE_FACTOR)
        self.significant_growth_threshold = tk.DoubleVar(value=SIGNIFICANT_GROWTH_THRESHOLD)
        self.skip_unsignificants = tk.BooleanVar(value=SKIP_UNSIGNIFICANTS)
        self.execute_analysis = tk.BooleanVar(value=EXECUTE_ANALYSIS)
        self.data_set = tk.StringVar(value=DATASET)

        self.setup_ui()

    def setup_ui(self):
        # Create main frame with scrollable content
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Parameters tab
        params_frame = ttk.Frame(notebook)
        notebook.add(params_frame, text="Parameters")

        # Console tab
        console_frame = ttk.Frame(notebook)
        notebook.add(console_frame, text="Console Output")

        # Results tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Results")

        self.setup_parameters_tab(params_frame)
        self.setup_console_tab(console_frame)
        self.setup_results_tab(results_frame)

    def setup_parameters_tab(self, parent):
        # Create scrollable frame for parameters
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # File selection section
        file_section = ttk.LabelFrame(scrollable_frame, text="File Selection", padding="10")
        file_section.pack(fill=tk.X, pady=5)

        tk.Button(file_section, text="Select JSON File", command=self.select_file).pack(pady=5)
        self.file_label = tk.Label(file_section, text=f"Selected: {self.file_path}", wraplength=600)
        self.file_label.pack()

        data_sets = [ "DBLP_V10", "Cit-HepTh", "Cit-HepPh" ]
        self.model_combo = ttk.Combobox(file_section, values=data_sets, textvariable=self.data_set, width=20)
        self.model_combo.pack(side=tk.LEFT)

        # Basic parameters section
        basic_section = ttk.LabelFrame(scrollable_frame, text="Basic Parameters", padding="10")
        basic_section.pack(fill=tk.X, pady=5)

        # Min Year
        min_year_frame = tk.Frame(basic_section)
        min_year_frame.pack(fill=tk.X, pady=2)
        tk.Label(min_year_frame, text="Minimum Year:", width=25, anchor='w').pack(side=tk.LEFT)
        tk.Entry(min_year_frame, textvariable=self.min_year, width=10).pack(side=tk.LEFT)
        tk.Label(min_year_frame, text="Papers before this year will be ignored", fg="gray").pack(side=tk.LEFT,
                                                                                                 padx=(10, 0))

        # Max Papers Per Year
        max_papers_frame = tk.Frame(basic_section)
        max_papers_frame.pack(fill=tk.X, pady=2)
        tk.Label(max_papers_frame, text="Max Papers Per Year:", width=25, anchor='w').pack(side=tk.LEFT)
        tk.Entry(max_papers_frame, textvariable=self.max_papers_per_year, width=10).pack(side=tk.LEFT)
        tk.Label(max_papers_frame, text="Maximum papers to process per year", fg="gray").pack(side=tk.LEFT,
                                                                                              padx=(10, 0))

        # Years of Interest
        years_frame = tk.Frame(basic_section)
        years_frame.pack(fill=tk.X, pady=2)
        tk.Label(years_frame, text="Years of Interest:", width=25, anchor='w').pack(side=tk.LEFT)
        tk.Entry(years_frame, textvariable=self.years_of_interest, width=20).pack(side=tk.LEFT)
        tk.Label(years_frame, text="Comma-separated years (e.g., 1995,2000,2005)", fg="gray").pack(side=tk.LEFT,
                                                                                                   padx=(10, 0))

        # Model Selection
        model_section = ttk.LabelFrame(scrollable_frame, text="Model Configuration", padding="10")
        model_section.pack(fill=tk.X, pady=5)

        model_frame = tk.Frame(model_section)
        model_frame.pack(fill=tk.X, pady=2)
        tk.Label(model_frame, text="Model:", width=25, anchor='w').pack(side=tk.LEFT)
        models = [
            "in_degree", "out_degree_centrality", "page_rank",
            "local_gravity", "eigenvector_centrality", "page_rank_gravity"
        ]
        self.model_combo = ttk.Combobox(model_frame, values=models, textvariable=self.model_var, width=20)
        self.model_combo.pack(side=tk.LEFT)

        # Plotter Selection
        plotter_frame = tk.Frame(model_section)
        plotter_frame.pack(fill=tk.X, pady=2)
        tk.Label(plotter_frame, text="Plotter:", width=25, anchor='w').pack(side=tk.LEFT)
        plotter_combo = ttk.Combobox(plotter_frame, values=["pyplot", "plotly"], textvariable=self.plotter_var,
                                     width=20)
        plotter_combo.pack(side=tk.LEFT)
        tk.Label(plotter_frame, text="pyplot (static) or plotly (dynamic)", fg="gray").pack(side=tk.LEFT, padx=(10, 0))

        # PageRank specific parameters
        pr_section = ttk.LabelFrame(scrollable_frame, text="PageRank Parameters", padding="10")
        pr_section.pack(fill=tk.X, pady=5)

        scale_pr_frame = tk.Frame(pr_section)
        scale_pr_frame.pack(fill=tk.X, pady=2)
        tk.Checkbutton(scale_pr_frame, text="Scale PageRank Results", variable=self.scale_pr).pack(side=tk.LEFT)

        pr_scale_frame = tk.Frame(pr_section)
        pr_scale_frame.pack(fill=tk.X, pady=2)
        tk.Label(pr_scale_frame, text="PR Scale Factor:", width=25, anchor='w').pack(side=tk.LEFT)
        tk.Entry(pr_scale_frame, textvariable=self.pr_scale_factor, width=15).pack(side=tk.LEFT)
        tk.Label(pr_scale_frame, text="Scaling factor for PageRank results", fg="gray").pack(side=tk.LEFT, padx=(10, 0))

        # Analysis parameters
        analysis_section = ttk.LabelFrame(scrollable_frame, text="Analysis Parameters", padding="10")
        analysis_section.pack(fill=tk.X, pady=5)

        threshold_frame = tk.Frame(analysis_section)
        threshold_frame.pack(fill=tk.X, pady=2)
        tk.Label(threshold_frame, text="Significant Growth Threshold:", width=25, anchor='w').pack(side=tk.LEFT)
        tk.Entry(threshold_frame, textvariable=self.significant_growth_threshold, width=15).pack(side=tk.LEFT)
        tk.Label(threshold_frame, text="Minimum change required to be plotted", fg="gray").pack(side=tk.LEFT,
                                                                                                padx=(10, 0))

        skip_frame = tk.Frame(analysis_section)
        skip_frame.pack(fill=tk.X, pady=2)
        tk.Checkbutton(skip_frame, text="Skip Unsignificant Papers", variable=self.skip_unsignificants).pack(
            side=tk.LEFT)

        execute_frame = tk.Frame(analysis_section)
        execute_frame.pack(fill=tk.X, pady=2)
        tk.Checkbutton(execute_frame, text="Execute Analysis", variable=self.execute_analysis).pack(side=tk.LEFT)

        # Run button
        run_frame = tk.Frame(scrollable_frame)
        run_frame.pack(fill=tk.X, pady=20)
        self.run_button = tk.Button(run_frame, text="Run Engine", command=self.run_engine_thread,
                                    bg="green", fg="white", font=("Arial", 12, "bold"), height=2)
        self.run_button.pack()

        self.progress = ttk.Progressbar(run_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def setup_console_tab(self, parent):
        # Console output
        console_frame = tk.Frame(parent)
        console_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tk.Label(console_frame, text="Console Output:", font=("Arial", 12, "bold")).pack(anchor='w')

        self.console_text = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD,
                                                      font=("Consolas", 9), bg="black", fg="green")
        self.console_text.pack(fill=tk.BOTH, expand=True, pady=5)

        # Clear console button
        clear_frame = tk.Frame(console_frame)
        clear_frame.pack(fill=tk.X, pady=5)
        tk.Button(clear_frame, text="Clear Console", command=self.clear_console).pack(side=tk.LEFT)
        tk.Button(clear_frame, text="Save Console Output", command=self.save_console).pack(side=tk.LEFT, padx=5)

    def setup_results_tab(self, parent):
        # Results visualization
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if path:
            self.file_path = path
            self.file_label.config(text=f"Selected: {self.file_path}")

    def clear_console(self):
        self.console_text.delete(1.0, tk.END)

    def save_console(self):
        content = self.console_text.get(1.0, tk.END)
        if content.strip():
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'w') as f:
                    f.write(content)
                messagebox.showinfo("Success", "Console output saved successfully!")

    def update_global_params(self):
        """Update global parameters in Influence_Engine module"""
        global MIN_YEAR, MAX_PAPERS_PER_YEAR, YEARS_OF_INTEREST, PLOTTER, MODEL_TO_EXECUTE
        global SCALE_PR, PR_SCALE_FACTOR, SIGNIFICANT_GROWTH_THRESHOLD, SKIP_UNSIGNIFICANTS, EXECUTE_ANALYSIS

        MIN_YEAR = self.min_year.get()
        MAX_PAPERS_PER_YEAR = self.max_papers_per_year.get()

        # Parse years of interest
        years_str = self.years_of_interest.get().strip()
        if years_str:
            try:
                years_list = [int(year.strip()) for year in years_str.split(',')]
                YEARS_OF_INTEREST = set(years_list)
            except ValueError:
                messagebox.showerror("Error",
                                     "Invalid years format. Use comma-separated integers (e.g., 1995,2000,2005)")
                return False

        PLOTTER = self.plotter_var.get()
        MODEL_TO_EXECUTE = self.model_var.get()
        SCALE_PR = self.scale_pr.get()
        PR_SCALE_FACTOR = self.pr_scale_factor.get()
        SIGNIFICANT_GROWTH_THRESHOLD = self.significant_growth_threshold.get()
        SKIP_UNSIGNIFICANTS = self.skip_unsignificants.get()
        EXECUTE_ANALYSIS = self.execute_analysis.get()
        DATASET = self.data_set.get()
        return True

    def run_engine_thread(self):
        """Run engine in a separate thread to prevent GUI freezing"""
        if not self.update_global_params():
            return

        self.run_button.config(state='disabled', text='Running...')
        self.progress.start()
        self.clear_console()

        # Create console redirect
        console_redirect = ConsoleRedirect(self.console_text)

        def run_analysis():
            try:
                # Redirect stdout and stderr to console
                with redirect_stdout(console_redirect), redirect_stderr(console_redirect):
                    self.console_text.insert(tk.END, "Starting analysis...\n")
                    self.console_text.insert(tk.END, f"Parameters:\n")
                    self.console_text.insert(tk.END, f"  File: {self.file_path}\n")
                    self.console_text.insert(tk.END, f"  Min Year: {MIN_YEAR}\n")
                    self.console_text.insert(tk.END, f"  Max Papers/Year: {MAX_PAPERS_PER_YEAR}\n")
                    self.console_text.insert(tk.END, f"  Years of Interest: {YEARS_OF_INTEREST}\n")
                    self.console_text.insert(tk.END, f"  Model: {MODEL_TO_EXECUTE}\n")
                    self.console_text.insert(tk.END, f"  Plotter: {PLOTTER}\n")
                    self.console_text.insert(tk.END, "-" * 50 + "\n")

                    self.run_influence_engine()

                    self.console_text.insert(tk.END, "\nAnalysis completed successfully!\n")

            except Exception as e:
                self.console_text.insert(tk.END, f"\nError occurred: {str(e)}\n")
                messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            finally:
                self.root.after(0, self.analysis_finished)

        # Start analysis in separate thread
        thread = threading.Thread(target=run_analysis)
        thread.daemon = True
        thread.start()

    def analysis_finished(self):
        """Called when analysis is finished"""
        self.run_button.config(state='normal', text='Run Analysis')
        self.progress.stop()

    def run_influence_engine(self):
        """Main analysis logic (adapted from original code)"""
        if not self.file_path or not os.path.exists(self.file_path):
            raise Exception("Please select a valid JSON file")

        self.ax.clear()

        print("Loading papers from dataset...")
        #papers_dict = get_papers_dict(self.file_path)
        if DATASET == "Cit-HepTh" or DATASET =="Cit-HepPh":
            papers_dict = get_papers_dict_CitHep_dataset(FILE_PATH)  # Extract data from dataset.

        if DATASET == "DBLP_V10":
            papers_dict = get_papers_dict_DBLP_dataset(FILE_PATH)  # Extract data from dataset.

        print("Initializing graph...")
        g = init_graph()
        tracked_papers = defaultdict(list)
        years_read_from_ds = []

        # Check if any years of interest exist in dataset
        available_years = set(papers_dict.keys())
        years_of_interest_available = YEARS_OF_INTEREST.intersection(available_years)

        if not years_of_interest_available:
            raise Exception(
                f"No papers found for years {YEARS_OF_INTEREST} in the dataset. Available years: {sorted(available_years)}")

        print(f"Found papers for years of interest: {sorted(years_of_interest_available)}")

        # Calculate scores for all years
        for year in sorted(papers_dict.keys()):
            print(f"Processing year {year}...")
            papers_of_year = papers_dict[year]
            add_papers_of_year(g, papers_of_year)
            scores = calculate_importance(g, MODEL_TO_EXECUTE)

            if year in YEARS_OF_INTEREST:
                for paper in papers_of_year:
                    tracked_papers[paper['id']] = []

            for paper_id in tracked_papers:
                tracked_papers[paper_id].append(scores.get(paper_id, 0))

            years_read_from_ds.append(year)

        print(f"Tracking {len(tracked_papers)} papers across {len(years_read_from_ds)} years")

        # Analysis section
        if EXECUTE_ANALYSIS:
            print("\nPerforming derivative analysis...")
            derivatives = {
                paper: np.gradient(extracted_scores)
                for paper, extracted_scores in tracked_papers.items()
            }

            if derivatives:
                most_growth_paper = max(derivatives.items(), key=lambda item: item[1][-1])
                print(f"Most rapidly growing paper: {most_growth_paper[0]}")

                avg_growth = {paper: np.mean(deriv) for paper, deriv in derivatives.items()}
                most_consistent_riser = max(avg_growth.items(), key=lambda item: item[1])
                print(f"Most consistent riser: {most_consistent_riser[0]}")

                max_single_jump = {paper: max(deriv) for paper, deriv in derivatives.items()}
                biggest_spike_paper = max(max_single_jump.items(), key=lambda item: item[1])
                print(f"Paper with biggest spike: {biggest_spike_paper[0]}")

                significant_papers = set()
                for paper_id, deriv in derivatives.items():
                    mean_deriv = np.mean(deriv)
                    if mean_deriv >= SIGNIFICANT_GROWTH_THRESHOLD:
                        significant_papers.add(paper_id)

                total_papers = len(derivatives)
                num_significant = len(significant_papers)
                print("=" * 40)
                print(f"Total papers: {total_papers}")
                print(f"Significant papers: {num_significant}")
                print(f"Percentage significant: {num_significant / total_papers * 100:.2f}%")
                print("=" * 40)
        else:
            significant_papers = set(tracked_papers.keys())

        # Plotting section
        print(f"\nGenerating plot using {PLOTTER}...")
        if PLOTTER == "pyplot":
            self.plot_with_matplotlib(tracked_papers, years_read_from_ds, significant_papers)
        elif PLOTTER == "plotly":
            self.plot_with_plotly(tracked_papers, years_read_from_ds, significant_papers)

    def plot_with_matplotlib(self, tracked_papers, years_read_from_ds, significant_papers):
        """Plot results using matplotlib in the GUI"""
        self.ax.clear()

        plotted_count = 0
        for paper_id, score_history in tracked_papers.items():
            if paper_id not in significant_papers and SKIP_UNSIGNIFICANTS:
                continue

            years_to_plot = years_read_from_ds[-len(score_history):]
            self.ax.plot(years_to_plot, score_history, marker='o', label=f'Paper {str(paper_id)[-6:]}', alpha=0.7)
            plotted_count += 1

            # Limit number of plotted lines to prevent overcrowding
            if plotted_count > 20:
                break

        self.ax.set_xlabel("Year")
        self.ax.set_ylabel("Importance Score")
        self.ax.set_title(f"Tracked Papers with Significant Growth (>{SIGNIFICANT_GROWTH_THRESHOLD})")

        if plotted_count <= 10:  # Only show legend if not too many lines
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        self.canvas.draw()
        print(f"Plotted {plotted_count} papers in matplotlib")

    def plot_with_plotly(self, tracked_papers, years_read_from_ds, significant_papers):
        """Plot results using plotly (external browser)"""
        try:
            import plotly.express as px
            import plotly.io as pio
            import pandas as pd

            pio.renderers.default = "browser"

            plot_data = []
            plotted_count = 0

            for paper_id, score_history in tracked_papers.items():
                if paper_id not in significant_papers and SKIP_UNSIGNIFICANTS:
                    continue

                years_to_plot = years_read_from_ds[-len(score_history):]
                for year, score in zip(years_to_plot, score_history):
                    plot_data.append({
                        'Year': year,
                        'Score': score,
                        'Paper ID': str(paper_id)[-6:],  # Shortened ID
                    })

                plotted_count += 1
                if plotted_count > 50:  # Limit for performance
                    break

            if plot_data:
                df = pd.DataFrame(plot_data)
                fig = px.line(df, x='Year', y='Score', color='Paper ID', line_group='Paper ID')
                fig.update_layout(
                    title=f"Importance Score Over Time (Significant Growth > {SIGNIFICANT_GROWTH_THRESHOLD})",
                    xaxis_title="Year",
                    yaxis_title="Importance Score",
                    hovermode='closest',
                )
                fig.show()
                print(f"Opened plotly visualization with {plotted_count} papers in browser")
            else:
                print("No data to plot with plotly")

        except ImportError:
            print("Plotly not available. Please install plotly: pip install plotly")
        except Exception as e:
            print(f"Error creating plotly visualization: {e}")


# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = CitationGUI(root)
    root.mainloop()