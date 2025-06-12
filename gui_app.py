import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
import threading
from contextlib import redirect_stdout, redirect_stderr
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from Influence_Engine import *
import queue
import os
import pandas as pd
from collections import defaultdict
import numpy as np


# Class for redirecting console output to the GUI text widget (used to capture stdout/stderr)
class ConsoleRedirect:
    """Redirects stdout/stderr to the GUI console"""

    def __init__(self, text_widget, root):
        self.text_widget = text_widget
        self.root = root

    def write(self, string):
        # Schedule GUI update on main thread
        self.root.after(0, self._write_to_console, string)

    def _write_to_console(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()

    def flush(self):
        pass

# Main GUI Class
class CitationGUI:
    def __init__(self, root):
        self.engine = None
        self.root = root
        self.root.title("Citation Importance Analyzer")
        self.root.geometry("1200x800")

        # Initialize parameters with defaults from Influence_Engine
        self.file_path = r'D:\Datasets\dblp.v10\dblp-ref\dblp-ref-0.json'
        self.min_year = tk.IntVar(value=1990)
        self.max_papers_per_year = tk.IntVar(value=10000)
        self.years_of_interest = tk.StringVar(value="1995")
        self.plotter_var = tk.StringVar(value="pyplot")
        self.model_var = tk.StringVar(value="local_gravity")
        self.scale_pr = tk.BooleanVar(value=False)
        self.pr_scale_factor = tk.DoubleVar(value=1e5)
        self.significant_growth_threshold = tk.DoubleVar(value=3e-06)
        self.skip_insignificant = tk.BooleanVar(value=False)
        self.execute_analysis = tk.BooleanVar(value=True)
        self.dataset = tk.StringVar(value="DBLP_V10")

        # Thread-safe queue for communication from worker thread back to GUI
        self.result_queue = queue.Queue()
        self.is_running = False

        # Build the full GUI
        self.setup_ui()

        # Handle window closing properly
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        """Handle application closing"""
        if self.is_running:
            if messagebox.askokcancel("Quit", "Analysis is running. Do you want to quit?"):
                self.is_running = False
                self.root.quit()
                self.root.destroy()
        else:
            self.root.quit()
            self.root.destroy()

    def setup_ui(self):
        """Create the main UI with tabs"""
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
        """Set up the full parameters tab"""
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

        data_sets = ["DBLP_V10", "Cit-HepTh", "Cit-HepPh"]
        self.dataset_combo = ttk.Combobox(file_section, values=data_sets, textvariable=self.dataset, width=20)
        self.dataset_combo.pack(side=tk.LEFT)

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
            "local_gravity", "eigenvector_centrality", "page_rank_gravity", "effective_distance_gravity_model"
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
        tk.Checkbutton(skip_frame, text="Skip Insignificant Papers", variable=self.skip_insignificant).pack(
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
        """Set up console output tab"""
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
        """Set up results tab"""
        # Results visualization
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def select_file(self):
        """Open a file dialog to select the JSON file; update the file path"""
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if path:
            self.file_path = path
            self.file_label.config(text=f"Selected: {self.file_path}")

    def clear_console(self):
        """Clear the console output in the console tab"""
        self.console_text.delete(1.0, tk.END)

    def save_console(self):
        """Save the console output into a text file"""
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

    def get_current_params(self):
        """Get current parameter values from the GUI input fields safely"""
        try:
            return {
                'file_path': self.file_path,
                'min_year': self.min_year.get(),
                'max_papers_per_year': self.max_papers_per_year.get(),
                'years_of_interest': set(map(int, self.years_of_interest.get().strip().split(','))),
                'plotter': self.plotter_var.get(),
                'model': self.model_var.get(),
                'scale_pr': self.scale_pr.get(),
                'pr_scale_factor': self.pr_scale_factor.get(),
                'significant_growth_threshold': self.significant_growth_threshold.get(),
                'skip_insignificant': self.skip_insignificant.get(),
                'execute_analysis': self.execute_analysis.get(),
                'dataset': self.dataset.get()
            }
        except tk.TclError:
            return None
  

    def run_engine_thread(self):
        """Run engine in a separate thread to prevent GUI freezing"""
        if self.is_running:
            return  # Prevent multiple runs simultaneously

        # Get parameters safely on main thread
        params = self.get_current_params()
        if params is None:
            messagebox.showerror("Error", "Could not read parameters")
            return

        self.is_running = True
        self.run_button.config(state='disabled', text='Running...')
        self.progress.start()
        self.clear_console()

        # Create console redirect
        console_redirect = ConsoleRedirect(self.console_text, self.root)

        def run_analysis():
            """Run the actual engine logic inside the thread"""
            try:
                # start engine with params
                self.engine = CitationNetworkAnalyzer(
                    min_year=params['min_year'],
                    max_papers_per_year=params['max_papers_per_year'],
                    years_of_interest=params['years_of_interest'],
                    plotter=params['plotter'],
                    file_path=params['file_path'],
                    model_to_execute=params['model'],
                    scale_pr=params['scale_pr'],
                    pr_scale_factor=params['pr_scale_factor'],
                    significant_growth_threshold=params['significant_growth_threshold'],
                    skip_insignificant=params['skip_insignificant'],
                    execute_analysis=params['execute_analysis'],
                    dataset=params['dataset']
                )

                # Redirect stdout and stderr to console
                with redirect_stdout(console_redirect), redirect_stderr(console_redirect):
                    print("Starting analysis...")
                    print(f"Parameters:")
                    print(f"  File: {params['file_path']}")
                    print(f"  Min Year: {params['min_year']}")
                    print(f"  Max Papers/Year: {params['max_papers_per_year']}")
                    print(f"  Years of Interest: {params['years_of_interest']}")
                    print(f"  Model: {params['model']}")
                    print(f"  Plotter: {params['plotter']}")
                    print("-" * 50)
                    
                    
                    result = self.run_influence_engine(params)

                    # Send result back to main thread
                    self.result_queue.put(('success', result))
                    print("\nAnalysis completed successfully!")

            except Exception as e:
                error_msg = f"Error occurred: {str(e)}"
                print(error_msg)
                self.result_queue.put(('error', error_msg))
            finally:
                # Schedule cleanup on main thread
                self.root.after(0, self.analysis_finished)

        # Start analysis in separate thread
        thread = threading.Thread(target=run_analysis, daemon=True)
        thread.start()

    def analysis_finished(self):
        """Called when analysis is finished. Reset the GUI state and handle any results or errors"""
        self.is_running = False
        self.run_button.config(state='normal', text='Run Engine')
        self.progress.stop()

        # Check for results
        try:
            while not self.result_queue.empty():
                result_type, result_data = self.result_queue.get_nowait()
                if result_type == 'error':
                    messagebox.showerror("Error", f"Analysis failed: {result_data}")
                elif result_type == 'success':
                    # Handle successful result if needed
                    pass
        except queue.Empty:
            pass

    def run_influence_engine(self, params):
        """
        Performes the main workflow:
        - loads data from selected dataset
        - constructs citation graph year by year
        - calculates importance scores per paper
        - saves the results to CSV
        - optionally analyzes growth trends
        - generates plots
        """

        # Unpack parameters from GUI input
        file_path = params['file_path']
        dataset = params['dataset']
        model = params['model']
        plotter = params['plotter']
        years_of_interest = params['years_of_interest']
        execute_analysis = params['execute_analysis']
        significant_growth_threshold = params['significant_growth_threshold']

        # Validate file path
        if not file_path or not os.path.exists(file_path):
            raise Exception("Please select a valid JSON file")

        print("Loading papers from dataset...")

        # Load papers dictionary depending on selected dataset
        if dataset == "Cit-HepTh" or dataset == "Cit-HepPh":
            papers_dict = self.engine.get_papers_dict_cithep_dataset(file_path)
        elif dataset == "DBLP_V10":
            papers_dict = self.engine.get_papers_dict_dblp_dataset(file_path)
        else:
            raise Exception(f"Unknown dataset: {dataset}")

        print("Initializing graph...")
        g = self.engine.init_graph()        # initialize empty graph
        tracked_papers = defaultdict(list)  # papers to track importance scores
        tracked_years = []                  # save tracked years for score saving
        years_read_from_ds = []             # list of years available in dataset

        # Validate years of interest
        available_years = set(papers_dict.keys())
        years_of_interest_available = available_years.intersection(years_of_interest)

        if not years_of_interest_available:
            raise Exception(
                f"No papers found for years {sorted(years_of_interest)} in the dataset. "
                f"Available years: {sorted(available_years)}")

        print(f"Found papers for years of interest: {sorted(years_of_interest_available)}")

        # Calculate scores for all years
        for year in sorted(papers_dict.keys()):
            print(f"Processing year {year}...")
            papers_of_year = papers_dict[year]
            self.engine.add_papers_of_year(graph= g,papers= papers_of_year)
            scores = self.engine.calculate_importance(model=model, graph=g)

            # Start tracking new papers that appeared in years of interest
            if year in years_of_interest:
                for paper in papers_of_year:
                    tracked_papers[paper['id']] = []

            # Save year for tracked papers
            if year >= sorted(years_of_interest)[0]:
                tracked_years.append(year)

            # Save scores for tracked papers
            for paper_id in tracked_papers:
                tracked_papers[paper_id].append(scores.get(paper_id, 0))

            years_read_from_ds.append(year)

        print(f"Tracking {len(tracked_papers)} papers across {len(years_read_from_ds)} years")

        # Convert the scores to DataFrame and save to csv
        print("Saving the scores to CSV file...")
        if tracked_papers:
            # Create columns names: paperID + year columns
            columns = ['paperID'] + [f'{year}' for year in tracked_years]
            
            # Prepare data for DataFrame
            data = []
            for paper_id, scores in tracked_papers.items():
                row = [paper_id] + scores
                data.append(row)
                
            # Create DataFrame
            df = pd.DataFrame(data, columns = columns)
            
            #Save to csv
            output_filename = f"paper_scores_{self.dataset.get()}_{self.model_var.get()}_year_{self.years_of_interest.get()}.csv"
            df.to_csv(output_filename, index=False)
            print(f"Results saved to {output_filename}")
        else:
            print("No papers were tracked - CSV not created")


        # Analysis section
        if execute_analysis:
            #If execute_analysis is True: compute derivatives and filter papers based on significant_growth_threshold
            print("\nPerforming derivative analysis...")
            derivatives = {
                paper: np.gradient(scores)
                for paper, scores in tracked_papers.items()
            }

            if derivatives:
                # Paper with largest derivative at last year (fastest growing recently)
                most_growth_paper = max(derivatives.items(), key=lambda item: item[1][-1])
                print(f"Most rapidly growing paper: {most_growth_paper[0]}")

                # Paper with highest average growth over period
                avg_growth = {paper: np.mean(deriv) for paper, deriv in derivatives.items()}
                most_consistent_riser = max(avg_growth.items(), key=lambda item: item[1])
                print(f"Most consistent riser: {most_consistent_riser[0]}")

                # Paper with highest single year growth spike
                max_single_jump = {paper: max(deriv) for paper, deriv in derivatives.items()}
                biggest_spike_paper = max(max_single_jump.items(), key=lambda item: item[1])
                print(f"Paper with biggest spike: {biggest_spike_paper[0]}")

                # Select papers that exceed growth threshold
                significant_papers = {
                    paper_id
                    for paper_id, deriv in derivatives.items()
                    if np.mean(deriv) >= significant_growth_threshold
                }

                total_papers = len(derivatives)
                num_significant = len(significant_papers)
                print("=" * 40)
                print(f"Total papers: {total_papers}")
                print(f"Significant papers: {num_significant}")
                print(f"Percentage significant: {num_significant / total_papers * 100:.2f}%")
                print("=" * 40)
            else:
                significant_papers = set()
        else:
            # If execute_analysis is False: then all papers in tracked_papers will be treated as significant by default.
            significant_papers = set(tracked_papers.keys())

        # Plotting section
        print(f"\nGenerating plot using {plotter}...")
        if plotter == "pyplot":
            self.root.after(0, self.plot_with_matplotlib, tracked_papers, years_read_from_ds, significant_papers)
        elif plotter == "plotly":
            self.root.after(0, self.plot_with_plotly, tracked_papers, years_read_from_ds, significant_papers)

        return {
            'tracked_papers': tracked_papers,
            'years_read_from_ds': years_read_from_ds,
            'significant_papers': significant_papers
        }

    def plot_with_matplotlib(self, tracked_papers, years_read_from_ds, significant_papers):
        """Plot results using matplotlib in the GUI"""
        try:
            self.ax.clear()

            plotted_count = 0
                        
            for paper_id, score_history in tracked_papers.items():
                if paper_id not in significant_papers and self.skip_insignificant:
                    continue

                years_to_plot = years_read_from_ds[-len(score_history):]
                self.ax.plot(years_to_plot, score_history, marker='o', label=f'Paper {str(paper_id)[-6:]}', alpha=0.7)
                plotted_count += 1

            self.ax.set_xlabel("Year")
            self.ax.set_ylabel("Importance Score")
            
            if(self.execute_analysis.get()):
                self.ax.set_title(f"Tracked Papers with Significant Growth (>{self.significant_growth_threshold.get()})")
            else:
                self.ax.set_title("Tracked Papers")

            if plotted_count <= 10:  # Only show legend if not too many lines
                self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            self.canvas.draw()
            print(f"Plotted {plotted_count} papers in matplotlib")
        except Exception as e:
            print(f"Error plotting with matplotlib: {e}")

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
                if paper_id not in significant_papers and self.skip_insignificant:                
                    continue

                years_to_plot = years_read_from_ds[-len(score_history):]
                for year, score in zip(years_to_plot, score_history):
                    plot_data.append({
                        'Year': year,
                        'Score': score,
                        'Paper ID': str(paper_id)[-6:],  # Shortened ID
                    })

                plotted_count += 1

            if plot_data:
                df = pd.DataFrame(plot_data)
                fig = px.line(df, x='Year', y='Score', color='Paper ID', line_group='Paper ID')
                
                if(self.execute_analysis.get()):
                    fig.update_layout(
                        title=f"Importance Score Over Time (Significant Growth > {self.significant_growth_threshold.get()})",
                        xaxis_title="Year",
                        yaxis_title="Importance Score",
                        hovermode='closest',
                    )
                else:
                    fig.update_layout(
                        title="Importance Score Over Time",
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