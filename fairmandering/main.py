# fairmandering/main.py

import logging
import sys
import threading
from tkinter import (
    Tk, Label, Entry, Button, StringVar, IntVar, DoubleVar, Text, Scrollbar, RIGHT, Y, END, LEFT, BOTH, Frame
)
from tkinter import messagebox
from .config import Config
from .data_processing import DataProcessor, DataProcessingError
from .optimization import optimize_districting, generate_ensemble_plans
from .fairness_evaluation import evaluate_fairness
from .visualization import (
    visualize_district_map,
    plot_fairness_metrics,
    visualize_district_characteristics,
    generate_explainable_report,
    visualize_trend_analysis
)
from .analysis import (
    analyze_districts,
    save_analysis_results,
    perform_sensitivity_analysis,
    compare_ensemble_plans,
    rank_plans
)
from .versioning import save_plan
import argparse

logger = logging.getLogger(__name__)

class FairmanderingGUI:
    def __init__(self, master):
        self.master = master
        master.title("Fairmandering Redistricting System")

        # Configure logging
        logging.basicConfig(
            filename=Config.LOG_FILE,
            level=Config.LOG_LEVEL,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # State FIPS Code
        self.state_fips_label = Label(master, text="State FIPS Code:")
        self.state_fips_label.pack()
        self.state_fips_var = StringVar(value=Config.STATE_FIPS)
        self.state_fips_entry = Entry(master, textvariable=self.state_fips_var)
        self.state_fips_entry.pack()

        # Start Button
        self.start_button = Button(master, text="Start Redistricting", command=self.start_process)
        self.start_button.pack(pady=10)

        # Status Text
        self.status_frame = Frame(master)
        self.status_frame.pack(fill=BOTH, expand=True)
        self.status_text = Text(self.status_frame, wrap='word')
        self.status_scroll = Scrollbar(self.status_frame, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=self.status_scroll.set)
        self.status_scroll.pack(side=RIGHT, fill=Y)
        self.status_text.pack(side=LEFT, fill=BOTH, expand=True)

    def log(self, message):
        self.status_text.insert(END, message + '\n')
        self.status_text.see(END)
        logger.info(message)

    def start_process(self):
        thread = threading.Thread(target=self.run_redistricting)
        thread.start()

    def run_redistricting(self):
        self.start_button.config(state='disabled')
        self.log("Starting the redistricting process.")

        # Perform system checks
        try:
            self.log("Performing system checks...")
            Config.validate()
            logger.info("Configuration validated successfully.")
            self.log("Configuration validated successfully.")

            # Check for required packages
            required_packages = [
                'pandas', 'geopandas', 'numpy', 'scipy', 'requests', 'census', 'pymoo',
                'matplotlib', 'seaborn', 'folium', 'python-dotenv', 'joblib', 'scikit-learn',
                'cryptography', 'us', 'plotly', 'redis', 'tkinter'
            ]
            for pkg in required_packages:
                __import__(pkg)
            self.log("All required packages are installed.")
            logger.info("All required packages are installed.")

        except Exception as e:
            error_message = f"System check failed: {e}"
            self.log(error_message)
            logger.error(error_message)
            messagebox.showerror("System Check Failed", error_message)
            self.start_button.config(state='normal')
            return

        # Parse arguments
        state_fips = self.state_fips_var.get()

        # Get the number of districts dynamically from the Census API
        try:
            self.log(f"Retrieving number of districts for state FIPS {state_fips}...")
            num_districts = Config.get_num_districts(state_fips)
            self.log(f"Number of districts for state FIPS {state_fips}: {num_districts}")
            logger.info(f"Number of districts for state FIPS {state_fips}: {num_districts}")
        except Exception as e:
            error_message = f"Failed to get the number of districts: {e}"
            self.log(error_message)
            logger.error(error_message)
            messagebox.showerror("District Retrieval Failed", error_message)
            self.start_button.config(state='normal')
            return

        # Data Processing
        processor = DataProcessor(state_fips, Config.STATE_NAME)
        try:
            self.log("Integrating data...")
            data = processor.integrate_data()
            self.log("Data integration complete.")
            logger.info("Data integration complete.")
        except DataProcessingError as e:
            error_message = f"Data processing failed: {e}"
            self.log(error_message)
            logger.error(error_message)
            messagebox.showerror("Data Processing Failed", error_message)
            self.start_button.config(state='normal')
            return
        except Exception as e:
            error_message = f"Unexpected error during data processing: {e}"
            self.log(error_message)
            logger.error(error_message)
            messagebox.showerror("Data Processing Error", error_message)
            self.start_button.config(state='normal')
            return

        # Optimization
        try:
            self.log("Starting optimization...")
            district_assignments, _ = optimize_districting(data, seeds=[1, 2, 3, 4, 5])
            best_assignment = district_assignments[0]  # For simplicity, use the first solution
            self.log("Optimization completed.")
            logger.info("Optimization completed.")
        except Exception as e:
            error_message = f"Optimization failed: {e}"
            self.log(error_message)
            logger.error(error_message)
            messagebox.showerror("Optimization Failed", error_message)
            self.start_button.config(state='normal')
            return

        # Fairness Evaluation
        try:
            self.log("Evaluating fairness...")
            fairness_metrics = evaluate_fairness(data, best_assignment)
            self.log("Fairness evaluation completed.")
            logger.info("Fairness evaluation completed.")
        except Exception as e:
            error_message = f"Fairness evaluation failed: {e}"
            self.log(error_message)
            logger.error(error_message)
            messagebox.showerror("Fairness Evaluation Failed", error_message)
            self.start_button.config(state='normal')
            return

        # Analysis
        try:
            self.log("Analyzing districts...")
            analysis_results = analyze_districts(data)
            save_analysis_results(analysis_results)
            self.log("Analysis completed and results saved.")
            logger.info("Analysis completed and results saved.")
        except Exception as e:
            error_message = f"Analysis failed: {e}"
            self.log(error_message)
            logger.error(error_message)
            messagebox.showerror("Analysis Failed", error_message)
            self.start_button.config(state='normal')
            return

        # Visualization
        try:
            self.log("Generating visualizations...")
            visualize_district_map(data, best_assignment)
            plot_fairness_metrics(fairness_metrics)
            visualize_district_characteristics(data)
            visualize_trend_analysis(data)
            generate_explainable_report(fairness_metrics, analysis_results)
            self.log("Visualizations generated and saved.")
            logger.info("Visualizations generated and saved.")
        except Exception as e:
            error_message = f"Visualization failed: {e}"
            self.log(error_message)
            logger.error(error_message)
            messagebox.showerror("Visualization Failed", error_message)
            self.start_button.config(state='normal')
            return

        # Versioning
        try:
            self.log("Saving districting plan version...")
            metadata = {'author': 'Your Name', 'description': 'Initial plan'}
            save_plan(best_assignment, metadata, version='1.0.0')
            self.log("Districting plan saved.")
            logger.info("Districting plan saved.")
        except Exception as e:
            error_message = f"Versioning failed: {e}"
            self.log(error_message)
            logger.error(error_message)
            messagebox.showerror("Versioning Failed", error_message)
            self.start_button.config(state='normal')
            return

        # Sensitivity Analysis
        try:
            self.log("Performing sensitivity analysis...")
            perform_sensitivity_analysis(data, best_assignment)
            self.log("Sensitivity analysis completed and results saved.")
            logger.info("Sensitivity analysis completed and results saved.")
        except Exception as e:
            error_message = f"Sensitivity analysis failed: {e}"
            self.log(error_message)
            logger.error(error_message)
            messagebox.showerror("Sensitivity Analysis Failed", error_message)
            self.start_button.config(state='normal')
            return

        # Ensemble Analysis
        try:
            self.log("Generating ensemble plans...")
            ensemble = generate_ensemble_plans(data, num_plans=5)
            self.log("Comparing ensemble plans...")
            metrics_df = compare_ensemble_plans(data, ensemble)
            weights = Config.OBJECTIVE_WEIGHTS
            self.log("Ranking plans based on metrics...")
            ranked_plans = rank_plans(metrics_df, weights)
            self.log("Ensemble analysis completed and ranked plans saved.")
            logger.info("Ensemble analysis completed and ranked plans saved.")
        except Exception as e:
            error_message = f"Ensemble analysis failed: {e}"
            self.log(error_message)
            logger.error(error_message)
            messagebox.showerror("Ensemble Analysis Failed", error_message)
            self.start_button.config(state='normal')
            return

        self.log("Redistricting process completed successfully.")
        messagebox.showinfo("Success", "Redistricting process completed successfully.")
        self.start_button.config(state='normal')


def main():
    root = Tk()
    gui = FairmanderingGUI(root)
    root.geometry("600x400")
    root.mainloop()


if __name__ == "__main__":
    main()
