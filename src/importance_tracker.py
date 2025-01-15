import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging


class ImportanceTracker:
    def __init__(self, base_output_dir: Path, n_runs: int):
        self.base_output_dir = Path(base_output_dir)
        self.n_runs = n_runs
        self.importances = []
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def add_run(self, run_number: int, importance_scores, task_names):
        run_df = pd.DataFrame({
            'task': task_names,
            'importance': importance_scores,
            'run': run_number
        })
        self.importances.append(run_df)

    def generate_final_analysis(self):
        # Combine all runs
        all_runs = pd.concat(self.importances, ignore_index=True)

        # Calculate statistics
        summary = all_runs.groupby('task').agg({
            'importance': ['mean', 'std', 'min', 'max', 'count']
        }).droplevel(0, axis=1)

        # Fill NaN values in std column with 0
        summary['std'] = summary['std'].fillna(0)

        # Sort by mean importance
        summary = summary.sort_values('mean', ascending=False)

        # Calculate top 10 frequency
        top_10_counts = {}
        for _, run_df in enumerate(self.importances):
            top_tasks = run_df.nlargest(10, 'importance')['task']
            for task in top_tasks:
                top_10_counts[task] = top_10_counts.get(task, 0) + 1

        # Add top 10 percentage and fill NaN with 0
        summary['top_10_percentage'] = pd.Series(top_10_counts) / self.n_runs * 100
        summary['top_10_percentage'] = summary['top_10_percentage'].fillna(0)

        # Save summary to CSV with better formatting
        summary.to_csv(self.base_output_dir / 'task_importance_summary.csv',
                       float_format='%.4f')

        # Create visualization
        plt.figure(figsize=(15, 12))
        top_20 = summary.head(20)

        # Create bar plot instead of boxplot for single runs
        if self.n_runs == 1:
            plt.barh(range(len(top_20)), top_20['mean'])
            plt.yticks(range(len(top_20)), top_20.index)
        else:
            # Use boxplot for multiple runs
            top_20_data = all_runs[all_runs['task'].isin(top_20.index)]
            sns.boxplot(data=top_20_data, x='importance', y='task',
                        order=top_20.index, whis=1.5)

        # Add error bars for multiple runs
        if self.n_runs > 1:
            plt.errorbar(x=top_20['mean'], y=range(len(top_20)),
                         xerr=top_20['std'], fmt='none', color='black',
                         capsize=5)

        # Add percentage annotations
        for i, (_, row) in enumerate(top_20.iterrows()):
            percentage = f"{row['top_10_percentage']:.1f}%"
            plt.text(plt.xlim()[1] * 1.05, i,
                     percentage,
                     verticalalignment='center',
                     fontsize=10)

        plt.xlabel('Importance Score')
        plt.ylabel('Task')
        plt.title('Task Importance Distribution Across All Runs (Top 20 Tasks)')
        plt.tight_layout()
        plt.savefig(self.base_output_dir / 'task_importance_distribution.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

        # Generate detailed report
        with open(self.base_output_dir / 'task_importance_report.txt', 'w') as f:
            f.write("Task Importance Analysis Summary\n")
            f.write(f"Based on {self.n_runs} experimental run{'s' if self.n_runs > 1 else ''}\n\n")
            f.write("Top 10 Most Important Tasks:\n\n")

            for task in summary.head(10).index:
                stats = summary.loc[task]
                f.write(f"{task}:\n")
                f.write(f"  Mean Importance: {stats['mean']:.4f}")
                if self.n_runs > 1:
                    f.write(f" ± {stats['std']:.4f}")
                f.write(f"\n  Range: [{stats['min']:.4f} - {stats['max']:.4f}]\n")
                f.write(f"  In Top 10: {stats['top_10_percentage']:.1f}% of runs\n\n")