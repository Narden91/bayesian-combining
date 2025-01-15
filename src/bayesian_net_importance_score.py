from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score
import logging


class BayesianImportanceTracker:
    def __init__(self, base_output_dir: Path, n_runs: int, n_permutations: int = 10):
        self.base_output_dir = Path(base_output_dir)
        self.n_runs = n_runs
        self.n_permutations = n_permutations
        self.node_importances = defaultdict(list)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def calculate_node_importance(self, model, data, target_node):
        """Calculate node importance using permutation-based approach."""
        try:
            inference = VariableElimination(model)
            baseline_predictions = []

            # Get baseline predictions
            for _, row in data.iterrows():
                evidence = {col: row[col] for col in data.columns if col != target_node}
                result = inference.query([target_node], evidence=evidence)
                pred = np.argmax(result.values)
                baseline_predictions.append(pred)

            baseline_score = accuracy_score(data[target_node], baseline_predictions)
            importances = {}

            # Calculate importance for each node
            for node in data.columns:
                if node != target_node:
                    importance = self._calculate_single_node_importance(
                        model, inference, data, node, target_node, baseline_score
                    )
                    importances[node] = importance

            return importances

        except Exception as e:
            logging.error(f"Error calculating node importance: {str(e)}")
            return {}

    def _calculate_single_node_importance(self, model, inference, data, node, target_node, baseline_score):
        """Calculate importance for a single node using permutation."""
        importance_scores = []

        for _ in range(self.n_permutations):
            permuted_data = data.copy()
            permuted_data[node] = np.random.permutation(permuted_data[node].values)

            predictions = []
            for _, row in permuted_data.iterrows():
                evidence = {col: row[col] for col in permuted_data.columns if col != target_node}
                result = inference.query([target_node], evidence=evidence)
                pred = np.argmax(result.values)
                predictions.append(pred)

            permuted_score = accuracy_score(data[target_node], predictions)
            importance = baseline_score - permuted_score
            importance_scores.append(importance)

        return np.mean(importance_scores)

    def add_run(self, run_number: int, model, data, target_node: str):
        """Record importance scores for a single run."""
        try:
            importances = self.calculate_node_importance(model, data, target_node)

            for node, importance in importances.items():
                self.node_importances[node].append({
                    'run': run_number,
                    'importance': importance
                })

            logging.info(f"Successfully recorded importance scores for run {run_number}")

        except Exception as e:
            logging.error(f"Error in add_run for run {run_number}: {str(e)}")

    def generate_final_analysis(self):
        """Generate final importance analysis report."""
        try:
            if not self.node_importances:
                logging.warning("No importance data collected. Skipping analysis.")
                return

            # Calculate mean importance across runs
            mean_importances = {}
            std_importances = {}

            for node, scores in self.node_importances.items():
                importance_values = [s['importance'] for s in scores]
                mean_importances[node] = np.mean(importance_values)
                std_importances[node] = np.std(importance_values)

            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'Node': mean_importances.keys(),
                'Mean_Importance': mean_importances.values(),
                'Std_Importance': std_importances.values()
            })

            # Sort by importance
            importance_df = importance_df.sort_values('Mean_Importance', ascending=False)

            # Save results
            importance_df.to_csv(self.base_output_dir / 'node_importance.csv', index=False)

            # Create visualization
            self._create_importance_plot(importance_df)

            # Generate text report
            self._generate_text_report(importance_df)

        except Exception as e:
            logging.error(f"Error in generate_final_analysis: {str(e)}")
            raise

    def _create_importance_plot(self, importance_df):
        """Create bar plot of node importances."""
        try:
            plt.figure(figsize=(12, 6))
            plt.errorbar(
                x=range(len(importance_df)),
                y=importance_df['Mean_Importance'],
                yerr=importance_df['Std_Importance'],
                fmt='o',
                capsize=5
            )
            plt.xticks(range(len(importance_df)),
                       importance_df['Node'],
                       rotation=45,
                       ha='right')
            plt.xlabel('Node')
            plt.ylabel('Importance Score')
            plt.title('Node Importance in Bayesian Network')
            plt.tight_layout()
            plt.savefig(self.base_output_dir / 'node_importance.png')
            plt.close()

        except Exception as e:
            logging.error(f"Error creating importance plot: {str(e)}")

    def _generate_text_report(self, importance_df):
        """Generate text report of node importances."""
        try:
            report = [
                "Bayesian Network Node Importance Analysis",
                f"Based on {self.n_runs} experimental runs",
                f"Using {self.n_permutations} permutations per node\n",
                "Top Nodes by Importance:\n"
            ]

            for _, row in importance_df.iterrows():
                report.append(
                    f"{row['Node']}:\n"
                    f"  Mean Importance: {row['Mean_Importance']:.4f}\n"
                    f"  Std Importance: {row['Std_Importance']:.4f}\n"
                )

            with open(self.base_output_dir / 'importance_report.txt', 'w') as f:
                f.write('\n'.join(report))

        except Exception as e:
            logging.error(f"Error generating text report: {str(e)}")