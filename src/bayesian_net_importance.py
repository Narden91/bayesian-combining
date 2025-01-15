import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import networkx as nx
from collections import defaultdict
import logging


class BayesianImportanceTracker:
    def __init__(self, base_output_dir: Path, n_runs: int, top_k: int = 20):
        """Initialize the Bayesian Network importance tracker."""
        self.base_output_dir = Path(base_output_dir)
        self.n_runs = n_runs
        self.top_k = top_k
        self.node_data = defaultdict(list)
        self.markov_blanket_counts = defaultdict(int)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def add_run(self, run_number: int, model, target_node: str, log_likelihood: float):
        """Record metrics for a single run with proper node name handling."""
        try:
            # Get Markov blanket for this run
            markov_blanket = set(model.get_markov_blanket(target_node))

            # Update Markov blanket counts with original node names
            for node in markov_blanket:
                self.markov_blanket_counts[str(node)] += 1

            # Create NetworkX graph for centrality analysis
            G = nx.DiGraph()
            for edge in model.edges():
                G.add_edge(str(edge[0]), str(edge[1]))

            # Calculate centrality metrics with string node names
            centrality_metrics = {
                'degree': nx.degree_centrality(G),
                'betweenness': nx.betweenness_centrality(G),
                'closeness': nx.closeness_centrality(G)
            }

            # Store node metrics with proper string conversion
            for node in G.nodes():
                node_str = str(node)
                if node_str != str(target_node):
                    # Ensure all metrics are valid floats
                    metrics_dict = {
                        'run': run_number,
                        'in_markov_blanket': node_str in {str(n) for n in markov_blanket},
                        'degree': float(centrality_metrics['degree'].get(node_str, 0.0)),
                        'betweenness': float(centrality_metrics['betweenness'].get(node_str, 0.0)),
                        'closeness': float(centrality_metrics['closeness'].get(node_str, 0.0)),
                        'num_parents': float(len(list(G.predecessors(node_str)))),
                        'num_children': float(len(list(G.successors(node_str))))
                    }
                    self.node_data[node_str].append(metrics_dict)

            logging.info(f"Successfully recorded metrics for run {run_number}")

        except Exception as e:
            logging.error(f"Error in add_run for run {run_number}: {str(e)}")
            raise

    def _prepare_node_metrics_df(self):
        """Prepare DataFrame with improved node name handling."""
        all_metrics = []

        for node, runs_data in self.node_data.items():
            # Clean up node name by removing any potential duplication
            clean_node = str(node).strip()
            if '_' in clean_node:
                # Split by underscore and take unique parts only
                parts = clean_node.split('_')
                unique_parts = []
                for part in parts:
                    if part not in unique_parts:
                        unique_parts.append(part)
                clean_node = '_'.join(unique_parts)

            if not clean_node:  # Skip empty node names
                continue

            for run_data in runs_data:
                try:
                    metrics = {
                        'node': clean_node,
                        'run': int(run_data['run']),
                        'mb_frequency': float(self.markov_blanket_counts[node]) / float(self.n_runs) * 100.0
                    }

                    # Ensure numeric values for all metrics with proper error handling
                    for metric in ['degree', 'betweenness', 'closeness', 'num_parents', 'num_children']:
                        try:
                            value = run_data.get(metric, 0.0)
                            metrics[metric] = float(value) if value is not None else 0.0
                        except (ValueError, TypeError):
                            metrics[metric] = 0.0
                            logging.warning(f"Invalid value for {metric} in node {clean_node}, defaulting to 0.0")

                    all_metrics.append(metrics)
                except Exception as e:
                    logging.warning(
                        f"Skipping invalid metrics for node {clean_node} in run {run_data.get('run')}: {str(e)}")
                    continue

        # Create DataFrame and handle any remaining non-numeric values
        df = pd.DataFrame(all_metrics)
        numeric_columns = ['mb_frequency', 'degree', 'betweenness', 'closeness', 'num_parents', 'num_children']

        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        return df

    def _create_importance_boxplots(self, metrics_df, target_node):
        """Create boxplots for different metrics with consistent coloring across platforms."""
        if metrics_df.empty:
            logging.warning("No valid metrics data available for creating boxplots")
            return

        try:
            # Ensure all metrics are numeric
            for col in ['degree', 'betweenness', 'closeness', 'num_children']:
                metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce').fillna(0.0)

            top_nodes = (metrics_df.groupby('node')['mb_frequency']
                         .mean()
                         .sort_values(ascending=False)
                         .head(self.top_k)
                         .index)

            if len(top_nodes) == 0:
                logging.warning("No valid nodes found for plotting")
                return

            plot_df = metrics_df[metrics_df['node'].isin(top_nodes)].copy()

            # Set the style and color palette
            plt.style.use('default')
            colors = plt.cm.Set3(np.linspace(0, 1, len(top_nodes)))  # Using Set3 colormap for distinct colors

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Node Importance Metrics Distribution (Top {self.top_k} Nodes)')

            metrics_to_plot = {
                (0, 0): ('degree', 'Degree Centrality'),
                (0, 1): ('betweenness', 'Betweenness Centrality'),
                (1, 0): ('closeness', 'Closeness Centrality'),
                (1, 1): ('num_children', 'Number of Children')
            }

            for (i, j), (metric, title) in metrics_to_plot.items():
                try:
                    # Create boxplot with explicit color palette
                    sns.boxplot(data=plot_df, x='node', y=metric, ax=axes[i, j],
                                order=top_nodes, palette=colors)

                    axes[i, j].set_title(title)
                    axes[i, j].set_xticklabels(axes[i, j].get_xticklabels(),
                                               rotation=45, ha='right')
                    axes[i, j].set_xlabel('Node')

                    # Adjust grid for better visibility
                    axes[i, j].grid(True, linestyle='--', alpha=0.7)

                except Exception as e:
                    logging.error(f"Error plotting {metric}: {str(e)}")
                    axes[i, j].text(0.5, 0.5, f"Error plotting {metric}",
                                    ha='center', va='center')

            plt.tight_layout()
            plt.savefig(self.base_output_dir / 'importance_distributions.png',
                        bbox_inches='tight', dpi=300)
            plt.close()

        except Exception as e:
            logging.error(f"Error in creating importance boxplots: {str(e)}")

    def _create_markov_blanket_frequency_plot(self):
        """Create bar plot of Markov blanket frequencies with improved error handling."""
        try:
            if not self.markov_blanket_counts:
                logging.warning("No Markov blanket data available for plotting")
                return

            frequencies = pd.Series(self.markov_blanket_counts) / float(self.n_runs) * 100.0
            frequencies = frequencies.dropna()

            if frequencies.empty:
                logging.warning("No valid frequencies to plot")
                return

            top_frequencies = frequencies.sort_values(ascending=False).head(self.top_k)

            plt.figure(figsize=(12, 6))
            sns.barplot(x=top_frequencies.index, y=top_frequencies.values)
            plt.title(f'Markov Blanket Frequency (Top {self.top_k} Nodes)')
            plt.xlabel('Node')
            plt.ylabel('Frequency in Markov Blanket (%)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(self.base_output_dir / 'markov_blanket_frequency.png',
                        bbox_inches='tight', dpi=300)
            plt.close()

        except Exception as e:
            logging.error(f"Error in creating Markov blanket frequency plot: {str(e)}")

    def _save_summary_statistics(self, metrics_df):
        """Save statistical summary of node metrics with improved error handling."""
        try:
            if metrics_df.empty:
                logging.warning("No metrics data available for summary statistics")
                return

            # Ensure all metrics are numeric
            numeric_columns = ['mb_frequency', 'degree', 'betweenness', 'closeness', 'num_parents', 'num_children']
            for col in numeric_columns:
                metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce').fillna(0.0)

            summary = metrics_df.groupby('node').agg({
                'mb_frequency': 'first',
                'degree': ['mean', 'std'],
                'betweenness': ['mean', 'std'],
                'closeness': ['mean', 'std'],
                'num_parents': ['mean', 'std'],
                'num_children': ['mean', 'std']
            }).round(4)

            summary.columns = ['mb_frequency', 'degree_mean', 'degree_std',
                               'betweenness_mean', 'betweenness_std',
                               'closeness_mean', 'closeness_std',
                               'num_parents_mean', 'num_parents_std',
                               'num_children_mean', 'num_children_std']

            summary = summary.sort_values('mb_frequency', ascending=False)
            summary.to_csv(self.base_output_dir / 'node_importance_summary.csv')

        except Exception as e:
            logging.error(f"Error in saving summary statistics: {str(e)}")

    def _generate_text_report(self, metrics_df, target_node):
        """Generate comprehensive text report with improved string handling."""
        try:
            if metrics_df.empty:
                logging.warning("No metrics data available for generating report")
                return

            # Clean up node names and calculate mean frequencies
            node_frequencies = (metrics_df.groupby('node')['mb_frequency']
                                .mean()
                                .sort_values(ascending=False))

            # Take top nodes
            top_nodes = node_frequencies.head(self.top_k)

            report = [
                "Bayesian Network Node Importance Analysis",
                f"Based on {self.n_runs} experimental runs\n",
                f"Target Node: {target_node}\n",
                f"Top {self.top_k} Most Important Nodes:\n"
            ]

            for node_name in top_nodes.index:
                # Get metrics for this node without attempting numeric conversion on the node name
                node_data = metrics_df[metrics_df['node'] == node_name]

                # Calculate mean metrics safely
                mean_metrics = {
                    'frequency': node_frequencies[node_name],
                    'degree': node_data['degree'].mean(),
                    'betweenness': node_data['betweenness'].mean(),
                    'closeness': node_data['closeness'].mean(),
                    'num_parents': node_data['num_parents'].mean(),
                    'num_children': node_data['num_children'].mean()
                }

                # Format the report entry
                report.extend([
                    f"\n{node_name}:",
                    f"  Markov Blanket Frequency: {mean_metrics['frequency']:.1f}%",
                    f"  Average Metrics:",
                    f"    - Degree Centrality: {mean_metrics['degree']:.3f}",
                    f"    - Betweenness Centrality: {mean_metrics['betweenness']:.3f}",
                    f"    - Closeness Centrality: {mean_metrics['closeness']:.3f}",
                    f"    - Number of Parents: {mean_metrics['num_parents']:.1f}",
                    f"    - Number of Children: {mean_metrics['num_children']:.1f}"
                ])

            # Write the report to file
            with open(self.base_output_dir / 'network_analysis_report.txt', 'w') as f:
                f.write('\n'.join(report))

            logging.info("Successfully generated text report")

        except Exception as e:
            logging.error(f"Error in generating text report: {str(e)}")
            raise

    def generate_final_analysis(self, target_node: str):
        """Generate comprehensive analysis of node importance across all runs with improved error handling."""
        try:
            logging.info("Starting final analysis generation")

            if not self.node_data:
                logging.warning("No data collected. Skipping analysis.")
                return

            # Prepare the metrics DataFrame
            metrics_df = self._prepare_node_metrics_df()
            if metrics_df.empty:
                logging.warning("No valid metrics data available for analysis")
                return

            logging.info(f"Prepared metrics for {len(metrics_df)} nodes")

            # Generate all visualizations and reports with error handling
            self._create_importance_boxplots(metrics_df, target_node)
            logging.info("Created importance boxplots")

            self._create_markov_blanket_frequency_plot()
            logging.info("Created Markov blanket frequency plot")

            self._save_summary_statistics(metrics_df)
            logging.info("Saved summary statistics")

            self._generate_text_report(metrics_df, target_node)
            logging.info("Generated text report")

            logging.info("Successfully completed final analysis")

        except Exception as e:
            logging.error(f"Error in generate_final_analysis: {str(e)}")
            raise
