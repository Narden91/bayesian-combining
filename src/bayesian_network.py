import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import graphviz
import numpy as np
import pandas as pd
from pgmpy.estimators import (
    HillClimbSearch, MmhcEstimator, PC, TreeSearch,
    BicScore, K2Score, BDeuScore, BDsScore, AICScore,
    BayesianEstimator
)
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.metrics import log_likelihood_score
from sklearn.metrics import mutual_info_score
from tqdm import tqdm
from .GES import FastGESEstimator
from .vanilla_GES import VanillaGESEstimator


def _initialize_structure_learner(cfg: Dict, data: pd.DataFrame) -> object:
    """
    Initialize the appropriate structure learning algorithm with proper configuration.

    Args:
        cfg: Configuration dictionary
        data: Training data

    Returns:
        Initialized structure learner
    """
    algorithm_mapping = {
        'HillClimb': HillClimbSearch,
        'MMHC': MmhcEstimator,
        'PC': PC,
        'Tree': TreeSearch,
        'GES': FastGESEstimator,
        'VanillaGES': VanillaGESEstimator
    }

    if cfg.bayesian_net.algorithm not in algorithm_mapping:
        raise ValueError(f"Invalid algorithm: {cfg.bayesian_net.algorithm}")

    if cfg.bayesian_net.algorithm == 'GES':
        return FastGESEstimator(
            data=data,
            n_jobs=cfg.bayesian_net.get('n_jobs', -1),
            cache_size=cfg.bayesian_net.get('cache_size', 1000),
            early_stopping_steps=cfg.bayesian_net.get('early_stopping_steps', 5),
            score_delta_threshold=cfg.bayesian_net.get('score_delta_threshold', 1e-4),
            min_improvement_ratio=cfg.bayesian_net.get('min_improvement_ratio', 0.001)
        )
    elif cfg.bayesian_net.algorithm == 'PC':
        return PC(data)
    else:
        return algorithm_mapping[cfg.bayesian_net.algorithm](data)


def _setup_scoring_method(cfg: Dict, data: pd.DataFrame) -> Optional[object]:
    """
    Set up the appropriate scoring method based on configuration.

    Args:
        cfg: Configuration dictionary
        data: Training data

    Returns:
        Initialized scoring method or None for PC algorithm
    """
    if cfg.bayesian_net.algorithm == 'PC':
        return None

    score_mapping = {
        'bic': BicScore,
        'k2': K2Score,
        'bdeu': BDeuScore,
        'bds': BDsScore,
        'aic': AICScore
    }

    score_metric = score_mapping.get(cfg.bayesian_net.score_metric)
    if score_metric is None:
        logging.warning(f"Invalid score metric: {cfg.bayesian_net.score_metric}, using BicScore")
        score_metric = BicScore

    return score_metric(data)


def _ensure_target_node(model: BayesianNetwork, target: str, data: pd.DataFrame) -> BayesianNetwork:
    """
    Ensure target node is present in the network and has meaningful connections.

    Args:
        model: BayesianNetwork model
        target: Target variable name
        data: Training data

    Returns:
        Updated BayesianNetwork with target node
    """
    if target not in model.nodes():
        # Add target node if missing
        model.add_node(target)
        logging.info(f"Added missing target node {target}")

        # Calculate mutual information between target and other variables
        target_data = data[target]
        mutual_info = {}
        for node in model.nodes():
            if node != target:
                mi_score = mutual_info_score(target_data, data[node])
                mutual_info[node] = mi_score

        # Connect target to nodes with highest mutual information
        if mutual_info:
            # Get top 3 most informative nodes or all if less than 3
            top_nodes = sorted(mutual_info.items(), key=lambda x: x[1], reverse=True)
            num_connections = min(3, len(top_nodes))

            for node, score in top_nodes[:num_connections]:
                try:
                    model.add_edge(target, node)
                    logging.info(f"Added edge from {target} to {node} (MI: {score:.4f})")
                except Exception as e:
                    logging.warning(f"Could not add edge {target}->{node}: {str(e)}")

    return model


def optimize_dataset_for_pc(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Optimize dataset before PC algorithm application.
    """
    # Create a copy to avoid modifying the original
    df = df.copy()

    # Convert numeric columns to float32 for faster computation
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].astype('float32')

    return df


def bayesian_network(cfg: Dict, run_number: int, run_folder: str,
                     df_predictions: pd.DataFrame, df_predictions_proba: pd.DataFrame,
                     df_test: pd.DataFrame, importance_tracker=None) -> Tuple:
    """
    Build a Bayesian Network using the predictions from the stacking models.

    Args:
        cfg: Configuration dictionary
        run_number: Current run number
        run_folder: Path to the run folder
        df_predictions: DataFrame with predictions
        df_predictions_proba: DataFrame with prediction probabilities
        df_test: Test data DataFrame
        importance_tracker: Optional BayesianImportanceTracker instance

    Returns:
        Tuple containing selected columns, results DataFrames, and metrics
    """
    try:
        # Initialize structure learner with proper configuration
        if cfg.bayesian_net.algorithm == 'PC':
            df_predictions = optimize_dataset_for_pc(df_predictions, cfg.data.target)

        bn_model = _initialize_structure_learner(cfg, df_predictions)
        logging.info(f"Initialized {cfg.bayesian_net.algorithm} structure learner")

        # Setup scoring method
        scoring_method = _setup_scoring_method(cfg, df_predictions)

        # Estimate network structure
        start_time = time.time()

        if cfg.bayesian_net.algorithm == 'PC':
            # Set up PC algorithm specific parameters
            estimate_params = {
                'variant': cfg.bayesian_net.get('variant', 'stable'),
                'alpha': cfg.bayesian_net.significance_level,
                'max_cond_vars': cfg.bayesian_net.get('max_cond_vars'),
                'show_progress': cfg.bayesian_net.get('show_progress', True)
            }
            best_model_stck = bn_model.estimate(**estimate_params)

            # Create initial model from PC results
            model = BayesianNetwork(best_model_stck.edges())

            # Add target node if missing and connect it properly
            if cfg.data.target not in model.nodes():
                model = _ensure_target_node(model, cfg.data.target, df_predictions)
                logging.info(f"Added target node {cfg.data.target} to the network")

            # Verify model is still a DAG after adding target
            if not isinstance(model, BayesianNetwork):
                raise ValueError("Invalid model structure after adding target node")
        else:
            # Handle different algorithms
            if cfg.bayesian_net.algorithm == 'Tree':
                # TreeSearch doesn't accept scoring_method
                best_model_stck = bn_model.estimate()
            else:
                # Other algorithms (HillClimb, MMHC, etc.)
                estimate_params = {
                    'scoring_method': scoring_method,
                    'max_indegree': cfg.bayesian_net.max_parents if cfg.bayesian_net.use_parents else None
                }
                best_model_stck = bn_model.estimate(**estimate_params)
            model = BayesianNetwork(best_model_stck.edges())

        elapsed_time = time.time() - start_time
        logging.info(f"Structure learning completed in {elapsed_time:.2f} seconds")

        if cfg.bayesian_net.prior_type == 'dirichlet':
            logging.info("Using K2 prior for parameter estimation")
            cfg.bayesian_net.prior_type = 'K2'

        # Validate and fit model
        if cfg.data.target not in model.nodes():
            raise ValueError(f"Target node {cfg.data.target} missing from network")

        model.fit(
            df_predictions,
            estimator=BayesianEstimator,
            prior_type=cfg.bayesian_net.prior_type
        )
        logging.info("Network parameters estimated successfully")

        # Get Markov blanket
        label_markov_blanket = model.get_markov_blanket(cfg.data.target)
        logging.info(f"Markov blanket size: {len(label_markov_blanket)}")

        # Calculate network scores
        bic_score = BicScore(df_predictions).score(model)
        log_likelihood = log_likelihood_score(model, df_predictions[list(model.nodes())])
        log_likelihood_test = log_likelihood_score(model, df_test[list(model.nodes())])

        # Update importance tracker if provided
        if importance_tracker is not None:
            importance_tracker.add_run(
                run_number=run_number,
                model=model,
                target_node=cfg.data.target,
                log_likelihood=log_likelihood
            )

        # Create visualizations and analysis
        create_bn_visualization(model, cfg.data.target, run_folder, run_number)
        analyze_network_structure(model, cfg)

        # Prepare selected columns and perform inference
        selected_columns = list(label_markov_blanket) + [cfg.data.target]
        selected_tasks_df = df_predictions[selected_columns]

        inference = VariableElimination(model)

        # Training data predictions
        predictions_train, probabilities_class_0_train, probabilities_class_1_train = perform_inference(
            inference, selected_tasks_df, cfg.data.target, label_markov_blanket
        )

        # Test data predictions
        test_columns = [col for col in selected_columns if col != cfg.data.target]
        test_tasks_df = df_test[test_columns]
        predictions_test, probabilities_class_0_test, probabilities_class_1_test = perform_inference(
            inference, test_tasks_df, cfg.data.target, label_markov_blanket
        )

        # Create results DataFrames
        results_df_train = pd.DataFrame({
            cfg.data.target: predictions_train,
            f"{cfg.data.target}_proba_-1": probabilities_class_0_train,
            f"{cfg.data.target}_proba_1": probabilities_class_1_train
        })
        results_df_train[cfg.data.target] = results_df_train[cfg.data.target].map({0: -1, 1: 1})

        results_df_test = pd.DataFrame({
            cfg.data.target: predictions_test,
            f"{cfg.data.target}_proba_-1": probabilities_class_0_test,
            f"{cfg.data.target}_proba_1": probabilities_class_1_test
        })
        results_df_test[cfg.data.target] = results_df_test[cfg.data.target].map({0: -1, 1: 1})

        return (selected_columns, results_df_train, results_df_test,
                bic_score, log_likelihood, log_likelihood_test)

    except Exception as e:
        logging.error(f"Error in bayesian_network: {str(e)}")
        raise


def perform_inference(inference, df: pd.DataFrame, target: str,
                      markov_blanket: set) -> Tuple[list, list, list]:
    """
    Perform inference on the Bayesian network.

    Args:
        inference: VariableElimination object
        df: Data to perform inference on
        target: Target variable name
        markov_blanket: Set of nodes in Markov blanket

    Returns:
        Tuple of predictions and class probabilities
    """
    predictions = []
    probabilities_class_0 = []
    probabilities_class_1 = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inferring"):
        evidence = {col: row[col] for col in markov_blanket if col != target}
        query_result = inference.query([target], evidence=evidence)
        probs = query_result.values
        prediction = np.argmax(probs)

        predictions.append(prediction)
        probabilities_class_0.append(probs[0])
        probabilities_class_1.append(probs[1])

    return predictions, probabilities_class_0, probabilities_class_1


def create_bn_visualization(model: BayesianNetwork, target: str,
                            run_folder: str, run_number: int) -> None:
    """Create and save visualization of the Bayesian network."""
    dot = graphviz.Digraph(comment=f'Bayesian Network {run_number + 1}')
    dot.attr(rankdir='LR')

    for node in model.nodes():
        shape = 'diamond' if node == target else 'ellipse'
        dot.node(node, node, shape=shape)

    for edge in model.edges():
        dot.edge(edge[0], edge[1])

    try:
        bn_path = Path(run_folder) / f"bayesian_network_{run_number + 1}"
        dot_source = dot.source
        with open(bn_path.with_suffix('.dot'), 'w') as f:
            f.write(dot_source)
    except Exception as e:
        logging.error(f"Failed to save network visualization: {str(e)}")


def analyze_network_structure(model: BayesianNetwork, cfg: Dict) -> None:
    """Analyze and log network structure details."""
    if not cfg.bayesian_net.verbose:
        return

    logging.info("\nNetwork Structure Analysis:")

    # Edge analysis
    logging.info("\nEdge Structure:")
    for edge in model.edges():
        logging.info(f"{edge[0]} -> {edge[1]}")

    # Node degree analysis
    logging.info("\nNode Degrees:")
    for node in model.nodes():
        in_degree = len(model.get_parents(node))
        out_degree = len(model.get_children(node))
        logging.info(f"{node}: In-degree={in_degree}, Out-degree={out_degree}")

    # Network statistics
    n_nodes = len(model.nodes())
    n_edges = len(model.edges())
    max_possible_edges = n_nodes * (n_nodes - 1) / 2
    density = n_edges / max_possible_edges

    logging.info("\nNetwork Statistics:")
    logging.info(f"Nodes: {n_nodes}")
    logging.info(f"Edges: {n_edges}")
    logging.info(f"Network density: {density:.4f}")