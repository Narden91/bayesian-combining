import logging
import random
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import graphviz
import numpy as np
import pandas as pd
from pgmpy.estimators import HillClimbSearch, ExhaustiveSearch, PC, TreeSearch, BayesianEstimator, BicScore
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.metrics import log_likelihood_score
from tqdm import tqdm


def bayesian_network(cfg: dict, run_number: int, run_folder: str, df_predictions: pd.DataFrame,
                     df_predictions_proba: pd.DataFrame, df_test: pd.DataFrame) -> (list, pd.DataFrame, pd.DataFrame):
    """
    Build a Bayesian Network using the predictions from the stacking models and perform classification on test data.
    Algorithm options: HillClimb, Exhaustive, PC, TreeSearch
    USE_PARENTS: True, False
    MAX_PARENTS: int
    Prior type: K2, BDeu

    :param cfg: dict
    :param run_number: int
    :param run_folder: str
    :param df_predictions: pd.DataFrame
    :param df_predictions_proba: pd.DataFrame
    :param df_test: pd.DataFrame
    :return: list, pd.DataFrame, pd.DataFrame
    """
    # Build Bayesian Network
    # Mapping algorithms and initializing the selected algorithm
    algorithm_mapping = {
        'HillClimb': HillClimbSearch,
        'Exhaustive': ExhaustiveSearch,
        'PC': PC,
        'TreeSearch': TreeSearch
    }

    # Ensure the selected algorithm is valid
    if cfg.bayesian_net.algorithm not in algorithm_mapping:
        raise ValueError(f"Invalid algorithm: {cfg.bayesian_net.algorithm}")

    # Initialize the Bayesian Network search object with df_predictions
    bn_model = algorithm_mapping[cfg.bayesian_net.algorithm](df_predictions)

    logging.info(f"Building Bayesian Network using {cfg.bayesian_net.algorithm} algorithm...")

    # Estimate the network structure
    if cfg.bayesian_net.use_parents:
        logging.info(f"Using max_parents = {cfg.experiment.max_parents}")
        best_model_stck = bn_model.estimate(max_indegree=cfg.experiment.max_parents)
    else:
        logging.info("No max_parents constraint applied")
        best_model_stck = bn_model.estimate()

    # Extract nodes from the learned structure
    nodes = set(sum(best_model_stck.edges(), ()))  # Flatten edges to get all nodes
    logging.info(f"Nodes in the Bayesian Network: {nodes}")

    # Check if the target node is in the set of nodes
    if cfg.data.target not in nodes:
        logging.warning(f"Target node {cfg.data.target} not found in the learned structure. Adding it manually.")

        # Add the node to the model and connect it minimally to ensure it's recognized
        best_model_stck.add_node(cfg.data.target)
        # Optional: Add a minimal edge to ensure the node is fully registered
        example_node = next(iter(nodes))  # Select any existing node
        best_model_stck.add_edge(cfg.data.target, example_node)  # Add a neutral connection

    # Verify node addition in the graph
    if cfg.data.target not in best_model_stck.nodes():
        logging.error(f"Failed to add target node {cfg.data.target} to the Bayesian Network.")
    else:
        logging.info(f"Successfully added target node {cfg.data.target} to the Bayesian Network.")

    # Initialize the Bayesian Network with the edges
    model = BayesianNetwork(best_model_stck.edges())

    # Fit the model with the given data
    model.fit(df_predictions, estimator=BayesianEstimator, prior_type=cfg.bayesian_net.prior_type)

    # Final check: ensure the node is in the fitted model
    if cfg.data.target not in model.nodes():
        logging.error(f"The node {cfg.data.target} is still not in the fitted Bayesian Network.")
        raise ValueError(f"The node {cfg.data.target} is missing from the Bayesian Network after fitting.")
    else:
        try:
            # Attempt to retrieve the Markov Blanket of the target node
            label_markov_blanket = model.get_markov_blanket(cfg.data.target)
            logging.info(f"Markov Blanket for {cfg.data.target}: {label_markov_blanket}")
        except Exception as e:
            logging.error(f"Error retrieving Markov Blanket for {cfg.data.target}: {str(e)}")

    # Calculate BIC score
    try:
        bic = BicScore(df_predictions)
        bic_score = bic.score(model)
        logging.info(f"BIC Score: {bic_score}")
    except Exception as e:
        logging.error(f"Error calculating BIC score: {str(e)}")
        bic_score = 0
        logging.info(f"BIC Score set to: {bic_score}")

    # Calculate log likelihood score on training data
    try:
        df_predictions_reordered = df_predictions[list(model.nodes())]
        log_likelihood = log_likelihood_score(model, df_predictions_reordered)
        logging.info(f"Log Likelihood Score: {log_likelihood}")
    except Exception as e:
        logging.error(f"Error calculating log likelihood: {str(e)}")
        log_likelihood = 0
        logging.info(f"Log Likelihood Score set to: {log_likelihood}")

    # Calculate log likelihood score on test data
    try:
        df_test_reordered = df_test[list(model.nodes())]
        log_likelihood_test = log_likelihood_score(model, df_test_reordered)
        logging.info(f"Log Likelihood Score (Test): {log_likelihood_test}")
    except Exception as e:
        logging.error(f"Error calculating log likelihood on test data: {str(e)}")
        log_likelihood_test = 0
        logging.info(f"Log Likelihood Score (Test) set to: {log_likelihood_test}")

    # Create and save Bayesian Network visualization
    create_bn_visualization(model, cfg.data.target, run_folder, run_number)

    # Analyze and log network structure
    analyze_network_structure(model, cfg)

    # Find Markov blanket for target
    label_markov_blanket = model.get_markov_blanket(cfg.data.target)
    logging.info(f"Markov Blanket for {cfg.data.target}: {label_markov_blanket}")

    # Select columns based on Markov blanket
    selected_columns = list(label_markov_blanket) + [cfg.data.target]
    selected_tasks_df = df_predictions[selected_columns]

    # Perform improved inference
    logging.info("Performing inference...")
    inference = VariableElimination(model)

    # Make predictions using probabilistic inference on training data
    predictions_train, probabilities_class_0_train, probabilities_class_1_train = perform_inference(inference,
                                                                                                    selected_tasks_df,
                                                                                                    cfg.data.target,
                                                                                                    label_markov_blanket)

    # Create DataFrame with predictions and probabilities for training data
    results_df_train = pd.DataFrame({
        cfg.data.target: predictions_train,
        f"{cfg.data.target}_proba_-1": probabilities_class_0_train,
        f"{cfg.data.target}_proba_1": probabilities_class_1_train
    })

    # Map Label 0 to -1 for training data
    results_df_train[cfg.data.target] = results_df_train[cfg.data.target].map({0: -1, 1: 1})

    # Perform inference on test data
    test_columns = [col for col in selected_columns if col != cfg.data.target]
    test_tasks_df = df_test[test_columns]
    predictions_test, probabilities_class_0_test, probabilities_class_1_test = perform_inference(inference,
                                                                                                 test_tasks_df,
                                                                                                 cfg.data.target,
                                                                                                 label_markov_blanket)

    # Create DataFrame with predictions and probabilities for test data
    results_df_test = pd.DataFrame({
        cfg.data.target: predictions_test,
        f"{cfg.data.target}_proba_-1": probabilities_class_0_test,
        f"{cfg.data.target}_proba_1": probabilities_class_1_test
    })

    # Map Label 0 to -1 for test data
    results_df_test[cfg.data.target] = results_df_test[cfg.data.target].map({0: -1, 1: 1})

    # return selected_columns, results_df_train, results_df_test
    return selected_columns, results_df_train, results_df_test, bic_score, log_likelihood, log_likelihood_test


def perform_inference(inference, df, target, markov_blanket):
    predictions = []
    probabilities_class_0 = []
    probabilities_class_1 = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inferring"):
        evidence = {col: row[col] for col in markov_blanket if col != target}
        query_result = inference.query([target], evidence=evidence)

        # Get the probabilities for both classes
        probs = query_result.values

        # Get the most probable class
        prediction = np.argmax(probs)
        predictions.append(prediction)

        # Store probabilities for both classes
        probabilities_class_0.append(probs[0])
        probabilities_class_1.append(probs[1])

    return predictions, probabilities_class_0, probabilities_class_1


# def create_bn_visualization(model, target, run_folder, run_number):
#     dot = graphviz.Digraph(comment=f'Bayesian Network {run_number + 1}')
#     dot.attr(rankdir='LR')
#
#     for node in model.nodes():
#         shape = 'diamond' if node == target else 'ellipse'
#         dot.node(node, node, shape=shape)
#
#     for edge in model.edges():
#         dot.edge(edge[0], edge[1])
#
#     bn_path = run_folder / f"bayesian_network_{run_number + 1}"
#     dot.render(bn_path, format='png', cleanup=True)


# def create_bn_visualization(model, target, run_folder, run_number, max_retries=5, base_delay=0.1):
#     """
#     Create and save a visualization of the Bayesian Network with manual file handling.
#
#     :param model: The Bayesian Network model
#     :param target: The target variable
#     :param run_folder: The folder to save the visualization
#     :param run_number: The run number
#     :param max_retries: Maximum number of retry attempts
#     :param base_delay: Base delay between retries (will be multiplied by attempt number)
#     """
#     dot = graphviz.Digraph(comment=f'Bayesian Network {run_number + 1}')
#     dot.attr(rankdir='LR')
#
#     for node in model.nodes():
#         shape = 'diamond' if node == target else 'ellipse'
#         dot.node(node, node, shape=shape)
#
#     for edge in model.edges():
#         dot.edge(edge[0], edge[1])
#
#     run_folder = Path(run_folder)
#     bn_path = run_folder / f"bayesian_network_{run_number + 1}_{uuid.uuid4().hex}"  # Ensure unique filename
#
#     for attempt in range(max_retries):
#         try:
#             # Generate DOT source
#             dot_source = dot.source
#
#             # Save DOT file
#             dot_file = bn_path.with_suffix('.dot')
#             with open(dot_file, 'w') as f:
#                 f.write(dot_source)
#
#             # Convert DOT to PNG using command-line tool
#             png_file = bn_path.with_suffix('.png')
#             subprocess.run(['dot', '-Tpng', str(dot_file), '-o', str(png_file)], check=True)
#
#             logging.info(f"Bayesian Network visualization saved successfully: {png_file}")
#             return
#         except Exception as e:
#             if attempt == max_retries - 1:
#                 logging.error(f"Failed to save Bayesian Network visualization after {max_retries} attempts: {str(e)}")
#                 raise
#             delay = base_delay * (attempt + 1) * (1 + random.random())
#             logging.warning(f"Attempt {attempt + 1} to save BN visualization failed. Retrying in {delay:.2f} seconds. Error: {str(e)}")
#             time.sleep(delay)


def create_bn_visualization(model, target, run_folder, run_number, max_retries=5, base_delay=0.1):
    """
    Create and save a visualization of the Bayesian Network with manual file handling.
    Continues execution even if an error occurs.

    :param model: The Bayesian Network model
    :param target: The target variable
    :param run_folder: The folder to save the visualization
    :param run_number: The run number
    :param max_retries: Maximum number of retry attempts
    :param base_delay: Base delay between retries (will be multiplied by attempt number)
    """
    dot = graphviz.Digraph(comment=f'Bayesian Network {run_number + 1}')
    dot.attr(rankdir='LR')

    for node in model.nodes():
        shape = 'diamond' if node == target else 'ellipse'
        dot.node(node, node, shape=shape)

    for edge in model.edges():
        dot.edge(edge[0], edge[1])

    run_folder = Path(run_folder)
    bn_path = run_folder / f"bayesian_network_{run_number + 1}_{uuid.uuid4().hex}"  # Ensure unique filename

    for attempt in range(max_retries):
        try:
            # Generate DOT source
            dot_source = dot.source

            # Save DOT file
            dot_file = bn_path.with_suffix('.dot')
            with open(dot_file, 'w') as f:
                f.write(dot_source)

            # Convert DOT to PNG using command-line tool
            png_file = bn_path.with_suffix('.png')
            subprocess.run(['dot', '-Tpng', str(dot_file), '-o', str(png_file)], check=True)

            logging.info(f"Bayesian Network visualization saved successfully: {png_file}")
            break  # Exit the loop if successful
        except Exception as e:
            delay = base_delay * (attempt + 1) * (1 + random.random())
            logging.warning(f"Attempt {attempt + 1} to save BN visualization failed. Retrying in {delay:.2f} seconds. Error: {str(e)}")
            time.sleep(delay)
    else:
        logging.error(f"Failed to save Bayesian Network visualization after {max_retries} attempts. Continuing execution.")


def analyze_network_structure(model, cfg):
    if cfg.bayesian_net.verbose:
        logging.info("\nNetwork Structure:")
        for edge in model.edges():
            logging.info(f"{edge[0]} -> {edge[1]}")

        logging.info("\nNode degrees:")
        for node in model.nodes():
            in_degree = len(model.get_parents(node))
            out_degree = len(model.get_children(node))
            logging.info(f"{node}: In-degree = {in_degree}, Out-degree = {out_degree}")

        logging.info("\nNetwork Statistics:")
        logging.info(f"Number of nodes: {len(model.nodes())}")
        logging.info(f"Number of edges: {len(model.edges())}")

        # Calculate network density
        max_edges = len(model.nodes()) * (len(model.nodes()) - 1) / 2
        density = len(model.edges()) / max_edges
        logging.info(f"Network density: {density:.4f}")
