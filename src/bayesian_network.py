import logging
import graphviz
import numpy as np
import pandas as pd
from pgmpy.estimators import HillClimbSearch, ExhaustiveSearch, PC, TreeSearch, BayesianEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from tqdm import tqdm


def bayesian_network(cfg: dict, run_number: int, run_folder: str, df_predictions: pd.DataFrame,
                     df_predictions_proba: pd.DataFrame) -> (list, pd.DataFrame):
    """
    Build a Bayesian Network using the predictions from the stacking models.
    :param cfg: dict
    :param run_number: int
    :param run_folder: str
    :param df_predictions: pd.DataFrame
    :param df_predictions_proba: pd.DataFrame
    :return: list, pd.DataFrame
    """
    # Build Bayesian Network
    algorithm_mapping = {
        'HillClimb': HillClimbSearch,
        'Exhaustive': ExhaustiveSearch,
        'PC': PC,
        'TreeSearch': TreeSearch
    }

    if cfg.bayesian_net.algorithm not in algorithm_mapping:
        raise ValueError(f"Invalid algorithm: {cfg.bayesian_net.algorithm}")

    bn_model = algorithm_mapping[cfg.bayesian_net.algorithm](df_predictions)

    logging.info(f"Building Bayesian Network using {cfg.bayesian_net.algorithm} algorithm...")

    if cfg.bayesian_net.use_parents:
        best_model_stck = bn_model.estimate(max_indegree=cfg.experiment.max_parents)
        logging.info(f"Using max_parents = {cfg.experiment.max_parents}")
    else:
        best_model_stck = bn_model.estimate()
        logging.info("No max_parents constraint applied")

    if cfg.bayesian_net.verbose:
        logging.info("Best model edges:")
        logging.info(best_model_stck.edges())

    model = BayesianNetwork(best_model_stck.edges())
    model.fit(df_predictions, estimator=BayesianEstimator, prior_type=cfg.bayesian_net.prior_type)

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

    # Make predictions using probabilistic inference
    predictions = []
    probabilities_class_0 = []
    probabilities_class_1 = []

    for _, row in tqdm(selected_tasks_df.iterrows(), total=len(selected_tasks_df), desc="Inferring"):
        evidence = {col: row[col] for col in label_markov_blanket}
        query_result = inference.query([cfg.data.target], evidence=evidence)

        # Get the probabilities for both classes
        probs = query_result.values

        # Get the most probable class
        prediction = np.argmax(probs)
        predictions.append(prediction)

        # Store probabilities for both classes
        probabilities_class_0.append(probs[0])
        probabilities_class_1.append(probs[1])

    # Create DataFrame with predictions and probabilities for both classes
    results_df = pd.DataFrame({
        cfg.data.target: predictions,
        f"{cfg.data.target}_proba_-1": probabilities_class_0,
        f"{cfg.data.target}_proba_1": probabilities_class_1
    })

    # Map Label 0 to 1
    results_df[cfg.data.target] = results_df[cfg.data.target].map({0: -1, 1: 1})

    return selected_columns, results_df


def create_bn_visualization(model, target, run_folder, run_number):
    dot = graphviz.Digraph(comment=f'Bayesian Network {run_number + 1}')
    dot.attr(rankdir='LR')

    for node in model.nodes():
        shape = 'diamond' if node == target else 'ellipse'
        dot.node(node, node, shape=shape)

    for edge in model.edges():
        dot.edge(edge[0], edge[1])

    bn_path = run_folder / f"bayesian_network_{run_number + 1}"
    dot.render(bn_path, format='png', cleanup=True)


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