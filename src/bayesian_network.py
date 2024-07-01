from pgmpy.estimators import HillClimbSearch, PC, BayesianEstimator, ExhaustiveSearch, TreeSearch
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
import graphviz
import logging
import pandas as pd


def bayesian_network(cfg: dict, run_number: int, run_folder: str, df_predictions: pd.DataFrame,
                     df_predictions_proba: pd.DataFrame) -> (list, pd.DataFrame):
    """
    Build a Bayesian Network using the predictions from the stacking models.
    :param cfg: dict
    :param run_number: int
    :param run_folder: str
    :param df_predictions: pd.DataFrame
    :param df_predictions_proba: pd.DataFrame
    :return: pd.DataFrame
    """
    # Build Bayesian Network
    if cfg.bayesian_net.algorithm == 'HillClimb':
        bn_model = HillClimbSearch(df_predictions)
    elif cfg.bayesian_net.algorithm == 'Exhaustive':
        bn_model = ExhaustiveSearch(df_predictions)
    elif cfg.bayesian_net.algorithm == 'PC':
        bn_model = PC(df_predictions)
    elif cfg.bayesian_net.algorithm == 'TreeSearch':
        bn_model = TreeSearch(df_predictions)
    else:
        raise ValueError(f"Invalid algorithm: {cfg.bayesian_net.algorithm}")

    logging.info("\nBuilding Bayesian Network using {} algorithm...".format(cfg.bayesian_net.algorithm))

    if cfg.bayesian_net.use_parents:
        best_model_stck = bn_model.estimate(max_indegree=cfg.experiment.max_parents)
        logging.info(f"Using max_parents = {cfg.experiment.max_parents}")
    else:
        best_model_stck = bn_model.estimate()
        logging.info("No max_parents constraint applied")

    logging.info("\nBest model edges:") if cfg.bayesian_net.verbose else None
    logging.info(best_model_stck.edges()) if cfg.bayesian_net.verbose else None

    model = BayesianNetwork(best_model_stck.edges())
    model.fit(df_predictions, estimator=BayesianEstimator, prior_type=cfg.bayesian_net.prior_type)

    # Create a Graphviz object
    dot = graphviz.Digraph(comment=f'Bayesian Network {run_number + 1}')
    dot.attr(rankdir='LR')

    # Add nodes
    for node in model.nodes():
        if node == cfg.data.target:
            dot.node(node, node, shape='diamond')
        else:
            dot.node(node, node)

    # Add edges
    for edge in model.edges():
        dot.edge(edge[0], edge[1])

    # Render the graph
    bn_path = run_folder / f"bayesian_network_{run_number + 1}"
    dot.render(bn_path, format='png', cleanup=True)

    # Print network structure
    if cfg.bayesian_net.verbose:
        logging.info("\nNetwork Structure:")
        for edge in model.edges():
            logging.info(f"{edge[0]} -> {edge[1]}")

        # Analyze network
        logging.info("\nNode degrees:")
        for node in model.nodes():
            in_degree = len(model.get_parents(node))
            out_degree = len(model.get_children(node))
            logging.info(f"{node}: In-degree = {in_degree}, Out-degree = {out_degree}")

    # Find Markov blanket for Label
    label_markov_blanket = model.get_markov_blanket(cfg.data.target)
    logging.info(f"Markov Blanket for {cfg.data.target}: {label_markov_blanket}")

    # Select columns based on Markov blanket
    selected_columns = list(label_markov_blanket) + [cfg.data.target]
    selected_tasks_df = df_predictions[selected_columns]

    # Perform inference
    logging.info("Performing inference...")
    inference = VariableElimination(model)

    # Make predictions using majority vote
    predictions = []
    for _, row in selected_tasks_df.iterrows():
        evidence = {col: row[col] for col in label_markov_blanket}
        prediction = inference.map_query([cfg.data.target], evidence=evidence)
        predictions.append(prediction[cfg.data.target])

    return selected_columns, pd.DataFrame(predictions, columns=[cfg.data.target])
    # return pd.DataFrame(predictions, columns=[cfg.data.target])
