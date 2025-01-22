from typing import List, Tuple, Optional, Set, Dict, Any
import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations
import logging
from pgmpy.base import DAG
from pgmpy.estimators import (
    BaseEstimator,
    BicScore, K2Score, BDeuScore, BDsScore, AICScore
)
from sklearn.metrics import mutual_info_score


class UniqueColumnTransformer:
    """Handles column name transformation to ensure uniqueness."""

    def __init__(self):
        self.original_to_unique = {}
        self.unique_to_original = {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform column names to ensure uniqueness."""
        new_columns = []
        for idx, col in enumerate(df.columns):
            unique_name = f"col_{idx}_{col}"
            self.original_to_unique[col] = unique_name
            self.unique_to_original[unique_name] = col
            new_columns.append(unique_name)

        transformed_df = df.copy()
        transformed_df.columns = new_columns
        return transformed_df

    def reverse_transform(self, model: DAG) -> DAG:
        """Reverse transform node names in the model."""
        new_model = DAG()

        # Add nodes with original names
        for node in model.nodes():
            original_name = self.unique_to_original.get(node, node)
            new_model.add_node(original_name)

        # Add edges with original names
        for u, v in model.edges():
            u_orig = self.unique_to_original.get(u, u)
            v_orig = self.unique_to_original.get(v, v)
            new_model.add_edge(u_orig, v_orig)

        return new_model


class ScoreCache:
    """Cache for storing local scores to avoid recomputation."""

    def __init__(self, scoring_method: Any, data: pd.DataFrame):
        self.scoring_method = scoring_method
        self.data = data
        self.cache = {}

    def local_score(self, node: str, parents: List[str]) -> float:
        """Calculate and cache local score for a node and its parents."""
        try:
            key = (node, tuple(sorted(parents)))
            if key not in self.cache:
                if node not in self.data.columns:
                    raise ValueError(f"Node {node} not found in data columns")
                valid_parents = [p for p in parents if p in self.data.columns]
                if len(valid_parents) != len(parents):
                    raise ValueError(f"Some parent nodes not found in data columns")

                self.cache[key] = self.scoring_method.local_score(node, valid_parents)
            return self.cache[key]
        except Exception as e:
            logging.debug(f"Error in local_score computation: {str(e)}")
            return float('-inf')


class VanillaGESEstimator(BaseEstimator):
    """
    Enhanced implementation of the Greedy Equivalence Search (GES) algorithm
    with improved handling of column names and data structures.
    """

    def __init__(self, data: pd.DataFrame, use_cache: bool = True, target_node: str = "Label"):
        """Initialize the estimator with data transformation."""
        self.transformer = UniqueColumnTransformer()
        self.target_node = target_node
        self.transformed_target = None

        # Store original data for mutual information computation
        self.original_data = data.copy()

        # Transform data
        transformed_data = self.transformer.transform(data)

        # Store transformed target node name
        for orig, trans in self.transformer.original_to_unique.items():
            if orig == target_node:
                self.transformed_target = trans
                break

        super().__init__(transformed_data)
        self.use_cache = use_cache
        self.logger = logging.getLogger("VanillaGES")
        self.nodes = list(self.data.columns)

    def _ensure_target_connectivity(self, model: DAG) -> DAG:
        """Ensure target node is properly connected in the network."""
        if self.transformed_target not in model.nodes():
            model.add_node(self.transformed_target)

        # If target node has no connections, add edges based on mutual information
        if len(list(model.predecessors(self.transformed_target))) == 0 and \
                len(list(model.successors(self.transformed_target))) == 0:

            # Calculate mutual information with other nodes
            mi_scores = []
            target_data = self.original_data[self.target_node]

            for node in self.original_data.columns:
                if node != self.target_node:
                    mi = mutual_info_score(target_data, self.original_data[node])
                    mi_scores.append((node, mi))

            # Sort by mutual information score
            mi_scores.sort(key=lambda x: x[1], reverse=True)

            # Calculate mean and std of MI scores
            mi_values = [score for _, score in mi_scores]
            mi_mean = np.mean(mi_values)
            mi_std = np.std(mi_values)

            # Select nodes with MI score above (mean + 0.5*std)
            # This threshold can be adjusted based on desired connectivity
            threshold = mi_mean + 0.5 * mi_std
            significant_nodes = [(node, score) for node, score in mi_scores if score > threshold]

            # Add edges for significant nodes
            for node, score in significant_nodes:
                transformed_node = self.transformer.original_to_unique[node]
                try:
                    # Check both directions for edge addition
                    model_copy = model.copy()

                    # Try node -> target
                    if not nx.has_path(model_copy, self.transformed_target, transformed_node):
                        model.add_edge(transformed_node, self.transformed_target)
                        self.logger.debug(f"Added edge {node} -> {self.target_node} (MI: {score:.4f})")
                        continue

                    # Try target -> node
                    model_copy = model.copy()
                    if not nx.has_path(model_copy, transformed_node, self.transformed_target):
                        model.add_edge(self.transformed_target, transformed_node)
                        self.logger.debug(f"Added edge {self.target_node} -> {node} (MI: {score:.4f})")

                except Exception as e:
                    self.logger.debug(f"Failed to add edge for node {node}: {str(e)}")
                    continue

            # Ensure we have at least some minimum connectivity
            if len(significant_nodes) < 5 <= len(mi_scores):
                for node, score in mi_scores[len(significant_nodes):5]:
                    transformed_node = self.transformer.original_to_unique[node]
                    try:
                        if not model.has_edge(transformed_node, self.transformed_target) and \
                                not model.has_edge(self.transformed_target, transformed_node):
                            model.add_edge(transformed_node, self.transformed_target)
                            self.logger.debug(f"Added additional edge {node} -> {self.target_node} (MI: {score:.4f})")
                    except Exception as e:
                        self.logger.debug(f"Failed to add additional edge for node {node}: {str(e)}")
                        continue

        return model

    def _legal_operations(self, model: DAG, operation_type: str) -> List[Tuple[str, str]]:
        """
        Identify legal operations (additions, removals, or flips) based on operation type.
        """
        try:
            if operation_type == 'addition':
                return self._legal_edge_additions(model)
            elif operation_type == 'removal':
                return list(model.edges())
            elif operation_type == 'flip':
                return self._legal_edge_flips(model)
            else:
                raise ValueError(f"Unknown operation type: {operation_type}")
        except Exception as e:
            self.logger.error(f"Error in _legal_operations: {str(e)}")
            return []

    def _legal_edge_additions(self, current_model: DAG) -> List[Tuple[str, str]]:
        """Find all legal edge additions that maintain acyclicity."""
        edges = []
        try:
            for u, v in combinations(current_model.nodes(), 2):
                if not (current_model.has_edge(u, v) or current_model.has_edge(v, u)):
                    current_model_copy = current_model.copy()
                    if not nx.has_path(current_model_copy, v, u):
                        edges.append((u, v))
                    if not nx.has_path(current_model_copy, u, v):
                        edges.append((v, u))
            return edges
        except Exception as e:
            self.logger.error(f"Error in _legal_edge_additions: {str(e)}")
            return []

    def _legal_edge_removals(self, current_model: DAG) -> List[Tuple[str, str]]:
        """Get all edges that can be legally removed."""
        try:
            return list(current_model.edges())
        except Exception as e:
            self.logger.error(f"Error in _legal_edge_removals: {str(e)}")
            return []

    def _legal_edge_flips(self, current_model: DAG) -> List[Tuple[str, str]]:
        """Find all legal edge flips that maintain acyclicity."""
        potential_flips = []
        try:
            edges = list(current_model.edges())
            for u, v in edges:
                current_model_copy = current_model.copy()
                current_model_copy.remove_edge(u, v)
                if not nx.has_path(current_model_copy, u, v):
                    potential_flips.append((v, u))
            return potential_flips
        except Exception as e:
            self.logger.error(f"Error in _legal_edge_flips: {str(e)}")
            return []

    def _compute_score_delta(self, score_fn: callable, node: str,
                             old_parents: List[str], new_parents: List[str]) -> float:
        """Compute the score delta for a node given old and new parents."""
        try:
            new_score = score_fn(node, new_parents)
            old_score = score_fn(node, old_parents)
            return new_score - old_score
        except Exception as e:
            self.logger.debug(f"Score computation failed: {str(e)}")
            return float('-inf')

    def _setup_scoring_method(self, scoring_method: Any) -> Any:
        """Configure the scoring method based on input."""
        supported_methods = {
            'k2': K2Score,
            'bdeu': BDeuScore,
            'bds': BDsScore,
            'bic': BicScore,
            'aic': AICScore
        }

        if isinstance(scoring_method, str):
            if scoring_method.lower() not in supported_methods:
                raise ValueError(f"Unknown scoring method: {scoring_method}")
            return supported_methods[scoring_method.lower()](self.data)
        return scoring_method

    def estimate(self,
                 scoring_method: str = "bic",
                 min_improvement: float = 1e-4,
                 max_indegree: Optional[int] = None,
                 **kwargs) -> DAG:
        """
        Estimate the DAG structure using the GES algorithm with transformed data.
        """
        try:
            score = self._setup_scoring_method(scoring_method)
            if self.use_cache:
                score_cache = ScoreCache(score, self.data)
                score_fn = score_cache.local_score
            else:
                score_fn = score.local_score

            # Initialize empty model
            current_model = DAG()
            current_model.add_nodes_from(self.nodes)

            # Forward Phase
            self.logger.info("Starting Forward Phase...")
            improved = True
            while improved:
                improved = False
                potential_edges = self._legal_operations(current_model, 'addition')
                best_score_delta = min_improvement
                best_edge = None

                for u, v in potential_edges:
                    if max_indegree and len(list(current_model.get_parents(v))) >= max_indegree:
                        continue

                    current_parents = list(current_model.get_parents(v))
                    score_delta = self._compute_score_delta(
                        score_fn, v, current_parents, current_parents + [u])

                    if score_delta > best_score_delta:
                        best_score_delta = score_delta
                        best_edge = (u, v)

                if best_edge:
                    current_model.add_edge(*best_edge)
                    improved = True
                    self.logger.debug(f"Added edge {best_edge[0]} -> {best_edge[1]}")

            # Backward Phase
            self.logger.info("Starting Backward Phase...")
            improved = True
            while improved:
                improved = False
                potential_removals = self._legal_operations(current_model, 'removal')
                best_score_delta = min_improvement
                best_edge = None

                for u, v in potential_removals:
                    current_parents = list(current_model.get_parents(v))
                    new_parents = [p for p in current_parents if p != u]

                    score_delta = self._compute_score_delta(
                        score_fn, v, current_parents, new_parents)

                    if score_delta > best_score_delta:
                        best_score_delta = score_delta
                        best_edge = (u, v)

                if best_edge:
                    current_model.remove_edge(*best_edge)
                    improved = True
                    self.logger.debug(f"Removed edge {best_edge[0]} -> {best_edge[1]}")

            # Edge Flipping Phase
            self.logger.info("Starting Edge Flipping Phase...")
            improved = True
            while improved:
                improved = False
                potential_flips = self._legal_operations(current_model, 'flip')
                best_score_delta = min_improvement
                best_flip = None

                for u, v in potential_flips:
                    v_parents = list(current_model.get_parents(v))
                    u_parents = list(current_model.get_parents(u))

                    v_new_parents = [p for p in v_parents if p != u]
                    u_new_parents = u_parents + [v]

                    score_delta = (
                            self._compute_score_delta(score_fn, v, v_parents, v_new_parents) +
                            self._compute_score_delta(score_fn, u, u_parents, u_new_parents)
                    )

                    if score_delta > best_score_delta:
                        best_score_delta = score_delta
                        best_flip = (u, v)

                if best_flip:
                    current_model.remove_edge(best_flip[1], best_flip[0])
                    current_model.add_edge(best_flip[0], best_flip[1])
                    improved = True
                    self.logger.debug(f"Flipped edge {best_flip[1]} -> {best_flip[0]}")

            # Ensure target node is properly connected
            current_model = self._ensure_target_connectivity(current_model)

            # Transform back to original column names
            final_model = self.transformer.reverse_transform(current_model)
            self.logger.info("GES estimation completed successfully")
            return final_model

        except Exception as e:
            self.logger.error(f"Error in GES estimation: {str(e)}")
            raise