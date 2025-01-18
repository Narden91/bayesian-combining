from typing import Set, List, Tuple, Dict, Optional, Union, FrozenSet
import numpy as np
import pandas as pd
from pgmpy.base import DAG
from pgmpy.estimators import BaseEstimator, BicScore
from itertools import combinations
import logging
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import networkx as nx
from collections import defaultdict, OrderedDict


@dataclass
class Operation:
    """Represents a graph operation with its score change."""
    source: str
    target: str
    operation_type: str  # 'add' or 'delete'
    score_diff: float
    parents_count: int


class ScoreCache:
    """LRU cache for score computations."""

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self.cache = OrderedDict()

    def get(self, key: Tuple) -> Optional[float]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: Tuple, value: float) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.maxsize:
                self.cache.popitem(last=False)
        self.cache[key] = value


class FastGESEstimator(BaseEstimator):
    """
    Optimized implementation of Greedy Equivalence Search (GES) algorithm
    for learning Bayesian Network structure.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 n_jobs: int = -1,
                 cache_size: int = 1000,
                 early_stopping_steps: int = 5,
                 score_delta_threshold: float = 1e-4,
                 min_improvement_ratio: float = 0.001):
        """
        Initialize the GES estimator with improved parameter validation.

        Args:
            data: Training data
            n_jobs: Number of parallel jobs (-1 for all cores)
            cache_size: Size of score cache
            early_stopping_steps: Steps without improvement before stopping
            score_delta_threshold: Minimum score improvement required
            min_improvement_ratio: Minimum ratio of improvement to continue
        """
        super().__init__(data)
        self._validate_init_params(cache_size, early_stopping_steps,
                                   score_delta_threshold, min_improvement_ratio)

        self.nodes = list(self.data.columns)
        self.node_indices = {node: idx for idx, node in enumerate(self.nodes)}
        self.n_jobs = n_jobs if n_jobs > 0 else None
        self.cache_size = cache_size
        self.early_stopping_steps = early_stopping_steps
        self.score_delta_threshold = score_delta_threshold
        self.min_improvement_ratio = min_improvement_ratio

        # Initialize optimized data structures
        self.data_matrix = data.to_numpy()
        self.score_cache = ScoreCache(cache_size)
        self.adj_matrix = np.zeros((len(self.nodes), len(self.nodes)), dtype=bool)
        self.parent_counts = defaultdict(int)

        self._init_logging()

    def _validate_init_params(self, cache_size: int, early_stopping_steps: int,
                              score_delta_threshold: float, min_improvement_ratio: float) -> None:
        """Validate initialization parameters."""
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        if early_stopping_steps <= 0:
            raise ValueError("early_stopping_steps must be positive")
        if score_delta_threshold <= 0:
            raise ValueError("score_delta_threshold must be positive")
        if not 0 < min_improvement_ratio < 1:
            raise ValueError("min_improvement_ratio must be between 0 and 1")

    def _init_logging(self) -> None:
        """Initialize logging configuration."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("FastGES")

    def _compute_local_score(self, node: str, parents: FrozenSet[str]) -> float:
        """
        Compute local score with improved caching.

        Args:
            node: Target node
            parents: Frozen set of parent nodes

        Returns:
            Local score value
        """
        cache_key = (node, parents)
        cached_score = self.score_cache.get(cache_key)

        if cached_score is not None:
            return cached_score

        score = self.scoring_method.local_score(node, list(parents))
        self.score_cache.put(cache_key, score)
        return score

    def _evaluate_operation(self, source: str, target: str,
                            operation_type: str, current_parents: Set[str]) -> Optional[Operation]:
        """
        Evaluate a single graph operation with statistical validation.

        Args:
            source: Source node
            target: Target node
            operation_type: Type of operation ('add' or 'delete')
            current_parents: Current parent set

        Returns:
            Operation object if valid, None otherwise
        """
        try:
            if operation_type == 'add':
                new_parents = current_parents | {source}
                if self.max_indegree and len(new_parents) > self.max_indegree:
                    return None
            else:
                new_parents = current_parents - {source}

            current_score = self._compute_local_score(target, frozenset(current_parents))
            new_score = self._compute_local_score(target, frozenset(new_parents))
            score_diff = new_score - current_score

            if score_diff > self.score_delta_threshold:
                return Operation(source, target, operation_type,
                                 score_diff, len(new_parents))
            return None

        except Exception as e:
            self.logger.error(f"Error evaluating operation: {str(e)}")
            return None

    def _check_cycle(self, edges: Set[Tuple[str, str]],
                     new_edge: Tuple[str, str]) -> bool:
        """
        Enhanced cycle detection using NetworkX.

        Args:
            edges: Existing edges
            new_edge: Proposed new edge

        Returns:
            True if adding edge creates cycle
        """
        G = nx.DiGraph()
        G.add_edges_from(edges)
        G.add_edge(*new_edge)
        try:
            nx.find_cycle(G)
            return True
        except nx.NetworkXNoCycle:
            return False

    def _generate_candidate_operations(self, dag: DAG,
                                       phase: str) -> List[Tuple[str, str, str, Set[str]]]:
        """
        Generate candidate operations with improved filtering.

        Args:
            dag: Current DAG
            phase: 'forward' or 'backward'

        Returns:
            List of candidate operations
        """
        operations = []
        if phase == 'forward':
            for source, target in combinations(self.nodes, 2):
                if not dag.has_edge(source, target) and not dag.has_edge(target, source):
                    current_parents = set(dag.predecessors(target))
                    if self.max_indegree is None or len(current_parents) < self.max_indegree:
                        if not self._check_cycle(set(dag.edges()), (source, target)):
                            operations.append((source, target, 'add', current_parents))
        else:  # backward phase
            for edge in dag.edges():
                source, target = edge
                current_parents = set(dag.predecessors(target))
                operations.append((source, target, 'delete', current_parents))

        return operations

    def _search_phase(self, dag: DAG, phase: str) -> Tuple[DAG, bool]:
        """
        Perform a search phase (forward or backward) with improved tracking.

        Args:
            dag: Current DAG
            phase: 'forward' or 'backward'

        Returns:
            Updated DAG and whether improvements were made
        """
        no_improvement_count = 0
        made_improvements = False
        total_improvement = 0

        while no_improvement_count < self.early_stopping_steps:
            candidates = self._generate_candidate_operations(dag, phase)
            if not candidates:
                break

            operations = []
            for candidate in candidates:
                op = self._evaluate_operation(*candidate)
                if op is not None:
                    operations.append(op)

            if not operations:
                break

            best_op = max(operations, key=lambda x: x.score_diff)
            relative_improvement = (best_op.score_diff / abs(total_improvement)
                                    if total_improvement != 0 else float('inf'))

            if (best_op.score_diff > self.score_delta_threshold and
                    relative_improvement > self.min_improvement_ratio):
                if best_op.operation_type == 'add':
                    dag.add_edge(best_op.source, best_op.target)
                    self.parent_counts[best_op.target] += 1
                else:
                    dag.remove_edge(best_op.source, best_op.target)
                    self.parent_counts[best_op.target] -= 1

                total_improvement += best_op.score_diff
                made_improvements = True
                no_improvement_count = 0

                self.logger.debug(f"{phase.capitalize()} phase: "
                                  f"{'Added' if best_op.operation_type == 'add' else 'Removed'} "
                                  f"edge {best_op.source}->{best_op.target}")
            else:
                no_improvement_count += 1

        return dag, made_improvements

    def estimate(self, scoring_method=None, max_indegree=None, **kwargs) -> DAG:
        """
        Estimate the DAG structure with improved validation and tracking.

        Args:
            scoring_method: Scoring method (defaults to BIC)
            max_indegree: Maximum number of parents per node
            **kwargs: Additional parameters

        Returns:
            Estimated DAG structure
        """
        self.scoring_method = scoring_method or BicScore(self.data)
        self.max_indegree = max_indegree

        # Initialize empty DAG
        current_dag = DAG()
        for node in self.nodes:
            current_dag.add_node(node)

        # Forward Phase
        self.logger.info("Starting Forward Phase...")
        current_dag, forward_improved = self._search_phase(current_dag, 'forward')

        # Backward Phase
        self.logger.info("Starting Backward Phase...")
        current_dag, backward_improved = self._search_phase(current_dag, 'backward')

        # Final validation
        self._validate_final_dag(current_dag)

        self.logger.info(f"GES completed: {len(current_dag.edges())} edges, "
                         f"avg parents: {np.mean(list(self.parent_counts.values())):.2f}")

        return current_dag

    def _validate_final_dag(self, dag: DAG) -> None:
        """Validate final DAG structure."""
        # Ensure all nodes are present
        missing_nodes = set(self.nodes) - set(dag.nodes())
        for node in missing_nodes:
            dag.add_node(node)
            self.logger.warning(f"Added missing node: {node}")

        # Verify acyclicity
        try:
            nx.find_cycle(dag)
            raise ValueError("Final DAG contains cycles")
        except nx.NetworkXNoCycle:
            pass