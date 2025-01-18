# GES (Greedy Equivalence Search) Algorithm Parameters Reference

## Table 1: Parameter Options

| Parameter              | Available Options                     | Default Value | Valid Range/Values                          |
|-----------------------|--------------------------------------|---------------|-------------------------------------------|
| n_jobs                | Integer value                        | -1            | -1 (all cores) or positive integer         |
| cache_size            | Integer value                        | 1000          | Positive integer                           |
| early_stopping_steps  | Integer value                        | 5             | Positive integer                           |
| score_delta_threshold | Float value                          | 1e-4          | Positive float                             |
| scoring_method        | 'bic', 'k2', 'bdeu', 'bds', 'aic'   | BicScore      | Any scoring method from pgmpy.estimators   |
| max_indegree         | Integer value                        | None          | None or positive integer                   |

## Table 2: Parameter Descriptions and Impact

| Parameter              | Description                                          | Impact on Model                              | When to Adjust                               |
|-----------------------|------------------------------------------------------|---------------------------------------------|---------------------------------------------|
| n_jobs                | Number of parallel jobs for score computation        | - Higher values: Faster on multicore systems<br>- -1: Uses all available cores<br>- 1: No parallelization | - Adjust based on available CPU cores<br>- Reduce if memory usage is too high<br>- Set to 1 for debugging |
| cache_size            | Maximum number of local scores to cache              | - Larger values: Better performance, more memory usage<br>- Smaller values: Less memory usage, more recomputation | - Increase for faster computation with memory trade-off<br>- Decrease if memory is limited<br>- Monitor cache hit rate |
| early_stopping_steps  | Number of iterations without improvement before stopping | - Larger values: More thorough search<br>- Smaller values: Faster convergence<br>- Affects search termination | - Increase for more thorough search<br>- Decrease for faster results<br>- Adjust based on convergence behavior |
| score_delta_threshold | Minimum score improvement required to continue search | - Larger values: Faster termination, fewer edges<br>- Smaller values: More thorough search, more edges | - Decrease if network is too sparse<br>- Increase if overfitting is suspected<br>- Tune based on score metric scale |
| scoring_method        | Scoring method used to evaluate network structure    | - BIC: Best for large samples<br>- K2: Good with prior knowledge<br>- BDeu: Good for discrete data<br>- AIC: Tends to create denser graphs | - Choose based on data characteristics<br>- Consider sample size<br>- Consider prior knowledge availability |
| max_indegree         | Maximum number of parents allowed for any node       | - Larger values: More complex relationships<br>- Smaller values: Simpler structure<br>- None: No restriction | - Limit based on domain knowledge<br>- Reduce for faster computation<br>- Increase if missing important relationships |

## Recommended Configurations

### 1. Fast Exploration Configuration
```yaml
n_jobs: -1
cache_size: 500
early_stopping_steps: 3
score_delta_threshold: 1e-3
scoring_method: BicScore
max_indegree: 3
```

### 2. Balanced Configuration
```yaml
n_jobs: -1
cache_size: 1000
early_stopping_steps: 5
score_delta_threshold: 1e-4
scoring_method: BicScore
max_indegree: 5
```

### 3. Thorough Analysis Configuration
```yaml
n_jobs: -1
cache_size: 2000
early_stopping_steps: 10
score_delta_threshold: 1e-5
scoring_method: BDeuScore
max_indegree: None
```

## Performance Implications

1. **Computational Complexity**
   - Major factors: 
     * max_indegree (O(n^k) where k is max_indegree)
     * Number of variables (O(n^2) candidate edges)
     * early_stopping_steps and score_delta_threshold (affect iterations)

2. **Memory Usage**
   - Primary drivers:
     * cache_size (linear relationship)
     * Number of variables (quadratic relationship)
     * n_jobs (linear with number of processes)

3. **Run Time Factors**
   - Forward phase typically slower than backward phase
   - Score cache hit rate significantly affects performance
   - Parallelization efficiency depends on problem size

## Algorithm Phases

1. **Forward Phase**
   - Starts with empty graph
   - Iteratively adds edges that maximize score
   - Checks cycle constraints and max_indegree
   - Stops when no improvement exceeds threshold

2. **Backward Phase**
   - Starts with forward phase result
   - Iteratively removes edges that improve score
   - More efficient than forward phase
   - Helps reduce overfitting

## Implementation Notes

1. **Score Caching**
   - Implements LRU-style cache for local scores
   - Clears when size exceeds cache_size
   - Critical for performance in dense graphs

2. **Cycle Detection**
   - Uses efficient DFS-based implementation
   - Prevents invalid DAG structures
   - Critical for maintaining acyclicity

3. **Early Stopping**
   - Monitors score improvements
   - Prevents unnecessary iterations
   - Separate counters for forward and backward phases