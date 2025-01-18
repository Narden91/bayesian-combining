# PC Algorithm Parameters Reference

## Table 1: Parameter Options

| Parameter           | Available Options                                    | Default Value | Valid Range/Values                          |
|--------------------|-----------------------------------------------------|---------------|-------------------------------------------|
| ci_test            | 'pearsonr', 'chi_square', 'g_sq', 'fisher_exact'    | 'pearsonr'    | Categorical choice from available options   |
| significance_level | Float value                                          | 0.05          | 0.01 to 0.1                               |
| max_cond_vars      | Integer value                                        | None          | 1 to n-2 (n = number of variables)        |
| stable             | Boolean                                              | True          | True/False                                |
| complete_dag_search| Boolean                                              | False         | True/False                                |
| show_progress      | Boolean                                              | True          | True/False                                |

## Table 2: Parameter Descriptions and Impact

| Parameter           | Description                                          | Impact on Model                              | When to Adjust                               |
|--------------------|------------------------------------------------------|---------------------------------------------|---------------------------------------------|
| ci_test            | Statistical test used to determine conditional independence between variables | - 'pearsonr': Best for continuous data with linear relationships<br>- 'chi_square': Optimal for categorical data<br>- 'g_sq': Versatile for mixed data types<br>- 'fisher_exact': Best for small samples with categorical data | - Change based on data type<br>- Switch if current test shows poor performance<br>- Consider computational resources |
| significance_level | Threshold for determining statistical significance in independence tests | - Lower values (e.g., 0.01): More conservative, fewer edges<br>- Higher values (e.g., 0.1): More liberal, more edges<br>- Affects network density | - Adjust if network is too sparse/dense<br>- Change based on domain knowledge requirements<br>- Modify if too many/few dependencies are found |
| max_cond_vars      | Maximum size of the conditioning set in independence tests | - Larger values: More thorough but slower<br>- Smaller values: Faster but might miss complex relationships<br>- Affects computational complexity | - Adjust based on computational resources<br>- Change if runtime is too long<br>- Modify based on expected relationship complexity |
| stable             | Whether to use the order-independent version of PC | - True: More reliable but slower<br>- False: Faster but results may depend on variable order | - Set to True for reproducible results<br>- Set to False for faster prototyping<br>- Consider when comparing different runs |
| complete_dag_search| Whether to search the complete DAG space | - True: More thorough but computationally intensive<br>- False: Faster but might miss some structures | - Use True for final analysis<br>- Use False for initial exploration<br>- Consider computational resources |
| show_progress      | Whether to display progress bars during computation | - True: Provides visual feedback<br>- False: Slightly faster execution | - Set to True during development<br>- Set to False in production<br>- Useful for long-running computations |

## Recommended Configurations

### 1. Fast Exploration Configuration
```yaml
ci_test: pearsonr
significance_level: 0.05
max_cond_vars: 3
stable: false
complete_dag_search: false
show_progress: true
```

### 2. Balanced Configuration
```yaml
ci_test: chi_square
significance_level: 0.05
max_cond_vars: 5
stable: true
complete_dag_search: false
show_progress: true
```

### 3. Thorough Analysis Configuration
```yaml
ci_test: g_sq
significance_level: 0.01
max_cond_vars: 7
stable: true
complete_dag_search: true
show_progress: true
```

## Performance Implications

1. **Computational Complexity**
   - Most intensive parameters: max_cond_vars, complete_dag_search
   - Moderate impact: stable, ci_test
   - Minimal impact: significance_level, show_progress

2. **Memory Usage**
   - High impact: max_cond_vars, complete_dag_search
   - Moderate impact: ci_test (especially with large datasets)
   - Low impact: other parameters

3. **Run Time Factors**
   - Increases exponentially with max_cond_vars
   - Doubles or more with complete_dag_search=True
   - Increases by 20-30% with stable=True