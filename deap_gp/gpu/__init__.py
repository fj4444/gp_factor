"""
GPU实现模块

包含:
- operators: GPU实现的算子库
- fitness: GPU实现的适应度评估
"""

from .operators import (
    setup_primitives, setup_advanced_primitives, calculate_expression,
    convert_to_torch_tensors, convert_to_numpy, DEVICE
)

from .fitness import (
    create_fitness_function, setup_deap_fitness, evaluate_individual, evaluate_population_batch,
    calculate_ic, calculate_icir, calculate_cumulative_return, calculate_sharpe_ratio
)
