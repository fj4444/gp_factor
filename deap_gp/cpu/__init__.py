"""
CPU实现模块

包含:
- operators: CPU实现的算子库
- fitness: CPU实现的适应度评估
"""

from .operators import (
    setup_advanced_primitives, calculate_expression
)

from .fitness import (
    setup_deap_fitness, evaluate_individual,
    calculate_ic, calculate_NDCG
)
