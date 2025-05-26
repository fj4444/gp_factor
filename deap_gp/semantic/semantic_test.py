"""
语义检查测试模块

提供测试语义检查功能的示例代码
"""

import random
import numpy as np
from deap import gp, creator, base, tools
from .semantic_wrapper import (
    wrap_genHalfAndHalf,
    wrap_genGrow,
    wrap_genFull,
    wrap_generate,
    wrap_cxOnePoint,
    wrap_mutUniform,
    wrap_mutShrink
)
from .base.primitives import create_pset
from .cpu.operators import setup_advanced_primitives

def test_semantic_checking():
    """
    测试语义检查功能
    """
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 创建特征名称列表
    feature_names = [f'x{i+1}' for i in range(10)]
    
    # 创建原始集
    pset = create_pset(feature_names)
    setup_advanced_primitives(pset)
    
    # 创建适应度类
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)
    
    # 创建工具箱
    toolbox = base.Toolbox()
    
    # 注册使用语义检查的函数
    toolbox.register("expr", wrap_genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # 注册遗传操作
    toolbox.register("mate", wrap_cxOnePoint)
    toolbox.register("expr_mut", wrap_genGrow, pset=pset, min_=0, max_=2)
    toolbox.register("mutate", wrap_mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.register("mutate_shrink", wrap_mutShrink)
    
    # 生成初始种群
    pop = toolbox.population(n=10)
    
    print("初始种群:")
    for i, ind in enumerate(pop):
        print(f"个体 {i+1}: {str(ind)}")
    
    # 测试交叉操作
    print("\n测试交叉操作:")
    ind1, ind2 = pop[0], pop[1]
    print(f"交叉前个体1: {str(ind1)}")
    print(f"交叉前个体2: {str(ind2)}")
    
    ind1_new, ind2_new = toolbox.mate(ind1, ind2)
    print(f"交叉后个体1: {str(ind1_new)}")
    print(f"交叉后个体2: {str(ind2_new)}")
    
    # 测试变异操作
    print("\n测试变异操作:")
    ind = pop[2]
    print(f"变异前个体: {str(ind)}")
    
    ind_mut, = toolbox.mutate(ind)
    print(f"变异后个体: {str(ind_mut)}")
    
    # 测试收缩变异
    print("\n测试收缩变异:")
    ind = pop[3]
    print(f"收缩前个体: {str(ind)}")
    
    ind_shrink, = toolbox.mutate_shrink(ind)
    print(f"收缩后个体: {str(ind_shrink)}")
    
    # 测试生成包含if_then_else的表达式
    print("\n测试生成包含if_then_else的表达式:")
    for _ in range(5):
        expr = wrap_genHalfAndHalf(pset, 2, 4)
        tree = gp.PrimitiveTree(expr)
        has_if = any(node.name == "if_then_else" for node in tree if isinstance(node, gp.Primitive))
        print(f"生成的表达式: {str(tree)}")
        print(f"包含if_then_else: {has_if}")
        print()

if __name__ == "__main__":
    test_semantic_checking()
