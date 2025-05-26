"""
语义检查包装模块

提供带有语义检查功能的遗传编程函数包装器
"""

from deap import gp
from typing import Callable, Optional, Any, List, Tuple
from .semantic_check import (
    genHalfAndHalf_with_semantics,
    genGrow_with_semantics,
    genFull_with_semantics,
    cxOnePoint_with_semantics,
    mutUniform_with_semantics,
    mutShrink_with_semantics
)

def wrap_genHalfAndHalf(pset: gp.PrimitiveSetTyped, min_: int, max_: int, 
                        type_: Optional[type] = None) -> List:
    """
    包装genHalfAndHalf函数，添加语义检查
    
    参数:
        pset: 原始集
        min_: 最小深度
        max_: 最大深度
        type_: 返回类型
        
    返回:
        生成的表达式
    """
    return genHalfAndHalf_with_semantics(pset, min_, max_, type_)

def wrap_genGrow(pset: gp.PrimitiveSetTyped, min_: int, max_: int, 
                type_: Optional[type] = None) -> List:
    """
    包装genGrow函数，添加语义检查
    
    参数:
        pset: 原始集
        min_: 最小深度
        max_: 最大深度
        type_: 返回类型
        
    返回:
        生成的表达式
    """
    return genGrow_with_semantics(pset, min_, max_, type_)

def wrap_genFull(pset: gp.PrimitiveSetTyped, min_: int, max_: int, 
                type_: Optional[type] = None) -> List:
    """
    包装genFull函数，添加语义检查
    
    参数:
        pset: 原始集
        min_: 最小深度
        max_: 最大深度
        type_: 返回类型
        
    返回:
        生成的表达式
    """
    return genFull_with_semantics(pset, min_, max_, type_)

def wrap_generate(pset: gp.PrimitiveSetTyped, min_: int, max_: int, 
                 condition: Callable, type_: Optional[type] = None) -> List:
    """
    包装generate函数，添加语义检查
    
    参数:
        pset: 原始集
        min_: 最小深度
        max_: 最大深度
        condition: 生成条件函数
        type_: 返回类型
        
    返回:
        生成的表达式
    """
    from .semantic_check import generate_with_semantics
    return generate_with_semantics(pset, min_, max_, condition, type_)

def wrap_cxOnePoint(ind1: gp.PrimitiveTree, ind2: gp.PrimitiveTree) -> Tuple[gp.PrimitiveTree, gp.PrimitiveTree]:
    """
    包装cxOnePoint函数，添加语义检查
    
    参数:
        ind1: 第一个个体
        ind2: 第二个个体
        
    返回:
        交叉后的两个个体
    """
    return cxOnePoint_with_semantics(ind1, ind2)

def wrap_mutUniform(individual: gp.PrimitiveTree, expr: Callable, 
                   pset: gp.PrimitiveSetTyped) -> Tuple[gp.PrimitiveTree]:
    """
    包装mutUniform函数，添加语义检查
    
    参数:
        individual: 要变异的个体
        expr: 生成表达式的函数
        pset: 原始集
        
    返回:
        变异后的个体
    """
    return mutUniform_with_semantics(individual, expr, pset)

def wrap_mutShrink(individual: gp.PrimitiveTree) -> Tuple[gp.PrimitiveTree]:
    """
    包装mutShrink函数，添加语义检查
    
    参数:
        individual: 要变异的个体
        
    返回:
        变异后的个体
    """
    return mutShrink_with_semantics(individual)
