"""
语义检查包装函数

为遗传编程中的树生成和变异操作提供语义检查功能
"""

import random
import numpy as np
from deap import gp
from collections import defaultdict
from typing import List, Callable, Any, Optional, Union, Tuple

def check_if_then_else_semantics(expr: List, pset: gp.PrimitiveSetTyped) -> bool:
    """
    检查if_then_else操作符的语义是否合法
    
    参数:
        expr: 表达式列表
        pset: 原始集
        
    返回:
        语义是否合法
    """
    # 如果表达式为空，直接返回True
    if not expr:
        return True
    
    # 遍历表达式中的每个节点
    for i, node in enumerate(expr):
        # 检查是否是if_then_else操作符
        if isinstance(node, gp.Primitive) and node.name == "if_then_else":
            # if_then_else操作符有4个参数：条件1, 条件2, 结果1, 结果2
            # 我们需要检查条件1和条件2是否有相同的量纲
            
            # 获取if_then_else的子树
            slice_ = gp.PrimitiveTree(expr).searchSubtree(i)
            subtree = expr[slice_]
            
            # 如果子树长度不足，说明树还没有完全构建好，跳过检查
            if len(subtree) < 5:  # if_then_else + 至少4个参数
                continue
            
            # 获取条件1和条件2的子树
            condition1_idx = i + 1  # if_then_else后的第一个参数
            condition1_slice = gp.PrimitiveTree(expr).searchSubtree(condition1_idx)
            condition1_subtree = expr[condition1_slice]
            
            # 条件2的位置取决于条件1子树的大小
            condition2_idx = condition1_slice.stop
            condition2_slice = gp.PrimitiveTree(expr).searchSubtree(condition2_idx)
            condition2_subtree = expr[condition2_slice]
            
            # 检查两个条件子树的"量纲"是否相同
            # 在这个简化的实现中，我们通过检查子树的根节点类型来判断
            # 实际应用中，可能需要更复杂的量纲检查逻辑
            
            # 获取条件1和条件2的根节点
            condition1_root = condition1_subtree[0]
            condition2_root = condition2_subtree[0]
            
            # 如果两个条件的根节点类型不同，返回False
            if isinstance(condition1_root, gp.Primitive) and isinstance(condition2_root, gp.Primitive):
                if condition1_root.ret != condition2_root.ret:
                    return False
            elif isinstance(condition1_root, gp.Terminal) and isinstance(condition2_root, gp.Terminal):
                if type(condition1_root.value) != type(condition2_root.value):
                    return False
            elif isinstance(condition1_root, gp.Primitive) and isinstance(condition2_root, gp.Terminal):
                return False
            elif isinstance(condition1_root, gp.Terminal) and isinstance(condition2_root, gp.Primitive):
                return False
    
    # 所有检查都通过，返回True
    return True

def generate_with_semantics(pset: gp.PrimitiveSetTyped, min_: int, max_: int, 
                           condition: Callable, type_: Optional[type] = None,
                           max_attempts: int = 50) -> List:
    """
    生成满足语义约束的表达式
    
    参数:
        pset: 原始集
        min_: 最小深度
        max_: 最大深度
        condition: 生成条件函数
        type_: 返回类型
        max_attempts: 最大尝试次数
        
    返回:
        生成的表达式
    """
    attempts = 0
    while attempts < max_attempts:
        expr = gp.generate(pset, min_, max_, condition, type_)
        if check_if_then_else_semantics(expr, pset):
            return expr
        attempts += 1
    
    # 如果达到最大尝试次数仍未生成有效表达式，返回最后一次生成的表达式
    return expr

def genFull_with_semantics(pset: gp.PrimitiveSetTyped, min_: int, max_: int, 
                          type_: Optional[type] = None) -> List:
    """
    使用Full方法生成满足语义约束的表达式
    
    参数:
        pset: 原始集
        min_: 最小深度
        max_: 最大深度
        type_: 返回类型
        
    返回:
        生成的表达式
    """
    def condition(height, depth):
        return depth == height
    
    return generate_with_semantics(pset, min_, max_, condition, type_)

def genGrow_with_semantics(pset: gp.PrimitiveSetTyped, min_: int, max_: int, 
                          type_: Optional[type] = None) -> List:
    """
    使用Grow方法生成满足语义约束的表达式
    
    参数:
        pset: 原始集
        min_: 最小深度
        max_: 最大深度
        type_: 返回类型
        
    返回:
        生成的表达式
    """
    def condition(height, depth):
        return depth == height or (depth >= min_ and random.random() < pset.terminalRatio)
    
    return generate_with_semantics(pset, min_, max_, condition, type_)

def genHalfAndHalf_with_semantics(pset: gp.PrimitiveSetTyped, min_: int, max_: int, 
                                 type_: Optional[type] = None) -> List:
    """
    使用Half-and-Half方法生成满足语义约束的表达式
    
    参数:
        pset: 原始集
        min_: 最小深度
        max_: 最大深度
        type_: 返回类型
        
    返回:
        生成的表达式
    """
    method = random.choice([genGrow_with_semantics, genFull_with_semantics])
    return method(pset, min_, max_, type_)

def cxOnePoint_with_semantics(ind1: gp.PrimitiveTree, ind2: gp.PrimitiveTree, 
                             max_attempts: int = 20) -> Tuple[gp.PrimitiveTree, gp.PrimitiveTree]:
    """
    带有语义检查的单点交叉操作
    
    参数:
        ind1: 第一个个体
        ind2: 第二个个体
        max_attempts: 最大尝试次数
        
    返回:
        交叉后的两个个体
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # 单节点树无法进行交叉
        return ind1, ind2

    # 尝试进行有效的交叉
    attempts = 0
    while attempts < max_attempts:
        # 复制个体，以便在检查失败时恢复
        ind1_copy = gp.PrimitiveTree(ind1)
        ind2_copy = gp.PrimitiveTree(ind2)
        
        # 执行标准的单点交叉
        if ind1.root.ret == gp.__type__:
            # 非强类型GP优化
            types1 = [gp.__type__] * len(ind1)
            types2 = [gp.__type__] * len(ind2)
            common_types = [gp.__type__]
        else:
            # 列出每个个体中所有可用的原始类型
            types1 = defaultdict(list)
            types2 = defaultdict(list)
            for idx, node in enumerate(ind1):
                types1[node.ret].append(idx)
            for idx, node in enumerate(ind2):
                types2[node.ret].append(idx)
            common_types = set(types1.keys()).intersection(set(types2.keys()))

        if len(common_types) > 0:
            type_ = random.choice(list(common_types))
            
            index1 = random.choice(types1[type_])
            index2 = random.choice(types2[type_])
            
            slice1 = ind1.searchSubtree(index1)
            slice2 = ind2.searchSubtree(index2)
            ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
            
            # 检查交叉后的个体是否满足语义约束
            if check_if_then_else_semantics(ind1, ind1.pset) and check_if_then_else_semantics(ind2, ind2.pset):
                return ind1, ind2
            
            # 如果不满足，恢复原始个体
            ind1[:] = ind1_copy
            ind2[:] = ind2_copy
        
        attempts += 1
    
    # 如果达到最大尝试次数仍未找到有效交叉，返回原始个体
    return ind1, ind2

def mutUniform_with_semantics(individual: gp.PrimitiveTree, expr: Callable, 
                             pset: gp.PrimitiveSetTyped, max_attempts: int = 20) -> Tuple[gp.PrimitiveTree]:
    """
    带有语义检查的均匀变异操作
    
    参数:
        individual: 要变异的个体
        expr: 生成表达式的函数
        pset: 原始集
        max_attempts: 最大尝试次数
        
    返回:
        变异后的个体
    """
    attempts = 0
    while attempts < max_attempts:
        # 复制个体，以便在检查失败时恢复
        individual_copy = gp.PrimitiveTree(individual)
        
        # 随机选择一个点进行变异
        index = random.randrange(len(individual))
        slice_ = individual.searchSubtree(index)
        type_ = individual[index].ret
        
        # 生成新的子树
        new_subtree = expr(pset=pset, type_=type_)
        
        # 替换子树
        individual[slice_] = new_subtree
        
        # 检查变异后的个体是否满足语义约束
        if check_if_then_else_semantics(individual, pset):
            return individual,
        
        # 如果不满足，恢复原始个体
        individual[:] = individual_copy
        attempts += 1
    
    # 如果达到最大尝试次数仍未找到有效变异，返回原始个体
    return individual,

def mutShrink_with_semantics(individual: gp.PrimitiveTree, max_attempts: int = 20) -> Tuple[gp.PrimitiveTree]:
    """
    带有语义检查的收缩变异操作
    
    参数:
        individual: 要变异的个体
        max_attempts: 最大尝试次数
        
    返回:
        变异后的个体
    """
    # 如果个体太小，无法进行收缩变异
    if len(individual) < 3 or individual.height <= 1:
        return individual,
    
    attempts = 0
    while attempts < max_attempts:
        # 复制个体，以便在检查失败时恢复
        individual_copy = gp.PrimitiveTree(individual)
        
        # 找到所有可以收缩的原始节点
        iprims = []
        for i, node in enumerate(individual[1:], 1):
            if isinstance(node, gp.Primitive) and node.ret in node.args:
                iprims.append((i, node))
        
        # 如果没有可收缩的节点，返回原始个体
        if len(iprims) == 0:
            return individual,
        
        # 随机选择一个节点进行收缩
        index, prim = random.choice(iprims)
        arg_idx = random.choice([i for i, type_ in enumerate(prim.args) if type_ == prim.ret])
        
        # 找到要替换的子树
        rindex = index + 1
        for _ in range(arg_idx + 1):
            rslice = individual.searchSubtree(rindex)
            subtree = individual[rslice]
            rindex += len(subtree)
        
        # 执行收缩变异
        slice_ = individual.searchSubtree(index)
        individual[slice_] = subtree
        
        # 检查变异后的个体是否满足语义约束
        if check_if_then_else_semantics(individual, individual.pset):
            return individual,
        
        # 如果不满足，恢复原始个体
        individual[:] = individual_copy
        attempts += 1
    
    # 如果达到最大尝试次数仍未找到有效变异，返回原始个体
    return individual,
