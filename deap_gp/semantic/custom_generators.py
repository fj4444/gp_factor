"""
自定义DEAP表达式生成器

提供针对特定类型约束的表达式生成函数
"""

import random
import sys
from deap import gp

def generate_with_type_constraints(pset, min_, max_, condition, type_=None):
    """
    生成受类型约束的表达式
    
    参数:
        pset: 原始集
        min_: 最小深度
        max_: 最大深度
        condition: 生成条件函数
        type_: 返回类型
        
    返回:
        生成的表达式
    """
    if type_ is None:
        type_ = pset.ret
    
    # 特殊处理整数类型 - 如果需要整数类型且没有返回整数的原始集,直接使用终端节点
    if type_ == int and (not pset.primitives[type_] or max_ <= 0):
        if not pset.terminals[type_]:
            raise ValueError(f"没有找到类型为 {type_} 的终端节点")
        term = random.choice(pset.terminals[type_])
        return term()
    
    # 正常的DEAP生成逻辑
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    
    while len(stack) != 0:
        depth, type_ = stack.pop()
        
        # 表达式已经达到最大深度或满足条件函数,使用终端节点
        if condition(height, depth):
            try:
                term = random.choice(pset.terminals[type_])
                expr.append(term())
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("没有找到类型为 {} 的终端节点".format(type_)).with_traceback(traceback)
        else:
            # 尝试使用原始集
            try:
                prim = random.choice(pset.primitives[type_])
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth + 1, arg))
            except IndexError:
                # 如果没有找到合适的原始集,使用终端节点
                term = random.choice(pset.terminals[type_])
                expr.append(term())
    
    return expr

def genFull_with_constraints(pset, min_, max_, type_=None):
    """
    使用Full方法生成表达式,带有类型约束
    
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
    
    return generate_with_type_constraints(pset, min_, max_, condition, type_)

def genGrow_with_constraints(pset, min_, max_, type_=None):
    """
    使用Grow方法生成表达式,带有类型约束
    
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
    
    return generate_with_type_constraints(pset, min_, max_, condition, type_)

def genHalfAndHalf_with_constraints(pset, min_, max_, type_=None):
    """
    使用Half-and-Half方法生成表达式,带有类型约束
    
    参数:
        pset: 原始集
        min_: 最小深度
        max_: 最大深度
        type_: 返回类型
        
    返回:
        生成的表达式
    """
    method = random.choice([genGrow_with_constraints, genFull_with_constraints])
    return method(pset, min_, max_, type_)
