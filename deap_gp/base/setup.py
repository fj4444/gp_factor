"""
基础原始集定义

提供通用的原始集创建和设置功能,被CPU和GPU实现共享
"""

from deap import gp
import numpy as np

def create_pset(feature_names, ret_type=float):
    """
    创建DEAP原始集
    
    参数:
        feature_names: 特征名称列表
        ret_type: 返回类型
        
    返回:
        DEAP原始集
    """
    # 创建原始集
    pset = gp.PrimitiveSetTyped("MAIN", [float] * len(feature_names), ret_type)
    
    # 重命名参数
    for i, name in enumerate(feature_names):
        pset.renameArguments(**{f"ARG{i}": f"x{i+1}"})
    
    return pset

def add_basic_primitives(pset, ret_type=int):
    """
    添加基本原始集元素
    
    参数:
        pset: DEAP原始集
        ret_type: 返回类型
    
    返回:
        添加了基本原始集元素的DEAP原始集
    """
    # 添加常量
    # pset.addEphemeralConstant("rand_const", lambda: np.random.uniform(-1, 1), ret_type)
    # pset.addTerminal(1, ret_type, name="const_1")
    pset.addTerminal(5, ret_type, name="const_5")
    pset.addTerminal(10, ret_type, name="const_10")
    pset.addTerminal(20, ret_type, name="const_20")
    pset.addTerminal(40, ret_type, name="const_40")
    pset.addTerminal(60, ret_type, name="const_60")
    
    return pset
