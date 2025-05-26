"""
语义检查使用示例

展示如何在不修改core.py的情况下使用语义检查功能
"""

import os
import numpy as np
import pandas as pd
import random
import json
import time
import warnings
import operator
from typing import Dict, List, Any, Optional, Union, Tuple
from tqdm.auto import tqdm
from deap import creator, base, gp, tools, algorithms

from . import select_device_strategy
from .base.primitives import create_pset
from .semantic_wrapper import (
    wrap_genHalfAndHalf,
    wrap_genGrow,
    wrap_genFull,
    wrap_generate,
    wrap_cxOnePoint,
    wrap_mutUniform,
    wrap_mutShrink
)

def setup_deap_toolbox_with_semantics(pset, data_dict, args, history, device_info):
    """
    设置带有语义检查的DEAP工具箱
    
    参数:
        pset: DEAP原始集
        data_dict: 数据字典
        args: 命令行参数
        history: 世代追踪工具模块
        device_info: 设备信息
        
    返回:
        DEAP工具箱
    """
    # 获取模块引用
    operators_module = device_info['operators_module']
    fitness_module = device_info['fitness_module']
    
    # 设置适应度类
    fitness_module.setup_deap_fitness(maximize=True)
    
    # 创建工具箱
    toolbox = base.Toolbox()
    
    # 注册个体和种群生成函数 - 使用带有语义检查的函数
    # 生成一个个体
    toolbox.register("expr", wrap_genHalfAndHalf, pset=pset, min_=1, max_=args.max_depth)
    # 调用toolbox.expr生成个体，然后把个体装进Individual容器里
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    # 重复获取若干个individual, 把它们装进list里
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # 创建适应度函数
    fitness_weights = None
    if args.fitness_metric == 'comprehensive' and args.fitness_weights is not None:
        try:
            fitness_weights = json.loads(args.fitness_weights)
        except json.JSONDecodeError:
            print("警告: 无法解析适应度权重JSON，使用默认权重")
    
    fitness_function = fitness_module.create_fitness_function(
        metric=args.fitness_metric,
        quantiles=args.fitness_quantiles if hasattr(args, 'fitness_quantiles') else 5,
        weights=fitness_weights
    )     
    
    # 从core.py复制的base_calculate和base_evaluate函数
    def base_calculate(device_name, pset, feature_data, ind):
        from .cpu import operators as operators_module
        return operators_module.calculate_expression(ind, pset=pset, feature_data=feature_data)

    def base_evaluate(device_name, returns, fitness_function, value):
        from .cpu import fitness as fitness_module
        factor_value, barra_values, barra_usage, weights = value
        return fitness_module.evaluate_individual(factor_value, returns=returns, fitness_function=fitness_function, barra_values=barra_values, barra_usage=barra_usage, weights=weights)
    
    if not args.use_gpu:
        toolbox.register("calculate", base_calculate, device_info['device'], pset, data_dict['train_feature_data'])
        toolbox.register("evaluate", base_evaluate, device_info['device'], data_dict['y_train'], fitness_function)
    
    # 注册遗传操作 - 使用带有语义检查的函数
    if args.parsimony:
        toolbox.register("select", tools.selDoubleTournament, fitness_size=args.tournament_size, parsimony_size=1.4, fitness_first=True)
    else:
        toolbox.register("select", tools.selTournament, tournsize=args.tournament_size)

    toolbox.register("mate", wrap_cxOnePoint)

    toolbox.register("expr_mut", wrap_genGrow, pset=pset, min_=0, max_=2)
    def mutUniformAndShrink(individual, expr, pset):
        method = random.choice((0,1,2,3))
        if method < 3:
            return wrap_mutUniform(individual=individual, expr=toolbox.expr_mut, pset=pset)
        else:
            return wrap_mutShrink(individual=individual)
        
    toolbox.register("mutate", mutUniformAndShrink, expr=toolbox.expr_mut, pset=pset)
    
    # 设置突变和交叉的约束
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=args.max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=args.max_depth))

    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)

    # 从core.py复制的process_map_with_tqdm函数
    def process_map_with_tqdm(func, task, global_pool=None, desc=None):
        total = len(task)
        if global_pool is not None:
            results = list(tqdm(
                global_pool.imap(func, task),
                total=total,
                desc=desc or "Processing"
            ))
        else:
            # 对于内置map，需要手动迭代并更新进度条
            warnings.warn(f"未获得进程池，{desc}任务单进程计算")
            results = []
            with tqdm(total=total, desc=desc or "Processing") as pbar:
                for item in task:
                    results.append(func(item))
                    pbar.update(1)
        return results

    toolbox.register("map", process_map_with_tqdm)
    return toolbox

def run_gp_with_semantics(args, data_dict, device_info):
    """
    运行带有语义检查的遗传规划算法
    
    参数:
        args: 命令行参数
        data_dict: 数据字典
        device_info: 设备信息
        
    返回:
        (hof, log): 名人堂和日志
    """
    print("正在设置带有语义检查的遗传规划参数...")
    
    # 设置随机种子
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    
    # 根据设备设置额外的随机种子
    if device_info['device'] == 'gpu' and hasattr(args, 'use_gpu') and args.use_gpu:
        import torch
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.random_seed)
    
    # 创建全局进程池（如果需要）
    global_pool = None
    if hasattr(args, 'n_jobs') and args.n_jobs > 1 and not args.use_gpu:
        try:
            import multiprocessing
            global_pool = multiprocessing.Pool(processes=args.n_jobs)
            print(f"已创建全局进程池，进程数: {args.n_jobs}")
        except Exception as e:
            print(f"警告: 无法创建全局进程池: {e}。使用单进程模式。")
    
    operators_module = device_info['operators_module']
    pset = create_pset(list(data_dict['feature_dict'].keys()))
    operators_module.setup_advanced_primitives(pset)
    history = tools.History()
    
    # 使用带有语义检查的工具箱
    toolbox = setup_deap_toolbox_with_semantics(pset, data_dict, args, history, device_info)
    
    # 从这里开始，可以直接调用core.py中的eaMuPlusLambdaWithEarlyStopping函数
    # 因为我们已经替换了工具箱中的遗传操作函数
    
    print("设置完成，现在可以调用core.py中的eaMuPlusLambdaWithEarlyStopping函数进行进化...")
    print("示例: 从core.py导入eaMuPlusLambdaWithEarlyStopping函数，然后调用它")
    print("from .core import eaMuPlusLambdaWithEarlyStopping")
    
    # 创建初始种群
    if args.use_warm_start:
        pop = toolbox.population(n=args.population_size * 5)
    else:
        pop = toolbox.population(n=args.population_size)
    
    print(f"已创建初始种群，大小: {len(pop)}")
    print("现在可以将这个种群传递给eaMuPlusLambdaWithEarlyStopping函数进行进化")
    
    return pop, toolbox

def main_with_semantics(args):
    """
    带有语义检查的主函数
    
    参数:
        args: 命令行参数
        
    返回:
        (pop, toolbox): 种群和工具箱
    """
    # 检查优化选项
    if hasattr(args, 'use_gpu') and args.use_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                warnings.warn("请求使用 GPU 加速，但 CUDA 不可用，将使用 CPU")
                args.use_gpu = False
        except ImportError:
            warnings.warn("请求使用 GPU 加速，但 PyTorch 不可用，将使用 CPU")
            args.use_gpu = False
    
    # 准备数据 - 这部分可以直接调用core.py中的prepare_data函数
    print("可以直接调用core.py中的prepare_data函数准备数据")
    print("from .core import prepare_data")
    print("data_dict, device_info = prepare_data(args)")
    
    # 运行带有语义检查的遗传规划算法
    print("然后调用run_gp_with_semantics函数运行带有语义检查的遗传规划算法")
    print("pop, toolbox = run_gp_with_semantics(args, data_dict, device_info)")
    
    # 这里只是示例，实际运行时需要先准备数据
    # 返回None作为占位符
    return None, None

if __name__ == "__main__":
    # 这里只是示例，实际运行时需要先解析命令行参数
    print("这是一个示例文件，展示如何在不修改core.py的情况下使用语义检查功能")
    print("实际使用时，需要先解析命令行参数，然后调用main_with_semantics函数")
