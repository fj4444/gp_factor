"""
核心算法实现，根据配置选择CPU或GPU实现
"""

import os
NUM_THREADS = '1'
os.environ['OMP_NUM_THREADS'] = NUM_THREADS
os.environ['MKL_NUM_THREADS'] = NUM_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = NUM_THREADS
os.environ['NUMEXPR_MAX_THREADS'] = NUM_THREADS
os.environ['NUM_INTER_THREADS'] = NUM_THREADS
os.environ['NUM_INTEA_THREADS'] = NUM_THREADS
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

from . import select_device_strategy, TORCH_AVAILABLE, NUMBA_AVAILABLE, RAPIDS_AVAILABLE
from .base.setup import create_pset
# from .custom_generators import genHalfAndHalf_with_constraints, genFull_with_constraints

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

def prepare_data(args):
    """
    准备数据
    
    参数:
        args: 参数对象，包含data_source和data_file等参数
        
    返回:
        {
        'y_train': y_train, #2D np array
        'time_train': time_train, #np array
        'y_test': y_test, #2D np array
        'time_test': time_test, #np array
        'feature_cols': feature_cols, #list
        'feature_dict': feature_dict, #{str(x1):str(lnto)}
        'train_feature_data': train_feature_data,#{str(x1):2D np array}
        'test_feature_data': test_feature_data,#{str(x1):2D np array}
        'dataset_index_column': [dataset_time_index, dataset_code_column]

        }, device_info
    """
    print("正在处理数据...")
    
    device_info = select_device_strategy(args.use_gpu)
    
    DataProcessorClass = device_info['data_processor']
    data_processor = DataProcessorClass(use_gpu=args.use_gpu)
    
    # 加载数据
    if hasattr(args, 'data') and args.data is not None:
        if not args.use_barra:
            data_processor.transform_data(args.data)
        else:
            data_processor.transform_data(args.data, args.barra_data, args.weights_data)
    else:
        raise ValueError("必须提供数据(args.data)")
    
    # 设置特征和目标
    if args.feature_cols is None:
        raise AttributeError("必须指定特征列列名")
    else:
        feature_cols = args.feature_cols
    
    if hasattr(args, 'target_col'):
        target_col = args.target_col
    else:
        # 使用原始收益率作为目标
        warnings.warn("预测目标为当期收益率,可能引入未来信息")
        target_col = args.return_col
    
    data_processor.set_features_and_target(features=feature_cols, target=target_col)
    # !这一步之后feature_cols就不应该再用了，因为feature_cols是config里设置的所有可能存在的特征，但如果只加载了一部分数据，真正能用的特征会是feature_cols的子集
    # !set_features_and_target里会给data_processor对象的features成员赋值为数据中真正包含的特征列表

    # 按时间分割原始特征数据和barra风格因子数据
    data_processor.split_data(
        test_size=args.test_size,
        time_series=True,
        time_col=args.time_col,
        id_col=args.id_col   
    )

    # 准备训练数据
    test_feature_data = data_processor.get_test_feature_dict()
    train_feature_data = data_processor.get_train_feature_dict()
    
    y_train = data_processor.get_train_target_vector()
    y_test = data_processor.get_test_target_vector()

    time_train = data_processor.get_train_time_vector()
    time_test = data_processor.get_test_time_vector()

    real_feature_list = data_processor.get_feature_list()
    feature_dict = {}
    for i, col in enumerate(real_feature_list):
        feature_dict[f'x{i+1}'] = col
    
    dataset_time_index = data_processor.get_dataset_index()
    dataset_code_column = data_processor.get_dataset_column()

    data_dict = {
        'y_train': y_train, #股票*时间的2D nparray
        'time_train': time_train, #np array
        'y_test': y_test, #股票*时间的2D nparray
        'time_test': time_test, #np array
        'feature_dict': feature_dict, #特征名称和特征别名的对应关系，例如{'x1':'lnto'}
        'train_feature_data': train_feature_data,#{str(x1):股票*时间的2D nparray}
        'test_feature_data': test_feature_data,#{str(x1):股票*时间的2D nparray}
        'dataset_index_column': [dataset_time_index, dataset_code_column]
    }

    if args.use_barra:
        data_dict['barra_train'] = data_processor.get_barra_train() #{'rv':股票*时间的2D nparray}
        data_dict['barra_test'] = data_processor.get_barra_test()
        data_dict['weights_train'] = data_processor.get_weights_train() #weights只有一个矩阵,不需要用dict的形式了，直接是股票*时间的2D nparry

    return data_dict, device_info

# 基本回填函数
def base_calculate(device_name, pset, feature_data, ind):
    if device_name=='gpu':
        from .gpu import operators as operators_module
    else:
        from .cpu import operators as operators_module
    return operators_module.calculate_expression(ind, pset=pset, feature_data=feature_data)

# 基本评估函数
def base_evaluate(device_name, returns, metric, value):
    """
    这个函数经过了多层包装，在toolbox的register获得了前三个参数，在toolbox.map才能获得最后一个参数，也只能获得一个参数
    所以因子值ndarray和包含风格因子值ndarray的字典作为一个tuple一起传到values里了，如果初始设置没有要求使用风格因子，那么value就是(factor_value,None)
    """
    if device_name=='gpu':
        from .gpu import fitness as fitness_module
    else:
        from .cpu import fitness as fitness_module
    factor_value, barra_values, barra_usage, weights = value
    return fitness_module.evaluate_individual(factor_value, returns=returns, metric=metric, barra_values=barra_values, barra_usage=barra_usage, weights=weights)

def setup_deap_toolbox(pset, data_dict, args, history, device_info):
    """
    设置DEAP工具箱
    
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
    fitness_module.setup_deap_fitness(maximize=(args.fitness_metric!='double'))
    
    # 创建工具箱
    toolbox = base.Toolbox()
    
    # 注册个体和种群生成函数
    # 生成一个个体
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=args.max_depth)
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
    
    if not args.use_gpu:
        toolbox.register("calculate", base_calculate, device_info['device'], pset, data_dict['train_feature_data'])
        toolbox.register("evaluate", base_evaluate, device_info['device'], data_dict['y_train'], args.fitness_metric)
    
    # 注册遗传操作
    if args.fitness_metric == 'double':
        toolbox.register("select",tools.selNSGA2)
    elif args.parsimony:
        toolbox.register("select", tools.selDoubleTournament, fitness_size=args.tournament_size, parsimony_size=1.4, fitness_first=True)
    else:
        toolbox.register("select", tools.selTournament, tournsize=args.tournament_size)

    toolbox.register("mate", gp.cxOnePoint)

    toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)
    def mutUniformAndShrink(individual, expr, pset, shrink_prob=0.25):
        assert shrink_prob <= 1.0, ("提升变异在变异操作中的概率必须小于1")
        op_choice = random.random()
        if op_choice < shrink_prob:
            return gp.mutShrink(individual=individual)
        else:
            return gp.mutUniform(individual=individual, expr=toolbox.expr_mut, pset=pset)            
        
    toolbox.register("mutate", mutUniformAndShrink, expr=toolbox.expr_mut, pset=pset, shrink_prob=args.shrink_mutation_rate)
    
    # 设置突变和交叉的约束
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=args.max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=args.max_depth))

    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)

    toolbox.register("map", process_map_with_tqdm)
    return toolbox

def calculate_correlation(factor_matrix_by_row):
    """
    计算矩阵中各行两两之间的相关系数
    
    参数:
        factor_matrix_by_row: 包含因子值/因子ic值的矩阵, 每个因子的数据占一行
    返回:
        因子间的相关矩阵
    """

    if np.isscalar(factor_matrix_by_row):
        return 0.0

    if isinstance(factor_matrix_by_row, pd.Series):
        # 将Series转换为2D矩阵 (时间 x 标的)
        factor_matrix_by_row = factor_matrix_by_row.unstack(level='InnerCode').values
    
    factor_matrix_by_row = np.nan_to_num(factor_matrix_by_row, nan=0.0)
    # 如果数据太少，返回0
    if factor_matrix_by_row.shape[0] < 5 or factor_matrix_by_row.shape[1] < 5:
        return 0.0
    else:
        try:
            return np.corrcoef(factor_matrix_by_row, rowvar=True)
        except Exception as e:
            print(f"计算相关性时出错: {e}")
            return 0.0

def filter_correlated_individuals_daily(population, factor_values, n_hof, pset, feature_data,  
                                        correlation_threshold=0.6, use_gpu=False, global_pool=None, n_jobs=1, factor_ic_values=None):
    """
    获得population中相互之间相关性&和名人堂个体之间相关性低于correlation_threshold的所有个体
    
    参数:
        population: 新生成的,需要计算相似度的个体群体
        factor_values: population中因子的值和hof中因子的值构成的列表
        n_hof: 名人堂个体的数量
        pset: 原始集
        feature_data: 特征数据
        correlation_threshold: 相关性阈值，超过此值的因子将被过滤
        use_gpu: 是否使用GPU
        global_pool: 全局进程池（可选）
        n_jobs: 进程数
        factor_ic_values: population中的因子和hof中的因子的ic序列构成的列表,如果不计算IC相关系数,则传入None
        
    返回:
        过滤后的个体群体,按照适应度从高到低排序
    """
    print(f"本次过滤前，population size:{len(population)}，hof size:{n_hof}")
    n_pop = len(population)

    # 按照时间切分，每天计算一次当天截面上全部因子两两之间的相关性
    correlation_tasks = []
    for i in range(len(factor_values[0])):
        current_day_factor_list = []
        for factor_value in factor_values:
            current_day_factor_list.append(factor_value[i,:])
        current_day_factor_matrix = np.vstack(current_day_factor_list)
        correlation_tasks.append(current_day_factor_matrix)
    print(f"开始相似度计算，任务数: {len(correlation_tasks)}")

    if global_pool is None and n_jobs > 1:
        warnings.warn(f"试图使用多线程，但未创建全局进程池，改用单线程计算相似度")

    correlation_results = process_map_with_tqdm(calculate_correlation, correlation_tasks, global_pool=global_pool, desc="相似度计算")
    correlation_matrix = np.nanmean(np.stack(correlation_results),axis=0)
    
    # 计算各因子的IC序列之间的相关性
    if factor_ic_values is not None:
        factor_ic_matrix = np.vstack(factor_ic_values)
        ic_correlation_matrix = calculate_correlation(factor_ic_matrix)
        correlation_matrix = np.maximum(correlation_matrix, ic_correlation_matrix)

    pop_correlation_matrix = correlation_matrix[:n_pop, :n_pop]

    # 基于相关性矩阵过滤种群
    # 按适应度排序，优先保留适应度高的个体
    sorted_indices = sorted(range(n_pop), 
                           key=lambda i: population[i].fitness.values[0], 
                           reverse=True)
    
    # 创建一个掩码，标记保留个体
    keep_mask = [True] * n_pop
    
    for i in range(len(sorted_indices)):
        idx_i = sorted_indices[i]
        if not keep_mask[idx_i]:
            continue  # 如果这个个体已经被标记为删除，跳过
        
        # 检查这个个体与其他个体的相关性
        for j in range(i+1, len(sorted_indices)):
            idx_j = sorted_indices[j]
            if not keep_mask[idx_j]:
                continue  # 如果这个个体已经被标记为删除，跳过
            
            # 直接使用排序后的索引检查相关性
            if abs(pop_correlation_matrix[idx_i, idx_j]) > correlation_threshold:
                keep_mask[idx_j] = False  # 标记为删除

    print(f"种群内过滤后剩余: {np.sum(keep_mask)}/{n_pop} 个体")
    
    # 然后处理与hof的相关性
    if n_hof > 0:
        pop_hof_correlation_matrix = correlation_matrix[:n_pop, n_pop:] 
        for i in range(n_pop):
            if not keep_mask[i]:
                continue
            for j in range(n_hof):
                if abs(pop_hof_correlation_matrix[i, j]) > correlation_threshold:
                    keep_mask[i] = False
        
        print(f"与名人堂过滤后剩余: {np.sum(keep_mask)}/{n_pop} 个体")
 
    filtered_population = sorted([ind for i, ind in enumerate(population) if keep_mask[i]], key=lambda ind: ind.fitness.values[0], reverse=True)

    # 如果过滤后没有个体，则保留原始过滤后种群中适应度最高的个体
    if len(filtered_population) == 0:
        sorted_population = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
        filtered_population = [sorted_population[0]] if sorted_population else []
        warnings.warn("过滤后没有个体，保留最高适应度个体")
    
    return filtered_population

def eaMuPlusLambdaWithEarlyStopping(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, history, stats=None, halloffame=None, verbose=__debug__, use_warm_start=False, 
                                    correlation_threshold=0.6, correlation_threshold_init=0.4, iccorr=True, dynamicProb=True, patience=5, min_delta=0.001,pset=None, feature_data=None, 
                                    use_gpu=False, n_jobs=1, global_pool=None, experiment_name=None, output_dir=None, barra_values=None, barra_usage='correlation', weights=None, competition=True):
    """
    带有早期停止功能的(μ+λ)进化策略
    
    参数:
        population: 初始种群
        toolbox: 包含进化操作的工具箱
        lambda_: 需要生成的子代数量
        mu: 从当前轮次的子代中选取mu个,亲代中选取lambda_-mu个,一同成为下一轮的亲代
        cxpb: 交叉概率
        mutpb: 变异概率
        ngen: 最大代数
        history: 亲代追踪模块
        stats: 统计对象
        halloffame: 名人堂对象
        verbose: 是否输出详细信息
        use_warm_start: 初始种群是否使用了热启动，如果未使用，跳过第一轮筛选
        correlation_threshold: 相关性阈值，超过此值的因子将被过滤
        correlation_threshold_init: 热启动后初始种群的相关性阈值
        iccorr: 是否用因子的IC序列计算相关系数并添加到相关性筛选中
        dynamicProb: 是否根据筛选后的多样性动态调整变异概率大小
        patience: 早期停止耐心值，连续多少代没有改善就停止 
        min_delta: 最小改善阈值，适应度提升小于此值视为没有显著改善
        pset: 原始集，用于计算因子相关性
        feature_data: 特征数据，用于计算因子相关性
        use_gpu: 是否使用GPU
        n_jobs: 进程数
        global_pool: 全局进程池（可选）,
        experiment_name: 实验名称，也是因子记录的文件名称
        output_dir: 记录输出目录
        barra_values: 训练集对应的风格因子值,是特征ndarray组成的dict, 如果启动参数里没有use_barra, 就是None
        barra_usage: barra风格因子的用法, correlation代表计算和因子的相关性作为惩罚项, neutralize代表将因子值WLS回归到风格因子上,用残差做因子值
        weights: 训练集对应的风格因子加权中性化的权重,设置为sqrt(float market value), 如果启动参数里没有use_barra, 就是None
        competition: 是否进行亲子竞争
    返回:
        (最终种群, 日志)
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # 评估初始种群
    if not use_gpu:
        factor_values = toolbox.map(toolbox.calculate, population, global_pool=global_pool, desc="初始种群-回填因子值")
        # 实验中发现诸如rank_sub(x54, x54)这样的因子，由于因子值只有0，在相关性计算的时候必然算出NaN，从而通过筛选留下，会持续污染population。应该把因子值只有少数几种值,缺少区分度的因子直接从种群里删掉
        useless_factor_indice = [i for i,factor_value in enumerate(factor_values) if (len(np.unique(factor_value))<=5)]
        
        for index in sorted(useless_factor_indice, reverse=True):
            del population[index]
            del factor_values[index]
            
        print(f"删去无用因子后后剩余: {len(population)}/{len(population)+len(useless_factor_indice)} 个体")

        #  factor_values是list, 每个元素都是一个因子值的ndarray, 应该把每个元素包装成(factor_value, barra_values)的元组,
        #  其中barra_values应该是用到的所有风格因子的ndarray组成的dict,如果设置里不要求风格因子,barra_values=None
        factor_barra_values = [(factor_value, barra_values, barra_usage, weights) for factor_value in factor_values]
        fitness, ic_array_tuple = list(zip(*toolbox.map(toolbox.evaluate, factor_barra_values, global_pool=global_pool, desc="初始种群-评估适应度")))

    ic_array_list = list(ic_array_tuple)
    for ind, fit in zip(population, fitness):
        ind.fitness.values = fit

    # 统计初始种群信息
    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    import copy
    old_pop = copy.deepcopy(population) #保留筛选前的population,其中元素和factor_values一一对应
    # 如果使用了热启动 筛选低相关性高适应度个体
    if use_warm_start:
        population = filter_correlated_individuals_daily(
            population, factor_values, 0, pset, feature_data, correlation_threshold_init, use_gpu, global_pool=global_pool, n_jobs=n_jobs, factor_ic_values=ic_array_list if iccorr else None)[:lambda_]
    
    experiment_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_dir,exist_ok=True)

    with open(os.path.join(experiment_dir, "generations.txt"), 'w', encoding='utf-8') as file:
        file.write("初始种群:\n")
        for ind in population:
            file.write(f"\t{str(ind)}\n")

    halloffame.update(population)
    # 找到hof个体在population中的index
    hofer_indices_pop = [old_pop.index(hofer) if hofer in old_pop else np.inf for hofer in halloffame]
    try:
        hof_factor_values = [factor_values[i] for i in hofer_indices_pop]
        hof_ic_array_list = [ic_array_list[i] for i in hofer_indices_pop]
    except Exception as e:
        print(f"初始hof个体在总体种群中的索引值有误: {e}")
    
    # 后续进化的setting
    # 早停设置
    best_fitness = min([ind.fitness.values[0] for ind in halloffame])
    no_improvement_count = 0
    
    # 进化代数计数
    gen = 1
    # 开始进化循环
    while gen < ngen + 1:
        print(f"本代有{len(population)}个个体作为亲代")
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)        
        # 评估子代
        # 这里对全部offspring中的个体都进行回填。严格来说，如果cxpb+mutpb<1，是有机会节省一些计算的，因为多余的概率会用于reproduction，直接从亲代里随机选几个变成子代
        # 这些直接变成子代的个体，本身已经回填过一次，但是因子值不方便找到，还是跟着重算一次
        factor_values = toolbox.map(toolbox.calculate, offspring, global_pool=global_pool, desc=f"第{gen}代-回填因子值")
        useless_factor_indice = [i for i,factor_value in enumerate(factor_values) if (len(np.unique(factor_value))<=5)]
        
        for index in sorted(useless_factor_indice, reverse=True):
            del offspring[index]
            del factor_values[index]
            
        print(f"删去无用因子后后剩余: {len(offspring)}/{len(offspring)+len(useless_factor_indice)} 个体")
        
        factor_barra_values = [(factor_value, barra_values, barra_usage, weights) for factor_value in factor_values]
        fitness, ic_array_tuple = zip(*toolbox.map(toolbox.evaluate, factor_barra_values, global_pool=global_pool, desc=f"第{gen}代-评估适应度"))

        ic_array_list = list(ic_array_tuple)

        for ind, fit in zip(offspring, fitness):
            ind.fitness.values = fit

        # 记录统计信息
        record = stats.compile(offspring) if stats else {}
        logbook.record(gen=gen, nevals=len(offspring), **record)
        if verbose:
            print(logbook.stream)

        factor_values.extend(hof_factor_values)
        ic_array_list.extend(hof_ic_array_list)

        filtered_offspring = filter_correlated_individuals_daily(
            offspring, factor_values, len(halloffame), pset, feature_data, correlation_threshold, use_gpu, global_pool=global_pool, n_jobs=n_jobs, factor_ic_values=ic_array_list if iccorr else None)[:lambda_]

        if dynamicProb:
        # 动态调整进化中的变异概率，在多样性不足时提高多样性
            if len(filtered_offspring) < mu:
                delta = (cxpb-0.2)*0.5
                mutpb += delta
                cxpb -= delta

        with open(os.path.join(experiment_dir, "generations.txt"), 'a', encoding='utf-8') as file:
            file.write(f"筛选后的第{gen}代种群:\n")
            for ind in filtered_offspring:
                file.write(f"\t{str(ind)}\n")
            file.write("\n")

        old_hof = copy.deepcopy(halloffame)
        halloffame.update(filtered_offspring)  
        offspring.extend(old_hof)
        hofer_indices = [offspring.index(hofer) if hofer in offspring else np.inf for hofer in halloffame]
        # hof个体都应该来自上一轮hof或者本轮生成的子代，index不应该出现np.inf，如果出现了np.inf会被catch并报错
        try:
            hof_factor_values = [factor_values[i] for i in hofer_indices]
            hof_ic_array_list = [ic_array_list[i] for i in hofer_indices]
        except Exception as e:
            print(f"第{gen}代hof个体在总体中的索引值有误: {e}")

        # 早期停止检查
        current_best = min([ind.fitness.values[0] for ind in halloffame])
        if current_best > best_fitness + min_delta:
            # 有显著改善，重置计数器
            best_fitness = current_best
            no_improvement_count = 0
        else:
            # 没有显著改善，增加计数器
            no_improvement_count += 1
            
        # 如果连续多代没有显著改善，提前停止
        if no_improvement_count >= patience:
            if verbose:
                print(f"早期停止：{patience}代内适应度提升小于{min_delta}")
            break

        # 选择下一次遗传的亲代
        if competition:
        # 父子竞争,对所有进入filtered_offspring的子代,如果发现适应度高于亲代,就把亲代从population中去掉
            parents_to_remove = []
            for ind in filtered_offspring:
                for parent_id in list(history.getGenealogy(ind,1).values())[0]:
                    parent_entity = history.genealogy_history[parent_id]
                    if ind.fitness.values[0] > parent_entity.fitness.values[0]:
                        if parent_entity not in parents_to_remove:
                            parents_to_remove.append(parent_entity)

            population = [entity for entity in population if entity not in parents_to_remove]
            print(f"在第{gen}代亲子竞争中,删去{len(parents_to_remove)}个亲代个体")
            if len(population)==0:
                print(f"亲子竞争后亲代全部被删除,重新生成{lambda_}个亲代")
                population = toolbox.population(n=lambda_)           
            
        population[:] = toolbox.select(population, (lambda_-mu)) + toolbox.select(filtered_offspring, mu)
        gen = gen + 1

    return logbook, gen

def run_gp(args, data_dict, device_info):
    """
    运行遗传规划算法
    
    参数:
        args: 命令行参数
        data_dict: 数据字典
        device_info: 设备信息
        
    返回:
        (hof, log): 名人堂和日志
    """
    print("正在设置遗传规划参数...")
    
    # 设置随机种子
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    
    # 根据设备设置额外的随机种子
    if device_info['device'] == 'gpu' and TORCH_AVAILABLE:
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
    toolbox = setup_deap_toolbox(pset, data_dict, args, history, device_info)
    if args.fitness_metric=='double':
        # from .base.LimitedParetoFront import LimitedParetoFront
        # hof = LimitedParetoFront(args.hall_of_fame_size if hasattr(args, 'hall_of_fame_size') else 20)
        hof = tools.ParetoFront()
    else:
        hof = tools.HallOfFame(args.hall_of_fame_size if hasattr(args, 'hall_of_fame_size') else 20)
    
    # 创建统计对象
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    print("开始进化...")
    start_time = time.time()
    
    # 创建初始种群,创建5倍于种群大小的初始种群并择优
    if args.use_warm_start:
        pop = toolbox.population(n=args.population_size * 5)
    else:
        pop = toolbox.population(n=args.population_size)

    # 确定是否添加风格因子到评估中
    barra_values = None
    weights = None
    if args.use_barra:
        barra_values = data_dict['barra_train']
        weights = data_dict['weights_train']

    # 运行遗传规划算法,返回最终名人堂个体的因子值和日志信息
    log, gen = eaMuPlusLambdaWithEarlyStopping(
        pop, 
        toolbox, 
        mu=int(args.population_size * 0.5),  # 亲代数量,从上一轮的父代和子代中各选mu个组成下一轮的亲代
        lambda_=int(args.population_size * 1),  # 子代数量
        cxpb=args.crossover_rate, 
        mutpb=args.mutation_rate, 
        ngen=args.max_generations, 
        history=history,
        stats=stats, 
        halloffame=hof, 
        verbose=args.verbose,
        use_warm_start=args.use_warm_start,
        correlation_threshold=args.correlation_threshold if hasattr(args, 'correlation_threshold') else 0.6,
        correlation_threshold_init=args.correlation_threshold_init if hasattr(args, 'correlation_threshold_init') else 0.4,
        iccorr=args.iccorr,
        dynamicProb=args.dynamicProb,
        patience=args.patience if hasattr(args, 'patience') else 5,
        min_delta=args.min_delta if hasattr(args, 'min_delta') else 0.001,
        pset=pset,  # 直接传递原始集
        feature_data=data_dict['train_feature_data'],  # 直接传递特征数据
        use_gpu=args.use_gpu if hasattr(args, 'use_gpu') else False,
        n_jobs=args.n_jobs if hasattr(args, 'n_jobs') else 1,
        global_pool=global_pool,
        experiment_name=args.experiment_name if hasattr(args, 'experiment_name') else 'test',
        output_dir=args.output_dir if hasattr(args, 'output_dir') else 'results',
        barra_values=barra_values,
        barra_usage=args.barra_usage,
        weights=weights,
        competition=args.competition
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"遗传规划完成，耗时: {elapsed_time:.2f}秒")
    
    # 评估所有名人堂个体在测试集上的表现
    fitness_module = device_info['fitness_module']
    hof_test_metrics = []
    
    print("评估所有名人堂个体在测试集上的表现...")
    
    # 准备GPU测试数据（如果使用GPU）
    # if device_info['device'] == 'gpu' and TORCH_AVAILABLE:
    #     from .gpu.operators import convert_to_torch_tensors
    #     torch_test_feature_data = convert_to_torch_tensors(data_dict['test_feature_data'])

    if global_pool is not None and args.n_jobs > 1:
        hof_factor_values = global_pool.starmap(base_calculate, [(device_info['device'], pset, data_dict['test_feature_data'], individual) for individual in hof])
        test_ic_arrays = global_pool.starmap(fitness_module.calculate_ic, [(hof_factor_value, data_dict['y_test']) for hof_factor_value in hof_factor_values])
    else:
        warnings.warn("未设置多进程，使用单进程回填并评估hof在测试集的表现")
        from itertools import starmap
        hof_factor_values = list(starmap(base_calculate, [[device_info['device'], pset, data_dict['test_feature_data'], individual] for individual in hof]))
        test_ic_arrays = list(starmap(fitness_module.calculate_ic, [[hof_factor_value, data_dict['y_test']] for hof_factor_value in hof_factor_values]))

    test_ic = [np.abs(np.mean(ic_array)) for ic_array in test_ic_arrays]
    
    # 关闭进程池
    if global_pool is not None:
        global_pool.close()
        global_pool.join()
        del global_pool
    if hasattr(toolbox, 'pool'):
        toolbox.pool.close()
        toolbox.pool.join()
        del toolbox.pool

    # 打印最佳个体的测试结果
    print(f"最佳表达式: {str(hof[0])}")
    print(f"最佳表达式 (原始特征名): {get_expression_with_original_names(str(hof[0]), data_dict['feature_dict'])}")
    print(f"训练集适应度: {hof[0].fitness.values[0]:.4f}")
    print(f"最佳个体测试集IC: {test_ic[0]:.4f}")
    
    # 保存结果
    save_results(args, pop, hof, hof_factor_values, log, pset, data_dict, {
        'best_test_ic': test_ic[0],
        'hof_test_ic': test_ic
    }, elapsed_time, gen, device_info)
    
    return hof, log

def get_expression_with_original_names(expression_str, feature_mapping):
    """
    将表达式中的重命名特征替换为原始特征名
    
    参数:
        expression_str: 表达式字符串，包含重命名的特征（如x1, x2等）
        feature_mapping: 特征映射字典，键为重命名特征，值为原始特征名
        
    返回:
        替换后的表达式字符串
    """
    # 创建一个排序后的特征名列表，确保先替换较长的名称（如x10应该在x1之前替换）
    renamed_features = sorted(feature_mapping.keys(), key=len, reverse=True)
    
    # 替换表达式中的重命名特征
    result = expression_str
    for renamed in renamed_features:
        original = feature_mapping[renamed]
        # 使用正则表达式确保只替换完整的特征名（避免x1替换x10中的一部分）
        import re
        result = re.sub(r'\b' + renamed + r'\b', original, result)
    
    return result

def save_results(args, pop, hof, hof_values, log, pset, data_dict, test_metrics, elapsed_time, gen, device_info):
    """
    保存结果
    
    参数:
        args: 命令行参数
        pop: 种群
        hof: 名人堂
        hof_values: 名人堂个体在全数据集上的因子值
        log: 日志
        pset: 原始集
        data_dict: 数据字典
        test_metrics: 测试指标
        elapsed_time: 运行时间
        gen: 进化代数
        device_info: 设备信息
    """
    output_dir = args.output_dir if hasattr(args, 'output_dir') else 'results'
    experiment_name = args.experiment_name if hasattr(args, 'experiment_name') else 'test'
    
    experiment_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # 获取模块引用
    operators_module = device_info['operators_module']
    
    # 保存最佳个体
    best_individual = hof[0]
    best_expression = str(best_individual)
    best_expression_original = get_expression_with_original_names(best_expression, data_dict['feature_dict'])
    best_fitness = best_individual.fitness.values[0]
    
    args_dict = vars(args).copy()
    if 'data' in args_dict:
        del args_dict['data']
    if 'barra_data' in args_dict:
        del args_dict['barra_data']
    if 'weights_data' in args_dict:
        del args_dict['weights_data']

    result = {
        'expression': best_expression,
        'expression_original': best_expression_original,
        'train_fitness': best_fitness,
        'test_ic': test_metrics['best_test_ic'],
        'elapsed_time': elapsed_time,
        'generation': gen,
        'parameters': args_dict,
        'feature_mapping': data_dict['feature_dict'],
        'device': device_info['device']
    }

    with open(os.path.join(experiment_dir, "best.json"), 'w') as f:
        json.dump(result, f, indent=2)
    
    # 保存所有名人堂个体
    hof_results = []
    for i, ind in enumerate(hof):
        expression = str(ind)
        expression_original = get_expression_with_original_names(expression, data_dict['feature_dict'])
        
        # 获取该个体在测试集上的表现
        test_ic_for_ind = test_metrics['hof_test_ic'][i]
        
        hof_results.append({
            'rank': i + 1,
            'expression': expression,
            'expression_original': expression_original,
            'fitness': ind.fitness.values[0],
            'test_ic': test_ic_for_ind,
        })
    
    with open(os.path.join(experiment_dir, "hof.json"), 'w') as f:
        json.dump(hof_results, f, indent=2)
    
    # 保存所有名人堂个体的测试集因子值
    icodetoscode = pd.read_csv("/data/home/jiamuxie/test/gp_proj_restructured/secucode_innercode.csv",usecols=['TradingDay','InnerCode','SecuCode'],dtype={'SecuCode':str})
    for i, ind_values in enumerate(hof_values):
        time_index = data_dict['dataset_index_column'][0]
        code_columns = data_dict['dataset_index_column'][1]
        factor_value_df = pd.DataFrame(data=ind_values, index=time_index, columns=code_columns)
        factor_value_df = factor_value_df.reset_index().melt(id_vars=time_index.name, var_name=code_columns.name, value_name='factor')
        factor_value_df = pd.merge(factor_value_df, icodetoscode, on=['TradingDay','InnerCode'])[['TradingDay','SecuCode','factor']]
        factor_value_df.to_csv(os.path.join(experiment_dir, f"hof_{i}_{args.random_seed}.csv"),index=False)

    # 保存日志
    log_dict = {
        'gen': log.select('gen'),
        'avg': log.select('avg'),
        'std': log.select('std'),
        'min': log.select('min'),
        'max': log.select('max'),

    }
    
    with open(os.path.join(experiment_dir, "log.json"), 'w') as f:
        json.dump(log_dict, f, indent=2)

    print(f"结果已保存到: {os.path.join(experiment_dir)}")

def main(args):
    """
    主函数
    
    参数:
        args: 命令行参数
        
    返回:
        (hof, log): 名人堂、日志
    """
    # 检查优化选项
    if args.use_gpu:
        if not TORCH_AVAILABLE:
            warnings.warn("请求使用 GPU 加速，但 PyTorch 不可用，将使用 CPU")
            args.use_gpu = False
    
    # 准备数据
    data_dict, device_info = prepare_data(args)
    
    # 运行遗传规划算法
    return run_gp(args, data_dict, device_info)
