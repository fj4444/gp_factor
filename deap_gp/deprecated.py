"""
开发过程中弃置的函数，留存供参考
"""
def filter_correlated_individuals(population, factor_values, halloffame, pset, feature_data, correlation_threshold_hof=0.7, correlation_threshold_pop=0.85, use_gpu=False, global_pool=None, n_jobs=1):
    """
    过滤掉与名人堂中个体高度相关的个体
    
    参数:
        population: 候选个体群体
        halloffame: 名人堂对象
        pset: 原始集
        feature_data: 特征数据
        correlation_threshold: 相关性阈值,超过此值的因子将被过滤
        use_gpu: 是否使用GPU
        global_pool: 全局进程池（可选）
        n_jobs: 进程数
        
    返回:
        过滤后的个体群体
    """
    # 导入tqdm
    from tqdm.auto import tqdm
    
    print(f"本次过滤前,population size:{len(population)},hof size:{len(halloffame)}")
    # 创建所有需要计算的种群内个体对
    correlation_tasks = []
    
    # 种群内部相关性任务
    for i in range(len(population)):
        for j in range(i+1, len(population)):
            # correlation_tasks.append((population[i], population[j], pset, feature_data, use_gpu))
            correlation_results.append(calculate_factor_correlation(population[i], population[j], pset, feature_data, use_gpu))

    # 与名人堂的相关性任务
    if halloffame is not None and len(halloffame) != 0:
        for i, ind in tqdm(enumerate(population)):
            for h, hof_ind in enumerate(halloffame):
                correlation_tasks.append((ind, hof_ind, pset, feature_data, use_gpu))
    
    if global_pool is not None and n_jobs > 1:
        # 使用全局进程池
        print(f"使用全局进程池计算相似度,任务数: {len(correlation_tasks)}")
        correlation_results = global_pool.starmap(calculate_factor_correlation, correlation_tasks)
    elif n_jobs > 1:
        # 创建新的进程池
        import multiprocessing
        # from tqdm.contrib.concurrent import process_map
        
        print(f"创建新进程池计算相似度,进程数: {n_jobs},任务数: {len(correlation_tasks)}")
        local_pool = multiprocessing.Pool(processes=args.n_jobs)
        correlation_results = local_pool.starmap(calculate_factor_correlation, correlation_tasks)
        
    else:
        # 单线程计算
        print(f"单线程计算相似度,任务数: {len(correlation_tasks)}")
        correlation_results = [
            calculate_factor_correlation(*task) 
            for task in tqdm(correlation_tasks, desc="计算因子相关性")
        ]
    
    # 处理结果
    # 首先处理population内部的相关性
    print("处理相关性结果...")

    n_pop = len(population)
    pop_correlation_matrix = np.zeros((n_pop, n_pop))
    task_idx = 0
    
    for i in range(n_pop):
        for j in range(i+1, n_pop):
            pop_correlation_matrix[i, j] = correlation_results[task_idx]
            pop_correlation_matrix[j, i] = pop_correlation_matrix[i, j]  # 矩阵是对称的
            task_idx += 1

    # 基于相关性矩阵过滤种群
    # 按适应度排序,优先保留适应度高的个体
    sorted_indices = sorted(range(n_pop), 
                           key=lambda i: population[i].fitness.values[0], 
                           reverse=True)
    
    # 创建一个掩码,标记保留个体
    keep_mask = [True] * n_pop
    
    print("过滤高相关性个体...")
    for i in range(len(sorted_indices)):
        idx_i = sorted_indices[i]
        if not keep_mask[idx_i]:
            continue  # 如果这个个体已经被标记为删除,跳过
        
        # 检查这个个体与其他个体的相关性
        for j in range(i+1, len(sorted_indices)):
            idx_j = sorted_indices[j]
            if not keep_mask[idx_j]:
                continue  # 如果这个个体已经被标记为删除,跳过
            
            # 直接使用排序后的索引检查相关性
            if abs(pop_correlation_matrix[idx_i, idx_j]) > correlation_threshold_pop:
                keep_mask[idx_j] = False  # 标记为删除
    
    # 应用掩码,只保留标记为保留的个体
    filtered_population = [ind for i, ind in enumerate(population) if keep_mask[i]]
    print(f"种群内过滤后剩余: {len(filtered_population)}/{len(population)} 个体")
    
    # 然后处理与hof的相关性
    if halloffame is not None and len(halloffame) > 0:
        final_filtered_population = []
        
        for ind in filtered_population:
            is_correlated = False
            for _ in halloffame:
                correlation = correlation_results[task_idx]
                task_idx += 1
                if abs(correlation) > correlation_threshold_hof:
                    is_correlated = True
                    break
            if not is_correlated:
                final_filtered_population.append(ind)
        
        print(f"与名人堂过滤后剩余: {len(final_filtered_population)}/{len(filtered_population)} 个体")
    else:
        final_filtered_population = filtered_population
        
    # 如果过滤后没有个体,则保留原始过滤后种群中适应度最高的个体
    if len(final_filtered_population) == 0:
        sorted_population = sorted(filtered_population, key=lambda ind: ind.fitness.values[0], reverse=True)
        final_filtered_population = [sorted_population[0]] if sorted_population else []
        print("警告: 过滤后没有个体,保留最高适应度个体")
    
    return final_filtered_population

def calculate_factor_correlation(ind1, ind2, pset, feature_data, use_gpu):
    """
    计算两个个体产生的因子之间的相关性
    
    参数:
        ind1: 第一个个体
        ind2: 第二个个体
        pset: 原始集
        feature_data: 特征数据
        
    返回:
        相关系数
    """
    from .cpu.operators import calculate_expression
    
    # 计算两个个体的因子值
    factor1 = calculate_expression(ind1, pset, feature_data)
    factor2 = calculate_expression(ind2, pset, feature_data)
    
    # 处理标量情况
    if np.isscalar(factor1) or np.isscalar(factor2):
        return 0.0
            
    # 确保输入是numpy数组
    if isinstance(factor1, pd.Series):
        # 将Series转换为2D矩阵 (时间 x 标的)
        factor1_matrix = factor1.unstack(level='InnerCode').values
    else:
        factor1_matrix = factor1
        
    if isinstance(factor2, pd.Series):
        factor2_matrix = factor2.unstack(level='InnerCode').values
    else:
        factor2_matrix = factor2
    
    # 用0填充NaN
    factor1_matrix = np.nan_to_num(factor1_matrix, nan=0.0)
    factor2_matrix = np.nan_to_num(factor2_matrix, nan=0.0)

    # 如果数据太少,返回0
    if factor1_matrix.shape[0] < 5 or factor1_matrix.shape[1] < 5:
        return 0.0
    if factor2_matrix.shape[0] < 5 or factor2_matrix.shape[1] < 5:
        return 0.0
    
    # TODO:对rapids实现非pca版本
    if RAPIDS_AVAILABLE and use_gpu:
        pass

    else:
        try:
            # return np.mean([np.corrcoef(factor1_matrix[i],factor2_matrix[i])[0,1] for i in range(0, len(factor1_matrix), 21)])
            return np.mean([np.corrcoef(factor1_matrix[i],factor2_matrix[i])[0,1] for i in range(len(factor1_matrix)-250, len(factor1_matrix))])
        except Exception as e:
            print(f"计算相关性时出错: {e}")
            return 0.0

def get_adjust_price(df_data, df_adj):
    """
    对价格和成交额数据进行复权处理
    
    参数:
        df_data: 包含价格和成交额数据的DataFrame,按股票代码分组
        df_adj: 复权因子DataFrame
    
    返回:
        复权后的DataFrame
    """
    # 获取当前股票的复权因子
    df_adj_spec = df_adj.loc[df_adj.SecuCode == df_data['SecuCode'].iloc[0]].sort_values(by='ExDiviDate', ascending=False)
    if df_adj_spec.shape[0] == 0:
        return df_data

    # 标记是否已复权
    df_data['IsReadjusted'] = False
    
    # 需要复权的字段
    cols2adj = ['OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice', 'PrevClosePrice']
    
    # 对每个除权除息日期,将该日期之前的数据乘以对应的复权因子
    for d, f in df_adj_spec[['ExDiviDate', 'AdjustingFactor']].values:
        mask = (df_data['TradingDay'] >= d) & ~df_data.IsReadjusted
        df_data.loc[mask, cols2adj] = df_data.loc[mask, cols2adj] * f
        df_data.loc[mask, 'IsReadjusted'] = True
    
    # 删除临时标记列
    del df_data['IsReadjusted']
    return df_data

def factor_pca(factor_value, use_gpu):
    """
    计算两个个体产生的因子之间PCA降维之后的相关性
    
    参数:
        ind: 个体
        pset: 原始集
        feature_data: 特征数据
        
    返回:
        相关系数
    """
    # 处理标量情况
    if np.isscalar(factor_value):
        return 0.0
            
    # 确保输入是numpy数组
    if isinstance(factor_value, pd.Series):
        # 将Series转换为2D矩阵 (时间 x 标的)
        factor_value = factor_value.unstack(level='InnerCode').values

    # 处理缺失值
    factor_value = np.nan_to_num(factor_value, nan=0.0)

    # 如果数据太少,返回0
    if factor_value.shape[0] < 5 or factor_value.shape[1] < 5:
        return 0.0
    
    # 根据是否可以使用RAPIDS选择PCA实现
    if RAPIDS_AVAILABLE and use_gpu:
        try:
            # 使用cuML实现GPU加速的PCA
            from cuml.decomposition import PCA as cuPCA
            import cudf
            
            # 转换为cuDF DataFrame
            factor_values = factor_matrix.values
            
            # 创建cuPCA对象
            cu_pca = cuPCA(n_components=1)
            
            # 对矩阵进行PCA降维
            factor_pca = cu_pca.fit_transform(factor_values)
            
            # 转换回NumPy数组并计算相关系数
            return factor_pca
        except Exception as e:
            print(f"使用cuML进行PCA时出错: {e}")

    else:
        try:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=1)

            factor_pca = pca.fit_transform(factor_value)

            return factor_pca
        except Exception as e:
            print(f"计算PCA时出错: {e}")
            return 0.0

def filter_correlated_individuals_pca(population, factor_values, halloffame, pset, feature_data, correlation_threshold_hof=0.7, correlation_threshold_pop=0.85, use_gpu=False, global_pool=None, n_jobs=1):
    """
    过滤掉与名人堂中个体高度相关的个体
    
    参数:
        population: 候选个体群体
        factor_values: population中因子的值
        halloffame: 名人堂对象
        pset: 原始集
        feature_data: 特征数据
        correlation_threshold: 相关性阈值,超过此值的因子将被过滤
        use_gpu: 是否使用GPU
        global_pool: 全局进程池（可选）
        n_jobs: 进程数
        
    返回:
        过滤后的个体群体
    """
    # 计算hof和pop全体的pc
    print(f"本次过滤前,population size:{len(population)},hof size:{len(halloffame)}")
    pca_tasks = []
    hof_full = False
    for factor_value in factor_values:
        pca_tasks.append((factor_value, use_gpu))
    if halloffame is not None and len(halloffame) != 0:
        hof_full = True
        if not use_gpu:
            from .cpu.operators import calculate_expression
        else:
            pass
        for hof_ind in halloffame:
            hof_factor_value = calculate_expression(hof_ind, pset, feature_data)
            pca_tasks.append((hof_factor_value, use_gpu))

    if global_pool is not None and n_jobs > 1:
        # 使用全局进程池
        print(f"使用全局进程池计算PCA,任务数: {len(pca_tasks)}")
        pca_results = global_pool.starmap(factor_pca, pca_tasks)
    elif n_jobs > 1:
        # 创建新的进程池
        import multiprocessing
        print(f"创建新进程池计算PCA,进程数: {n_jobs},任务数: {len(pca_tasks)}")
        with multiprocessing.Pool(processes=n_jobs) as pool:
            pca_results = pool.starmap(factor_pca, pca_tasks)
    else:
        # 单线程计算
        print(f"单线程计算PCA,任务数: {len(pca_tasks)}")
        pca_results = [factor_pca(*task) for task in pca_tasks]
    
    
    n_pop = len(population)
    pop_correlation_matrix = np.zeros((n_pop, n_pop))
    # 计算相关性
    for i in range(n_pop):
        for j in range(i+1, n_pop):
            # 计算主成分的相关性,应该不必多进程了,主成分也就是1000*1左右的向量
            # 注意pca之后的主成分的返回值总是二维的,即使只保留一个主成分,结果也会是(日期,1),需要ravel变成向量
            pop_correlation_matrix[i][j] = np.corrcoef(pca_results[i].ravel(),pca_results[j].ravel())[0,1]
            pop_correlation_matrix[j][i] = pop_correlation_matrix[i][j]

    # 基于相关性矩阵过滤种群
    # 按适应度排序,优先保留适应度高的个体
    sorted_indices = sorted(range(n_pop), 
                           key=lambda i: population[i].fitness.values[0], 
                           reverse=True)
    
    # 创建一个掩码,标记哪些个体应该被保留
    keep_mask = [True] * n_pop
    
    # 遍历排序后的个体
    for i in range(len(sorted_indices)):
        idx_i = sorted_indices[i]
        if not keep_mask[idx_i]:
            continue  # 如果这个个体已经被标记为删除,跳过
        
        # 检查这个个体与其他个体的相关性
        for j in range(i+1, len(sorted_indices)):
            idx_j = sorted_indices[j]
            if not keep_mask[idx_j]:
                continue  # 如果这个个体已经被标记为删除,跳过
            
            if abs(pop_correlation_matrix[idx_i, idx_j]) > correlation_threshold_pop:
                keep_mask[idx_j] = False  # 标记为删除
    
    # 应用掩码,只保留标记为保留的个体
    # 保留的个体：
    filtered_population = [ind for i, ind in enumerate(population) if keep_mask[i]]
    # 保留的个体的主成分
    filtered_pop_pc = [ind for i, ind in enumerate(pca_results[:n_pop]) if keep_mask[i]]
    print(f"种群内过滤后剩余: {len(filtered_population)}/{len(population)} 个体")
    
    # 然后处理与hof的相关性
    if hof_full:
        final_filtered_population = []
        
        for i in range(len(filtered_population)):
            is_correlated = False
            for j in range(len(halloffame)):
                correlation = np.corrcoef(filtered_pop_pc[i].ravel(), pca_results[n_pop+j].ravel())[0,1]
                if abs(correlation) > correlation_threshold_hof:
                    is_correlated = True
                    break
            if not is_correlated:
                final_filtered_population.append(filtered_population[i])
        
        print(f"与名人堂过滤后剩余: {len(final_filtered_population)}/{len(filtered_population)} 个体")
    else:
        final_filtered_population = filtered_population
        
    # 如果过滤后没有个体,则保留原始过滤后种群中适应度最高的个体
    if len(final_filtered_population) == 0:
        sorted_population = sorted(filtered_population, key=lambda ind: ind.fitness.values[0], reverse=True)
        final_filtered_population = [sorted_population[0]] if sorted_population else []
        print("警告: 和hof过滤后没有个体,保留种群内最高适应度个体")
    
    print(f"最终保留: {len(final_filtered_population)} 个体")
    
    return final_filtered_population


# icodetoscode.py

import os
import pandas as pd
def get_all(folder_path, keyword=".pkl"):
    target_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(keyword):  
                target_list.append(os.path.join(root,file))
    return target_list


# from tqdm.notebook import tqdm as tqdm
from tqdm import tqdm
def process_map_with_tqdm(func, task, pool, desc=None):
    total = len(task)
    results = list(tqdm(
        pool.imap(func, task),
        total=total,
        desc=desc or "Processing",
        leave=False
    ))
    return results

def codemapping(args):
    csv, save_path, icodetoscode = args
    try:
        hof = pd.read_csv(csv,usecols=['TradingDay','InnerCode','factor'])
        hof = pd.merge(hof, icodetoscode, on=['TradingDay','InnerCode'])[['TradingDay','SecuCode','factor']]
        save_path = csv.replace('results','results_secucode')
        os.makedirs('/'+os.path.join(*(save_path.split('/')[:-1])),exist_ok=True)
        hof.to_csv(save_path,index=False)
    except Exception as e:
        print(f"exception occured:{e}, while processing file:{csv}")

import multiprocessing
pool = multiprocessing.Pool(processes=30)

icodetoscode = pd.read_csv("/data/home/jiamuxie/test/gp_proj_restructured/secucode_innercode.csv",usecols=['TradingDay','InnerCode','SecuCode'],dtype={'SecuCode':str})
work_dir = "/data/home/jiamuxie/test/gp_proj_restructured/results/selected_weird_perform"
csv_list = get_all(work_dir, ".csv")
task_list = []
for csv in csv_list:
    save_path = csv.replace('results','results_secucode')
    if os.path.exists(save_path):
        continue
    task_list.append((csv,save_path,icodetoscode))

process_map_with_tqdm(codemapping,task_list,pool)
