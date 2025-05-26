"""
CPU适应度评估模块

"""

import numpy as np
import pandas as pd
from typing import Dict, Callable, List, Union, Optional, Any, Tuple
import numba as nb
from scipy import stats
from deap import creator, base, gp
import statsmodels.api as sm

from .operators import calculate_expression

# ====================== Numba 优化的核心计算函数 ======================
@nb.jit(nopython=True)
def _filter_valid_data_numba(x, y):
    """
    过滤有效数据 (Numba 实现)
    
    参数:
        x: 第一个数组
        y: 第二个数组
        
    返回:
        (x_clean, y_clean): 过滤后的数组
    """
    valid_indices = []
    for i in range(len(x)):
        if not (np.isnan(x[i]) or np.isnan(y[i])):
            valid_indices.append(i)
    
    if len(valid_indices) < 2:  # 至少需要两个有效值
        return np.array([0.0]), np.array([0.0])
    
    x_clean = np.array([x[i] for i in valid_indices])
    y_clean = np.array([y[i] for i in valid_indices])

    return x_clean, y_clean

@nb.jit(nopython=True)
def _calculate_rank_numba(x):
    """
    计算排名 (Numba 实现), nan值不参与排序, 结果仍然是nan
    
    参数:
        x: 输入数组
        
    返回:
        排名数组
    """
    n = len(x)
    valid_mask = ~np.isnan(x)
    result = np.full_like(x, np.nan)

    ranks = np.argsort(np.argsort(x[valid_mask]))
    result[valid_mask] = ranks / (np.sum(valid_mask) - 1)

    return result

@nb.jit(nopython=True)
def _calculate_partition_numba(x,quantiles=10):
    """
    计算分组 (Numba 实现), nan值已经去掉了
    
    参数:
        x: 输入array
        
    返回:
        包含每个元素的分组后的组序号的array
    """
    n = len(x)
    step = n // quantiles
    kth = np.arange(step, n, step)[:quantiles-1]
    partition_indices = np.argpartition(x, kth)
    
    partition = np.zeros(n, dtype=np.int32)
    for i, k in enumerate(kth):
        partition[partition_indices[k:]] = i + 1
    
    return partition

@nb.jit(nopython=True)
def _calculate_correlation_numba(x_ranks, y_ranks):
    """
    计算相关系数 (Numba 实现)
    
    参数:
        x_ranks: x 的排名
        y_ranks: y 的排名
        
    返回:
        相关系数
    """
    valid_mask = (~np.isnan(x_ranks)) & (~np.isnan(y_ranks))
    x_ranks = x_ranks[valid_mask]
    y_ranks = y_ranks[valid_mask]
    n = np.sum(valid_mask)
    
    # 计算均值
    mean_x = np.sum(x_ranks) / n
    mean_y = np.sum(y_ranks) / n
    
    # 计算协方差和方差
    numerator = 0.0
    denom_x = 0.0
    denom_y = 0.0
    
    for i in range(n):
        dx = x_ranks[i] - mean_x
        dy = y_ranks[i] - mean_y
        numerator += dx * dy
        denom_x += dx * dx
        denom_y += dy * dy
    
    # 避免除以零
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    
    # 计算相关系数
    correlation = numerator / np.sqrt(denom_x * denom_y)
    
    return correlation

@nb.jit(nopython=True)
def _calculate_ic_numba(factor_values, returns):
    """
    计算一日的IC (Numba 实现)
    
    参数:
        factor_values: 因子值数组
        returns: 收益率数组
        
    返回:
        IC值
    """
    # 过滤有效数据
    factor_clean, returns_clean = _filter_valid_data_numba(factor_values, returns)
    
    if len(factor_clean) < 2:  # 至少需要两个有效值
        return 0.0
    
    # 计算排名
    factor_ranks = _calculate_rank_numba(factor_clean)
    returns_ranks = _calculate_rank_numba(returns_clean)
    
    # 计算相关系数
    ic = _calculate_correlation_numba(factor_ranks, returns_ranks)
    
    return ic

@nb.jit(nopython=True)
def _calculate_ndcg_numba(factor_values, returns, quantiles=10):
    """
    计算一日的ndcg (Numba 实现)
    
    参数:
        factor_values: 因子值数组
        returns: 收益率数组
        quantiles: 分位数数量
        
    返回:
        NDCG@k值, k=quantiles/2
    """
    # 过滤有效数据
    factor_clean, returns_clean = _filter_valid_data_numba(factor_values, returns)
    
    if len(factor_clean) < quantiles:  # 确保有足够的样本进行分组
        return 0.0
    # 计算分组,结果类似于[0,1,1,2,1],表示前五个值分别被分到了第0,1,1,2,1组
    factor_partitions = _calculate_partition_numba(factor_clean, quantiles=quantiles)
    # 获取每组的平均收益的排名, 结果类似于[0,2,1,3], 表示按照因子值升序排出来的各组的平均收益率的排名
    group_returns = np.zeros(quantiles,dtype=np.int32)
    for i in range(quantiles):
        ith_group_mask = (factor_partitions == i)
        group_returns[i] = np.mean(returns[ith_group_mask])
    group_returns_rank = np.argsort(np.argsort(group_returns))

    # 因为不确定因子是正向还是负向, 所以计算DCG@K的时候，对(因子值分组的)前k组和后k组都计算一遍, 并且把后k组的组排名倒置(比如10分组的后5组,排名从10,9,8,7,6改成1,2,3,4,5),取其中较大的
    k = quantiles//2
    indices = np.arange(k)
    dcg_k_asc_values = (2**group_returns_rank[indices] - 1) / np.log2(indices + 2)
    dcg_k_asc = np.sum(dcg_k_asc_values)
    dcg_k_dsc_values = (2**group_returns_rank[quantiles - indices] - 1) / np.log2(indices + 2)
    dcg_k_dsc = np.sum(dcg_k_dsc_values)
    dcg_k = max(dcg_k_asc, dcg_k_dsc)

    idcg_values = (2**(quantiles - indices - 1)) / np.log2(indices + 2)
    idcg = np.sum(idcg_values)
    ndcg_k = dcg_k/idcg
    
    return ndcg_k

# ====================== 适应度基础指标计算函数 ======================
def format_conv(factor_values, returns):
    
    # 转换pandas Series为numpy数组
    if isinstance(factor_values, pd.Series):
        factor_array = factor_values.values
    else:
        factor_array = np.asarray(factor_values)
    
    if isinstance(returns, pd.Series):
        returns_array = returns.values
    else:
        returns_array = np.asarray(returns)

    return factor_array, returns_array

def calculate_ic(factor_values, returns):
    """
    计算raw ic数据,是一个长度与天数相等的array
    
    参数:
        factor_value: 因子值数组
        returns: 收益率数组
        
    返回:
        由因子每日ic值组成的array
    """
    
    factor_values, returns = format_conv(factor_values, returns)

    ics = []
    for t_factor,t_returns in zip(factor_values,returns):
        ic = _calculate_ic_numba(t_factor, t_returns)
        if not np.isnan(ic):
            ics.append(ic)
    
    if len(ics) == 0:
        return np.array([0.0])
    
    return np.array(ics)

def calculate_NDCG(factor_values, returns, quantiles=10):
    """
    计算raw NDCG@k数据,是一个长度与天数相等的array,k=int(quantiles/2)
    
    参数:
        factor_values: 因子值数组
        returns: 收益率数组
        quantiles: 按照因子将股票分成的组数
        
    返回:
        由因子每日NDCG@k值组成的array
    """
    factor_values, returns = format_conv(factor_values, returns)

    ndcgs = []
    for t_factor,t_returns in zip(factor_values,returns):
        ndcg = _calculate_ndcg_numba(t_factor, t_returns, quantiles)
        if not np.isnan(ndcg):
            ndcgs.append(ndcg)
    
    if len(ndcgs) == 0:
        return np.array([0.0])
    
    return np.array(ndcgs)


# ====================== 风格因子计算 ======================

def calculate_barra_correlation(factor_values, barra_values=None):
    """
    基于 IC 适应度评估方法进行的因子值与风格因子（秩）相关性计算, 这个函数总会被调用, 但是如果入口设置里没有要求对风格因子算相关性, barra_values会是None
    
    参数:
        factor_values: 单个因子值
        barra_values: 风格因子ndarray组成的dict或者None
        
    返回:
        秩相关性
    """
    if barra_values is not None:
        corr_dict = {}
        for barra_key, barra_value in barra_values.items():
            #barra_values是dict
            # factor_values, barra_value = format_conv(factor_values, barra_value)
            # 计算每个时间点的 IC,然后取平均
            corrs = calculate_group_ic(factor_values, barra_value)
            corr_dict[barra_key] = np.mean(corrs)
        return corr_dict

factors_neut = [
    # "lncap",
    # "beta",
    "rv",
    # "nls",
    # "bp",
    # "ey",
    "liquidity"#,
    # "leverage",
    # "growth",
    # "momentum",
    # "short_rev",
    # "煤炭",
    # "石油石化",
    # "传媒",
    # "电力设备",
    # "电子",
    # "房地产",
    # "纺织服饰",
    # "非银金融",
    # "钢铁",
    # "公用事业",
    # "国防军工",
    # "环保",
    # "机械设备",
    # "基础化工",
    # "计算机",
    # "家用电器",
    # "建筑材料",
    # "建筑装饰",
    # "交通运输",
    # "美容护理",
    # "农林牧渔",
    # "汽车",
    # "轻工制造",
    # "商贸零售",
    # "社会服务",
    # "食品饮料",
    # "通信",
    # "医药生物",
    # "银行",
    # "有色金属",
    # "综合",
]

def neutralize_matrix(factor, barra1, barra2, weights=None, add_constant=True, standardize_after_neut=True):
    """
    对矩阵的每一行(时间点)分别进行回归中性化
    
    参数:
    - factor: 形状为(T, N)的原始矩阵，T是时间点数，N是股票数
    - barra1: 形状为(T, N)的第一个因子矩阵
    - barra2: 形状为(T, N)的第二个因子矩阵
    - weights: 可选的权重矩阵，形状为(T, N)
    - add_constant: 是否添加常数项
    - standardize_after_neut: 是否对残差进行zscore标准化
    
    返回:
    - 形状为(T, N)的残差矩阵
    """
    T, N = factor.shape
    residuals = np.full_like(factor, np.nan)
    
    # 对每个时间点分别进行回归
    for t in range(T):
        # 提取当前时间点的数据
        y = factor[t, :]  # 当前时间点的收益率(N,)
        X = np.column_stack([barra1[t, :], barra2[t, :]])  # 因子数据(N, 2)
        
        # 添加常数项
        if add_constant:
            X = sm.add_constant(X)
        
        # 处理缺失值
        mask = ~(np.isnan(y) | np.isnan(X).any(axis=1))
        if np.sum(mask) <= X.shape[1]:  # 确保有足够的数据点进行回归
            continue
            
        y_valid = y[mask]
        X_valid = X[mask, :]
        
        # 处理权重
        if weights is not None:
            w_valid = weights[t, :][mask]
            # 使用加权最小二乘回归
            model = sm.WLS(y_valid, X_valid, weights=w_valid).fit()
        else:
            # 使用普通最小二乘回归
            model = sm.OLS(y_valid, X_valid).fit()
        
        # 为所有股票计算预测值和残差(包括有缺失值的股票)
        X_full = X.copy()
        if np.isnan(X_full).any():
            X_full = np.nan_to_num(X_full)
        
        y_pred = model.predict(X_full)
        residuals[t, :] = y - y_pred
        
        # 对于原始数据中的NaN，残差也应为NaN
        residuals[t, ~mask] = np.nan
    
    if standardize_after_neut:
        standardized_residuals = np.zeros_like(residuals)
        for t in range(T):
            valid_mask = ~np.isnan(residuals[t, :])
            if np.sum(valid_mask) > 1:  # 确保有足够的数据点计算均值和标准差
                residuals_t = residuals[t, valid_mask]
                standardized_residuals[t, valid_mask] = (residuals_t - residuals_t.mean()) / residuals_t.std()
        residuals = standardized_residuals
    return residuals

# ====================== DEAP 适应度设置 ======================

def setup_deap_fitness(maximize=True):
    """
    设置DEAP适应度类
    
    参数:
        maximize: 是否最大化适应度
    """
    # 创建适应度类
    if maximize:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    else:
        creator.create("FitnessDouble", base.Fitness, weights=(1.0,1.0))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessDouble)

# ====================== DEAP 适应度计算 ======================

def evaluate_individual(factor_values, returns, metric, quantiles=10, barra_values=None, barra_usage='correlation', weights=None):
    """
    评估个体适应度
    
    参数:
        factor_values: 因子值ndarray
        returns: 收益率ndarray
        metric: 适应度指标
        quantiles: 计算NDCG时分组数量
        barra_values: 风格因子ndarray构成的字典
        barra_usage: 风格因子的用法，计算correlation或者用wls neutralize
        weights: WLS用到的权重
    返回:
        适应度值元组 (DEAP要求返回元组以支持多目标优化)
    """
    # 检查因子值是否为标量
    if np.isscalar(factor_values):
        # 对于常数因子,适应度为0
        return (0.0,)

    from scipy.stats.mstats import winsorize
    factor_values_flattened = factor_values.flatten()
    factor_values_winsorized = winsorize(factor_values_flattened, limits=(0.003,0.003))
    factor_values = factor_values_winsorized.reshape(factor_values.shape)

    punish = False
    if barra_values is not None:
        if barra_usage == 'neutralize':
            # 对因子进行风格中性化
            factor_values = neutralize_matrix(factor_values,barra_values['rv'],barra_values['liquidity'],weights=weights)

        else:
        # 计算适应度
            barra_correlations = list(calculate_barra_correlation(factor_values, barra_values).values())
            # barra_correlations是当前因子和所有风格因子计算出的相关性的dict，或者是None
            for corr in barra_correlations:
                if corr>0.4:
                    punish = True
                    break

    ic_array = calculate_ic(factor_values, returns)

    if metric == 'ic':
        ic_mean = np.abs(np.mean(ic_array))
        if punish:
            fitness = (ic_mean - np.sum(np.abs(barra_correlations))/30, )
        else:
            fitness = (ic_mean, )
    
    elif metric == 'icir':
        ic_mean = np.abs(np.mean(ic_array))
        ic_std = np.std(ic_array)
        icir = abs(ic_mean) / ic_std
        if ic_std == 0 or np.isnan(icir):
            fitness = (0.0,)
        else:
            fitness = (icir,)
    
    else:
        ndcg_array = calculate_NDCG(factor_values, returns)
        ndcg_mean = np.mean(ndcg_array)
        if metric == 'NDCG':
            fitness = (ndcg_mean,)
        elif metric == 'double':
            ic_mean = np.abs(np.mean(ic_array))
            if punish:
                fitness = (ic_mean - np.sum(np.abs(barra_correlations))/30, ndcg_mean)
            else:
                fitness = (ic_mean, ndcg_mean)
        else:
            print(f"不受支持的metric: {metric}")

    return fitness, ic_array