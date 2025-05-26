"""
GPU适应度评估模块

包含:
- 使用PyTorch实现的IC计算
- 使用PyTorch实现的ICIR计算
- 使用PyTorch实现的收益率计算
- 使用PyTorch实现的夏普比率计算
"""

import numpy as np
import torch
from typing import Dict, Callable, List, Union, Optional, Any, Tuple
from deap import creator, base, gp

from .operators import calculate_expression, DEVICE, convert_to_torch_tensors

# ====================== PyTorch 实现的核心计算函数 ======================

def _filter_valid_data_torch(x, y):
    """
    过滤有效数据 (PyTorch 实现)
    
    参数:
        x: 第一个张量
        y: 第二个张量
        
    返回:
        (x_clean, y_clean): 过滤后的张量
    """
    # 创建掩码,标记非NaN的位置
    mask = ~(torch.isnan(x) | torch.isnan(y))
    
    if torch.sum(mask) < 2:  # 至少需要两个有效值
        return torch.tensor([0.0], device=x.device), torch.tensor([0.0], device=y.device)
    
    # 过滤数据
    x_clean = x[mask]
    y_clean = y[mask]
    
    return x_clean, y_clean

def _calculate_rank_torch(x):
    """
    计算排名 (PyTorch 实现)
    
    参数:
        x: 输入张量
        
    返回:
        排名张量
    """
    # 获取排序索引
    _, indices = torch.sort(x)
    
    # 创建排名张量
    ranks = torch.zeros_like(x)
    ranks[indices] = torch.arange(len(x), dtype=torch.float32, device=x.device)
    
    return ranks

def _calculate_correlation_torch(x_ranks, y_ranks):
    """
    计算相关系数 (PyTorch 实现)
    
    参数:
        x_ranks: x 的排名
        y_ranks: y 的排名
        
    返回:
        相关系数
    """
    # 计算均值
    mean_x = torch.mean(x_ranks)
    mean_y = torch.mean(y_ranks)
    
    # 计算协方差和方差
    dx = x_ranks - mean_x
    dy = y_ranks - mean_y
    
    numerator = torch.sum(dx * dy)
    denom_x = torch.sum(dx * dx)
    denom_y = torch.sum(dy * dy)
    
    # 避免除以零
    if denom_x == 0.0 or denom_y == 0.0:
        return torch.tensor(0.0, device=x_ranks.device)
    
    # 计算相关系数
    correlation = numerator / torch.sqrt(denom_x * denom_y)
    
    return correlation

def _calculate_ic_torch(factor_values, returns):
    """
    计算信息系数 (IC) (PyTorch 实现)
    
    参数:
        factor_values: 因子值张量
        returns: 收益率张量
        
    返回:
        IC值
    """
    # 过滤有效数据
    factor_clean, returns_clean = _filter_valid_data_torch(factor_values, returns)
    
    if len(factor_clean) < 2:  # 至少需要两个有效值
        return torch.tensor(0.0, device=factor_values.device)
    
    # 计算排名
    factor_ranks = _calculate_rank_torch(factor_clean)
    returns_ranks = _calculate_rank_torch(returns_clean)
    
    # 计算相关系数
    ic = _calculate_correlation_torch(factor_ranks, returns_ranks)
    
    return ic

def _calculate_group_ic_torch(factor_values, returns, time_index):
    """
    计算分组 IC (PyTorch 实现)
    
    参数:
        factor_values: 因子值张量
        returns: 收益率张量
        time_index: 时间索引张量
        
    返回:
        IC张量
    """
    # 获取唯一的时间点
    unique_times = torch.unique(time_index)
    ics = []
    
    for t in unique_times:
        # 获取当前时间点的数据
        t_mask = (time_index == t)
        t_factor = factor_values[t_mask]
        t_returns = returns[t_mask]
        
        # 计算当前时间点的 IC
        if len(t_factor) >= 2:  # 至少需要两个有效值
            ic = _calculate_ic_torch(t_factor, t_returns)
            if not torch.isnan(ic):
                ics.append(ic)
    
    if len(ics) == 0:
        return torch.tensor([0.0], device=factor_values.device)
    
    return torch.stack(ics)

def _calculate_quantile_returns_torch(factor_values, returns, quantiles=5):
    """
    计算分位数收益率 (PyTorch 实现)
    
    参数:
        factor_values: 因子值张量
        returns: 收益率张量
        quantiles: 分位数数量
        
    返回:
        (top_returns, bottom_returns): 顶部和底部分位数收益率
    """
    # 过滤有效数据
    factor_clean, returns_clean = _filter_valid_data_torch(factor_values, returns)
    
    if len(factor_clean) < quantiles:  # 确保有足够的样本进行分组
        return torch.tensor([0.0], device=factor_values.device), torch.tensor([0.0], device=factor_values.device)
    
    # 计算排名
    factor_ranks = _calculate_rank_torch(factor_clean)
    
    # 计算分位数边界
    n = len(factor_ranks)
    quantile_size = n // quantiles
    
    if quantile_size == 0:
        return torch.tensor([0.0], device=factor_values.device), torch.tensor([0.0], device=factor_values.device)
    
    # 获取顶部和底部分位数的掩码
    top_mask = factor_ranks < quantile_size
    bottom_mask = factor_ranks >= (n - quantile_size)
    
    # 计算顶部和底部分位数的收益率
    top_returns = returns_clean[top_mask]
    bottom_returns = returns_clean[bottom_mask]
    
    if len(top_returns) == 0 or len(bottom_returns) == 0:
        return torch.tensor([0.0], device=factor_values.device), torch.tensor([0.0], device=factor_values.device)
    
    return top_returns, bottom_returns

def _calculate_group_quantile_returns_torch(factor_values, returns, time_index):
    """
    计算分组分位数收益率 (PyTorch 实现)
    
    参数:
        factor_values: 因子值张量
        returns: 收益率张量
        time_index: 时间索引张量
        quantiles: 分位数数量
        
    返回:
        (top_returns, bottom_returns): 顶部和底部分位数收益率张量
    """
    # 获取唯一的时间点
    unique_times = torch.unique(time_index)
    top_returns_list = []
    bottom_returns_list = []
    
    for t in unique_times:
        # 获取当前时间点的数据
        t_mask = (time_index == t)
        t_factor = factor_values[t_mask]
        t_returns = returns[t_mask]
        
        # 计算当前时间点的分位数收益率
        if len(t_factor) >= 5:  # 确保有足够的样本进行分组
            t_top_returns, t_bottom_returns = _calculate_quantile_returns_torch(t_factor, t_returns, 5)
            
            # 如果有有效的收益率,则添加到列表中
            if len(t_top_returns) > 0 and len(t_bottom_returns) > 0:
                top_returns_list.append(torch.mean(t_top_returns))
                bottom_returns_list.append(torch.mean(t_bottom_returns))
    
    if len(top_returns_list) == 0 or len(bottom_returns_list) == 0:
        return torch.tensor([0.0], device=factor_values.device), torch.tensor([0.0], device=factor_values.device)
    
    return torch.stack(top_returns_list), torch.stack(bottom_returns_list)

# ====================== 适应度计算函数 ======================

def calculate_ic(factor_values, returns, time_index=None):
    """
    基于IC的适应度评估
    
    参数:
        factor_values: 因子值数组或张量
        returns: 收益率数组或张量
        time_index: 时间索引数组或张量,如果为None则计算整体IC
        
    返回:
        适应度值
    """
    # 确保输入是PyTorch张量
    if not isinstance(factor_values, torch.Tensor):
        factor_values = torch.tensor(factor_values, dtype=torch.float32, device=DEVICE)
    if not isinstance(returns, torch.Tensor):
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    
    if time_index is None:
        import warnings
        warnings.warn("计算ic的数据不包含时间")
        return
    else:
        # 确保time_index是PyTorch张量
        if not isinstance(time_index, torch.Tensor):
            time_index = torch.tensor(time_index, dtype=torch.int64, device=DEVICE)
        
        # 计算每个时间点的IC,然后取平均
        ics = _calculate_group_ic_torch(factor_values, returns, time_index)
        if len(ics) == 0:
            return 0.0
        
        # 取IC绝对值的平均
        return torch.abs(torch.mean(ics)).item()

def calculate_icir(factor_values, returns, time_index):
    """
    计算信息系数信息比率 (Information Coefficient Information Ratio, ICIR)
    
    参数:
        factor_values: 因子值数组或张量
        returns: 收益率数组或张量
        time_index: 时间索引数组或张量
        
    返回:
        ICIR值
    """
    # 确保输入是PyTorch张量
    if not isinstance(factor_values, torch.Tensor):
        factor_values = torch.tensor(factor_values, dtype=torch.float32, device=DEVICE)
    if not isinstance(returns, torch.Tensor):
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    if not isinstance(time_index, torch.Tensor):
        time_index = torch.tensor(time_index, dtype=torch.int64, device=DEVICE)
    
    # 计算每个时间点的IC
    ics = _calculate_group_ic_torch(factor_values, returns, time_index)
    
    # 计算ICIR
    if len(ics) < 2:  # 至少需要两个IC值
        return 0.0
    
    ic_mean = torch.mean(ics)
    ic_std = torch.std(ics)
    
    # 避免除以零
    if ic_std == 0:
        return 0.0
    
    icir = ic_mean / ic_std
    
    # 处理NaN结果
    if torch.isnan(icir):
        return 0.0
    
    # 转换为标量
    if isinstance(icir, torch.Tensor):
        icir = icir.item()
    
    return icir

def calculate_cumulative_return(factor_values, returns, time_index):
    """
    计算因子分组累积收益率
    
    参数:
        factor_values: 因子值数组或张量
        returns: 收益率数组或张量
        time_index: 时间索引数组或张量
        quantiles: 分组数量
        
    返回:
        多空组合累积收益率
    """
    # 确保输入是PyTorch张量
    if not isinstance(factor_values, torch.Tensor):
        factor_values = torch.tensor(factor_values, dtype=torch.float32, device=DEVICE)
    if not isinstance(returns, torch.Tensor):
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    if not isinstance(time_index, torch.Tensor):
        time_index = torch.tensor(time_index, dtype=torch.int64, device=DEVICE)
    
    # 计算每个时间点的分位数收益率
    top_returns, bottom_returns = _calculate_group_quantile_returns_torch(
        factor_values, returns, time_index
    )
    
    # 计算多空组合收益率
    if len(top_returns) == 0 or len(bottom_returns) == 0:
        return 0.0
    
    long_short_returns = top_returns - bottom_returns
    
    # 计算累积收益率
    cumulative_return = torch.prod(1 + long_short_returns) - 1
    
    # 处理NaN结果
    if torch.isnan(cumulative_return):
        return 0.0
    
    # 转换为标量
    if isinstance(cumulative_return, torch.Tensor):
        cumulative_return = cumulative_return.item()
    
    return cumulative_return

def calculate_sharpe_ratio(factor_values, returns, time_index, risk_free_rate=0.0, annualization_factor=252):
    """
    计算因子分组夏普比率
    
    参数:
        factor_values: 因子值数组或张量
        returns: 收益率数组或张量
        time_index: 时间索引数组或张量
        quantiles: 分组数量
        risk_free_rate: 无风险利率
        annualization_factor: 年化因子
        
    返回:
        多空组合夏普比率
    """
    # 确保输入是PyTorch张量
    if not isinstance(factor_values, torch.Tensor):
        factor_values = torch.tensor(factor_values, dtype=torch.float32, device=DEVICE)
    if not isinstance(returns, torch.Tensor):
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    if not isinstance(time_index, torch.Tensor):
        time_index = torch.tensor(time_index, dtype=torch.int64, device=DEVICE)
    
    # 计算每个时间点的分位数收益率
    top_returns, bottom_returns = _calculate_group_quantile_returns_torch(
        factor_values, returns, time_index
    )
    
    # 计算多空组合收益率
    if len(top_returns) == 0 or len(bottom_returns) == 0 or len(top_returns) < 2:
        return 0.0
    
    long_short_returns = top_returns - bottom_returns
    
    # 计算超额收益率
    excess_returns = long_short_returns - risk_free_rate
    
    # 计算夏普比率
    mean_excess_return = torch.mean(excess_returns)
    std_excess_return = torch.std(excess_returns)
    
    # 避免除以零
    if std_excess_return == 0:
        return 0.0
    
    # 年化夏普比率
    sharpe_ratio = mean_excess_return / std_excess_return * torch.sqrt(torch.tensor(annualization_factor, dtype=torch.float32, device=DEVICE))
    
    # 处理NaN结果
    if torch.isnan(sharpe_ratio):
        return 0.0
    
    # 转换为标量
    if isinstance(sharpe_ratio, torch.Tensor):
        sharpe_ratio = sharpe_ratio.item()
    
    return sharpe_ratio

def calculate_fitness_comprehensive(factor_values, returns, time_index, **kwargs):
    """
    综合适应度评估
    
    参数:
        factor_values: 因子值数组或张量
        returns: 收益率数组或张量
        time_index: 时间索引数组或张量
        kwargs:
            quantiles: 分组数量
            weights: 各指标权重字典,默认为None (使用均等权重)
        
    返回:
        适应度值
    """
    # 计算各指标
    quantiles = kwargs.get('quantiles', 5)
    weights = kwargs.get('weights', {
            'ic': 0.25,
            'icir': 0.25,
            'return': 0.25,
            'sharpe': 0.25
        })

    ic = calculate_ic(factor_values, returns, time_index)
    icir = calculate_icir(factor_values, returns, time_index)
    ret = calculate_cumulative_return(factor_values, returns, time_index, quantiles)
    sharpe = calculate_sharpe_ratio(factor_values, returns, time_index, quantiles)
    
    # 计算加权得分
    score = (
        weights.get('ic', 0.0) * ic +
        weights.get('icir', 0.0) * icir +
        weights.get('return', 0.0) * ret +
        weights.get('sharpe', 0.0) * sharpe
    )
    
    return score

def create_fitness_function(metric='ic', **kwargs):
    """
    创建适应度函数
    
    参数:
        metric: 适应度指标 ('ic', 'icir', 'return', 'sharpe', 'comprehensive')
        
    返回:
        适应度函数
    """
    if metric == 'ic':
        return calculate_ic
    elif metric == 'icir':
        return calculate_icir
    elif metric == 'return':
        return calculate_cumulative_return
    elif metric == 'sharpe':
        return calculate_sharpe_ratio
    elif metric == 'comprehensive':
        return calculate_fitness_comprehensive
    else:
        raise ValueError(f"不支持的适应度指标: {metric}")

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
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

def evaluate_individual(factor_value, returns, time_index, fitness_function):
    """
    评估个体适应度
    
    参数:
        individual: DEAP个体
        pset: DEAP原始集
        feature_data: 特征数据字典,键为特征名,值为数组
        returns: 收益率数组
        time_index: 时间索引数组
        fitness_function: 适应度函数
        
    返回:
        适应度值元组 (DEAP要求返回元组)
    """
    # 将输入数据转换为PyTorch张量
    torch_returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    torch_time_index = torch.tensor(time_index, dtype=torch.int64, device=DEVICE)
    
    # 计算适应度
    fitness = fitness_function(factor_value, torch_returns, torch_time_index)
    
    return (fitness,)  # 返回元组

# ====================== 批量评估函数 ======================

def evaluate_population_batch(population, pset, feature_data, returns, time_index, fitness_function, batch_size=64):
    """
    批量评估种群
    
    参数:
        population: 种群个体列表
        pset: DEAP原始集
        feature_data: 特征数据字典
        returns: 收益率数组
        time_index: 时间索引数组
        fitness_function: 适应度函数
        batch_size: 批处理大小
        
    返回:
        适应度值列表
    """
    # 将输入数据转换为PyTorch张量
    torch_feature_data = convert_to_torch_tensors(feature_data)
    torch_returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    torch_time_index = torch.tensor(time_index, dtype=torch.int64, device=DEVICE)
    
    n = len(population)
    fitness_values = []
    
    # 批量处理
    for i in range(0, n, batch_size):
        batch = population[i:min(i+batch_size, n)]
        batch_results = []
        
        # 并行评估批次中的每个个体
        for individual in batch:
            factor_values = calculate_expression(individual, pset, torch_feature_data, use_gpu=True)
            batch_results.append(factor_values)
        
        # 计算适应度
        for j, factor_values in enumerate(batch_results):
            fitness = fitness_function(factor_values, torch_returns, torch_time_index)
            fitness_values.append((fitness,))  # DEAP 要求返回元组
    
    return fitness_values
