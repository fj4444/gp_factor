"""
GPU算子库

包含:
- 使用PyTorch实现的基本算术运算算子
- 使用PyTorch实现的基本数学函数算子
- 批量表达式评估功能
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union, Callable
import warnings
from deap import gp

from ..base.setup import create_pset, add_basic_primitives

# 全局设置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 保护函数,处理异常情况
def protected_div(x, y):
    """
    保护除法,避免除以零 (PyTorch 实现)
    
    参数:
        x: 分子
        y: 分母
        
    返回:
        除法结果,当分母为零时返回1
    """
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        # 创建掩码,标记分母接近零的位置
        mask = torch.abs(y) < 1e-10
        # 将掩码位置的分母设为 1.0,避免除以零
        safe_y = torch.where(mask, torch.ones_like(y), y)
        # 计算除法结果
        result = x / safe_y
        # 将掩码位置的结果设为 1.0
        result = torch.where(mask, torch.ones_like(result), result)
        return result
    else:
        # 回退到 numpy 实现
        try:
            if np.isscalar(y) and abs(y) < 1e-10:
                return 1.0
            return x / y
        except (ZeroDivisionError, FloatingPointError, ValueError):
            return 1.0

def protected_log(x):
    """
    保护对数,避免对负数或零取对数 (PyTorch 实现)
    
    参数:
        x: 输入值
        
    返回:
        对数结果,当输入小于等于零时返回0
    """
    if isinstance(x, torch.Tensor):
        # 创建掩码,标记小于等于零的位置
        mask = x <= 0
        # 将掩码位置的输入设为 1.0,避免对负数或零取对数
        safe_x = torch.where(mask, torch.ones_like(x), x)
        # 计算对数结果
        result = torch.log(safe_x)
        # 将掩码位置的结果设为 0.0
        result = torch.where(mask, torch.zeros_like(result), result)
        return result
    else:
        # 回退到 numpy 实现
        try:
            if np.isscalar(x) and x <= 0:
                return 0.0
            return np.log(x)
        except (ValueError, FloatingPointError):
            return 0.0

def protected_sqrt(x):
    """
    保护平方根,避免对负数取平方根 (PyTorch 实现)
    
    参数:
        x: 输入值
        
    返回:
        平方根结果,当输入小于零时返回0
    """
    if isinstance(x, torch.Tensor):
        # 创建掩码,标记小于零的位置
        mask = x < 0
        # 将掩码位置的输入设为 0.0,避免对负数取平方根
        safe_x = torch.where(mask, torch.zeros_like(x), x)
        # 计算平方根结果
        result = torch.sqrt(safe_x)
        return result
    else:
        # 回退到 numpy 实现
        try:
            if np.isscalar(x) and x < 0:
                return np.sqrt(abs(x))
            return np.sqrt(x)
        except (ValueError, FloatingPointError):
            return 0.0

def protected_exp(x):
    """
    保护指数函数,避免溢出 (PyTorch 实现)
    
    参数:
        x: 输入值
        
    返回:
        指数结果,当结果溢出时返回最大浮点数
    """
    if isinstance(x, torch.Tensor):
        # 限制输入范围,避免溢出
        clipped_x = torch.clamp(x, -88.0, 88.0)  # torch.exp 的安全范围
        # 计算指数结果
        result = torch.exp(clipped_x)
        return result
    else:
        # 回退到 numpy 实现
        try:
            if np.isscalar(x) and x > 100:
                return np.finfo(np.float64).max
            return np.exp(x)
        except (OverflowError, FloatingPointError):
            return np.finfo(np.float64).max

# 算术运算算子
def add(x, y):
    """加法运算 (PyTorch 实现)"""
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return x + y
    else:
        return x + y

def subtract(x, y):
    """减法运算 (PyTorch 实现)"""
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return x - y
    else:
        return x - y

def multiply(x, y):
    """乘法运算 (PyTorch 实现)"""
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return x * y
    else:
        return x * y

def divide(x, y):
    """除法运算 (保护版本) (PyTorch 实现)"""
    return protected_div(x, y)

def power(x, y):
    """
    幂运算 (保护版本) (PyTorch 实现)
    """
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        # 限制指数范围,避免计算过大的幂
        y_clipped = torch.clamp(y, -10.0, 10.0)
        # 对于负底数,使用绝对值
        x_safe = torch.abs(x)
        # 计算幂
        result = torch.pow(x_safe, y_clipped)
        return result
    else:
        # 回退到 numpy 实现
        try:
            if np.isscalar(y):
                # 限制指数范围,避免计算过大的幂
                y_clipped = np.clip(y, -10, 10)
                if np.isscalar(x) and x < 0:
                    # 对于负底数,使用绝对值
                    return np.power(abs(x), y_clipped)
                return np.power(x, y_clipped)
            return np.power(abs(x), y)
        except (ValueError, OverflowError, FloatingPointError):
            return 1.0

# 数学函数算子
def log(x):
    """自然对数 (保护版本) (PyTorch 实现)"""
    return protected_log(x)

def sqrt(x):
    """平方根 (保护版本) (PyTorch 实现)"""
    return protected_sqrt(x)

def exp(x):
    """指数函数 (保护版本) (PyTorch 实现)"""
    return protected_exp(x)

def abs_val(x):
    """绝对值 (PyTorch 实现)"""
    if isinstance(x, torch.Tensor):
        return torch.abs(x)
    else:
        return abs(x)

def neg(x):
    """取负 (PyTorch 实现)"""
    if isinstance(x, torch.Tensor):
        return -x
    else:
        return -x

def sigmoid(x):
    """Sigmoid函数 (PyTorch 实现)"""
    if isinstance(x, torch.Tensor):
        return torch.sigmoid(x)
    else:
        # 回退到 numpy 实现
        try:
            if np.isscalar(x) and x > 100:
                return 1.0
            elif np.isscalar(x) and x < -100:
                return 0.0
            return 1.0 / (1.0 + np.exp(-x))
        except (OverflowError, FloatingPointError):
            if x > 0:
                return 1.0
            return 0.0

def sign(x):
    """符号函数 (PyTorch 实现)"""
    if isinstance(x, torch.Tensor):
        return torch.sign(x)
    else:
        try:
            return np.sign(x)
        except (ValueError, FloatingPointError):
            return 0.0

# 条件算子
def if_then_else(input1, input2, output1, output2):
    input = (input1 >= input2)
    if input: return output1
    else: return output2

def setup_primitives(pset, ret_type=float):
    """
    设置DEAP原始集
    
    参数:
        pset: DEAP原始集
        ret_type: 返回类型
    """
    # 添加算术运算算子
    pset.addPrimitive(add, [ret_type, ret_type], ret_type)
    pset.addPrimitive(subtract, [ret_type, ret_type], ret_type)
    pset.addPrimitive(multiply, [ret_type, ret_type], ret_type)
    pset.addPrimitive(divide, [ret_type, ret_type], ret_type)
    pset.addPrimitive(power, [ret_type, ret_type], ret_type)
    
    # 添加数学函数算子
    pset.addPrimitive(log, [ret_type], ret_type)
    pset.addPrimitive(sqrt, [ret_type], ret_type)
    pset.addPrimitive(exp, [ret_type], ret_type)
    pset.addPrimitive(abs_val, [ret_type], ret_type)
    pset.addPrimitive(neg, [ret_type], ret_type)
    pset.addPrimitive(sigmoid, [ret_type], ret_type)
    pset.addPrimitive(sign, [ret_type], ret_type)
    # pset.addPrimitive(if_then_else, [ret_type, ret_type, ret_type, ret_type], ret_type)
    
    # 添加基本原始集元素 (常量等)
    add_basic_primitives(pset, ret_type)
    
    return pset

def setup_advanced_primitives(pset, include_time_series=True, include_cross_sectional=True, ret_type=float):
    """
    设置高级DEAP原始集,包括时序和截面算子
    
    参数:
        pset: DEAP原始集
        include_time_series: 是否包含时序算子
        include_cross_sectional: 是否包含截面算子
        ret_type: 返回类型,默认为float
        
    注意: 时序和截面算子需要特殊处理,因为它们操作的是整个序列
    """
    # 首先设置基本原始集
    setup_primitives(pset, ret_type)
    
    # 添加时序算子
    if include_time_series:
        # 这些算子需要在评估时特殊处理
        # 这里只是添加占位符
        pset.addPrimitive(lag, [ret_type], ret_type)
        pset.addPrimitive(diff, [ret_type], ret_type)
        pset.addPrimitive(pct_change, [ret_type], ret_type)
        pset.addPrimitive(rolling_mean, [ret_type], ret_type)
        pset.addPrimitive(rolling_std, [ret_type], ret_type)
        pset.addPrimitive(ewm, [ret_type], ret_type)
    
    # 添加截面算子
    if include_cross_sectional:
        # 这些算子也需要在评估时特殊处理
        pset.addPrimitive(rank, [ret_type], ret_type)
        pset.addPrimitive(zscore, [ret_type], ret_type)
        pset.addPrimitive(min_max_scale, [ret_type], ret_type)
    
    return pset

# 时序算子占位符
def lag(x):
    """滞后算子占位符"""
    return x

def diff(x):
    """差分算子占位符"""
    return x

def pct_change(x):
    """百分比变化算子占位符"""
    return x

def rolling_mean(x):
    """滚动平均算子占位符"""
    return x

def rolling_std(x):
    """滚动标准差算子占位符"""
    return x

def ewm(x):
    """指数加权移动平均算子占位符"""
    return x

# 截面算子占位符
def rank(x):
    """排名算子占位符"""
    return x

def zscore(x):
    """Z分数标准化算子占位符"""
    return x

def min_max_scale(x):
    """最小-最大缩放算子占位符"""
    return x

# TODO:来自复现项目的算子,还需要适配这里的格式
def rank_div(x, y):
    """
    Multiply the cross-sectional rankings of two indicators
    """
    epsilon = 1e-10
    # Rank the two series in cross-section (for each date)
    x_rank = x.groupby(level='TradingDay').rank(pct=True)
    y_rank = y.groupby(level='TradingDay').rank(pct=True)+epsilon
    # Multiply the rankings
    return x_rank / y_rank

def power3(x):
    """
    Raise the input to the power of 3
    """
    return x ** 3

def mul(x, y):
    return x * y

def series_max(x, y):
    return np.maximum(x, y)

def ts_max(series, window):
    """
    Calculate the maximum value over a rolling window
    """
    return series.groupby(level='InnerCode').rolling(window=window, min_periods = 1).max().droplevel(0)

def ts_max_mean(series, window, num):
    """
    Calculate the average of the largest num values over a rolling window
    """
    num = num
    # 自定义函数获取前 N 个最大值的均值
    def top_n_max_mean(x):
        return np.mean(sorted(x, reverse=True)[:num])

    # 使用 rolling() 和 apply()
    result = series.groupby(level='InnerCode').rolling(window=window, min_periods = num).apply(top_n_max_mean, raw=False).droplevel(0)

    # return series.groupby(level='InnerCode').rolling(window=window, min_periods = num).max().droplevel(0)
    return result

def if_then_else(condition, value_if_true, value_if_false):
    """
    Conditional function: if condition is True, return value_if_true, else return value_if_false
    """
    result = pd.Series(
        np.where(condition, value_if_true, value_if_false),
        index=condition.index
    )
    return result

def ts_mean(series, window, min_periods=None):
    """
    Calculate the moving average over a rolling window
    """
    if min_periods is None:
        min_periods = int(0.75*window)
    return series.groupby(level='InnerCode').rolling(window=window, min_periods=min_periods).mean().droplevel(0)

def curt(x):
    """
    Calculate the cube root
    """
    return np.cbrt(x)

def inv(x):
    """
    Calculate the inverse (1/x)
    """
    return (1 / x).replace([np.inf, -np.inf], np.nan)

def umr(x1, x2):
    """
    Custom operator: (x1-mean(x1))*x2
    """
    # Calculate mean of x1 for each trading day
    x1_mean = x1.groupby(level='TradingDay').mean()
    
    # Broadcast the mean to match the original index
    x1_mean_broadcast = pd.Series(
        index=x1.index,
        data=[x1_mean.loc[date] for date in x1.index.get_level_values('TradingDay')]
    )
    # Calculate (x1-mean(x1))*x2
    return (x1 - x1_mean_broadcast) * x2

def ts_cov(x, y, window):
    """
    Calculate the covariance between two series over a rolling window for each InnerCode
    """
    # Create a list to store results
    cov_results = []
    
    # Process each stock separately
    for code, group_x in x.groupby(level='InnerCode'):
        # Get corresponding group for y
        group_y = y.loc[group_x.index]
        
        # Sort by TradingDay
        group_x = group_x.sort_index(level='TradingDay')
        group_y = group_y.sort_index(level='TradingDay')
        
        # Calculate rolling covariance
        cov = group_x.rolling(window=window).cov(group_y)
        
        # Add InnerCode back as index
        cov = pd.Series(cov.values, index=pd.MultiIndex.from_tuples(
            [(day, code) for day in group_x.index.get_level_values('TradingDay')],
            names=['TradingDay', 'InnerCode']
        ))
        
        cov_results.append(cov)
    
    # Combine all covariance results
    if cov_results:
        return pd.concat(cov_results)
    else:
        return pd.Series()

def rank_sub(x, y):
    """
    Multiply the cross-sectional rankings of two indicators
    """
    # Rank the two series in cross-section (for each date)
    x_rank = x.groupby(level='TradingDay').rank(pct=True)
    y_rank = y.groupby(level='TradingDay').rank(pct=True)
    
    # Multiply the rankings
    return x_rank - y_rank

def rank_mul(x, y):
    """
    Multiply the cross-sectional rankings of two indicators
    """
    # Rank the two series in cross-section (for each date)
    x_rank = x.groupby(level='TradingDay').rank(pct=True)
    y_rank = y.groupby(level='TradingDay').rank(pct=True)
    
    # Multiply the rankings
    return x_rank * y_rank

import warnings

def ts_to_wm(series, window):
    """
    Apply linear decay weighting to the past 'window' days - Optimized version
    """
    # Define a function to apply to each rolling window
    def weighted_max_div_mean(values):
        # Get the max value in the window
        max_val = np.max(values)
        # Calculate weighted average
        weights = np.arange(1, len(values)+1)
        weights = weights / weights.sum()
        weighted_avg = np.sum(values * weights)
        # Return max divided by weighted average
        return max_val / (weighted_avg + 1e-10)
    
    # Group by InnerCode and apply the rolling calculation
    result = series.groupby(level='InnerCode').apply(
        lambda x: x.sort_index(level='TradingDay').rolling(
            window=window, min_periods=int(0.75*window)
        ).apply(weighted_max_div_mean, raw=True)
    )
    
    # Handle the MultiIndex after groupby.apply
    if isinstance(result.index, pd.MultiIndex) and len(result.index.names) > 2:
        result = result.droplevel(0)
    
    return result

def ts_max_to_min(series, window):
    maxseries = series.groupby(level='InnerCode').rolling(window=window, min_periods = 1).max().droplevel(0)
    minseries = series.groupby(level='InnerCode').rolling(window=window, min_periods = 1).min().droplevel(0)
    return maxseries - minseries

def mean2(x, y):
    return (x+y)/2

def ts_sum(series, window):
    """
    Calculate the sum over a rolling window
    """
    return series.groupby(level='InnerCode').rolling(window=window).sum().droplevel(0)


def calculate_expression(individual, pset, feature_data, use_gpu=True):
    """
    评估表达式 (PyTorch 实现)
    
    参数:
        individual: DEAP个体
        pset: DEAP原始集
        feature_data: 特征数据字典,键为特征名,值为数组
        use_gpu: 是否使用 GPU
        
    返回:
        表达式计算结果
    """
    # 编译表达式
    func = gp.compile(individual, pset)
    
    # 准备输入数据
    if use_gpu and torch.cuda.is_available():
        # 将输入数据转换为 PyTorch 张量并移至 GPU
        inputs = []
        for arg in pset.arguments:
            if arg in feature_data:
                tensor = torch.tensor(feature_data[arg], dtype=torch.float32).to(DEVICE)
                inputs.append(tensor)
            else:
                # 如果找不到参数,使用零张量
                shape = next(iter(feature_data.values())).shape
                inputs.append(torch.zeros(shape, dtype=torch.float32).to(DEVICE))
    else:
        # 使用原始 numpy 数组
        inputs = []
        for arg in pset.arguments:
            if arg in feature_data:
                inputs.append(feature_data[arg])
            else:
                # 如果找不到参数,使用零数组
                shape = next(iter(feature_data.values())).shape
                inputs.append(np.zeros(shape))
    
    # 评估表达式
    try:
        result = func(*inputs)
        
        # 处理无效值
        if isinstance(result, torch.Tensor):
            # 将 NaN 和 Inf 替换为 0.0
            result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            # 如果需要,将结果转回 CPU 和 numpy
            if use_gpu and torch.cuda.is_available():
                result = result.cpu().numpy()
        elif isinstance(result, np.ndarray):
            result[np.isnan(result)] = 0.0
            result[np.isinf(result)] = 0.0
        elif np.isnan(result) or np.isinf(result):
            result = 0.0
            
        return result
    except Exception as e:
        print(f"表达式评估错误: {e}")
        # 返回全零数组或零值
        if len(inputs) > 0:
            if isinstance(inputs[0], torch.Tensor):
                return torch.zeros_like(inputs[0]).cpu().numpy()
            elif isinstance(inputs[0], np.ndarray):
                return np.zeros_like(inputs[0])
        return 0.0


# def evaluate_population_batch(population, pset, feature_data, returns, time_index, fitness_function, batch_size=64, use_gpu=True):
#     """
#     批量评估种群
    
#     参数:
#         population: 种群个体列表
#         pset: DEAP原始集
#         feature_data: 特征数据字典
#         returns: 收益率数组
#         time_index: 时间索引数组
#         fitness_function: 适应度函数
#         batch_size: 批处理大小
#         use_gpu: 是否使用 GPU
        
#     返回:
#         适应度值列表
#     """
#     n = len(population)
#     fitness_values = []
    
#     # 将输入数据转换为 PyTorch 张量
#     if use_gpu and torch.cuda.is_available():
#         torch_feature_data = {k: torch.tensor(v, dtype=torch.float32).to(DEVICE) 
#                              for k, v in feature_data.items()}
#     else:
#         torch_feature_data = feature_data
    
#     # 批量处理
#     for i in range(0, n, batch_size):
#         batch = population[i:min(i+batch_size, n)]
#         batch_results = []
        
#         # 并行评估批次中的每个个体
#         for individual in batch:
#             factor_values = calculate_expression(individual, pset, torch_feature_data, use_gpu)
#             batch_results.append(factor_values)
        
#         # 计算适应度
#         for j, factor_values in enumerate(batch_results):
#             fitness = fitness_function(factor_values, returns, time_index)
#             fitness_values.append((fitness,))  # DEAP 要求返回元组
    
#     return fitness_values



def convert_to_torch_tensors(data_dict, device=None):
    """
    将数据字典转换为 PyTorch 张量
    
    参数:
        data_dict: 数据字典,键为特征名,值为 numpy 数组
        device: PyTorch 设备,如果为 None 则使用全局 DEVICE
        
    返回:
        转换后的数据字典,键为特征名,值为 PyTorch 张量
    """
    if device is None:
        device = DEVICE
        
    torch_dict = {}
    for k, v in data_dict.items():
        if isinstance(v, np.ndarray):
            torch_dict[k] = torch.tensor(v, dtype=torch.float32).to(device)
        else:
            torch_dict[k] = v
            
    return torch_dict


def convert_to_numpy(tensor):
    """
    将 PyTorch 张量转换为 numpy 数组
    
    参数:
        tensor: PyTorch 张量
        
    返回:
        numpy 数组
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    else:
        return tensor
