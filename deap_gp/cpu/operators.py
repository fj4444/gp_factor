"""
CPU算子库

包含:
- 基本算术运算算子 (加减乘除等)
- 基本数学函数算子 (log、sqrt等)
- 基本时序算子 (移动平均、滞后等)
- 基本截面算子 (排名、标准化等)
"""

import numpy as np
import pandas as pd
import bottleneck as bn
import warnings
from deap import gp
import numba
from ..base.setup import create_pset, add_basic_primitives

# 保护函数,处理异常情况
def protected_div(x, y):
    """
    保护除法,避免除以零
    
    参数:
        x: 分子
        y: 分母
        
    返回:
        除法结果,当分母为零时返回1
    """
    try:
        if np.isscalar(y) and abs(y) < 1e-10:
            return 1.0
        return x / y
    except (ZeroDivisionError, FloatingPointError, ValueError):
        return 1.0

def protected_log(x):
    """
    保护对数,避免对负数或零取对数
    
    参数:
        x: 输入值
        
    返回:
        对数结果,当输入小于等于零时返回0
    """
    try:
        if np.isscalar(x) and x <= 0:
            return 0.0
        return np.log(x)
    except (ValueError, FloatingPointError):
        return 0.0

def protected_sqrt(x):
    """
    保护平方根,避免对负数取平方根
    
    参数:
        x: 输入值
        
    返回:
        平方根结果,当输入小于零时返回0
    """
    try:
        if np.isscalar(x) and x < 0:
            return np.sqrt(abs(x))
        return np.sqrt(x)
    except (ValueError, FloatingPointError):
        return 0.0

# 算术运算算子
def add(x, y):
    """加法运算"""
    return x + y

def subtract(x, y):
    """减法运算"""
    return x - y

def multiply(x, y):
    """乘法运算"""
    return x * y

def divide(x, y):
    """除法运算 (保护版本)"""
    return protected_div(x, y)

def power(x, y):
    """
    幂运算 (保护版本)
    """
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
    """自然对数 (保护版本)"""
    return protected_log(x)

def sqrt(x):
    """平方根 (保护版本)"""
    return protected_sqrt(x)

def abs_val(x):
    """绝对值"""
    return abs(x)

def neg(x):
    """取负"""
    return -x

def sigmoid(x):
    """Sigmoid函数"""
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

def hardsigmoid(x):
    """
    Hard Sigmoid函数 - 分段线性近似的Sigmoid
    
    参数:
        x: 输入值
        
    返回:
        Hard Sigmoid结果: max(0, min(1, (x+1)/2))
    """
    try:
        if np.isscalar(x):
            return max(0.0, min(1.0, (x + 1.0) / 2.0))
        else:
            # 向量化处理: max(0, min(1, (x+1)/2))
            # 按照公式顺序实现: 先计算(x+1)/2,再取min(1,结果),最后取max(0,结果)
            result = (x + 1.0) / 2.0
            result = np.minimum(1.0, result)
            result = np.maximum(0.0, result)
            return result
    except (OverflowError, FloatingPointError):
        if x > 0:
            return min(1.0, (x + 1.0) / 2.0)
        return 0.0

# TODO:让alpha变成可变参数
def leakyrelu(x, alpha=0.1):
    """
    Leaky ReLU函数 - 允许负值有小梯度的ReLU变体
    
    参数:
        x: 输入值
        alpha: 负值区域的斜率,默认为0.01
        
    返回:
        Leaky ReLU结果: x if x > 0 else alpha * x
    """
    try:
        if np.isscalar(x):
            return x if x > 0 else alpha * x
        else:
            # 向量化处理
            return np.where(x > 0, x, alpha * x)
    except (OverflowError, FloatingPointError):
        return 0.0

def gelu(x):
    """
    GELU函数 - Gaussian Error Linear Unit
    
    参数:
        x: 输入值
        
    返回:
        GELU结果: x * Φ(x),其中Φ是标准正态分布的累积分布函数
    """
    try:
        # 使用近似公式: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    except (OverflowError, FloatingPointError):
        return 0.0

def sign(x):
    """符号函数"""
    try:
        return np.sign(x)
    except (ValueError, FloatingPointError):
        return 0.0

def power2(x):
    """
    Raise the input to the power of 2
    """
    return power(x,2)

def power3(x):
    """
    Raise the input to the power of 3
    """
    return power(x,3)

def curt(x):
    """
    Calculate the cube root
    """
    return np.cbrt(x)

def inv(x):
    """
    Calculate the inverse (1/x)
    """
    return divide(1,x)

def mean2(x, y):
    return (x+y)/2

# 条件算子
def if_then_else(input1, input2, output1, output2):
    mask = (input1 >= input2)
    return np.where(mask, output1, output2)

def series_max(x, y):
    return np.maximum(x, y)

# 时序算子
def ts_lag(x, periods=1):
    """滞后算子"""
    if not np.isscalar(periods):
        periods = int(periods.item(0))
        
    if isinstance(x, pd.Series):
        # 使用Series的索引进行分组滞后
        if x.index.nlevels > 1:
            # 按标的分组应用滞后操作
            return x.groupby(level='InnerCode').shift(periods)
        else:
            # 单级索引,直接滞后
            return x.shift(periods)
    elif isinstance(x, np.ndarray):   
        result = np.empty_like(x)     
        if x.ndim == 1:  # 一维数组
            result[:periods] = np.nan
            result[periods:] = x[:-periods]
        elif x.ndim == 2:  # 二维数组 (时间 × 股票)
            result[:periods, :] = np.nan  # 所有列的前periods行设为NaN
            result[periods:, :] = x[:-periods, :]  # 所有列同时进行滞后操作
        else:
            # 更高维数组,返回原始数组
            return x
        return result
    return x

def ts_diff(x, periods=1):
    """差分算子"""
    if not np.isscalar(periods):
        periods = int(periods.item(0))
        
    if isinstance(x, pd.Series):
        # 使用Series的索引进行分组差分
        if x.index.nlevels > 1:
            # 按标的分组应用差分操作
            return x.groupby(level='InnerCode').diff(periods)
        else:
            # 单级索引,直接差分
            return x.diff(periods)
    elif isinstance(x, np.ndarray):
        # 一维二维都是一个算法
        lagged = ts_lag(x, periods)
        result = x - lagged
        return result
    return x

def ts_pct_change(x, periods=1):
    """百分比变化算子"""
    if not np.isscalar(periods):
        periods = int(periods.item(0))
        
    if isinstance(x, pd.Series):
        # 使用Series的索引进行分组百分比变化
        if x.index.nlevels > 1:
            # 按标的分组应用百分比变化操作
            result = x.groupby(level='InnerCode').ffill().pct_change(periods)
        else:
            # 单级索引,直接百分比变化
            result = x.pct_change(periods)
        # 处理无效值
        result = result.fillna(0.0)
        result = result.replace([np.inf, -np.inf], 0.0)
        return result
    elif isinstance(x, np.ndarray):
        lagged = ts_lag(x, periods)
        with np.errstate(divide='ignore', invalid='ignore'):
            return (x / lagged) - 1
    return x

def ts_mean(x, window=5):
    """滚动平均算子 (移动平均)"""
    if not np.isscalar(window):
        window = int(window.item(0))
        
    if isinstance(x, pd.Series):
        # 使用Series的索引进行分组滚动平均
        if x.index.nlevels > 1:
            # 按标的分组应用滚动平均操作
            return x.groupby(level='InnerCode').rolling(window=window, min_periods=1).mean().reset_index(level='TradingDay', drop=True)
        else:
            # 单级索引,直接滚动平均
            return x.rolling(window=window, min_periods=1).mean()
    elif isinstance(x, np.ndarray):
        result = np.full_like(x,np.nan)
        if x.ndim == 1:
            result = bn.move_mean(x, window=window, min_count=1)
        elif x.ndim == 2:
            result = bn.move_mean(x, window=window, min_count=1, axis=0)
        return result
    return x

def ts_std(x, window=5):
    """滚动标准差算子"""
    if not np.isscalar(window):
        window = int(window.item(0))
        
    if isinstance(x, pd.Series):
        # 使用Series的索引进行分组滚动标准差
        if x.index.nlevels > 1:
            # 按标的分组应用滚动标准差操作
            return x.groupby(level='InnerCode').rolling(window=window, min_periods=1).std().reset_index(level='TradingDay', drop=True)
        else:
            # 单级索引,直接滚动标准差
            return x.rolling(window=window, min_periods=1).std()
    elif isinstance(x, np.ndarray):
        result = np.full_like(x,np.nan)
        if x.ndim == 1:
            result = bn.move_std(x, window=window, min_count=2)
        elif x.ndim == 2:
            result = bn.move_std(x, window=window, min_count=2, axis=0)
            
        return result
    return x

@numba.jit(nopython=True)
def calc_ts_ewm_vectorized(x, alpha):
    """numba 优化的 ts_ewm 计算核心 - 向量化实现"""
    result = np.empty_like(x)
    
    if x.ndim == 1:
        result[0] = x[0]
        for i in range(1, len(x)):
            result[i] = alpha * x[i] + (1 - alpha) * result[i-1]
    elif x.ndim == 2:
        # 第一行直接赋值
        result[0, :] = x[0, :]
        
        # 对后续每一行进行向量化计算
        for i in range(1, x.shape[0]):
            result[i, :] = alpha * x[i, :] + (1 - alpha) * result[i-1, :]
    
    return result

def ts_ewm(x, halflife = 1):
    """指数加权移动平均算子 - 使用 numba 优化的向量化实现"""
    if not np.isscalar(halflife):
        halflife = int(halflife.item(0))
        
    if halflife <= 0:
        warnings.warn("halflife应该大于0,已设置为1")
        halflife = 1

    if isinstance(x, pd.Series):
        # 使用Series的索引进行分组指数加权移动平均
        if x.index.nlevels > 1:
            # 按标的分组应用指数加权移动平均操作
            return x.groupby(level='InnerCode').ewm(halflife=halflife, min_periods=1).mean().reset_index(level='TradingDay', drop=True)
        else:
            # 单级索引,直接指数加权移动平均
            return x.ewm(halflife=halflife, min_periods=1).mean()
    elif isinstance(x, np.ndarray):      
        alpha = 1 - np.exp(-np.log(2)/halflife)
        return calc_ts_ewm_vectorized(x, alpha)
    return x

def ts_max(x, window=5):
    """
    Calculate the maximum value over a rolling window
    """
    if not np.isscalar(window):
        window = int(window.item(0))
        
    if isinstance(x, pd.Series):
        # 使用Series的索引进行分组滚动
        if x.index.nlevels > 1:
            # 按标的分组应用滚动操作
            return x.groupby(level='InnerCode').rolling(window=window, min_periods=1).max().reset_index(level='TradingDay', drop=True)
        else:
            # 单级索引,直接滚动
            return x.rolling(window=window, min_periods=1).max()
    elif isinstance(x, np.ndarray):
        result = np.full_like(x,np.nan)
        if x.ndim == 1:
            result = bn.move_max(x, window=window, min_count=1)
        elif x.ndim == 2:
            result = bn.move_max(x, window=window, min_count=1, axis=0)
        return result
    return x

def ts_min(x, window=5):
    """
    Calculate the maximum value over a rolling window
    """
    if not np.isscalar(window):
        window = int(window.item(0))
        
    if isinstance(x, pd.Series):
        # 使用Series的索引进行分组滚动
        if x.index.nlevels > 1:
            # 按标的分组应用滚动操作
            return x.groupby(level='InnerCode').rolling(window=window, min_periods=1).min().reset_index(level='TradingDay', drop=True)
        else:
            # 单级索引,直接滚动
            return x.rolling(window=window, min_periods=1).min()
    elif isinstance(x, np.ndarray):
        result = np.full_like(x,np.nan)
        if x.ndim == 1:
            result = bn.move_min(x, window=window, min_count=1)
        elif x.ndim == 2:
            result = bn.move_min(x, window=window, min_count=1, axis=0)
        return result
    return x

def ts_argmin(x, window=5):
    """
    Calculate the position of the minimum value over a rolling window
    """
    if not np.isscalar(window):
        window = int(window.item(0))
        
    if isinstance(x, np.ndarray) and x.ndim==2:
        result = np.full_like(x,np.nan)
        if x.ndim == 1:
            result = bn.move_argmin(x, window=window, min_count=1)
        elif x.ndim == 2:
            result = bn.move_argmin(x, window=window, min_count=1, axis=0)
        return result
    else:
        raise ValueError("输入不是二维ndarray")

def ts_argmax(x, window=5):
    """
    Calculate the position of the maximum value over a rolling window
    """
    if not np.isscalar(window):
        window = int(window.item(0))
        
    if isinstance(x, np.ndarray) and x.ndim==2:
        result = np.full_like(x,np.nan)
        if x.ndim == 1:
            result = bn.move_argmax(x, window=window, min_count=1)
        elif x.ndim == 2:
            result = bn.move_argmax(x, window=window, min_count=1, axis=0)
        return result
    else:
        raise ValueError("输入不是二维ndarray")

def ts_max_to_min(x, window=5):
    if not np.isscalar(window):
        window = int(window.item(0))
        
    return ts_max(x, window) - ts_min(x, window)

def ts_sum(x, window=5):
    """
    Calculate the sum over a rolling window
    """
    if not np.isscalar(window):
        window = int(window.item(0))
        
    if isinstance(x, pd.Series):
        # 使用Series的索引进行分组滚动
        if x.index.nlevels > 1:
            # 按标的分组应用滚动操作
            return x.groupby(level='InnerCode').rolling(window=window, min_periods=1).sum().reset_index(level='TradingDay', drop=True)
        else:
            # 单级索引,直接滚动
            return x.rolling(window=window, min_periods=1).sum()
    elif isinstance(x, np.ndarray):
        result = np.full_like(x,np.nan)
        if x.ndim == 1:
            result = bn.move_sum(x, window=window, min_count=1)
        elif x.ndim == 2:
            result = bn.move_sum(x, window=window, min_count=1, axis=0)
        return result
    return x

@numba.jit(nopython=True, parallel=True)
def calc_ts_max_mean(x, window=5, num=3):    
    result = np.full_like(x, np.nan)
    
    # 并行处理不同的股票
    for j in numba.prange(x.shape[1]):
        start = int(window - 1)
        end = x.shape[0]
        for i in range(start, end):
            window_data = x[max(0, i-window+1):i+1, j]
            
            valid_mask = ~np.isnan(window_data)
            valid_count = np.sum(valid_mask)
            valid_data = window_data[valid_mask]

            if valid_count <= num:
                # 如果有效值数量小于等于num，直接使用所有有效值
                result[i, j] = np.mean(valid_data)
            else:
                # 使用partition只部分排序数组
                pivot = valid_count - num
                np.partition(valid_data, pivot)
                result[i, j] = np.mean(valid_data[pivot:])
    
    return result

def ts_max_mean(x, window=5, num=3):
    """
    计算滚动窗口内最大的num个值的平均值
    
    参数:
        x: 输入序列
        window: 滚动窗口大小,默认为5
        num: 取最大的几个值,默认为3
        
    返回:
        滚动窗口内最大的num个值的平均值
    """
    if not np.isscalar(window):
        window = int(window.item(0))
    if not np.isscalar(num):
        num = int(num.item(0))
    window = max(window,num)
    num = min(window,num)

    if isinstance(x, np.ndarray):
        return calc_ts_max_mean(x, window, num)
    return x

def ts_cov(x, y, window=5):
    # !慢
    """
    计算两个序列在滚动窗口内的协方差,按InnerCode分组
    
    参数:
        x: 第一个输入序列
        y: 第二个输入序列
        window: 滚动窗口大小,默认为5
        
    返回:
        按InnerCode分组的滚动协方差
    """
    if not np.isscalar(window):
        window = int(window.item(0))

    if isinstance(x, pd.Series) and isinstance(y, pd.Series):
        # 创建列表存储结果
        cov_results = []
        
        # 按股票分别处理
        for code, group_x in x.groupby(level='InnerCode'):
            # 获取对应的y组
            group_y = y.loc[group_x.index]
            
            # 按TradingDay排序
            group_x = group_x.sort_index(level='TradingDay')
            group_y = group_y.sort_index(level='TradingDay')
            
            # 计算滚动协方差
            cov = group_x.rolling(window=window, min_periods=2).cov(group_y)
            
            # 将InnerCode添加回索引
            cov = pd.Series(cov.values, index=pd.MultiIndex.from_tuples(
                [(day, code) for day in group_x.index.get_level_values('TradingDay')],
                names=['TradingDay', 'InnerCode']
            ))
            
            cov_results.append(cov)
        
        # 合并所有协方差结果
        if cov_results:
            return pd.concat(cov_results)
        else:
            return pd.Series()
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        result = np.full_like(x, np.nan)
        
        # 对于numpy数组,第一维是时间,第二维是股票
        if x.ndim == 2 and y.ndim == 2:
            # 对每个股票计算滚动协方差
            for j in range(x.shape[1]):
                for i in range(window-1, x.shape[0]):
                    x_window = x[max(0, i-window+1):i+1, j]
                    y_window = y[max(0, i-window+1):i+1, j]
                    
                    # 找出两个窗口中都有效的数据点
                    valid_mask = ~np.isnan(x_window) & ~np.isnan(y_window)
                    if np.sum(valid_mask) >= 2:  # 至少需要2个点才能计算协方差
                        result[i, j] = np.cov(x_window[valid_mask], y_window[valid_mask])[0, 1]
        
        return result
    return x

@numba.jit(nopython=True, parallel=True)
def calc_ts_corr(x, y, window=5):
    result = np.full_like(x, np.nan)
    
    # 并行处理不同的股票
    for j in numba.prange(x.shape[1]):
        start = int(window - 1)
        end = x.shape[0]
        for i in range(start, end):
            x_window = x[max(0, i-window+1):i+1, j]
            y_window = y[max(0, i-window+1):i+1, j]
            
            # 找出两个窗口中都有效的数据点
            valid_mask = ~(np.isnan(x_window) | np.isnan(y_window))
            valid_count = np.sum(valid_mask)
            
            if valid_count >= 2:
                x_valid = x_window[valid_mask]
                y_valid = y_window[valid_mask]
                
                # 计算相关系数
                mean_x = np.mean(x_valid)
                mean_y = np.mean(y_valid)
                num = 0.0
                den_x = 0.0
                den_y = 0.0
                
                dx = x_valid - mean_x
                dy = y_valid - mean_y
                
                num = np.sum(dx * dy)
                den_x = np.sum(dx * dx)
                den_y = np.sum(dy * dy)
                
                if den_x > 0 and den_y > 0:
                    result[i, j] = num / np.sqrt(den_x * den_y)
    
    return result

def ts_corr(x, y, window=5):
    """
    计算两个序列在滚动窗口内的相关系数,按InnerCode分组
    
    参数:
        x: 第一个输入序列
        y: 第二个输入序列
        window: 滚动窗口大小,默认为5
        
    返回:
        按InnerCode分组的滚动相关系数
    """
    if not np.isscalar(window):
        window = int(window.item(0))

    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return calc_ts_corr(x,y,window)
    return x

@numba.jit(nopython=True, parallel=True)
def calc_ts_rankcorr(x, y, window=5):
    result = np.full_like(x, np.nan)
    
    # 并行处理不同的股票
    for j in numba.prange(x.shape[1]):
        start = int(window - 1)
        end = x.shape[0]
        for i in range(start, end):
            x_window = x[max(0, i-window+1):i+1, j]
            y_window = y[max(0, i-window+1):i+1, j]
            
            # 找出两个窗口中都有效的数据点
            valid_mask = ~(np.isnan(x_window) | np.isnan(y_window))
            valid_count = np.sum(valid_mask)
            
            if valid_count >= 2:
                x_valid = x_window[valid_mask]
                y_valid = y_window[valid_mask]
                
                # 计算排名
                x_ranks = np.argsort(np.argsort(x_valid))
                y_ranks = np.argsort(np.argsort(y_valid))
                
                # 计算相关系数
                mean_x = np.mean(x_ranks)
                mean_y = np.mean(y_ranks)
                num = 0.0
                den_x = 0.0
                den_y = 0.0
                
                dx = x_valid - mean_x
                dy = y_valid - mean_y
                
                num = np.sum(dx * dy)
                den_x = np.sum(dx * dx)
                den_y = np.sum(dy * dy)
                
                if den_x > 0 and den_y > 0:
                    result[i, j] = num / np.sqrt(den_x * den_y)
    
    return result

def ts_rankcorr(x, y, window=5):
    """
    计算两个序列在滚动窗口内的秩相关系数,按InnerCode分组
    
    参数:
        x: 第一个输入序列
        y: 第二个输入序列
        window: 滚动窗口大小,默认为5
        
    返回:
        按InnerCode分组的滚动秩相关系数
    """
    if not np.isscalar(window):
        window = int(window.item(0))

    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return calc_ts_rankcorr(x,y,window)
    return x

@numba.jit(nopython=True, parallel=True)
def calc_ts_to_wm(x, window=5):
    result = np.full_like(x, np.nan)

    weights_template = np.arange(1, window+1, dtype=np.float64)
    
    # 并行处理不同的股票
    for j in numba.prange(x.shape[1]):
        start = int(window - 1)
        end = x.shape[0]
        for i in range(start, end):
            window_data = x[max(0, i-window+1):i+1, j]
            
            valid_mask = ~np.isnan(window_data)
            valid_count = np.sum(valid_mask)
            
            if valid_count >= max(1, int(0.75*window)):
                valid_data = window_data[valid_mask]
                max_val = np.max(valid_data)
                
                # 使用对应的权重
                weights = weights_template[valid_mask]
                weights = weights / np.sum(weights)  # 归一化
                
                weighted_avg = np.sum(valid_data * weights)
                result[i, j] = max_val / (weighted_avg + 1e-10)
    
    return result

def ts_to_wm(x, window=5):
    """
    对过去'window'天应用线性衰减加权 - 优化版本
    
    参数:
        x: 输入序列
        window: 滚动窗口大小,默认为5
        
    返回:
        窗口内最大值除以加权平均值,权重为线性递增
    """
    if not np.isscalar(window):
        window = int(window.item(0))
    
    if isinstance(x, pd.Series):
        # 按InnerCode分组并应用滚动计算
        result = x.groupby(level='InnerCode').apply(
            lambda x: x.sort_index(level='TradingDay').rolling(
                window=window, min_periods=max(1, int(0.75*window))
            ).apply(weighted_max_div_mean, raw=True)
        )
        
        # 处理groupby.apply后的MultiIndex
        if isinstance(result.index, pd.MultiIndex) and len(result.index.names) > 2:
            result = result.droplevel(0)
        
        return result
    elif isinstance(x, np.ndarray):
        return calc_ts_to_wm(x,window)
    return x

def ts_rank(x, window=5):
    if not np.isscalar(window):
        window = int(window.item(0))
    if isinstance(x, np.ndarray) and x.ndim==2:
        result = np.full_like(x,np.nan)
        if x.ndim == 1:
            result = bn.move_rank(x, window=window, min_count=1)
        elif x.ndim == 2:
            result = bn.move_rank(x, window=window, min_count=1, axis=0)
        # move_rank 返回[-1,1], 标准化到[0,1]
        return (result+1)/2
    else:
        raise ValueError("输入数据不是二维ndarray")

def ts_median(x, window=5):
    if not np.isscalar(window):
        window = int(window.item(0))
    if isinstance(x, np.ndarray) and x.ndim==2:
        result = np.full_like(x,np.nan)
        if x.ndim == 1:
            result = bn.move_median(x, window=window, min_count=1)
        elif x.ndim == 2:
            result = bn.move_median(x, window=window, min_count=1, axis=0)
        return result
    else:
        raise ValueError("输入数据不是二维ndarray")

# 截面算子 (这些在DEAP中也需要特殊处理)
def rank(x):
    """排名算子 - 支持二维numpy数组"""
    if isinstance(x, pd.Series):
        # 原有pandas实现保持不变
        if x.index.nlevels > 1:
            return x.groupby(level='TradingDay').rank(pct=True)
        else:
            return x.rank(pct=True)
    elif isinstance(x, np.ndarray):
        if x.ndim == 1:
            # 一维数组处理
            valid_mask = ~np.isnan(x)
            result = np.full_like(x,np.nan)
            if np.sum(valid_mask) > 0:
                # 计算百分比排名 (0到1之间)
                ranks = np.argsort(np.argsort(x[valid_mask]))
                result[valid_mask] = ranks / (np.sum(valid_mask) - 1) if np.sum(valid_mask) > 1 else 0.5
            return result
        elif x.ndim == 2:
            # 二维数组处理 - 按行(时间)进行排名
            result = np.full_like(x,np.nan)
            for t in range(x.shape[0]):
                valid_mask = ~np.isnan(x[t, :])
                if np.sum(valid_mask) > 1:  # 至少需要2个有效值才能计算有意义的排名
                    # 计算百分比排名
                    ranks = np.argsort(np.argsort(x[t, valid_mask]))
                    result[t, valid_mask] = ranks / (np.sum(valid_mask) - 1)
                elif np.sum(valid_mask) == 1:
                    # 只有一个有效值时,排名为0.5
                    result[t, valid_mask] = 0.5
                # 无效值保持为0
            return result
    return x

def rank_div(x, y):
    return divide(rank(x),rank(y))

def rank_sub(x, y):
    return rank(x) - rank(y)

def rank_mul(x, y):
    return rank(x) * rank(y)

def zscore(x):
    """Z分数标准化算子 - 支持二维numpy数组"""
    if isinstance(x, pd.Series):
        # 原有pandas实现保持不变
        if x.index.nlevels > 1:
            grouped = x.groupby(level='TradingDay')
            mean = grouped.transform('mean')
            std = grouped.transform('std')
            std = std.replace(0, 1)
            return (x - mean) / std
        else:
            mean = x.mean()
            std = x.std()
            if std == 0:
                return pd.Series(0, index=x.index)
            return (x - mean) / std
    elif isinstance(x, np.ndarray):
        if x.ndim == 1:
            # 一维数组处理
            valid_mask = ~np.isnan(x)
            result = np.full_like(x,np.nan)
            if np.sum(valid_mask) > 0:
                mean = np.mean(x[valid_mask])
                std = np.std(x[valid_mask])
                if std > 0:
                    result[valid_mask] = (x[valid_mask] - mean) / std
            return result
        elif x.ndim == 2:
            # 二维数组处理 - 按行(时间)进行标准化
            result = np.full_like(x,np.nan)
            
            # 计算每行的均值和标准差 (忽略NaN)
            # 使用nanmean和nanstd可以避免显式循环
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                means = np.nanmean(x, axis=1, keepdims=True)
                stds = np.nanstd(x, axis=1, keepdims=True)
            
            # 处理标准差为0或NaN的情况
            stds[stds == 0] = 1.0
            stds[np.isnan(stds)] = 1.0
            
            # 向量化计算Z分数
            valid_mask = ~np.isnan(x)
            result[valid_mask] = (x[valid_mask] - means.repeat(x.shape[1], axis=1)[valid_mask]) / stds.repeat(x.shape[1], axis=1)[valid_mask]
            
            return result
    return x

def min_max_scale(x):
    """最小-最大缩放算子 - 支持二维numpy数组"""
    if isinstance(x, pd.Series):
        # 原有pandas实现保持不变
        if x.index.nlevels > 1:
            grouped = x.groupby(level='TradingDay')
            min_val = grouped.transform('min')
            max_val = grouped.transform('max')
            denominator = max_val - min_val
            denominator = denominator.replace(0, 1)
            return (x - min_val) / denominator
        else:
            min_val = x.min()
            max_val = x.max()
            if max_val == min_val:
                return pd.Series(0, index=x.index)
            return (x - min_val) / (max_val - min_val)
    elif isinstance(x, np.ndarray):
        if x.ndim == 1:
            # 一维数组处理
            valid_mask = ~np.isnan(x)
            result = np.full_like(x,np.nan)
            if np.sum(valid_mask) > 0:
                min_val = np.min(x[valid_mask])
                max_val = np.max(x[valid_mask])
                if max_val > min_val:
                    result[valid_mask] = (x[valid_mask] - min_val) / (max_val - min_val)
            return result
        elif x.ndim == 2:
            # 二维数组处理 - 按行(时间)进行缩放
            result = np.full_like(x,np.nan)
            
            # 计算每行的最小值和最大值 (忽略NaN)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                min_vals = np.nanmin(x, axis=1, keepdims=True)
                max_vals = np.nanmax(x, axis=1, keepdims=True)
            
            # 计算分母并处理为0的情况
            denominators = max_vals - min_vals
            denominators[denominators == 0] = 1.0
            denominators[np.isnan(denominators)] = 1.0
            
            # 向量化计算缩放值
            valid_mask = ~np.isnan(x)
            result[valid_mask] = (x[valid_mask] - min_vals.repeat(x.shape[1], axis=1)[valid_mask]) / denominators.repeat(x.shape[1], axis=1)[valid_mask]
            
            return result
    return x

def umr(x1, x2):
    """
    自定义算子: (x1-mean(x1))*x2
    
    参数:
        x1: 第一个输入序列
        x2: 第二个输入序列
        
    返回:
        (x1-mean(x1))*x2,其中mean(x1)是x1在每个交易日的截面均值
    """
    if isinstance(x1, pd.Series) and isinstance(x2, pd.Series):
        # 计算x1在每个交易日的均值
        x1_mean = x1.groupby(level='TradingDay').mean()
        
        # 将均值广播到原始索引
        x1_mean_broadcast = pd.Series(
            index=x1.index,
            data=[x1_mean.loc[date] for date in x1.index.get_level_values('TradingDay')]
        )
        # 计算 (x1-mean(x1))*x2
        return (x1 - x1_mean_broadcast) * x2
    elif isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray):
        result = np.full_like(x1, np.nan)
        
        # 对于numpy数组,第一维是时间,第二维是股票
        if x1.ndim == 2 and x2.ndim == 2:
            for t in range(x1.shape[0]):
                # 计算每个时间点的截面均值
                valid_mask = ~np.isnan(x1[t, :])
                if np.sum(valid_mask) > 0:
                    x1_mean = np.mean(x1[t, valid_mask])
                    # 计算 (x1-mean(x1))*x2
                    result[t, valid_mask] = (x1[t, valid_mask] - x1_mean) * x2[t, valid_mask]
        
        return result
    return x1 * x2  # 默认情况下简单相乘

def regress_residual(x, y):
    """
    使用 NumPy 矩阵运算进行向量化的截面回归,实验发现比statmodel的算法快五倍
    
    返回:
    resid: 形状为 (time, assets) 的 2D numpy 数组,回归残差
    """
    if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
        raise ValueError("截面回归算子输入数据不是二维ndarray")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("输入数组必须是二维的")

    resid = np.full_like(y, np.nan)
    for t in range(x.shape[0]):
        valid_mask = (~np.isnan(x[t, :])) & (~np.isnan(y[t, :]))
        if np.sum(valid_mask) > 0:
            x_t = x[t, valid_mask]
            y_t = y[t, valid_mask]
            # 添加常数项
            X = np.column_stack([np.ones(x_t.shape[0]), x_t])
            # 使用 NumPy 的矩阵运算计算 OLS
            # (X'X)^(-1)X'y
            try:
                beta = np.linalg.solve(X.T @ X, X.T @ y_t)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(X.T @ X) @ (X.T @ y_t)
            # 计算拟合值和残差
            y_pred = X @ beta
            resid[t, valid_mask] = y_t - y_pred
    
    return resid

def sortreverse(x, n=10):
    """
    使用 NumPy 进行向量化的截面计算,X 截面前&后 n 名对应的 X 乘以-1

    返回:
    result: 形状为 (time, assets) 的 2D numpy 数组,是乘以-1之后的因子值
    """
    if not (isinstance(x, np.ndarray)):
        raise ValueError("截面回归算子输入数据不是二维ndarray")
    if x.ndim != 2:
        raise ValueError("输入数组必须是二维的")

    # 计算每一行非NaN的数量,并把width填充成x的形状
    width = np.sum(~np.isnan(x),axis=1)
    width = np.expand_dims(width, axis=1)
    width = np.repeat(width,x.shape[1], axis=1)

    # 每个值在当前截面上的排名
    ranking = np.argsort(np.argsort(x,axis=1,),axis=1)#这个排名其实把NaN排成最大的了
    small_mask = (ranking < n) #最小的n个值的位置
    large_mask = (ranking >= width - n) #最大的n个值的位置
    neg_mask = small_mask|large_mask
    return np.where(neg_mask, x*-1, x)

# def return_const_1(x):
#     return np.full_like(x,int(1))

def return_const_5(x):
    return np.full_like(x,int(5))

def return_const_10(x):
    return np.full_like(x,int(10))

def return_const_20(x):
    return np.full_like(x,int(20))

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
    pset.addPrimitive(power2, [ret_type], ret_type)
    pset.addPrimitive(power3, [ret_type], ret_type)
    pset.addPrimitive(curt, [ret_type], ret_type)
    pset.addPrimitive(inv, [ret_type], ret_type)
    pset.addPrimitive(mean2, [ret_type, ret_type], ret_type)
    
    # 添加数学函数算子
    pset.addPrimitive(log, [ret_type], ret_type)
    pset.addPrimitive(sqrt, [ret_type], ret_type)
    pset.addPrimitive(abs_val, [ret_type], ret_type)
    pset.addPrimitive(neg, [ret_type], ret_type)
    pset.addPrimitive(sigmoid, [ret_type], ret_type)
    pset.addPrimitive(hardsigmoid, [ret_type], ret_type)
    pset.addPrimitive(leakyrelu, [ret_type], ret_type)
    pset.addPrimitive(gelu, [ret_type], ret_type)
    pset.addPrimitive(sign, [ret_type], ret_type)

    # 添加条件算子
    pset.addPrimitive(if_then_else, [ret_type, ret_type, ret_type, ret_type], ret_type)
    pset.addPrimitive(series_max, [ret_type, ret_type], ret_type)
    
    # 添加返回常量的函数（权宜之计）
    # pset.addPrimitive(return_const_1, [ret_type], int)
    pset.addPrimitive(return_const_5, [ret_type], int)
    pset.addPrimitive(return_const_10, [ret_type], int)
    pset.addPrimitive(return_const_20, [ret_type], int)

    # 添加基本原始集元素 (常量等)
    add_basic_primitives(pset, int)
    
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
        pset.addPrimitive(ts_lag, [ret_type, int], ret_type)
        pset.addPrimitive(ts_diff, [ret_type, int], ret_type)
        pset.addPrimitive(ts_pct_change, [ret_type, int], ret_type)
        pset.addPrimitive(ts_mean, [ret_type, int], ret_type)
        pset.addPrimitive(ts_std, [ret_type, int], ret_type)
        pset.addPrimitive(ts_ewm, [ret_type, int], ret_type)
        pset.addPrimitive(ts_max, [ret_type, int], ret_type)
        pset.addPrimitive(ts_min, [ret_type, int], ret_type)
        pset.addPrimitive(ts_argmin, [ret_type, int], ret_type)
        pset.addPrimitive(ts_argmax, [ret_type, int], ret_type)
        pset.addPrimitive(ts_max_to_min, [ret_type, int], ret_type)
        pset.addPrimitive(ts_sum, [ret_type, int], ret_type)

        # !这五个都有两层循环,算起来可能很慢,用numba改进了四个
        pset.addPrimitive(ts_max_mean, [ret_type, int, int], ret_type)
        # pset.addPrimitive(ts_cov, [ret_type, ret_type, int], ret_type)
        pset.addPrimitive(ts_corr, [ret_type, ret_type, int], ret_type)
        pset.addPrimitive(ts_rankcorr, [ret_type, ret_type, int], ret_type)
        pset.addPrimitive(ts_to_wm, [ret_type, int], ret_type)

        pset.addPrimitive(ts_rank, [ret_type, int], ret_type)
        pset.addPrimitive(ts_median, [ret_type, int], ret_type)
    
    # 添加截面算子
    if include_cross_sectional:
        pset.addPrimitive(rank, [ret_type], ret_type)
        pset.addPrimitive(rank_div, [ret_type, ret_type], ret_type)
        pset.addPrimitive(rank_sub, [ret_type, ret_type], ret_type)
        pset.addPrimitive(rank_mul, [ret_type, ret_type], ret_type)
        pset.addPrimitive(zscore, [ret_type], ret_type)
        pset.addPrimitive(min_max_scale, [ret_type], ret_type)
        pset.addPrimitive(umr, [ret_type, ret_type], ret_type)
        pset.addPrimitive(regress_residual, [ret_type, ret_type], ret_type)
        pset.addPrimitive(sortreverse, [ret_type, int], ret_type)
    
    return pset

def calculate_expression(individual, pset, feature_data):
    """
    评估表达式
    
    参数:
        individual: DEAP个体
        pset: DEAP原始集
        feature_data: 特征数据字典,键为特征名,值为股票*时间的2D nparray
        
    返回:
        表达式计算结果
    """
    # 编译表达式
    func = gp.compile(individual, pset)
    # 准备输入数据
    inputs = [feature_data[arg] for arg in pset.arguments]

    # 评估表达式
    try:
        # 执行表达式计算
        result = func(*inputs)
        
        # 处理无效值
        if np.isscalar(result):
            return np.zeros_like(inputs[0])

        elif isinstance(result, pd.Series):
            # 对于pandas Series,使用pandas方法处理无效值并转化成np array
            result = result.fillna(0.0)
            result = result.replace([np.inf, -np.inf], 0.0)
            result = result.values 

        elif isinstance(result, np.ndarray):
            # 对于numpy数组,使用numpy方法处理无效值
            result[np.isnan(result)] = 0.0
            result[np.isinf(result)] = 0.0
            
        autocorr_mtx = np.corrcoef(result, rowvar=False)
        autocorr_5d = np.nanmean(np.diag(autocorr_mtx, k=5))

        import bottleneck as bn
        if autocorr_5d < 0.7:
            return bn.move_mean(result, window=20, min_count=1, axis=0)
        else:
            return result

    except Exception as e:
        import traceback,sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(f"表达式评估错误: {e}")
        for tb in traceback.extract_tb(exc_traceback):
            print(f"  File: {tb.filename}, Line: {tb.lineno}, Context: {tb.line}")

        # 返回全零数组或零值
        if len(inputs) > 0:
            if isinstance(inputs[0], pd.Series):
                return pd.Series(0.0, index=inputs[0].index)
            elif isinstance(inputs[0], np.ndarray):
                return np.zeros_like(inputs[0])
        return 0.0
