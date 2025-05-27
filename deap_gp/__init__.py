"""
DEAP-GP: 基于DEAP库实现的遗传规划因子挖掘框架

主要模块:
- data: 数据处理模块
- cpu: CPU实现的算子和适应度计算
- gpu: GPU实现的算子和适应度计算
- base: 基础组件
- core: 核心算法实现
"""

import warnings

__version__ = '0.1.0'

# 定义优化flag
TORCH_AVAILABLE = False
NUMBA_AVAILABLE = False
RAPIDS_AVAILABLE = False

# 尝试导入优化库
try:
    import torch
    TORCH_AVAILABLE = True
    print("PyTorch 导入成功")
except ImportError:
    warnings.warn("PyTorch 不可用，gpu计算可能无法加速")

# 尝试导入 RAPIDS 库
try:
    import cudf
    import cupy as cp
    RAPIDS_AVAILABLE = True
except ImportError:
    warnings.warn("RAPIDS 不可用，gpu计算可能无法加速")

try:
    import numba as nb
    NUMBA_AVAILABLE = True
except ImportError:
    warnings.warn("Numba 不可用，部分加速将失效")


# 设备选择函数
def select_device_strategy(use_gpu=False):
    """
    根据配置和可用性选择设备策略
    
    参数:
        use_gpu: 是否使用GPU
        
    返回:
        device_info: 包含设备信息的字典
    """
    if use_gpu and TORCH_AVAILABLE:
        from .gpu import operators, fitness
        device_info = {
            'device': 'gpu',
            'operators_module': operators,
            'fitness_module': fitness
        }
        
        if RAPIDS_AVAILABLE:
            from .data.rapids_processor import RapidsDataProcessor
            device_info['data_processor'] = RapidsDataProcessor
        else:
            from .data.pandas_processor import PandasDataProcessor
            device_info['data_processor'] = PandasDataProcessor
            warnings.warn("RAPIDS不可用,将在GPU模式下使用Pandas进行数据处理")
    else:
        from .cpu import operators, fitness
        from .data.pandas_processor import PandasDataProcessor
        
        device_info = {
            'device': 'cpu',
            'operators_module': operators,
            'fitness_module': fitness,
            'data_processor': PandasDataProcessor
        }
        
        if use_gpu:
            warnings.warn("不使用GPU或PyTorch不可用,将使用CPU模式")
    
    return device_info

# 导出主要接口
from .core import main
from .cpu import operators as cpu_ops
from .gpu import operators as gpu_ops
