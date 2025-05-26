"""
数据处理模块

包含:
- processor_base: 数据处理器基类
- pandas_processor: Pandas实现的数据处理器
- rapids_processor: RAPIDS实现的数据处理器
"""

from .processor_base import DataProcessorBase
from .pandas_processor import PandasDataProcessor

# 尝试导入RAPIDS实现
try:
    from .rapids_processor import RapidsDataProcessor
    RAPIDS_PROCESSOR_AVAILABLE = True
except ImportError:
    RAPIDS_PROCESSOR_AVAILABLE = False
