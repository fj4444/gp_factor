"""
RAPIDS数据处理器实现

使用RAPIDS (cuDF/CuPy) 实现GPU加速的数据处理功能
"""

import numpy as np
import pandas as pd
import cudf
import cupy as cp
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings

from .processor_base import DataProcessorBase

class RapidsDataProcessor(DataProcessorBase):
    """
    RAPIDS数据处理器
    
    使用RAPIDS (cuDF/CuPy) 实现GPU加速的数据处理功能
    """
    
    def __init__(self, use_gpu=True):
        """
        初始化RAPIDS数据处理器
        
        参数:
            use_gpu: 是否使用GPU加速
        """
        super().__init__(use_gpu=True)  # RAPIDS必须使用GPU
        print("RapidsDataProcessor 将使用 GPU 加速")
    
    def transform_data(self, data, barra_data=None):
        """
        转换数据
        
        参数:
            data: 数据源,可以是文件路径(str)或已加载的数据框(DataFrame)
            
        返回:
            转换后的数据框
        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            # 转换为cuDF DataFrame
            self.data = cudf.DataFrame.from_pandas(data)
        elif isinstance(data, cudf.DataFrame):
            self.data = data.copy()
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")
            
        print(f"生成cuDF数据完成,形状: {self.data.shape}")
        return self.data
    
    def preprocess_data(self, time_col, id_col, drop_na_threshold=None):
        """
        预处理数据,涉及缺失值填充,先保持DataFrame形式
        
        参数:
            fillna_method: 填充缺失值的方法,可选 'forward', 'backward', 'mean', 'median', 'zero', None
            drop_duplicates: 是否删除重复行
            drop_na_threshold: 删除缺失值比例超过阈值的列,None 表示不删除
            
        返回:
            预处理后的数据框
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 复制数据,避免修改原始数据
        data = self.data.copy()
        
        # 删除缺失值比例超过阈值的列
        if drop_na_threshold is not None:
            # cuDF 版本
            na_ratio = data.isnull().mean()._index_map(lambda x: x > drop_na_threshold)
            cols_to_drop = na_ratio[na_ratio].index.to_pandas().tolist()
            
            if cols_to_drop:
                data = data.drop(columns=cols_to_drop)
                print(f"删除缺失值比例超过 {drop_na_threshold} 的列: {cols_to_drop}")
                print(f"删除后,形状: {data.shape}")
        
        # TODO 填充缺失值
        
        # 此时剩下缺失值应该都是rolling窗口导致的,应该整行删掉
        data = data.dropna(axis=0)
        print(f"删除缺失行后,剩余行数: {len(data)}")
        
        self.data = data
        return self.data

    def split_data(self, test_size=0.3, time_series=True, time_col=None, id_col=None, random_state=42):
        """
        分割数据为训练集和测试集
        
        参数:
            test_size: 测试集比例
            time_series: 是否按时间序列分割
            time_col: 时间列名,仅在 time_series=True 时使用
            random_state: 随机种子,仅在 time_series=False 时使用
            id_col: 标的列名,让InnerCode从小到大排序
            
        返回:
            (训练集, 测试集) 元组
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        if self.features is None or self.target is None:
            raise ValueError("请先设置特征和目标")
        
        if id_col is None or id_col not in self.data.columns:
            raise ValueError("InnerCode列不存在或未指定")

        if time_series:
            if time_col is None:
                raise ValueError("按时间序列分割时,必须指定时间列")
            
            if time_col not in self.data.columns:
                raise ValueError(f"时间列 {time_col} 不存在")

            #TODO:这部分是从pandas_processor直接搬过来的,没有针对cudf做适配
            # 按时间排序并获取唯一日期
            data_sorted = self.data.sort_values(by=[time_col,id_col])
            unique_dates = data_sorted[time_col].unique()
            
            # 按日期分割,确保完整日期在训练集或测试集
            split_date_idx = int(len(unique_dates) * (1 - test_size))
            train_dates = unique_dates[:split_date_idx]
            test_dates = unique_dates[split_date_idx:]
            
            # 分割数据
            self._train_data = data_sorted[data_sorted[time_col].isin(train_dates)]
            self._test_data = data_sorted[data_sorted[time_col].isin(test_dates)]

        else:
            # 随机分割
            # cuDF 版本
            # 生成随机索引
            indices = cp.random.permutation(len(self.data))
            split_idx = int(len(indices) * (1 - test_size))
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]
            
            # 分割数据
            self._train_data = self.data.iloc[train_indices]
            self._test_data = self.data.iloc[test_indices]
        
        print(f"训练集形状: {self._train_data.shape}")
        print(f"测试集形状: {self._test_data.shape}")
        
        # 准备时间序列,特征矩阵和目标向量
        self._prepare_feature_dict(time_col=time_col,id_col=id_col)
    
    def _prepare_feature_dict(self, time_col='TradingDay', id_col='InnerCode'):

        """准备时间序列,特征字典和目标向量"""
        if self._train_data is None or self._test_data is None:
            raise ValueError("请先分割数据")

        self.time_train = self._train_data[time_col].values
        self.time_test = self._test_data[time_col].values
        
        # 提取特征和目标,转换为CuPy数组
        self.X_train = self._train_data[self.features].values
        self.y_train = self._train_data[self.target].values
        self.X_test = self._test_data[self.features].values
        self.y_test = self._test_data[self.target].values
        
        # 保存原始索引信息,以便在需要时重建DataFrame
        self.train_indices = {
            'time': self._train_data[time_col].values,
            'id': self._train_data[id_col].values
        }
        self.test_indices = {
            'time': self._test_data[time_col].values,
            'id': self._test_data[id_col].values
        }
        
