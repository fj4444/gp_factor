"""
Pandas数据处理器实现

使用Pandas实现数据处理功能
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import warnings

from .processor_base import DataProcessorBase

class PandasDataProcessor(DataProcessorBase):
    """
    Pandas数据处理器
    
    使用Pandas实现数据处理功能
    """
    
    def __init__(self, use_gpu=False):
        """
        初始化Pandas数据处理器
        
        参数:
            use_gpu: 是否使用GPU加速 (对Pandas实现无效)
        """
        super().__init__(use_gpu=False)  # Pandas不支持GPU加速
        print("PandasDataProcessor 将使用 CPU")
    
    def transform_data(self, data, barra_data=None, weights_data=None):
        """
        转换数据
        
        参数:
            data: 原始特征数据，应该是dataframe
            barra_data: 风格因子数据，应该是dataframe 
            weights_data: 用于风格中性化WLS回归的权重，应该是dataframe
            
        返回:
            无
        """
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            raise TypeError(f"data是不支持的数据类型: {type(data)}")

        if barra_data is not None:
            if isinstance(barra_data, pd.DataFrame):
                self.barra_data = barra_data.copy()
            else:
                raise TypeError(f"barra_data是不支持的数据类型: {type(barra_data)}")
            
            if isinstance(weights_data, pd.DataFrame):
                self.weights_data = weights_data.copy()
            else:
                raise TypeError(f"weights_data是不支持的数据类型: {type(weights_data)}")
        
        print(f"加载DataFrame数据完成, 数据形状: {self.data.shape}")

    def split_data(self, test_size=0.3, time_series=True, time_col=None, id_col=None):
        """
        分割特征数据和风格因子数据为训练集和测试集
        
        参数:
            test_size: 测试集比例
            time_series: 是否按时间序列分割
            time_col: 时间列名，仅在 time_series=True 时使用
            id_col: 标的列名，让InnerCode从小到大排序
            
        返回:
            无
        """

        if self.data is None:
            raise ValueError("请先加载数据")
        
        if self.features is None or self.target is None:
            raise ValueError("请先设置特征和目标")
        
        if id_col is None or id_col not in self.data.columns:
            raise ValueError("InnerCode列不存在或未指定")

        if time_series:
            if time_col is None:
                raise ValueError("按时间序列分割时，必须指定时间列")
            
            if time_col not in self.data.columns:
                raise ValueError(f"时间列 {time_col} 不存在")
            
            # 按时间排序并获取唯一日期
            data_sorted = self.data.sort_values(by=[time_col,id_col])
            unique_dates = data_sorted[time_col].unique()
            
            # 按日期分割，确保完整日期在训练集或测试集
            split_date_idx = int(len(unique_dates) * (1 - test_size))
            train_dates = unique_dates[:split_date_idx]
            test_dates = unique_dates[split_date_idx:]
            
            # 分割数据
            self._train_data = data_sorted[data_sorted[time_col].isin(train_dates)]
            self._test_data = data_sorted[data_sorted[time_col].isin(test_dates)]

            if self.barra_data is not None:
                barra_data_sorted = self.barra_data.sort_values(by=[time_col,id_col])
                self._barra_train_data = barra_data_sorted[barra_data_sorted[time_col].isin(train_dates)]
                self._barra_test_data = barra_data_sorted[barra_data_sorted[time_col].isin(test_dates)]

                weights_data_sorted = self.weights_data.sort_values(by=[time_col,id_col])
                self._weights_train_data = weights_data_sorted[weights_data_sorted[time_col].isin(train_dates)]
                self._weights_test_data = weights_data_sorted[weights_data_sorted[time_col].isin(test_dates)]
        else:
            # 滚动分割
            pass
        
        print(f"训练集形状: {self._train_data.shape}")
        print(f"测试集形状: {self._test_data.shape}")

        # 准备时间序列,特征矩阵和目标向量
        self._prepare_feature_dict(time_col= time_col, id_col= id_col)
    
    def _prepare_feature_dict(self, time_col=None, id_col=None):
        """准备特征字典和目标向量"""
        if self._train_data is None or self._test_data is None:
            raise ValueError("请先分割数据")

        if time_col is None or id_col is None:
            warnings.warn("特征字典和目标向量未包含时间和标的序列")

        # 创建时间索引
        self.time_train = self._train_data[time_col].values
        self.time_test = self._test_data[time_col].values

        # 创建特征字典
        self.X_train = {}
        self.X_test = {}

        for i, col in enumerate(self.features):
            # 训练集特征
            pivot_train = self._train_data.pivot_table(
                values=col, index=time_col, columns=id_col, aggfunc='first')
            self.X_train[f'x{i+1}'] = pivot_train.values
            
            # 测试集特征
            pivot_test = self._test_data.pivot_table(
                values=col, index=time_col, columns=id_col, aggfunc='first')
            self.X_test[f'x{i+1}'] = pivot_test.values
            if i == 0:
                # full_data = self.data.sort_values(by=[time_col,id_col])
                # pivot_full = full_data.pivot_table(values=col, index=time_col, columns='SecuCode', aggfunc='first')
                self.time_index = pivot_test.index
                self.code_columns = pivot_test.columns
            

        # 目标向量
        self.y_train = self._train_data.pivot_table(
            values=self.target, index=time_col, columns=id_col, aggfunc='first').values
        
        self.y_test = self._test_data.pivot_table(
            values=self.target, index=time_col, columns=id_col, aggfunc='first').values



        if self.barra_data is not None:
            self.barra_train = {}
            self.barra_test = {}
            barra_factor_names = [col for col in self.barra_data.columns if col not in['TradingDay','SecuCode','InnerCode']]
            for i, col in enumerate(barra_factor_names):
                # 风格因子
                pivot_train = self._barra_train_data.pivot_table(
                    values=col, index=time_col, columns = id_col, aggfunc='first')
                self.barra_train[col] = pivot_train.values

                pivot_test = self._barra_test_data.pivot_table(
                    values=col, index=time_col, columns = id_col, aggfunc='first')
                self.barra_test[col] = pivot_test.values
            
            # 风格中性化的权重数据
            self.weights_train = self._weights_train_data.pivot_table(
                values='weights', index=time_col, columns = id_col, aggfunc='first').values

            self.weights_test = self._weights_test_data.pivot_table(
                values='weights', index=time_col, columns = id_col, aggfunc='first').values
            
