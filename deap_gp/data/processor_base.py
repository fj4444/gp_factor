"""
数据处理器基类

定义数据处理器的通用接口，被具体实现继承
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

class DataProcessorBase(ABC):
    """
    数据处理器基类
    
    定义数据处理器的通用接口，被具体实现继承
    """
    
    def __init__(self, use_gpu=False):
        """
        初始化数据处理器
        
        参数:
            use_gpu: 是否使用GPU加速
        """
        self.use_gpu = use_gpu
        self.data = None #dataframe,无index
        self.barra_data = None #无index的dataframe, 目前应该只需要liquidity和rv两个风格因子
        self.barra_train = None #dict, key是barra风格因子名,value是特征值的2D numpy array,维度分别是日期和股票
        self.barra_test = None #dict, key是barra风格因子名,value是特征值的2D numpy array,维度分别是日期和股票
        self.time_train = None #np array,训练集的时间序列
        self.time_test = None #np array
        self.features = None #list,包含feature的名字
        self.target = None #str,预测目标列的列名
        self.X_train = None #dict, key是改名之后的feature名(x1,x2...),value是特征值的2D numpy array,维度分别是日期和股票
        self.X_test = None #同上
        self.y_train = None #预测目标的2D numpy array,维度分别是日期和股票
        self.y_test = None
        self.time_index = None #把测试集的原始dataframe数据转换成2D ndarray之前，先保存下来2D数据对应的行和列标签，最后hof个体的数据还得变回去
        self.code_columns = None
        self.weights_data = None #WLS权重(市值开根号)的2D numpy array,维度分别是日期和股票
        self.weights_train = None
        self.weights_test = None

        # 私有成员
        self._train_data = None #dataframe,无index
        self._test_data = None #dataframe,无index
        self._barra_train_data = None #dataframe,无index
        self._barra_test_data = None #dataframe,无index
        self._weights_train_data = None
        self._weights_test_data = None
    
    @abstractmethod
    def transform_data(self, data, barra_data, weights_data):
        """
        转换数据
        
        参数:
            data: 原始特征数据，应该是dataframe
            barra_data: 风格因子数据，应该是dataframe 
            
        返回:
            无
        """
        pass
    
    @abstractmethod
    def preprocess_data(self, time_col, id_col, fillna_method='forward', drop_duplicates=True, drop_na_threshold=None):
        """
        预处理数据
        
        参数:
            fillna_method: 填充缺失值的方法
            drop_duplicates: 是否删除重复行
            drop_na_threshold: 删除缺失值比例超过阈值的列
            
        返回:
            预处理后的数据
        """
        pass
    
    def set_features_and_target(self, features, target):
        """
        设置特征和目标
        
        参数:
            features: 特征列名列表
            target: 目标列名
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        # 检查目标列是否存在
        if target not in self.data.columns:
            raise ValueError(f"目标列{target}不存在")

        # 根据特征列名list和输入数据中包含的列的交集确定本次遗传规划使用的特征列
        config_feature_set = set(features)
        data_feature_set = set(self.data.columns)
        common_features = list(config_feature_set.intersection(data_feature_set))
        # 把从set转换回来的common features列表重新按照原始feature_cols列表的顺序排列，这样每次运行规划程序时，特征的别名(x1,x2)和特征的原名的对应关系都能尽量保持一致
        common_features = sorted(common_features, key=lambda x: features.index(x))
        config_only_features = list(config_feature_set - data_feature_set)
        data_only_features = list(data_feature_set - config_feature_set)
        
        self.features = common_features
        self.target = target
        print(f"设置特征: {self.features}")
        print(f"设置目标: {self.target}")
        print(f"输入数据中还有 {data_only_features} 未作为特征")
        if len(config_only_features) > 0:
            print(f"配置中的 {config_only_features} 在输入数据中不存在")
    
    @abstractmethod
    def split_data(self, test_size=0.3, time_series=True, time_col=None, random_state=42):
        """
        分割数据为训练集和测试集
        
        参数:
            test_size: 测试集比例
            time_series: 是否按时间序列分割
            time_col: 时间列名
            random_state: 随机种子
            
        返回:
            (训练集, 测试集) 元组
        """
        pass
    
    @abstractmethod
    def _prepare_feature_dict(self):
        pass
    
    def get_train_feature_dict(self):
        if self.X_train is None:
            raise ValueError("请先分割数据")
        return self.X_train
    
    def get_train_target_vector(self):
        if self.y_train is None:
            raise ValueError("请先分割数据")
        return self.y_train
    
    def get_test_feature_dict(self):
        if self.X_test is None:
            raise ValueError("请先分割数据")
        return self.X_test
    
    def get_test_target_vector(self):
        if self.y_test is None:
            raise ValueError("请先分割数据")
        return self.y_test

    def get_barra_train(self):
        if self.barra_train is None:
            raise ValueError("请先分割数据")
        return self.barra_train

    def get_barra_test(self):
        if self.barra_test is None:
            raise ValueError("请先分割数据")
        return self.barra_test

    def get_weights_train(self):
        if self.weights_train is None:
            raise ValueError("请先分割数据")
        return self.weights_train

    def get_weights_test(self):
        if self.weights_test is None:
            raise ValueError("请先分割数据")
        return self.weights_test

    def get_train_time_vector(self):
        if self.time_train is None:
            raise ValueError("请先分割数据")
        return self.time_train

    def get_test_time_vector(self):
        if self.time_test is None:
            raise ValueError("请先分割数据")
        return self.time_test

    def get_feature_list(self):
        if self.features is None:
            raise ValueError("请先设置特征")
        return self.features

    def get_dataset_index(self):
        if self.time_index is None:
            raise ValueError("请先设置特征")
        return self.time_index

    def get_dataset_column(self):
        if self.code_columns is None:
            raise ValueError("请先设置特征")
        return self.code_columns