"""
遗传规划统一入口脚本 (重构版)

提供多种数据输入方式：
1. 从文件加载数据
2. 从数据库加载数据
3. 直接接收DataFrame数据

集成了 PyTorch、Numba 和 RAPIDS 优化,并根据配置自动选择最佳实现
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import connectorx as cx
import pickle
import time
from typing import Dict, List, Any, Optional, Union, Tuple

# 导入配置模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import get_config

# 导入核心算法模块
from deap_gp import main as gp_main

# 导入特征工程模块
from feature_engineering import feature_engineering, clear_data

def load_data_from_db(start_date=None, end_date=None, connection_string=None):
    """
    从数据库加载因子计算所需的数据
    
    参数:
        start_date: 开始日期,格式为 'YYYY-MM-DD'
        end_date: 结束日期,格式为 'YYYY-MM-DD'
        connection_string: 数据库连接字符串
        
    返回:
        加载的DataFrame
    """
    # 获取配置
    db_config = get_config('db')
    
    # 使用参数或配置中的值
    conn = connection_string or db_config['connection_string']
    start_date = start_date or db_config['default_start_date']
    end_date = end_date or db_config['default_end_date']
    
    print(f"开始从数据库加载数据,时间范围: {start_date} 至 {end_date}")
    # 非交易日全部去掉,停牌和ST暂时留下,待计算完预测目标列(未来若干天的累积收益率后再手动删去)
    sql_query = f'''
        SELECT 
            rd.TradingDay, rd.SecuCode, rd.InnerCode, 
            rd.OpenPrice, rd.ClosePrice, rd.HighPrice, rd.LowPrice, rd.PrevClosePrice,
            rd.TurnoverVolume, rd.TurnoverValue, rd.NonRestrictedShares, rd.DailyReturn, rd.IfSuspended, rd.IfSpecialTrade,
            idx.ClosePrice as MarketClosePrice, idx.PrevClosePrice as MarketPrevClosePrice, idx.ChangePCT as MarketChangePCT
        FROM smartquant.ReturnDaily rd
        LEFT JOIN jydb.QT_IndexQuote idx ON rd.TradingDay = idx.TradingDay AND idx.InnerCode = 4089
        WHERE
            rd.TradingDay BETWEEN '{start_date}' AND '{end_date}'
            AND rd.IfTradingDay = 1
        ORDER BY rd.TradingDay, rd.InnerCode;
    '''
    print("正在从数据库加载日频数据...")
    data = cx.read_sql(conn, sql_query)
    
    # 转换TradingDay为datetime
    data['TradingDay'] = pd.to_datetime(data['TradingDay'])
    return data

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='遗传规划因子挖掘统一入口 (重构版)')
    
    # 从配置中获取默认值
    data_config = get_config('data')
    gp_config = get_config('gp')
    gpu_config = get_config('gpu')
    path_config = get_config('path')
    db_config = get_config('db')
    barra_config = get_config('barra')
    
    # 数据源参数
    data_source = parser.add_argument_group('数据源参数')
    data_source.add_argument('--data_file', type=str, default=os.path.join(os.getcwd(),'features.parquet'), help='数据文件路径')
    data_source.add_argument('--use_db', action='store_true', help='是否从数据库加载数据')
    data_source.add_argument('--use_barra', action='store_true', help='是否加载风格因子')
    
    # 数据库参数
    db_group = parser.add_argument_group('数据库参数 (仅在 --use_db 时使用)')
    db_group.add_argument('--connection_string', type=str, default=db_config['connection_string'], help='数据库连接字符串')

    # 风格因子参数
    barra_group = parser.add_argument_group('风格因子参数 (仅在 --use_barra 时使用)')
    barra_group.add_argument('--barra_file', type=str, default=barra_config['barra_file'], help='风格因子文件路径')
    barra_group.add_argument('--barra_usage', type=str, default=barra_config['barra_usage'], choices=['correlation','neutralize'], help='barra风格惩罚的用法, 相关系数惩罚或者回归中性化')
    barra_group.add_argument('--weights_file', type=str, default=barra_config['weights_file'], help='权重文件路径')
    
    # 数据参数
    data_group = parser.add_argument_group('数据参数')
    data_group.add_argument('--start_date', type=str, default=data_config['default_start_date'], help='开始日期,格式为 YYYY-MM-DD')
    data_group.add_argument('--end_date', type=str, default=data_config['default_end_date'], help='结束日期,格式为 YYYY-MM-DD')
    data_group.add_argument('--time_col', type=str, default=data_config['time_col'], help='时间列名')
    data_group.add_argument('--id_col', type=str, default=data_config['id_col'], help='证券编号列名')
    data_group.add_argument('--return_col', type=str, default=data_config['return_col'], help='收益率列名')
    data_group.add_argument('--target_col', type=str, default=data_config['target_col'], help='预测目标列名')
    data_group.add_argument('--feature_cols', type=str, nargs='+', default=data_config['feature_cols'], help='特征列名列表')
    data_group.add_argument('--cumulative_window', type=int, default=data_config['cumulative_window'], help='预测目标延迟期数')
    data_group.add_argument('--test_size', type=float, default=data_config['test_size'], help='测试集比例')
    data_group.add_argument('--feature_engineer', action='store_true', help='是否进行特征工程')
    
    # 遗传规划参数
    gp_group = parser.add_argument_group('遗传规划参数')
    gp_group.add_argument('--use_warm_start', action='store_true', help='是否使用热启动')
    gp_group.add_argument('--population_size', type=int, default=gp_config['population_size'], help='种群大小')
    gp_group.add_argument('--max_generations', type=int, default=gp_config['max_generations'], help='最大代数')
    gp_group.add_argument('--max_depth', type=int, default=gp_config['max_depth'], help='树的最大深度')
    gp_group.add_argument('--parsimony',action='store_true', help='是否进行包含复杂度压力的双重锦标赛')
    gp_group.add_argument('--tournament_size', type=int, default=gp_config['tournament_size'], help='锦标赛选择的大小')
    gp_group.add_argument('--crossover_rate', type=float, default=gp_config['crossover_rate'], help='交叉概率')
    gp_group.add_argument('--mutation_rate', type=float, default=gp_config['mutation_rate'], help='变异概率')
    gp_group.add_argument('--shrink_mutation_rate', type=float, default=gp_config['shrink_mutation_rate'], help='提升变异在变异操作中的概率')
    #如果mutation_rate设置为0.4,shrink_mutation_rate设置为0.25,则最终会有0.6的交叉概率, 0.3的子树变异概率, 0.1的提升变异概率
    gp_group.add_argument('--hall_of_fame_size', type=int, default=gp_config['hall_of_fame_size'], help='名人堂大小')
    gp_group.add_argument('--patience', type=int, default=gp_config['patience'], help='早停轮次限制')
    gp_group.add_argument('--min_delta', type=float, default=gp_config['min_delta'], help='早停提升度限制')
    gp_group.add_argument('--correlation_threshold_init', type=float, default=gp_config['correlation_threshold_init'], help='初始种群热启动相关性阈值,超过此值的因子将被过滤')
    gp_group.add_argument('--correlation_threshold', type=float, default=gp_config['correlation_threshold'], help='迭代过程中因子相关性阈值,超过此值的因子将被过滤')
    
    # 适应度参数
    fitness_group = parser.add_argument_group('适应度参数')
    fitness_group.add_argument('--fitness_metric', type=str, default='ic', choices=['ic','icir','NDCG','double'], help='适应度指标')
    fitness_group.add_argument('--fitness_quantiles', type=int, default=10, help='适应度指标计算中的分组数 (仅用于NDCG指标)')
    
    # 输出参数
    output_group = parser.add_argument_group('输出参数')
    output_group.add_argument('--output_dir', type=str, default=path_config['output_dir'], help='输出目录')
    output_group.add_argument('--experiment_name', type=str, default=f"deap_gp_experiment_{time.strftime('%Y%m%d_%H%M%S')}", help='实验名称')
    
    # 并行参数
    parallel_group = parser.add_argument_group('并行参数')
    parallel_group.add_argument('--n_jobs', type=int, default=1, help='并行进程数')
    
    # 优化参数
    optimization_group = parser.add_argument_group('优化参数')
    optimization_group.add_argument('--use_gpu', action='store_true', help='是否使用GPU加速 (需要PyTorch或RAPIDS)')
    optimization_group.add_argument('--batch_size', type=int, default=gpu_config['batch_size'], help='批处理大小 (用于GPU加速)')
    
    # 其他参数
    other_group = parser.add_argument_group('其他参数')
    other_group.add_argument('--random_seed', type=int, default=gp_config['random_seed'], help='随机种子')
    other_group.add_argument('--verbose', action='store_true', help='是否输出详细信息')
    
    return parser.parse_args()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    args = parse_args()
 
    if not args.use_db and args.data_file is None:
        raise ValueError("必须指定数据源 (--data_file 或 --use_db)")

    # 加载数据
    if args.use_db:
    # 从数据库加载数据
        db_data = load_data_from_db(
            start_date=args.start_date,
            end_date=args.end_date,
            connection_string=args.connection_string
        )

    if args.data_file:
        # 从文件加载数据
        file_path = args.data_file
        
        # 根据文件扩展名确定加载方法
        if file_path.endswith('.csv'):
            file_data = pd.read_csv(file_path,dtype={'SecuCode':str})
        elif file_path.endswith('.parquet'):
            file_data = pd.read_parquet(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")

        file_data = file_data[(file_data['TradingDay']>pd.to_datetime(args.start_date))&(file_data['TradingDay']<=pd.to_datetime(args.end_date))]

    # 合并特征数据
    data = pd.DataFrame()
    if args.use_db and args.data_file is not None:
        data = pd.merge(db_data, file_data, how="inner", on=['TradingDay','SecuCode','IfSuspended','DailyReturn'], suffixes=('','_adj'))

    elif args.use_db:
        data = db_data
    else:
        data = file_data

    # 先针对parquet数据已知的情况做一些简单处理,这里的数据包括ST,应该删掉,不过删除操作全部合并到一起了，放在特征工程的之后
    # 20个runner特征里,在ST标的之外的非停牌日也有不少NaN,因为已经zscore标准化,这些NaN填0
    runner_features = [feature_name for feature_name in data.columns if 'runner' in feature_name]
    mask = data['IfSuspended'] == 0
    data.loc[mask,runner_features]=np.nan_to_num(data.loc[mask,runner_features],nan=0.0)
    # S M B SB四个特征在2015-05-01之前全部为空, 之后的空值则全部出现在ST和停牌上
    # 如果规划涉及早年的数据,这四个特征不加入训练
    if pd.to_datetime(args.start_date)<=pd.to_datetime('2015-04-30'):
        args.feature_cols = [feature for feature in args.feature_cols if feature not in ['S','M','B','SB']]
    # 否则防御性地填0,虽然应该靠停牌和ST的筛选把NaN都筛掉了
    else:
        data.loc[mask,['S','M','B','SB']]=np.nan_to_num(data.loc[mask,['S','M','B','SB']],nan=0.0)

    if args.feature_engineer:
        # 计算特征
        data = feature_engineering(args, data)
    else:
        print("跳过特征工程")
        data = clear_data(data)

    if args.use_barra:
        # 获取barra数据
        barra_file_data = pd.read_csv(args.barra_file,dtype={'SecuCode':str})
        barra_file_data['TradingDay'] = pd.to_datetime(barra_file_data['TradingDay'])
        barra_file_data = barra_file_data[(barra_file_data['TradingDay']>pd.to_datetime(args.start_date))&(barra_file_data['TradingDay']<=pd.to_datetime(args.end_date))]

        weights_file_data = pd.read_csv(args.weights_file,dtype={'SecuCode':str})
        weights_file_data['TradingDay'] = pd.to_datetime(weights_file_data['TradingDay'])
        weights_file_data = weights_file_data[(weights_file_data['TradingDay']>pd.to_datetime(args.start_date))&(weights_file_data['TradingDay']<=pd.to_datetime(args.end_date))]

        # 合并barra数据和特征数据，保持两者条目对应
        data_columns = data.columns
        barra_data_columns = barra_file_data.columns
        weighst_data_columns = weights_file_data.columns
        data_with_barra = pd.merge(data,barra_file_data, how='left', on=['TradingDay','InnerCode','SecuCode'])
        data_with_barra_n_weights = pd.merge(data_with_barra, weights_file_data, how='left', on=['TradingDay','InnerCode','SecuCode'])
        data = data_with_barra_n_weights[data_columns]
        barra_data = data_with_barra_n_weights[barra_data_columns]
        weights_data = data_with_barra_n_weights[weighst_data_columns]
        args.barra_data = barra_data
        args.weights_data = weights_data

    args.data = data

    # 运行遗传规划
    hof, log = gp_main(args)
