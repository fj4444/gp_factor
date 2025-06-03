"""
特征工程模块

提供各种特征计算函数,包括：
1. 基础特征计算
2. 滚动窗口特征计算
3. 技术指标计算
"""

import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any, Optional, Union, Tuple

from .hf_feature import calculate_arpp, calculate_rskew, calculate_rvol, calculate_vhhi
from .apb import apb

def max_n_mean(series, window=20, n=3):
    """计算滚动窗口内最大n个值的均值"""
    return series.rolling(window=window, min_periods=n).apply(lambda x: np.mean(np.sort(x)[-n:]), raw=True)

def calc_ivol(returns, market_returns, window=20):
    """计算特质波动率"""
    # 使用滚动窗口计算beta
    def rolling_beta(y, x, window):
        # 计算协方差
        cov = (y * x).rolling(window, min_periods=2).mean() - y.rolling(window, min_periods=2).mean() * x.rolling(window, min_periods=2).mean()
        # 计算方差
        var = x.rolling(window, min_periods=2).var()
        return cov / var
    
    # 计算beta
    beta = rolling_beta(returns, market_returns, window)
    
    # 计算残差
    residuals = returns - beta * market_returns

    # 计算残差的年化标准差
    return residuals.rolling(window, min_periods=2).std()*np.sqrt(243)

def process_group(group_data, market_returns):
    """处理单个股票组的时序特征"""
    result = pd.DataFrame(index=group_data.index)
    
    group_data.loc[group_data['IfSuspended']==1,['DailyReturn','ClosePrice','lnto','amihud']]=np.nan
    # 计算各种滚动特征
    result['ret20'] = group_data['DailyReturn'].rolling(window=20, min_periods=1).sum() #停牌DailyReturn=0 不影响
    result['vol20'] = group_data['DailyReturn'].rolling(window=20, min_periods=2).std() 
    result['ppreversal'] = group_data['ClosePrice_adj'].rolling(window=5, min_periods=1).mean() / group_data['ClosePrice_adj'].rolling(window=60, min_periods=1).mean() - 1 #closeprice没有空值也没有0, 停牌不影响
    result['maxret20'] = max_n_mean(group_data['DailyReturn'], 20, 3)
    
    # 特质波动率
    group_market_returns = market_returns.loc[group_data.index]
    result['ivol20'] = calc_ivol(group_data['DailyReturn'], group_market_returns, 20)

    # 特异度
    result['ivr20'] = result['ivol20'] / result['vol20']
    
    # 换手率对数均值
    result['lnto_20d'] = group_data['lnto'].rolling(window=20, min_periods=1).mean()
    
    # Amihud非流动性对数
    result['lnamihud20'] = np.log(group_data['Amihud'].rolling(window=20, min_periods=1).mean()) # amihud带绝对值
    
    return result

def process_group_wrapper(args):
    """多进程包装函数,用于解包参数"""
    id_val, group_data, market_returns_data = args
    try:
        return id_val, process_group(group_data, market_returns_data)
    except Exception as e:
        print(f"处理股票 {id_val} 时出错: {e}")
        return id_val, None

def calculate_future_cumulative_returns(df, return_col, window, time_col, id_col, new_col_name='FutureCumReturn'):
        """
        计算未来window天的收益率之和,从t+2算到t+window+1,并且删掉停牌数据
        
        参数:
            return_col: 收益率列名
            window: 累积窗口大小
            time_col: 时间列名
            id_col: 证券编号列名
            new_col_name: 新列名
            
        返回:
            带未来收益率的datapanel
        """
        if return_col not in df.columns:
            raise ValueError(f"收益率列 {return_col} 不存在")
        
        if time_col not in df.columns:
            raise ValueError(f"时间列 {time_col} 不存在")
        
        # 计算未来累积收益
        if id_col in df.columns:
            df = df.sort_values([id_col, time_col])
            df[new_col_name] = 0
            for i in range(2, window + 2):
                df[new_col_name] += df.groupby(id_col)[return_col].shift(-i)
        else:
            # 只按时间排序计算
            df = df.sort_values(time_col)
            df[new_col_name] = 0
            for i in range(1, window + 1):
                df[new_col_name] += df[return_col].shift(-i)

        print(f"累积收益率计算完成,数据形状: {df.shape}")        
        return df

def clear_data(data):
    """负责整个预处理过程的全部异常数据删除,其他地方的删除操作都整合到这里

    参数：
        data: DataFrame格式的数据panel
    """
    # 去掉ST
    data = data[data['IfSpecialTrade']==0]
    # 删除累积收益为NaN的行和停牌的记录
    data = data.dropna(subset=['FutureCumReturn'])
    data = data[data['IfSuspended']==0]
    print(f"删除ST,停牌和累积收益为NaN的值后,数据形状: {data.shape}")
    # 此时剩下缺失值应该都是rolling窗口导致的，是每支票最早的若干交易日，应该整行删掉
    data = data.dropna(axis=0)
    print(f"删除缺失行后,数据形状: {data.shape}")
    return data

def feature_engineering(args, data):
    """计算特征
    
    参数:
        args: 参数字典
        data: 特征工程依赖的基础数据
    """

    #!停牌的记录对普通特征的影响仅限于停牌当日当前股票,可以很容易地删掉；
    #!但是对时序特征可能影响当前股票其他日期的特征值,比如DailyReturn停牌日是0,不会影响ret20但是会影响vol20

    if args.n_jobs is None or args.n_jobs <= 0:
        args.n_jobs = max(1, multiprocessing.cpu_count() - 1)
    
    # 计算基础特征
    print("计算基础特征...")
    data['lnto'] = np.log(data['TurnoverVolume'] / data['NonRestrictedShares']) #停牌的无交易量
    # data['swing'] = (data['HighPrice'] - data['LowPrice']) / data['LowPrice'] #停牌的High,Low都是0, 一字涨跌停的High=low
    data['lncret'] = np.log(data['DailyReturn'] + 1) #ST的return有极端值,退市前可能-0.96
    data['lncoret'] = np.log(data['ClosePrice'] / data['OpenPrice']) #停牌的OpenPrice是0,还有一些正常的日子确实close=open
    data['lnhlret'] = np.log(data['HighPrice'] / data['LowPrice'])
    
    # 计算真实波动范围
    data['tr'] = np.maximum(
        data['HighPrice_adj'] - data['LowPrice_adj'],
        np.maximum(
            abs(data['HighPrice_adj'] - data['PrevClosePrice_adj']),
            abs(data['LowPrice_adj'] - data['PrevClosePrice_adj'])
        )
    )/data['PrevClosePrice_adj'] #涨跌停的High=Low,早年也有21条High=Low=PrevClose的数据;完全不存在PrevClose=0的数据
    data['ex_lnret'] = data['lncret'] - np.log(data['MarketChangePCT']/100 + 1) #marketchangePCT没有极端值
    

    # 计算apb
    print("正在计算APB...")
    df_apb = apb(data)
    data = pd.merge(data, df_apb, on=['TradingDay', 'InnerCode'], how='left')
    
    # 加载高频数据
    print("正在加载高频数据...")
    
    # 存储高频数据
    intraday_data = {
        'skew': [],
        'vhhi': [],
        'rvol': [],
        'arpp': []
    }
    
    # 处理每个交易日的高频数据
    trading_days = data['TradingDay'].unique()
    total_days = len(trading_days)
    
    for i, date in enumerate(trading_days):
        print(f"处理高频数据: {date.strftime('%Y-%m-%d')} ({i+1}/{total_days})")
        
        # 获取日内收益率偏度
        try:
            skew_data = calculate_rskew(date)
            if skew_data is not None and not skew_data.empty:
                intraday_data['skew'].append(skew_data)
        except Exception as e:
            print(f"处理日内偏度数据时出错 {date}: {e}")
        
        # 获取VHHI
        try:
            vhhi_data = calculate_vhhi(date)
            if vhhi_data is not None and not vhhi_data.empty:
                intraday_data['vhhi'].append(vhhi_data)
        except Exception as e:
            print(f"处理VHHI数据时出错 {date}: {e}")
        
        # 获取日内波动率
        try:
            rvol_data = calculate_rvol(date)
            if rvol_data is not None and not rvol_data.empty:
                intraday_data['rvol'].append(rvol_data)
        except Exception as e:
            print(f"处理日内波动率数据时出错 {date}: {e}")
        
        # 获取ARPP
        try:
            arpp_data = calculate_arpp(date)
            if arpp_data is not None and not arpp_data.empty:
                intraday_data['arpp'].append(arpp_data)
        except Exception as e:
            print(f"处理ARPP数据时出错 {date}: {e}")
    
    # 合并高频数据
    high_freq_dfs = {}
    
    for data_type, data_list in intraday_data.items():
        if data_list:
            high_freq_dfs[data_type] = pd.concat(data_list, ignore_index=True)
            # 合并到日频数据
            data = pd.merge(data, high_freq_dfs[data_type], on=['TradingDay', 'SecuCode'], how='left')
        else:
            print(f"没有处理到{data_type}数据。请检查数据可用性。")
    
    print(f"数据加载完成")

    # 将市场收益率转换为小数
    market_returns = data['MarketChangePCT'] / 100
    
    # 按股票分组
    groups = data.groupby(args.id_col)
    unique_ids = list(groups.groups.keys())
    
    # 根据args.n_jobs选择计算方式
    if args.n_jobs <= 1:
        # 普通计算
        print("开始单进程计算时序特征...")
        start_time = time.time()
        
        # 普通循环计算每个股票的特征
        results = {}
        for id_val in unique_ids:
            try:
                group_data = groups.get_group(id_val)
                market_returns_data = market_returns.loc[group_data.index]
                results[id_val] = process_group(group_data, market_returns_data)
            except Exception as e:
                print(f"处理股票 {id_val} 时出错: {e}")
    else:
        # 多进程计算
        import multiprocessing
        print(f"开始多进程计算时序特征,使用{args.n_jobs}个进程...")
        start_time = time.time()
        
        tasks = []
        for id_val in unique_ids:
            group_data = groups.get_group(id_val)
            market_returns_data = market_returns.loc[group_data.index]
            tasks.append((id_val, group_data, market_returns_data))

        with multiprocessing.Pool(processes=args.n_jobs) as pool:
            results_list = pool.map(process_group_wrapper, tasks)
            pool.close()
            pool.join()
            del pool

        results = {id_val: result for id_val, result in results_list if result is not None}
        
    # 合并结果到原始DataFrame
    # TODO:目前大多数列都是float64,可以考虑换小一点,多数特征不需要双精度
    for feature in ['ret20', 'vol20', 'ppreversal', 'maxret20', 'ivol20', 'ivr20', 'lnto_20d', 'lnamihud20']:
        data[feature] = np.nan
        for id_val, result_df in results.items():
            if feature in result_df.columns:
                data.loc[result_df.index, feature] = result_df[feature]
    
    end_time = time.time()
    print(f"特征计算完成,耗时: {end_time - start_time:.2f}秒")

    print(f"计算未来{args.cumulative_window}天的累积收益率...")
    data = calculate_future_cumulative_returns(
        df=data,
        return_col=args.return_col,
        window=args.cumulative_window,
        time_col=args.time_col,
        id_col=args.id_col,
        new_col_name=args.target_col
    )
    data.to_parquet("features.parquet")

    data = clear_data(data)

    print("特征处理完毕")

    return data