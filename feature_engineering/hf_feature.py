import pandas as pd
import os
import duckdb
def calculate_rskew(date, data_type='five_minute'):
    """
    读取5分钟频率数据并计算收益率偏度
    """
    dir_path = f"/data/HighFreqData/MinuteQuote/{data_type}"
    file_path = f"{dir_path}/{pd.to_datetime(date).strftime('%Y%m%d')}.parquet"
    if os.path.exists(file_path):
        # 直接在SQL中计算偏度
        sql = f"""
        WITH intraday_returns AS (
            SELECT 
                security_code as SecuCode,
                trading_day as TradingDay,
                start_time as StartTime,
                close_price,
                (close_price - LAG(close_price) OVER (PARTITION BY security_code ORDER BY start_time)) / 
                NULLIF(LAG(close_price) OVER (PARTITION BY security_code ORDER BY start_time), 0) AS return
            FROM read_parquet('{file_path}')
            WHERE turnover_volume != 0
        )
        SELECT 
            SecuCode,
            TradingDay,
            CASE 
                WHEN COUNT(DISTINCT return) = 1 THEN 0  -- 如果return列的唯一值数量为1,表示它是常数列,即一天内价格没有变化,可能存在两种情况,
                -- 一种是每个五分钟都有交易,但是价格一直没变,close_price一直是同一个值,也可能有些五分钟区间没有交易,会被turnover_volume=0筛选掉。不过计算不筛选掉,close_price在原始数据里是NaN,duckdb处理后看作NULL,导致return列里有0也有NULL,但是count(distinct)不对NULL进行计数
                -- 两种情况下都可以通过这样的判断把rskew赋值为0
                WHEN COUNT(close_price) = 1 THEN 0 --还有可能全天只有一个五分钟有成交,所以只有一个close_price有值,其他都是NULL,这种也应该把skew当作0
                ELSE SKEWNESS(return) 
            END AS rskew
        FROM intraday_returns
        WHERE return IS NOT NULL
        GROUP BY SecuCode, TradingDay
        """
        try:
            data = duckdb.query(sql).df()
            data['sqrskew'] = data['rskew']*data['rskew']
            return data
        except Exception as e:
            print(f"Error calculating rskew for {date}: {e}")
    return None

def calculate_vskew(date, data_type='five_minute'):
    """
    计算交易量的日内偏度
    """
    dir_path = f"/data/HighFreqData/MinuteQuote/{data_type}"
    file_path = f"{dir_path}/{pd.to_datetime(date).strftime('%Y%m%d')}.parquet"
    
    if os.path.exists(file_path):
        # 直接在SQL中计算日内波动率
        sql = f"""
        SELECT 
            security_code as SecuCode,
            trading_day as TradingDay,
            SKEWNESS(turnover_volume) as vvol
        FROM read_parquet('{file_path}')
        GROUP BY SecuCode, TradingDay
        """
        try:
            data = duckdb.query(sql).df()
            return data
        except Exception as e:
            print(f"Error calculating intraday volume skewness for {date}: {e}")
    
    return None

def calculate_rkurt(date, data_type='five_minute'):
    """
    读取5分钟频率数据并计算收益率峰度
    """
    dir_path = f"/data/HighFreqData/MinuteQuote/{data_type}"
    file_path = f"{dir_path}/{pd.to_datetime(date).strftime('%Y%m%d')}.parquet"
    if os.path.exists(file_path):
        # 直接在SQL中计算偏度
        sql = f"""
        WITH intraday_returns AS (
            SELECT 
                security_code as SecuCode,
                trading_day as TradingDay,
                start_time as StartTime,
                close_price,
                (close_price - LAG(close_price) OVER (PARTITION BY security_code ORDER BY start_time)) / 
                NULLIF(LAG(close_price) OVER (PARTITION BY security_code ORDER BY start_time), 0) AS return
            FROM read_parquet('{file_path}')
            WHERE turnover_volume != 0
        )
        SELECT 
            SecuCode,
            TradingDay,
            CASE 
                WHEN COUNT(DISTINCT return) = 1 THEN 0  -- 如果return列的唯一值数量为1,表示它是常数列,即一天内价格没有变化,可能存在两种情况,
                -- 一种是每个五分钟都有交易,但是价格一直没变,close_price一直是同一个值,也可能有些五分钟区间没有交易,会被turnover_volume=0筛选掉。不过计算不筛选掉,close_price在原始数据里是NaN,duckdb处理后看作NULL,导致return列里有0也有NULL,但是count(distinct)不对NULL进行计数
                -- 两种情况下都可以通过这样的判断把rskew赋值为0
                WHEN COUNT(close_price) = 1 THEN 0 --还有可能全天只有一个五分钟有成交,所以只有一个close_price有值,其他都是NULL,这种也应该把skew当作0
                ELSE kurtosis(return)
            END AS rkurt
        FROM intraday_returns
        WHERE return IS NOT NULL
        GROUP BY SecuCode, TradingDay
        """
    
        try:
            data = duckdb.query(sql).df()
            return data
        except Exception as e:
            print(f"Error calculating rkurt for {date}: {e}")
    return None

def calculate_vhhi(date, data_type='five_minute'):
    """
    计算日内交易量的HHI指数
    """
    dir_path = f"/data/HighFreqData/MinuteQuote/{data_type}"
    file_path = f"{dir_path}/{pd.to_datetime(date).strftime('%Y%m%d')}.parquet"
    
    if os.path.exists(file_path):
        # 直接在SQL中计算HHI指数
        sql = f"""
        WITH volume_shares AS (
            SELECT 
                security_code as SecuCode,
                trading_day as TradingDay,
                start_time,
                turnover_volume,
                SUM(turnover_volume) OVER (PARTITION BY security_code) as total_volume
            FROM read_parquet('{file_path}')
            WHERE turnover_volume != 0
        ),
        squared_shares AS (
            SELECT
                SecuCode,
                TradingDay,
                start_time,
                POWER(turnover_volume / NULLIF(total_volume, 0), 2) as squared_share
            FROM volume_shares
        )
        SELECT 
            SecuCode,
            TradingDay,
            SUM(squared_share) as vhhi
        FROM squared_shares
        GROUP BY SecuCode, TradingDay
        """
        try:
            data = duckdb.query(sql).df()
            return data
        except Exception as e:
            print(f"Error calculating VHHI for {date}: {e}")
    
    return None

def calculate_rvol(date, data_type='five_minute'):
    """
    计算日内波动率
    """
    dir_path = f"/data/HighFreqData/MinuteQuote/{data_type}"
    file_path = f"{dir_path}/{pd.to_datetime(date).strftime('%Y%m%d')}.parquet"
    
    if os.path.exists(file_path):
        # 直接在SQL中计算日内波动率
        sql = f"""
        WITH intraday_returns AS (
            SELECT 
                security_code as SecuCode,
                trading_day as TradingDay,
                close_price,
                (close_price - LAG(close_price) OVER (PARTITION BY security_code ORDER BY start_time)) / 
                NULLIF(LAG(close_price) OVER (PARTITION BY security_code ORDER BY start_time), 0) AS return
            FROM read_parquet('{file_path}')
            WHERE turnover_volume != 0
        )
        SELECT 
            SecuCode,
            TradingDay,
            CASE
                WHEN COUNT(DISTINCT close_price) = 1 THEN 0
                ELSE STDDEV(return) --如果五分钟数据里没有连在一起的有效数字,return列算出来就全是NULL,导致rvol也是NULL,所以干脆检测到close_price只有一个值或者NULL,直接rvol为0
            END AS rvol
        FROM intraday_returns
        WHERE return IS NOT NULL
        GROUP BY SecuCode, TradingDay
        """
        try:
            data = duckdb.query(sql).df()
            data['sqrvol'] = data['rvol']*data['rvol']
            return data
        except Exception as e:
            print(f"Error calculating intraday volatility for {date}: {e}")
    
    return None

def calculate_vvol(date, data_type='five_minute'):
    """
    计算交易量的日内波动率
    """
    dir_path = f"/data/HighFreqData/MinuteQuote/{data_type}"
    file_path = f"{dir_path}/{pd.to_datetime(date).strftime('%Y%m%d')}.parquet"
    
    if os.path.exists(file_path):
        # 直接在SQL中计算日内波动率
        sql = f"""
        SELECT 
            security_code as SecuCode,
            trading_day as TradingDay,
            STDDEV(turnover_volume) as vvol
        FROM read_parquet('{file_path}')
        GROUP BY SecuCode, TradingDay
        """
        try:
            data = duckdb.query(sql).df()
            data['sqvvol'] = data['vvol']*data['vvol']
            return data
        except Exception as e:
            print(f"Error calculating intraday volume volatility for {date}: {e}")
    
    return None

def calculate_arpp(date, data_type='five_minute'):
    """
    计算时间加权平均价格的相对价格位置
    """
    dir_path = f"/data/HighFreqData/MinuteQuote/{data_type}"
    file_path = f"{dir_path}/{pd.to_datetime(date).strftime('%Y%m%d')}.parquet"
    
    if os.path.exists(file_path):
        # 直接在SQL中计算TWAP和相对价格位置
        sql = f"""
        WITH price_data AS (
            WITH filled_data AS (
                SELECT 
                    security_code AS SecuCode,
                    trading_day AS TradingDay,
                    close_price AS price,
                    -- 使用窗口函数进行前向填充
                    LAST_VALUE(price IGNORE NULLS) OVER (
                        PARTITION BY security_code 
                        ORDER BY start_time 
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS filled_price,
                    start_time
                FROM read_parquet('{file_path}')
            )

            SELECT 
                SecuCode,
                TradingDay,
                AVG(filled_price) as twap,  -- 时间加权平均价格（简单平均收盘价）
                MIN(filled_price) as min_price,
                MAX(filled_price) as max_price
            FROM filled_data
            WHERE start_time < 1457
            GROUP BY SecuCode, TradingDay
        )
        SELECT 
            SecuCode,
            TradingDay,
            -- 计算相对价格位置：(TWAP - min) / (max - min)
            (twap - min_price) / NULLIF(max_price - min_price, 0) as arpp
        FROM price_data
        """
        try:
            data = duckdb.query(sql).df()
            return data
        except Exception as e:
            print(f"Error calculating ARPP for {date}: {e}")
    
    return None

def calculate_retjump_retmod(date, data_type='five_minute'):
    """
    计算日内对数极端收益之和和温和收益之和，对正态分布的Xi, 样本点足够多，则样本中位数数收敛于样本均值，1.483 倍的 MAD 收敛于样本标准差。
        𝑚𝑑 = 𝑚𝑒𝑑𝑖𝑎𝑛(𝑥𝑖,𝑖 = 1,2, … , 𝑛)
        𝑀𝐴𝐷 = 𝑚𝑒𝑑𝑖𝑎𝑛(|𝑥𝑖 − 𝑚𝑑|, 𝑖 = 1,2, … , 𝑛)
        𝑀𝐴𝐷𝑒 = 1.483 × 𝑀𝐴𝐷
    在 5%的置信度下，与中位数 md 距离在 1.96 倍𝑀𝐴𝐷𝑒以上的样本点即为异常点（𝑝𝑝𝑓(1 − 5%⁄2) =1.96, 𝑝𝑝𝑓是标准正太分布累计分布函数的反函数）。
    """
    dir_path = f"/data/HighFreqData/MinuteQuote/{data_type}"
    file_path = f"{dir_path}/{pd.to_datetime(date).strftime('%Y%m%d')}.parquet"

    if os.path.exists(file_path):
        # 直接在SQL中计算极端收益和温和收益
        sql = f"""
        WITH intraday_returns AS (
            SELECT 
                security_code as SecuCode,
                trading_day as TradingDay,
                start_time,
                LN(close_price / NULLIF(LAG(close_price) OVER (PARTITION BY security_code ORDER BY start_time), 0)) AS log_return
            FROM read_parquet('{file_path}')
            WHERE turnover_volume != 0
        ),
        -- 首先计算每个证券每天的收益率中位数
        median_returns AS (
            SELECT 
                SecuCode,
                TradingDay,
                MEDIAN(log_return) AS md
            FROM intraday_returns
            WHERE log_return IS NOT NULL
            GROUP BY SecuCode, TradingDay
        ),
        -- 然后计算每个收益率与中位数的绝对偏差
        abs_deviations AS (
            SELECT
                ir.SecuCode,
                ir.TradingDay,
                ir.log_return,
                mr.md,
                ABS(ir.log_return - mr.md) AS abs_dev
            FROM intraday_returns ir
            JOIN median_returns mr ON ir.SecuCode = mr.SecuCode AND ir.TradingDay = mr.TradingDay
            WHERE ir.log_return IS NOT NULL
        ),
        -- 计算MAD (Median Absolute Deviation)
        mad_stats AS (
            SELECT
                SecuCode,
                TradingDay,
                MEDIAN(abs_dev) AS mad
            FROM abs_deviations
            GROUP BY SecuCode, TradingDay
        ),
        -- 最后计算极端收益和温和收益
        classified_returns AS (
            SELECT
                ad.SecuCode,
                ad.TradingDay,
                ad.log_return,
                ad.md,
                ms.mad,
                1.483 * ms.mad AS made,
                CASE
                    WHEN ABS(ad.log_return - ad.md) > 1.96 * 1.483 * ms.mad THEN ad.log_return
                    ELSE 0
                END AS extreme_return,
                CASE
                    WHEN ABS(ad.log_return - ad.md) <= 1.96 * 1.483 * ms.mad THEN ad.log_return
                    ELSE 0
                END AS moderate_return
            FROM abs_deviations ad
            JOIN mad_stats ms ON ad.SecuCode = ms.SecuCode AND ad.TradingDay = ms.TradingDay
        )
        SELECT 
            SecuCode,
            TradingDay,
            SUM(extreme_return) AS retjump,
            SUM(moderate_return) AS retmod
        FROM classified_returns
        GROUP BY SecuCode, TradingDay
        """
        try:
            data = duckdb.query(sql).df()
            return data
        except Exception as e:
            print(f"Error calculating extreme and moderate returns for {date}: {e}")
    
    return None
