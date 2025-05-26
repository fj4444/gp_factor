"""
配置模块

提供全局配置和配置管理功能
"""

import os
import json
from typing import Dict, Any, Optional

# 默认配置
DEFAULT_CONFIG = {
    'data': {
        'default_start_date': '2018-01-01',
        'default_end_date': '2025-04-01',
        'time_col': 'TradingDay',
        'id_col': 'InnerCode',
        'return_col': 'DailyReturn',
        'target_col': 'FutureCumReturn',
        'feature_cols': 
            ['OpenPrice', 'ClosePrice', 'HighPrice', 'LowPrice', 'PrevClosePrice', 'TurnoverVolume', 'TurnoverValue', 
            'NonRestrictedShares', 'MarketClosePrice', 'MarketPrevClosePrice', 'MarketChangePCT', 'lnto', 'lncret', 'lncoret', 'lnhlret', 
            'tr', 'ex_lnret', 'ret20','vol20','ppreversal','maxret20','ivol20','ivr20','lnto_20d', 'lnamihud20', 'HighPrice_adj', 
            'LowPrice_adj', 'OpenPrice_adj', 'ClosePrice_adj', 'LnTurnoverVolume', 'LnTurnoverValue', 'Overnight', 'TurnoverRate', 
            'Amplitude', 'VWAP', 'Amihud', 'early_ret', 'tail_ret', 'max_ret', 'min_ret', 'mean_ret',
            'runner_value_321', 'runner_value_320', 'runner_value_186', 'runner_value_212', 'runner_value_211', 'runner_value_268',
            'runner_value_283', 'runner_value_195', 'runner_value_177', 'runner_value_199', 'runner_value_314', 'runner_value_191',
            'runner_value_292', 'runner_value_304', 'runner_value_201', 'runner_value_297', 'runner_value_188', 'runner_value_315',
            'runner_value_193', 'runner_value_322', 'runner_value_350', 'runner_value_179', 'runner_value_286', 'runner_value_197',
            'runner_value_318', 'runner_value_269', 'runner_value_319', 'runner_value_181', 'runner_value_347', 'runner_value_183',
            'runner_value_302', 'runner_value_267', 
            'S', 'M', 'B', 'SB','rvol', 'sqrskew', 'apb', 'arpp', 'vhhi', 'day_freq_apb', 'sqrvol', 'rskew','vvol','vskew','rkurt','sqvvol','retjump','retmod'],
        'cumulative_window': 5,
        'test_size': 0.3
    },
    'gp': {
        'population_size': 100,
        'max_generations': 20,
        'max_depth': 6,
        'tournament_size': 13,
        'crossover_rate': 0.65,
        'mutation_rate': 0.35,
        'hall_of_fame_size': 15,
        'random_seed': 42,
        'patience': 5,
        'min_delta': 0.001,
        'correlation_threshold': 0.6,  # 迭代内部相关性阈值
        'correlation_threshold_init': 0.4   # 热启动种群相关性阈值
    },
    'gpu': {
        'device': 'cuda',
        'batch_size': 64
    },
    'path': {
        'output_dir': 'results'
    },
    'db': {
        'connection_string': 'mysql://jiamu_xie%40public%23Thetis:fxtU4iM0@192.168.55.161:2883'
    },
    'barra': {
        'barra_file': "/data/home/jiamuxie/test/gp_proj_restructured/barra/barra.csv",
        'barra_usage': "correlation",
        'weights_file': "/data/home/jiamuxie/test/gp_proj_restructured/barra/weights.csv",
    }
}

# 配置文件路径
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')

# 全局配置
_config = None

def load_config() -> Dict[str, Any]:
    """
    加载配置
    
    返回:
        配置字典
    """
    global _config
    _config = DEFAULT_CONFIG
    return _config

def get_config(section: Optional[str] = None) -> Dict[str, Any]:
    """
    获取配置
    
    参数:
        section: 配置节名称,如果为None则返回整个配置
        
    返回:
        配置字典或配置节
    """
    config = load_config()

    if section is None:
        return config
    
    if section not in config:
        print(f"警告: 配置节 {section} 不存在,使用默认配置")
        return DEFAULT_CONFIG.get(section, {})
    
    return config[section]

def reset_config() -> None:
    """
    重置配置为默认值
    """
    global _config
    
    _config = DEFAULT_CONFIG
    
    # 保存默认配置到文件
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(_config, f, indent=2)
        print(f"配置已重置为默认值并保存到 {CONFIG_FILE}")
    except Exception as e:
        print(f"保存配置失败: {e}")