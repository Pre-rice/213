import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
from typing import Callable, Dict, List, Tuple

# ========== 数据加载函数（优化版） ==========

def load_day_data(day_path: str) -> pd.DataFrame:
    """
    加载单个交易日的数据，将五只股票按时间对齐合并。
    """
    stocks = ['A', 'B', 'C', 'D', 'E']
    stock_dfs = []
    
    for stock in stocks:
        file_path = os.path.join(day_path, f"{stock}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")
        
        df = pd.read_csv(file_path)
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        df = df.dropna(subset=['Time']).sort_values('Time').set_index('Time')
        df = df[~df.index.duplicated(keep='first')]
        df.columns = [f"{stock}_{col}" for col in df.columns]
        stock_dfs.append(df)
    
    all_times = sorted(set.union(*[set(df.index) for df in stock_dfs]))
    aligned_dfs = []
    for df in stock_dfs:
        df_aligned = df.reindex(all_times, method='ffill')
        aligned_dfs.append(df_aligned)
    
    merged = pd.concat(aligned_dfs, axis=1)
    merged.reset_index(inplace=True)
    merged.rename(columns={'index': 'Time'}, inplace=True)
    merged['Time'] = merged['Time'].astype('int64')
    # 复制以消除碎片化警告
    return merged.copy()

def load_all_data(base_dir: str = 'data') -> Dict[str, pd.DataFrame]:
    """
    加载所有交易日的数据。
    """
    day_dirs = [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    day_dirs.sort(key=int)
    
    data_by_day = {}
    for day in day_dirs:
        day_path = os.path.join(base_dir, day)
        print(f"Loading day {day} from {day_path}")
        data_by_day[day] = load_day_data(day_path)
    return data_by_day

# ========== 特征函数定义 ==========

def spread_ratio(row):
    """相对买卖价差"""
    ask1 = row.get('E_AskPrice1', np.nan)
    bid1 = row.get('E_BidPrice1', np.nan)
    if pd.isna(ask1) or pd.isna(bid1) or ask1 + bid1 == 0:
        return np.nan
    mid = (ask1 + bid1) / 2
    return (ask1 - bid1) / mid

def weighted_mid_price(row):
    """加权中间价"""
    bid_p = row.get('E_BidPrice1', np.nan)
    ask_p = row.get('E_AskPrice1', np.nan)
    bid_v = row.get('E_BidVolume1', np.nan)
    ask_v = row.get('E_AskVolume1', np.nan)
    if any(pd.isna(x) for x in [bid_p, ask_p, bid_v, ask_v]) or bid_v + ask_v == 0:
        return np.nan
    return (bid_p * ask_v + ask_p * bid_v) / (bid_v + ask_v)

def order_imbalance(row):
    """订单流不平衡"""
    buy = row.get('E_OrderBuyVolume', np.nan)
    sell = row.get('E_OrderSellVolume', np.nan)
    if pd.isna(buy) or pd.isna(sell) or buy + sell == 0:
        return np.nan
    return (buy - sell) / (buy + sell)

def trade_imbalance(row):
    """主动成交不平衡"""
    buy = row.get('E_TradeBuyVolume', np.nan)
    sell = row.get('E_TradeSellVolume', np.nan)
    if pd.isna(buy) or pd.isna(sell) or buy + sell == 0:
        return np.nan
    return (buy - sell) / (buy + sell)

def depth_pressure_ratio(row):
    """深度压力比"""
    total_bid = sum(row.get(f'E_BidVolume{i}', 0) for i in range(1,6))
    total_ask = sum(row.get(f'E_AskVolume{i}', 0) for i in range(1,6))
    if total_ask == 0:
        return np.nan
    return total_bid / total_ask

def weighted_depth_imbalance(row):
    """加权深度不平衡"""
    imbalance = 0.0
    total_weight = 0.0
    for i in range(1,6):
        bid = row.get(f'E_BidVolume{i}', 0)
        ask = row.get(f'E_AskVolume{i}', 0)
        if bid + ask > 0:
            imb_i = (bid - ask) / (bid + ask)
            weight = 1.0 / i
            imbalance += imb_i * weight
            total_weight += weight
    if total_weight == 0:
        return np.nan
    return imbalance / total_weight

def log_return_last_price(row):
    """最新成交价对数收益率（依赖滞后列）"""
    last = row.get('E_LastPrice', np.nan)
    last_lag = row.get('E_LastPrice_lag1', np.nan)
    if pd.notna(last) and pd.notna(last_lag) and last_lag > 0:
        return np.log(last / last_lag)
    return np.nan

def buy_sell_volume_ratio(row):
    """订单买卖量比率"""
    buy = row.get('E_OrderBuyVolume', np.nan)
    sell = row.get('E_OrderSellVolume', np.nan)
    if pd.isna(buy) or pd.isna(sell) or buy + sell == 0:
        return np.nan
    return buy / (buy + sell)

def trade_buy_amount_ratio(row):
    """主动买入成交额占比"""
    buy_amt = row.get('E_TradeBuyAmount', np.nan)
    sell_amt = row.get('E_TradeSellAmount', np.nan)
    if pd.isna(buy_amt) or pd.isna(sell_amt) or buy_amt + sell_amt == 0:
        return np.nan
    return buy_amt / (buy_amt + sell_amt)

def sector_spread_mean(row):
    """板块其他股票的平均相对价差"""
    spreads = []
    for stock in ['A','B','C','D']:
        ask = row.get(f'{stock}_AskPrice1', np.nan)
        bid = row.get(f'{stock}_BidPrice1', np.nan)
        if pd.notna(ask) and pd.notna(bid) and ask + bid > 0:
            mid = (ask + bid) / 2
            spreads.append((ask - bid) / mid)
    if not spreads:
        return np.nan
    return np.mean(spreads)

def sector_order_imbalance_mean(row):
    """板块其他股票的订单流不平衡均值"""
    imbs = []
    for stock in ['A','B','C','D']:
        buy = row.get(f'{stock}_OrderBuyVolume', np.nan)
        sell = row.get(f'{stock}_OrderSellVolume', np.nan)
        if pd.notna(buy) and pd.notna(sell) and buy + sell > 0:
            imbs.append((buy - sell) / (buy + sell))
    if not imbs:
        return np.nan
    return np.mean(imbs)

def sector_trade_imbalance_std(row):
    """板块其他股票成交不平衡的标准差"""
    imbs = []
    for stock in ['A','B','C','D']:
        buy = row.get(f'{stock}_TradeBuyVolume', np.nan)
        sell = row.get(f'{stock}_TradeSellVolume', np.nan)
        if pd.notna(buy) and pd.notna(sell) and buy + sell > 0:
            imbs.append((buy - sell) / (buy + sell))
    if len(imbs) < 2:
        return np.nan
    return np.std(imbs)

def sector_weighted_mid_diff(row):
    """板块加权中间价与E的加权中间价之差"""
    wmid_e = weighted_mid_price(row)
    wmid_sector = []
    for stock in ['A','B','C','D']:
        bid_p = row.get(f'{stock}_BidPrice1', np.nan)
        ask_p = row.get(f'{stock}_AskPrice1', np.nan)
        bid_v = row.get(f'{stock}_BidVolume1', np.nan)
        ask_v = row.get(f'{stock}_AskVolume1', np.nan)
        if pd.notna(bid_p) and pd.notna(ask_p) and pd.notna(bid_v) and pd.notna(ask_v) and bid_v + ask_v > 0:
            wmid = (bid_p * ask_v + ask_p * bid_v) / (bid_v + ask_v)
            wmid_sector.append(wmid)
    if pd.isna(wmid_e) or not wmid_sector:
        return np.nan
    return wmid_e - np.mean(wmid_sector)

def sector_order_volume_total(row):
    """板块其他股票的订单总成交量之和"""
    total_vol = 0
    for stock in ['A','B','C','D']:
        buy = row.get(f'{stock}_OrderBuyVolume', 0)
        sell = row.get(f'{stock}_OrderSellVolume', 0)
        total_vol += (buy + sell)
    return total_vol if total_vol > 0 else np.nan

def sector_buy_pressure_rank(row):
    """股票E的订单流不平衡在板块中的标准化得分（z-score）"""
    e_imb = order_imbalance(row)
    sector_imbs = []
    for stock in ['A','B','C','D']:
        buy = row.get(f'{stock}_OrderBuyVolume', np.nan)
        sell = row.get(f'{stock}_OrderSellVolume', np.nan)
        if pd.notna(buy) and pd.notna(sell) and buy + sell > 0:
            sector_imbs.append((buy - sell) / (buy + sell))
    if pd.isna(e_imb) or len(sector_imbs) < 1:
        return np.nan
    all_imbs = sector_imbs + [e_imb]
    mean_imb = np.mean(all_imbs)
    std_imb = np.std(all_imbs)
    if std_imb == 0:
        return np.nan
    return (e_imb - mean_imb) / std_imb

def volume_ma_ratio(row, col='E_OrderBuyVolume', window=10):
    """当前成交量与过去window期平均成交量的比率（依赖移动平均列）"""
    ma_col = f'{col}_ma{window}'
    val = row.get(col, np.nan)
    ma = row.get(ma_col, np.nan)
    if pd.notna(val) and pd.notna(ma) and ma != 0:
        return val / ma
    return np.nan

def price_reversion_signal(row):
    """均值回归信号：(LastPrice - 20期均线) / 20期均线"""
    last = row.get('E_LastPrice', np.nan)
    ma = row.get('E_LastPrice_ma20', np.nan)
    if pd.notna(last) and pd.notna(ma) and ma != 0:
        return (last - ma) / ma
    return np.nan

def volatility_ratio(row):
    """波动率比率：当前价差与20期平均价差的比率"""
    spread = spread_ratio(row)
    spread_ma = row.get('spread_ma20', np.nan)
    if pd.notna(spread) and pd.notna(spread_ma) and spread_ma != 0:
        return spread / spread_ma
    return np.nan

# ========== 特征评估函数 ==========

def evaluate_feature_ic(data: pd.DataFrame, feature_func: Callable, target_col: str = 'E_Return5min') -> Tuple[float, float]:
    """
    计算特征与目标变量的整体IC（皮尔逊相关系数）和p值。
    """
    features = []
    targets = []
    for idx, row in data.iterrows():
        try:
            f_val = feature_func(row)
            t_val = row[target_col]
            if pd.notna(f_val) and pd.notna(t_val):
                features.append(f_val)
                targets.append(t_val)
        except Exception:
            continue
    if len(features) < 3:
        return np.nan, np.nan
    ic, p = pearsonr(features, targets)
    return ic, p

def evaluate_feature_daily_ic(data_by_day: Dict[str, pd.DataFrame], feature_func: Callable, target_col: str = 'E_Return5min') -> float:
    """
    按日计算IC并返回平均值。
    """
    daily_ics = []
    for day, df in data_by_day.items():
        ic_day, _ = evaluate_feature_ic(df, feature_func, target_col)
        if not np.isnan(ic_day):
            daily_ics.append(ic_day)
    return np.mean(daily_ics) if daily_ics else np.nan

# ========== 主程序 ==========

if __name__ == "__main__":
    # 1. 加载所有交易日数据
    data_by_day = load_all_data('data')
    
    # 2. 合并所有天数据（用于整体IC计算），同时保留day列
    combined_list = []
    for day, df in data_by_day.items():
        df['day'] = int(day)
        combined_list.append(df)
    full_data = pd.concat(combined_list, ignore_index=True).copy()  # 复制消除碎片化
    
    print("数据加载完成，开始预处理滚动特征...")
    
    # 3. 预先计算需要历史窗口的特征列（按日分组，避免跨天）
    # 3.1 滞后1期的LastPrice
    full_data['E_LastPrice_lag1'] = full_data.groupby('day')['E_LastPrice'].shift(1)
    
    # 3.2 20期移动平均
    full_data['E_LastPrice_ma20'] = full_data.groupby('day')['E_LastPrice'].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )
    
    # 3.3 20期价差移动平均（需要先计算价差）
    # 临时计算价差列，但不存储为特征，仅用于滚动平均
    full_data['spread_temp'] = (full_data['E_AskPrice1'] - full_data['E_BidPrice1']) / ((full_data['E_AskPrice1'] + full_data['E_BidPrice1'])/2)
    full_data['spread_ma20'] = full_data.groupby('day')['spread_temp'].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )
    full_data.drop(columns=['spread_temp'], inplace=True)
    
    # 3.4 10期移动平均成交量（用于volume_ma_ratio）
    for col in ['E_OrderBuyVolume', 'E_OrderSellVolume']:
        full_data[f'{col}_ma10'] = full_data.groupby('day')[col].transform(
            lambda x: x.rolling(10, min_periods=1).mean()
        )
    
    # 4. 定义特征列表
    feature_list = [
        ("spread_ratio", spread_ratio),
        ("weighted_mid_price", weighted_mid_price),
        ("order_imbalance", order_imbalance),
        ("trade_imbalance", trade_imbalance),
        ("depth_pressure_ratio", depth_pressure_ratio),
        ("weighted_depth_imbalance", weighted_depth_imbalance),
        ("log_return_last_price", log_return_last_price),
        ("buy_sell_volume_ratio", buy_sell_volume_ratio),
        ("trade_buy_amount_ratio", trade_buy_amount_ratio),
        ("sector_spread_mean", sector_spread_mean),
        ("sector_order_imbalance_mean", sector_order_imbalance_mean),
        ("sector_trade_imbalance_std", sector_trade_imbalance_std),
        ("sector_weighted_mid_diff", sector_weighted_mid_diff),
        ("sector_order_volume_total", sector_order_volume_total),
        ("sector_buy_pressure_rank", sector_buy_pressure_rank),
        ("volume_ma_ratio_buy", lambda row: volume_ma_ratio(row, 'E_OrderBuyVolume', 10)),
        ("volume_ma_ratio_sell", lambda row: volume_ma_ratio(row, 'E_OrderSellVolume', 10)),
        ("price_reversion_signal", price_reversion_signal),
        ("volatility_ratio", volatility_ratio),
    ]
    
    # 5. 循环计算每个特征的IC
    results = []
    for name, func in feature_list:
        print(f"正在计算特征: {name} ...")
        # 整体IC
        ic_overall, p_overall = evaluate_feature_ic(full_data, func)
        # 按日平均IC
        avg_daily_ic = evaluate_feature_daily_ic(data_by_day, func)
        results.append({
            "feature": name,
            "overall_IC": ic_overall,
            "p_value": p_overall,
            "avg_daily_IC": avg_daily_ic
        })
        print(f"  overall IC: {ic_overall:.6f}, p={p_overall:.4f}, avg daily IC: {avg_daily_ic:.6f}")
    
    # 6. 输出结果表格
    result_df = pd.DataFrame(results)
    print("\n========== 特征IC评估结果 ==========")
    print(result_df.to_string(index=False))
    
    # 可选：保存到CSV
    result_df.to_csv("feature_ic_results.csv", index=False)
    print("\n结果已保存至 feature_ic_results.csv")