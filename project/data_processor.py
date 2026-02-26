import pandas as pd
import numpy as np
import os
from typing import Dict, List

# ========== 数据清洗函数 ==========

def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗单个股票的数据
    """
    # 1. 删除完全重复的行
    df = df.drop_duplicates()
    
    # 2. 处理缺失值
    # 对于价格和成交量字段，缺失值用前向填充
    price_cols = [col for col in df.columns if 'Price' in col]
    volume_cols = [col for col in df.columns if 'Volume' in col]
    numeric_cols = price_cols + volume_cols + (['LastPrice'] if 'LastPrice' in df.columns else [])
    
    for col in numeric_cols:
        if col in df.columns:
            # 使用ffill和bfill替代fillna(method)
            df[col] = df[col].ffill()
            df[col] = df[col].bfill()
    
    # 3. 处理异常值
    for col in numeric_cols:
        if col in df.columns:
            # 价格和成交量不能为负
            df.loc[df[col] < 0, col] = np.nan
            # 再次填充
            df[col] = df[col].ffill().bfill()
    
    # 4. 确保买卖价格合理（AskPrice > BidPrice）
    for i in range(1, 6):
        ask_col = f'AskPrice{i}'
        bid_col = f'BidPrice{i}'
        if ask_col in df.columns and bid_col in df.columns:
            # 如果卖价 <= 买价，调整
            invalid_mask = df[ask_col] <= df[bid_col]
            if invalid_mask.any():
                df.loc[invalid_mask, ask_col] = df.loc[invalid_mask, bid_col] + 1
    
    return df

def load_and_clean_all_data(base_dir: str = 'data') -> Dict[str, pd.DataFrame]:
    """
    加载并清洗所有交易日的数据，返回按天对齐的DataFrame字典
    只保留股票E的原始字段，其他股票只保留用于板块特征计算的必要字段
    """
    day_dirs = [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    day_dirs.sort(key=int)
    
    data_by_day = {}
    stocks = ['A', 'B', 'C', 'D', 'E']
    
    # 定义需要的字段（只保留必要的字段以节省内存）
    e_required_fields = ['Time', 'BidPrice1', 'AskPrice1', 'BidVolume1', 'AskVolume1',
                         'BidVolume2', 'BidVolume3', 'BidVolume4', 'BidVolume5',
                         'AskVolume2', 'AskVolume3', 'AskVolume4', 'AskVolume5',
                         'OrderBuyVolume', 'OrderSellVolume', 'TradeBuyVolume', 'TradeSellVolume',
                         'TradeBuyAmount', 'TradeSellAmount', 'LastPrice', 'Return5min']
    
    sector_required_fields = ['Time', 'BidPrice1', 'AskPrice1', 'BidVolume1', 'AskVolume1',
                              'OrderBuyVolume', 'OrderSellVolume', 'TradeBuyVolume', 'TradeSellVolume']
    
    for day in day_dirs:
        print(f"Processing day {day}...")
        day_path = os.path.join(base_dir, day)
        stock_dfs = []
        
        for stock in stocks:
            file_path = os.path.join(day_path, f"{stock}.csv")
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found, skipping...")
                continue
            
            # 读取数据
            df = pd.read_csv(file_path)
            
            # 根据股票类型选择保留的字段
            if stock == 'E':
                # 股票E保留所有需要的字段
                available_fields = [col for col in e_required_fields if col in df.columns]
                df = df[available_fields].copy()
            else:
                # 其他股票只保留板块特征需要的字段
                available_fields = [col for col in sector_required_fields if col in df.columns]
                df = df[available_fields].copy()
            
            # 确保Time列为整数
            df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
            
            # 删除Time为NaN的行
            df = df.dropna(subset=['Time'])
            
            # 按时间排序
            df = df.sort_values('Time').reset_index(drop=True)
            
            # 清洗数据
            df = clean_stock_data(df)
            
            # 重命名列，添加股票前缀（除了Time列）
            rename_dict = {col: f"{stock}_{col}" for col in df.columns if col != 'Time'}
            df = df.rename(columns=rename_dict)
            
            stock_dfs.append(df)
        
        # 如果某个股票缺失，跳过这一天
        if len(stock_dfs) < 5:
            print(f"Day {day}: Missing some stocks, skipping...")
            continue
        
        # 对齐所有股票的时间
        all_times = set()
        for df in stock_dfs:
            all_times.update(df['Time'].values)
        all_times = sorted(all_times)
        
        # 对每个股票重索引
        aligned_dfs = []
        for df in stock_dfs:
            df = df.set_index('Time')
            df = df.reindex(all_times, method='ffill')
            aligned_dfs.append(df)
        
        # 合并所有股票
        merged = pd.concat(aligned_dfs, axis=1)
        merged.reset_index(inplace=True)
        merged.rename(columns={'index': 'Time'}, inplace=True)
        
        # 添加日期标记
        merged['day'] = int(day)
        
        data_by_day[day] = merged.copy()
        
    return data_by_day

# ========== 特征计算函数（只计算股票E的特征） ==========

def calculate_order_imbalance(row):
    """股票E的订单流不平衡: (OrderBuyVolume - OrderSellVolume) / (OrderBuyVolume + OrderSellVolume)"""
    buy = row.get('E_OrderBuyVolume', np.nan)
    sell = row.get('E_OrderSellVolume', np.nan)
    if pd.isna(buy) or pd.isna(sell) or buy + sell == 0:
        return np.nan
    return (buy - sell) / (buy + sell)

def calculate_depth_pressure_ratio(row):
    """股票E的深度压力比: (前五档买单总量) / (前五档卖单总量)"""
    total_bid = sum(row.get(f'E_BidVolume{i}', 0) for i in range(1, 6))
    total_ask = sum(row.get(f'E_AskVolume{i}', 0) for i in range(1, 6))
    if total_ask == 0:
        return np.nan
    return total_bid / total_ask

def calculate_weighted_depth_imbalance(row):
    """股票E的加权深度不平衡"""
    imbalance = 0.0
    total_weight = 0.0
    for i in range(1, 6):
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

def calculate_trade_imbalance(row):
    """股票E的主动成交不平衡"""
    buy = row.get('E_TradeBuyVolume', np.nan)
    sell = row.get('E_TradeSellVolume', np.nan)
    if pd.isna(buy) or pd.isna(sell) or buy + sell == 0:
        return np.nan
    return (buy - sell) / (buy + sell)

def calculate_trade_amount_ratio(row):
    """股票E的主动买入成交额占比"""
    buy_amt = row.get('E_TradeBuyAmount', np.nan)
    sell_amt = row.get('E_TradeSellAmount', np.nan)
    if pd.isna(buy_amt) or pd.isna(sell_amt) or buy_amt + sell_amt == 0:
        return np.nan
    return buy_amt / (buy_amt + sell_amt)

# ========== 板块特征计算函数（使用A、B、C、D） ==========

def calculate_sector_spread_mean(row):
    """板块其他股票的平均相对价差（A、B、C、D）"""
    spreads = []
    for stock in ['A', 'B', 'C', 'D']:
        ask = row.get(f'{stock}_AskPrice1', np.nan)
        bid = row.get(f'{stock}_BidPrice1', np.nan)
        if pd.notna(ask) and pd.notna(bid) and ask + bid > 0:
            mid = (ask + bid) / 2
            spreads.append((ask - bid) / mid)
    if not spreads:
        return np.nan
    return np.mean(spreads)

def calculate_sector_order_imbalance_mean(row):
    """板块其他股票的订单流不平衡均值"""
    imbs = []
    for stock in ['A', 'B', 'C', 'D']:
        buy = row.get(f'{stock}_OrderBuyVolume', np.nan)
        sell = row.get(f'{stock}_OrderSellVolume', np.nan)
        if pd.notna(buy) and pd.notna(sell) and buy + sell > 0:
            imbs.append((buy - sell) / (buy + sell))
    if not imbs:
        return np.nan
    return np.mean(imbs)

def calculate_sector_trade_imbalance_mean(row):
    """板块其他股票的主动成交不平衡均值"""
    imbs = []
    for stock in ['A', 'B', 'C', 'D']:
        buy = row.get(f'{stock}_TradeBuyVolume', np.nan)
        sell = row.get(f'{stock}_TradeSellVolume', np.nan)
        if pd.notna(buy) and pd.notna(sell) and buy + sell > 0:
            imbs.append((buy - sell) / (buy + sell))
    if not imbs:
        return np.nan
    return np.mean(imbs)

def calculate_sector_order_volume_total(row):
    """板块其他股票的订单总成交量之和"""
    total_vol = 0
    for stock in ['A', 'B', 'C', 'D']:
        buy = row.get(f'{stock}_OrderBuyVolume', 0)
        sell = row.get(f'{stock}_OrderSellVolume', 0)
        total_vol += (buy + sell)
    return total_vol if total_vol > 0 else np.nan

def calculate_sector_buy_pressure_rank(row):
    """股票E的订单流不平衡在板块中的标准化得分"""
    e_imb = calculate_order_imbalance(row)
    sector_imbs = []
    for stock in ['A', 'B', 'C', 'D']:
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

def calculate_sector_weighted_mid_diff(row):
    """板块加权中间价与E的加权中间价之差"""
    # 计算E的加权中间价
    bid_p_e = row.get('E_BidPrice1', np.nan)
    ask_p_e = row.get('E_AskPrice1', np.nan)
    bid_v_e = row.get('E_BidVolume1', np.nan)
    ask_v_e = row.get('E_AskVolume1', np.nan)
    
    if any(pd.isna(x) for x in [bid_p_e, ask_p_e, bid_v_e, ask_v_e]) or bid_v_e + ask_v_e == 0:
        return np.nan
    
    wmid_e = (bid_p_e * ask_v_e + ask_p_e * bid_v_e) / (bid_v_e + ask_v_e)
    
    # 计算板块平均加权中间价
    wmid_sector = []
    for stock in ['A', 'B', 'C', 'D']:
        bid_p = row.get(f'{stock}_BidPrice1', np.nan)
        ask_p = row.get(f'{stock}_AskPrice1', np.nan)
        bid_v = row.get(f'{stock}_BidVolume1', np.nan)
        ask_v = row.get(f'{stock}_AskVolume1', np.nan)
        if pd.notna(bid_p) and pd.notna(ask_p) and pd.notna(bid_v) and pd.notna(ask_v) and bid_v + ask_v > 0:
            wmid = (bid_p * ask_v + ask_p * bid_v) / (bid_v + ask_v)
            wmid_sector.append(wmid)
    
    if not wmid_sector:
        return np.nan
    
    return wmid_e - np.mean(wmid_sector)

# ========== 主处理函数 ==========

def process_and_save_features(input_base_dir: str = 'data', output_base_dir: str = 'processed_data'):
    """
    处理所有数据，添加特征，并保存到新的目录
    """
    # 创建输出目录
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    # 加载并清洗所有数据
    print("Loading and cleaning data...")
    data_by_day = load_and_clean_all_data(input_base_dir)
    
    # 特征函数映射（只包含股票E的特征和板块特征）
    feature_functions = {
        # 股票E的自身特征
        'E_order_imbalance': calculate_order_imbalance,
        'E_depth_pressure_ratio': calculate_depth_pressure_ratio,
        'E_weighted_depth_imbalance': calculate_weighted_depth_imbalance,
        'E_trade_imbalance': calculate_trade_imbalance,
        'E_trade_amount_ratio': calculate_trade_amount_ratio,
        
        # 板块特征
        'sector_spread_mean': calculate_sector_spread_mean,
        'sector_order_imbalance_mean': calculate_sector_order_imbalance_mean,
        'sector_trade_imbalance_mean': calculate_sector_trade_imbalance_mean,
        'sector_order_volume_total': calculate_sector_order_volume_total,
        'sector_buy_pressure_rank': calculate_sector_buy_pressure_rank,
        'sector_weighted_mid_diff': calculate_sector_weighted_mid_diff,
    }
    
    # 处理每一天的数据
    for day, df in data_by_day.items():
        print(f"Adding features for day {day}...")
        
        # 逐行计算特征
        for feature_name, feature_func in feature_functions.items():
            df[feature_name] = df.apply(feature_func, axis=1)
        
        # 处理特征中的缺失值
        feature_cols = list(feature_functions.keys())
        for col in feature_cols:
            # 用0填充缺失值（假设缺失表示无交易/平衡状态）
            df[col] = df[col].fillna(0)
        
        # 保存到新的CSV文件
        output_file = os.path.join(output_base_dir, f'day_{day}_with_features.csv')
        df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")
        
        # 显示一些统计信息
        print(f"Shape: {df.shape}")
        print(f"Features added: {feature_cols}")
        print("-" * 50)
    
    # 同时保存一个合并所有天的文件（可选）
    print("Creating combined file...")
    all_days_df = pd.concat(data_by_day.values(), ignore_index=True)
    combined_output = os.path.join(output_base_dir, 'all_days_with_features.csv')
    all_days_df.to_csv(combined_output, index=False)
    print(f"Combined file saved to {combined_output}")
    
    return data_by_day

# ========== 数据验证函数 ==========

def validate_features(df: pd.DataFrame, feature_names: List[str]):
    """
    验证添加的特征是否合理
    """
    print("\n=== Feature Validation ===")
    for feat in feature_names:
        if feat in df.columns:
            stats = df[feat].describe()
            print(f"\n{feat}:")
            print(f"  Count: {stats['count']:.0f}")
            print(f"  Mean: {stats['mean']:.6f}")
            print(f"  Std: {stats['std']:.6f}")
            print(f"  Min: {stats['min']:.6f}")
            print(f"  Max: {stats['max']:.6f}")
        else:
            print(f"  {feat} not found")

# ========== 主程序 ==========

if __name__ == "__main__":
    # 设置输入输出目录
    INPUT_DIR = 'data'           # 原始数据目录
    OUTPUT_DIR = 'processed_data' # 输出目录
    
    print("Starting feature engineering pipeline...")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Note: Only calculating features for stock E, other stocks used for sector features")
    
    # 执行处理
    processed_data = process_and_save_features(INPUT_DIR, OUTPUT_DIR)
    
    # 验证最后一天的数据
    if processed_data:
        last_day = max(processed_data.keys())
        print(f"\nValidating features for day {last_day}:")
        feature_list = [
            'E_order_imbalance', 'E_depth_pressure_ratio', 'E_weighted_depth_imbalance',
            'E_trade_imbalance', 'E_trade_amount_ratio',
            'sector_spread_mean', 'sector_order_imbalance_mean', 'sector_trade_imbalance_mean',
            'sector_order_volume_total', 'sector_buy_pressure_rank', 'sector_weighted_mid_diff'
        ]
        validate_features(processed_data[last_day], feature_list)
    
    print("\nFeature engineering completed!")
    print(f"Processed files are saved in '{OUTPUT_DIR}' directory")
    print("\nOutput files contain:")
    print("  - All original fields for stock E")
    print("  - Selected fields for stocks A,B,C,D (only those needed for sector features)")
    print("  - New calculated features (prefixed with 'E_' for stock E features)")