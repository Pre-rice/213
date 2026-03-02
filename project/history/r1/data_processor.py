# 数据预处理与特征工程

import numpy as np
import pandas as pd
from utils import get_day_folders, load_day_data

def generate_features_for_stock(df):
    data = df.copy()
    data['MidPrice'] = (data['BidPrice1'] + data['AskPrice1']) / 2.0
    data['TotalBidVol'] = data['BidVolume1'] + data['BidVolume2'] + data['BidVolume3'] + \
                          data['BidVolume4'] + data['BidVolume5']
    
    tb, ts = data['TradeBuyVolume'], data['TradeSellVolume']
    data['TradeImbalance'] = (tb - ts) / (tb + ts + 1e-6)
    
    ta_buy, ta_sell = data['TradeBuyAmount'], data['TradeSellAmount']
    data['AmountImbalance'] = (ta_buy - ta_sell) / (ta_buy + ta_sell + 1e-6)
    
    b1, a1 = data['BidVolume1'], data['AskVolume1']
    data['OrderImbalance'] = (b1 - a1) / (b1 + a1 + 1e-6)
    
    return data

def process_day_data(day_data):
    processed = {}
    for s in ['A','B','C','D','E']:
        processed[s] = generate_features_for_stock(day_data[s])
    
    df_e = processed['E'].copy()
    
    # --- 1. 长期特征 (600 ticks = 5 min) ---
    for col in ['TradeImbalance', 'AmountImbalance']:
        df_e[f'{col}_mean_600'] = df_e[col].rolling(600, min_periods=10).mean()
    
    # 差分 (长期)
    df_e['TradeImbalance_diff'] = df_e['TradeImbalance'] - df_e['TradeImbalance_mean_600']
    
    # 板块委托失衡
    s_order_sum = 0
    for s in ['A','B','C','D']:
        s_order_sum += processed[s]['OrderImbalance'].values
    df_e['Sector_OrderImbalance'] = s_order_sum / 4.0
    
    # --- 2. 短期特征 (60 ticks = 30 sec) ---
    # 捕捉短期资金脉冲
    df_e['TradeImbalance_mean_60'] = df_e['TradeImbalance'].rolling(60, min_periods=5).mean()
    
    # --- 3. 脉冲差值 (短期 - 长期) ---
    # 正值表示近期买盘突然增强
    df_e['TradeImbalance_pulse'] = df_e['TradeImbalance_mean_60'] - df_e['TradeImbalance_mean_600']
    
    # 清洗
    df_e = df_e.replace([np.inf, -np.inf], 0).fillna(0)
    
    # 最终特征列表
    features = [
        'TotalBidVol', 
        'TradeImbalance_mean_600', 
        'TradeImbalance_diff',
        'AmountImbalance_mean_600',
        'Sector_OrderImbalance',
        'TradeImbalance_pulse'        # 新增：稳健的脉冲特征
    ]
    
    return df_e[['Time'] + features + ['Return5min']]

def run_processor():
    print("开始预处理 (脉冲优化版)...")
    days = get_day_folders("./data")
    all_data = []
    for d in days:
        print(f"Processing Day {d}...")
        day_data = load_day_data("./data", d)
        all_data.append(process_day_data(day_data))
    
    total = pd.concat(all_data, ignore_index=True)
    total.to_csv("train.csv", index=False)
    print("Done. Shape:", total.shape)

if __name__ == "__main__":
    run_processor()