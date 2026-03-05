# data_processor.py
import numpy as np
import pandas as pd
from utils import get_day_folders, load_day_data

def calculate_imbalances(df):
    """计算基础失衡指标"""
    data = df.copy()
    
    # 1. TotalBidVol
    data['TotalBidVol'] = data['BidVolume1'] + data['BidVolume2'] + data['BidVolume3'] + \
                          data['BidVolume4'] + data['BidVolume5']
    
    # 2. TradeImbalance (成交失衡)
    tb, ts = data['TradeBuyVolume'], data['TradeSellVolume']
    data['TradeImbalance'] = (tb - ts) / (tb + ts + 1e-6)
    
    # 3. OrderImbalance (委托失衡)
    b1, a1 = data['BidVolume1'], data['AskVolume1']
    data['OrderImbalance'] = (b1 - a1) / (b1 + a1 + 1e-6)
    
    return data

def process_day_data(day_data):
    # 1. 处理 E 股票
    df_e = calculate_imbalances(day_data['E'])
    
    # === E 自身特征 ===
    # 特征 1: TotalBidVol (已有)
    
    # 特征 2: TradeImbalance_mean_600
    df_e['TradeImbalance_mean_600'] = df_e['TradeImbalance'].rolling(600, min_periods=1).mean()
    
    # 特征 3: TradeImbalance_diff
    df_e['TradeImbalance_diff'] = df_e['TradeImbalance'] - df_e['TradeImbalance_mean_600']
    
    # 特征 4: TradeImbalance_pulse (短期 - 长期)
    df_e['TradeImbalance_mean_60'] = df_e['TradeImbalance'].rolling(60, min_periods=1).mean()
    df_e['TradeImbalance_pulse'] = df_e['TradeImbalance_mean_60'] - df_e['TradeImbalance_mean_600']
    
    # === 板块特征 (关键修改：对齐数据) ===
    # 初始化板块特征列
    df_e['Sector_OrderImbalance'] = 0.0
    df_e['Sector_TradeImbalance'] = 0.0
    
    # 遍历 A, B, C, D，累加特征
    for s in ['A', 'B', 'C', 'D']:
        df_s = calculate_imbalances(day_data[s])
        
        # 提取需要的列，并重命名防止冲突
        temp = df_s[['Time', 'OrderImbalance', 'TradeImbalance']].copy()
        temp.columns = ['Time', f'{s}_Order', f'{s}_Trade']
        
        # 以 Time 为键合并到 df_e (Left Join 保证 E 的行数不变)
        df_e = pd.merge(df_e, temp, on='Time', how='left')
        
        # 累加 (缺失值 fillna(0) 视为无贡献)
        df_e['Sector_OrderImbalance'] += df_e[f'{s}_Order'].fillna(0)
        df_e['Sector_TradeImbalance'] += df_e[f'{s}_Trade'].fillna(0)
        
        # 删除临时列
        df_e.drop([f'{s}_Order', f'{s}_Trade'], axis=1, inplace=True)
        
    # 计算均值
    df_e['Sector_OrderImbalance'] /= 4.0
    df_e['Sector_TradeImbalance'] /= 4.0
    
    # 特征 5: Sector_TradeImb_pulse
    df_e['Sector_TradeImb_mean_600'] = df_e['Sector_TradeImbalance'].rolling(600, min_periods=1).mean()
    df_e['Sector_TradeImb_mean_60'] = df_e['Sector_TradeImbalance'].rolling(60, min_periods=1).mean()
    df_e['Sector_TradeImb_pulse'] = df_e['Sector_TradeImb_mean_60'] - df_e['Sector_TradeImb_mean_600']
    
    # 清洗
    df_e = df_e.replace([np.inf, -np.inf], 0).fillna(0)
    
    features = [
        'TotalBidVol', 
        'TradeImbalance_mean_600', 
        'TradeImbalance_diff',
        'Sector_OrderImbalance',
        'TradeImbalance_pulse',
        'Sector_TradeImb_pulse'
    ]
    
    return df_e[['Time'] + features + ['Return5min']]

def run_processor():
    print("开始预处理...")
    data_path = "./data"
    days = get_day_folders(data_path)
    all_data = []
    
    for d in days:
        print(f"Processing Day {d}...")
        day_data = load_day_data(data_path, d)
        day_df = process_day_data(day_data)
        # 添加 Day 标签
        day_df['Day'] = int(d)
        all_data.append(day_df)
    
    total = pd.concat(all_data, ignore_index=True)
    total.to_csv("train.csv", index=False)
    print("Done. Shape:", total.shape)

if __name__ == "__main__":
    run_processor()