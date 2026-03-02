# 评估output与原数据return5min的5日IC值
# 运行本程序会自动先运行main.py
# 第1-4日为训练集，第5日为测试集，重点关注第5日IC，以及IC的稳定性，ICIR为综合衡量指标

import os
import pandas as pd
import numpy as np
from utils import get_day_folders, evaluate_ic
from main import *
data_path = "data"
pred_path = "output"

def load_predictions(pred_path, day_folder):
    pred_file = os.path.join(pred_path, day_folder, 'E.csv')
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    pred_df = pd.read_csv(pred_file)
    pred_df['Time'] = pred_df['Time'].astype(int)
    return pred_df

def load_ground_truth(data_path, day_folder):
    e_file = os.path.join(data_path, day_folder, 'E.csv')
    if not os.path.exists(e_file):
        raise FileNotFoundError(f"Ground truth file not found: {e_file}")
    truth_df = pd.read_csv(e_file, usecols=['Time', 'Return5min'])
    truth_df['Time'] = truth_df['Time'].astype(int)
    return truth_df

def align_and_evaluate(pred_df, truth_df):
    merged = pd.merge(pred_df, truth_df, on='Time', how='inner')
    merged = merged.sort_values('Time')
    preds = merged['Predict'].values
    truth = merged['Return5min'].values
    ic = evaluate_ic(preds, truth)  
    return ic, merged

def main():
    day_folders = get_day_folders(data_path)
    all_results = []
    daily_ics = []
    for day in day_folders:
        pred_df = load_predictions(pred_path, day)
        truth_df = load_ground_truth(data_path, day)
        ic, merged = align_and_evaluate(pred_df, truth_df)          
        if not np.isnan(ic):
            daily_ics.append(ic)
            result = {
            'day': day,
                'ic': ic,
                'n_samples': len(merged),
                'pred_mean': merged['Predict'].mean(),
                'pred_std': merged['Predict'].std(),
                'truth_mean': merged['Return5min'].mean(),
                'truth_std': merged['Return5min'].std()
        }
            all_results.append(result)

    results_df = pd.DataFrame(all_results)
    mean_ic = np.mean(daily_ics)
    std_ic = np.std(daily_ics)  
    print("\n") 
    for _, row in results_df.iterrows():
        print(f"第{row['day']}天: IC={row['ic']:.6f}")
    print(f"\n平均IC: {mean_ic:.6f} (+/- {std_ic:.6f})")
    icir = mean_ic / std_ic if std_ic > 0 else 0
    print(f"ICIR: {icir:.4f}")

if __name__ == "__main__":
    run_test()
    main()