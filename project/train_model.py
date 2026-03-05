# train_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
import warnings

warnings.filterwarnings('ignore')

def ic_score(y_true, y_pred):
    mask = ~np.isnan(y_true)
    if np.sum(mask) == 0: return 0.0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if np.std(y_pred) < 1e-6 or np.std(y_true) < 1e-6: return 0.0
    return np.corrcoef(y_true, y_pred)[0, 1]

def train():
    df = pd.read_csv("train.csv")
    
    # 指定的特征组合
    features = [
        'TotalBidVol', 
        'TradeImbalance_mean_600', 
        'TradeImbalance_diff',
        'Sector_OrderImbalance',
        'TradeImbalance_pulse',
        'Sector_TradeImb_pulse'
    ]
    target = 'Return5min'
    
    X = df[features].values
    y = df[target].values
    groups = df['Day'].values
    
    gkf = GroupKFold(n_splits=5)
    
    results = []
    
    print("开始 5 折交叉验证 ...")
    print(f"{'Fold':<6} | {'Test Day':<8} | {'IC':<10}")
    print("-" * 30)
    
    # 5折交叉验证
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # 获取测试集对应的 Day
        test_day = groups[test_idx][0]
        
        # 训练线性模型
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train, y_train)
        
        # 预测
        pred = model.predict(X_test)
        
        # 计算 IC
        ic = ic_score(y_test, pred)
        
        print(f"Fold {fold+1}  | Day {test_day}    | {ic:.6f}")
        results.append(ic)
        
    print("-" * 30)
    
    # 计算统计指标
    mean_ic = np.mean(results)
    std_ic = np.std(results)
    icir = mean_ic / std_ic if std_ic > 0 else 0
    
    print(f"平均 IC: {mean_ic:.6f}")
    print(f"ICIR: {icir:.4f}")

if __name__ == "__main__":
    train()