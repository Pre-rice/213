import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import warnings

warnings.filterwarnings('ignore')

# ── 特征列表（与 data_processor.py 保持一致）──────────────────────────────────
FEATURE_COLS = [
    'TotalBidVol', 'TradeImb_600', 'TradeImb_diff',
    'Sect_OBI1', 'E_TI_rel_600',
    'TradeImb_p60', 'TradeImb_p40', 'Sect_TI_p40',
    'OVI_p15', 'OVI_p60', 'Sect_OVI_p20',
    'TradeImb_ep60', 'OVI_ep15', 'TNI_ep15',
]

# Ridge 正则化系数（通过 5 折交叉验证调参确定）
RIDGE_ALPHA = 150


def ic_score(y_true, y_pred):
    mask = ~np.isnan(y_true)
    if mask.sum() == 0:
        return 0.0
    y_t, y_p = y_true[mask], y_pred[mask]
    if np.std(y_p) < 1e-9 or np.std(y_t) < 1e-9:
        return 0.0
    return float(np.corrcoef(y_t, y_p)[0, 1])


def train():
    df = pd.read_csv("train.csv")

    X = df[FEATURE_COLS].values
    y = df['Return5min'].values
    groups = df['Day'].values

    gkf = GroupKFold(n_splits=5)

    results = []

    print("开始 5 折交叉验证 (Ridge Regression)...")
    print(f"特征数量: {len(FEATURE_COLS)}, Ridge alpha: {RIDGE_ALPHA}")
    print(f"{'Fold':<6} | {'Test Day':<8} | {'IC':<10}")
    print("-" * 32)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        test_day = groups[test_idx][0]

        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        ic = ic_score(y_test, pred)

        print(f"Fold {fold+1}  | Day {test_day}    | {ic:.6f}")
        results.append(ic)

    print("-" * 32)

    mean_ic = np.mean(results)
    std_ic  = np.std(results)
    icir    = mean_ic / std_ic if std_ic > 1e-9 else 0.0

    print(f"平均 IC: {mean_ic:.6f}")
    print(f"IC Std : {std_ic:.6f}")
    print(f"ICIR   : {icir:.4f}")
    return results


if __name__ == "__main__":
    train()