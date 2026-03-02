# 模型训练与保存

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer
import warnings

warnings.filterwarnings('ignore')

def ic_score(y_true, y_pred):
    y_pred = np.clip(y_pred, -0.02, 0.02)
    if np.std(y_pred) < 1e-6 or np.std(y_true) < 1e-6:
        return 0.0
    return np.corrcoef(y_true, y_pred)[0, 1]

ic_scorer = make_scorer(ic_score, greater_is_better=True)

def train():
    df = pd.read_csv("train.csv")
    features = [
        'TotalBidVol', 
        'TradeImbalance_mean_600', 
        'TradeImbalance_diff',
        'AmountImbalance_mean_600',
        'Sector_OrderImbalance',
        'TradeImbalance_pulse'
    ]
    target = 'Return5min'
    
    X = df[features].values
    y = df[target].values
    
    split = 27601 * 4
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    tscv = TimeSeriesSplit(n_splits=4)
    
    # 参数网格：保持保守，尝试微小的结构优化
    param_grid = {
        'num_leaves': [3, 5],          # 尝试稍微复杂一点
        'min_child_samples': [300, 500],
        'learning_rate': [0.01],                   
        'n_estimators': [1000],                   
        'reg_alpha': [1.0, 2.0],            
        'reg_lambda': [0.1, 0.5],          
        'feature_fraction': [0.9],            
        'bagging_fraction': [0.7],           
        'bagging_freq': [1]
    }
    
    estimator = lgb.LGBMRegressor(
        objective='regression',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    print("开始 Grid Search...")
    
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=ic_scorer,
        cv=tscv,
        verbose=1,
        n_jobs=-1,
        refit=True
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\n========================================")
    print(f"Best CV Score: {grid_search.best_score_:.5f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    print("========================================")
    
    best_model = grid_search.best_estimator_
    
    pred_test = best_model.predict(X_test)
    ic_test = ic_score(y_test, pred_test)
    print(f"Final Validation IC (Day 5): {ic_test:.5f}")
    
    best_model.booster_.save_model("model.txt")
    print("模型已保存")
    print("Feature Importances:", best_model.feature_importances_)

if __name__ == "__main__":
    train()