# 核心特征LightGBM模型 r1

import numpy as np
import lightgbm as lgb
from collections import deque

class MyModel:
    def __init__(self):
        self.model = lgb.Booster(model_file='model.txt')
        self.feature_cols = [
            'TotalBidVol', 
            'TradeImbalance_mean_600', 
            'TradeImbalance_diff',
            'AmountImbalance_mean_600',
            'Sector_OrderImbalance',
            'TradeImbalance_pulse'
        ]
        
        self.trade_imb_history = deque(maxlen=600)
        self.amount_imb_history = deque(maxlen=600)

    def reset(self):
        self.trade_imb_history.clear()
        self.amount_imb_history.clear()

    def online_predict(self, E_row, sector_rows):
        # 1. 计算即时值
        total_bid_vol = sum(E_row.get(f'BidVolume{i}', 0) for i in range(1, 6))
        
        tb, ts = E_row['TradeBuyVolume'], E_row['TradeSellVolume']
        trade_imb = (tb - ts) / (tb + ts + 1e-6)
        
        ta_buy, ta_sell = E_row['TradeBuyAmount'], E_row['TradeSellAmount']
        amount_imb = (ta_buy - ta_sell) / (ta_buy + ta_sell + 1e-6)
        
        s_order_sum = 0
        for r in sector_rows:
            b1, a1 = r['BidVolume1'], r['AskVolume1']
            s_order_sum += (b1 - a1) / (b1 + a1 + 1e-6)
        sector_order = s_order_sum / len(sector_rows)
        
        # 2. 更新缓存
        self.trade_imb_history.append(trade_imb)
        self.amount_imb_history.append(amount_imb)
        
        # 3. 计算特征
        # 长期均值
        feat_trade_mean_600 = np.mean(self.trade_imb_history) if self.trade_imb_history else 0
        feat_amount_mean = np.mean(self.amount_imb_history) if self.amount_imb_history else 0
        
        # 差分
        feat_trade_diff = trade_imb - feat_trade_mean_600
        
        # 短期均值 (30秒)
        # 取最后 60 个元素，如果不足则全取
        recent_trade_imb = list(self.trade_imb_history)[-60:]
        feat_trade_mean_60 = np.mean(recent_trade_imb) if recent_trade_imb else 0
        
        # 脉冲
        feat_pulse = feat_trade_mean_60 - feat_trade_mean_600
        
        # 4. 组装
        X = np.array([
            total_bid_vol, 
            feat_trade_mean_600, 
            feat_trade_diff,
            feat_amount_mean,
            sector_order,
            feat_pulse
        ]).reshape(1, -1)
        
        pred = self.model.predict(X)[0]
        pred = np.clip(pred, -0.01, 0.01)
        
        return float(pred)