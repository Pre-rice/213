# 线性模型baseline r0

import numpy as np
from collections import deque

class MyModel:
    """
    在线预测模型
    使用线性回归，特征：
    1. total_bid_vol: 当前tick买方五档总挂单量
    2. trade_imbalance_vol_mean_5min: 过去5分钟主动成交不平衡率的均值（已修正符号）
    """
    
    def __init__(self):
        # 线性回归系数（从训练获得）
        self.coef_total_bid_vol = 7.1160777412900405e-08   # 约 0.00000007116
        self.coef_imbalance = 0.02067537961916712          # 约 0.02068
        self.intercept = 0.0                                # 假设无截距
        
        # 用于计算滚动均值的缓存（每个tick的主动成交不平衡率）
        self.imb_history = deque(maxlen=600)  # 5分钟 * 120 tick/分钟 = 600
        
    def reset(self):
        """每个交易日开始时调用，重置模型状态"""
        self.imb_history.clear()
        
    def online_predict(self, E_row, sector_rows):
        """
        在线预测接口
        
        Args:
            E_row: dict, 当前 tick 股票 E 的数据
                   例如: {'Time': 93000000, 'BidPrice1': 100, ...}
            sector_rows: list[dict], 其他股票数据 [A_row, B_row, C_row, D_row]
        
        Returns:
            float: 预测股票 E 的 Return5min
        """
        # 1. 计算 total_bid_vol（买方五档总挂单量）
        total_bid_vol = sum(E_row.get(f'BidVolume{i}', 0) for i in range(1, 6))
        
        # 2. 计算当前tick的主动成交不平衡率
        trade_buy = E_row.get('TradeBuyVolume', 0)
        trade_sell = E_row.get('TradeSellVolume', 0)
        total_trade = trade_buy + trade_sell
        
        if total_trade > 0:
            imb = (trade_buy - trade_sell) / total_trade
        else:
            imb = 0.0  # 无成交时不平衡率为0
        
        # 3. 将当前不平衡率加入历史缓存
        self.imb_history.append(imb)
        
        # 4. 计算过去5分钟平均不平衡率（若缓存不足，用已有均值代替）
        if len(self.imb_history) > 0:
            mean_imb = np.mean(self.imb_history)
        else:
            mean_imb = 0.0
        
        # 5. 修正符号（原始特征负相关，乘以 -1 使与目标正相关）
        feature_imbalance = -mean_imb
        
        # 6. 线性回归预测
        pred = (self.coef_total_bid_vol * total_bid_vol +
                self.coef_imbalance * feature_imbalance +
                self.intercept)
        
        return float(pred)