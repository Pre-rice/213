"""
MyModel.py - 最小化实现版本
"""

import pickle
import numpy as np
from typing import Dict, List

class MyModel:
    """最小化实现的预测模型"""
    
    def __init__(self):
        """初始化模型"""
        self.model = None
        self.features = []
        
        # 尝试加载预训练模型
        try:
            with open('simple_model.pkl', 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.features = data['feature_columns']
                print("✅ 模型加载成功")
        except:
            print("⚠️  模型加载失败，将使用随机预测")
    
    def reset(self):
        """重置模型状态"""
        # 这个方法必须存在，即使不需要重置任何东西
        pass
    
    def _get_features(self, E_row: Dict, sector_rows: List[Dict]) -> np.ndarray:
        """提取特征（最小化版本）"""
        try:
            # 只计算3个最基本特征
            bid = float(E_row.get('BidPrice1', 0))
            ask = float(E_row.get('AskPrice1', 0))
            
            # 1. 相对价差
            spread = ask - bid
            mid = (bid + ask) / 2
            rel_spread = spread / mid if mid > 0 else 0
            
            # 2. 订单不平衡
            buy_vol = float(E_row.get('OrderBuyVolume', 0))
            sell_vol = float(E_row.get('OrderSellVolume', 0))
            order_imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol + 0.001)
            
            # 3. 中间价
            mid_price = mid
            
            # 返回特征向量
            return np.array([[rel_spread, order_imbalance, mid_price]])
            
        except:
            return np.array([[0, 0, 0]])
    
    def online_predict(self, E_row: Dict, sector_rows: List[Dict]) -> float:
        """在线预测"""
        try:
            if self.model is not None:
                # 提取特征
                features = self._get_features(E_row, sector_rows)
                # 预测
                pred = self.model.predict(features)[0]
                return float(pred)
            else:
                # 后备：返回0（中性预测）
                return 0.0
        except:
            # 出错时返回0
            return 0.0