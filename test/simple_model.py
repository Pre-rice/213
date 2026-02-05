"""
最简单的股票收益率预测模型 - 添加模型保存功能
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle  # 用于保存模型
import os

class SimpleStockPredictor:
    """最简单的股票预测器（带保存功能）"""
    
    def __init__(self, model_path="simple_model.pkl"):
        """
        初始化预测器
        
        Args:
            model_path: 模型保存路径
        """
        self.model = LinearRegression()
        self.feature_columns = []  # 记录使用的特征列
        self.model_path = model_path
        
    def save_model(self):
        """
        保存模型到文件
        """
        try:
            # 准备要保存的数据
            model_data = {
                'model': self.model,
                'feature_columns': self.feature_columns,
                'coef': self.model.coef_ if hasattr(self.model, 'coef_') else None,
                'intercept': self.model.intercept_ if hasattr(self.model, 'intercept_') else None
            }
            
            # 使用pickle保存
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✅ 模型已保存到: {self.model_path}")
            print(f"   文件大小: {os.path.getsize(self.model_path)/1024:.2f} KB")
            return True
            
        except Exception as e:
            print(f"❌ 保存模型失败: {e}")
            return False
    
    def load_model(self):
        """
        从文件加载模型
        """
        try:
            if not os.path.exists(self.model_path):
                print(f"❌ 模型文件不存在: {self.model_path}")
                return False
            
            # 使用pickle加载
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # 恢复模型状态
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            
            print(f"✅ 模型加载成功: {self.model_path}")
            print(f"   特征列: {self.feature_columns}")
            print(f"   系数: {self.model.coef_}")
            print(f"   截距: {self.model.intercept_:.6f}")
            return True
            
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            return False
    
    def predict_new(self, new_features):
        """
        使用加载的模型进行预测
        
        Args:
            new_features: 新的特征数据，可以是：
                - 字典：{'Spread': 1.5, 'OrderImbalance': 0.2, 'MidPrice': 100.0}
                - 列表：[1.5, 0.2, 100.0]
                - numpy数组
        Returns:
            预测的Return5min值
        """
        if not hasattr(self.model, 'coef_'):
            print("❌ 模型未训练或未加载！")
            return None
        
        try:
            # 转换输入为合适的格式
            if isinstance(new_features, dict):
                # 确保顺序与训练时一致
                X_new = np.array([[new_features.get(col, 0) for col in self.feature_columns]])
            elif isinstance(new_features, list):
                X_new = np.array([new_features])
            else:
                X_new = new_features.reshape(1, -1) if len(new_features.shape) == 1 else new_features
            
            # 进行预测
            prediction = self.model.predict(X_new)[0]
            print(f"📊 预测结果: Return5min = {prediction:.6f}")
            return prediction
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return None
    
    # 原来的其他方法保持不变...
    def load_and_prepare_data(self, data_path):
        """
        加载数据并准备特征和标签
        """
        print("1. 加载数据...")
        df = pd.read_csv(data_path)
        
        # 显示数据基本信息
        print(f"  数据形状: {df.shape}")
        print(f"  数据列名: {list(df.columns)}")
        
        # 2. 计算几个最简单的特征
        print("\n2. 计算基本特征...")
        
        # 使用已经预处理好的特征，如果没有就计算
        if 'MidPrice' not in df.columns:
            df['MidPrice'] = (df['BidPrice1'] + df['AskPrice1']) / 2
        
        if 'Spread' not in df.columns:
            df['Spread'] = df['AskPrice1'] - df['BidPrice1']
        
        if 'OrderImbalance' not in df.columns:
            df['OrderImbalance'] = (df['OrderBuyVolume'] - df['OrderSellVolume']) / (
                df['OrderBuyVolume'] + df['OrderSellVolume'] + 1e-10)
        
        # 3. 选择特征 - 使用最简单的3个特征
        print("\n3. 选择特征...")
        self.feature_columns = [
            'Spread',          # 买卖价差（流动性）
            'OrderImbalance',  # 订单流不平衡
            'MidPrice'         # 中间价（注意：实际比赛中要谨慎使用）
        ]
        
        # 创建特征矩阵X
        X = df[self.feature_columns].values
        
        # 4. 准备标签y - Return5min
        print("\n4. 准备标签...")
        y = df['Return5min'].values
        
        # 检查数据
        print(f"  特征形状: {X.shape}")
        print(f"  标签形状: {y.shape}")
        print(f"  使用的特征: {self.feature_columns}")
        
        return X, y, df
    
    def split_data(self, X, y, test_size=0.2):
        """
        按时间顺序划分训练集和测试集
        """
        print(f"\n6. 划分数据（测试集比例: {test_size*100}%）...")
        
        # 计算分割点
        split_idx = int(len(X) * (1 - test_size))
        
        # 划分训练集和测试集
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"  训练集: {X_train.shape[0]} 个样本")
        print(f"  测试集: {X_test.shape[0]} 个样本")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        训练线性回归模型
        """
        print("\n7. 训练模型...")
        
        # 创建并训练模型
        self.model.fit(X_train, y_train)
        
        print("  模型训练完成!")
        print(f"  系数: {self.model.coef_}")
        print(f"  截距: {self.model.intercept_:.6f}")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        评估模型性能
        """
        print("\n8. 评估模型...")
        
        # 在测试集上预测
        y_pred_test = self.model.predict(X_test)
        
        # 计算测试集指标
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)
        
        # 计算IC值（皮尔森相关系数）
        ic_test = np.corrcoef(y_test, y_pred_test)[0, 1]
        
        print("  测试集结果:")
        print(f"    MSE: {mse_test:.6f}")
        print(f"    R²: {r2_test:.6f}")
        print(f"    IC: {ic_test:.6f}")
        
        return y_pred_test, ic_test
    
    def run_training_pipeline(self, data_path):
        """
        运行完整的训练流程（训练+保存）
        """
        print("=" * 60)
        print("开始训练模型")
        print("=" * 60)
        
        # 1. 加载和准备数据
        X, y, df = self.load_and_prepare_data(data_path)
        
        # 2. 划分数据
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size=0.2)
        
        # 3. 训练模型
        self.train_model(X_train, y_train)
        
        # 4. 评估模型
        y_pred, ic_value = self.evaluate_model(X_test, y_test)
        
        # 5. 保存模型
        self.save_model()
        
        print("\n" + "=" * 60)
        print(f"训练完成！最终IC值: {ic_value:.6f}")
        print("=" * 60)
        
        return ic_value


def test_save_load():
    """测试保存和加载功能"""
    print("\n" + "=" * 60)
    print("测试模型保存和加载功能")
    print("=" * 60)
    
    # 1. 创建一个简单的模型并训练
    print("\n1. 创建和训练模型...")
    test_predictor = SimpleStockPredictor("test_model.pkl")
    
    # 创建一些虚拟数据用于测试
    np.random.seed(42)
    X_dummy = np.random.randn(100, 3)  # 100个样本，3个特征
    y_dummy = np.random.randn(100)     # 100个标签
    test_predictor.feature_columns = ['Feature1', 'Feature2', 'Feature3']
    
    # 训练模型
    test_predictor.model.fit(X_dummy, y_dummy)
    print(f"   训练完成，系数: {test_predictor.model.coef_}")
    
    # 2. 保存模型
    print("\n2. 保存模型...")
    test_predictor.save_model()
    
    # 3. 创建新的预测器并加载模型
    print("\n3. 创建新预测器并加载模型...")
    new_predictor = SimpleStockPredictor("test_model.pkl")
    new_predictor.load_model()
    
    # 4. 测试预测
    print("\n4. 测试预测功能...")
    test_input = [1.0, 0.5, 2.0]  # 3个特征值
    prediction = new_predictor.predict_new(test_input)
    
    print("\n✅ 保存和加载测试完成！")
    
    # 清理测试文件
    if os.path.exists("test_model.pkl"):
        os.remove("test_model.pkl")
        print("🧹 已清理测试文件")


def main():
    """主函数"""
    
    # 数据路径
    data_path = "processed_data/E_processed.csv"
    if not os.path.exists(data_path):
        data_path = "train_data/E.csv"
    
    print("请选择操作:")
    print("1. 训练新模型并保存")
    print("2. 加载已有模型并进行预测")
    print("3. 测试保存/加载功能")
    
    choice = input("\n请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 训练新模型
        predictor = SimpleStockPredictor("simple_model.pkl")
        ic_value = predictor.run_training_pipeline(data_path)
        
        # 询问是否测试预测
        test_pred = input("\n是否测试新数据的预测？ (y/n): ").strip().lower()
        if test_pred == 'y':
            print("\n测试预测（输入3个特征值）:")
            try:
                spread = float(input("Spread (买卖价差): "))
                imbalance = float(input("OrderImbalance (订单不平衡): "))
                midprice = float(input("MidPrice (中间价): "))
                
                predictor.predict_new({
                    'Spread': spread,
                    'OrderImbalance': imbalance,
                    'MidPrice': midprice
                })
            except:
                print("输入无效，跳过测试")
    
    elif choice == "2":
        # 加载已有模型
        model_path = input("请输入模型路径 (默认: simple_model.pkl): ").strip()
        if not model_path:
            model_path = "simple_model.pkl"
        
        predictor = SimpleStockPredictor(model_path)
        
        if predictor.load_model():
            # 进行预测
            print("\n开始预测...")
            while True:
                try:
                    print("\n输入特征值 (输入 'q' 退出):")
                    spread = input("Spread: ")
                    if spread.lower() == 'q':
                        break
                    
                    imbalance = input("OrderImbalance: ")
                    midprice = input("MidPrice: ")
                    
                    predictor.predict_new([
                        float(spread),
                        float(imbalance),
                        float(midprice)
                    ])
                except ValueError:
                    print("请输入有效的数字！")
                except KeyboardInterrupt:
                    print("\n退出预测")
                    break
    
    elif choice == "3":
        # 测试保存加载功能
        test_save_load()
    
    else:
        print("无效选择")


if __name__ == "__main__":
    main()