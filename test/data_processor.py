# 文件: simple_preprocess.py
import pandas as pd
import numpy as np

# 1. 加载数据
print("1. 加载数据...")
df = pd.read_csv("data/E.csv")
print(f"数据形状: {df.shape}")
print(f"数据列: {list(df.columns)}")

# 2. 简单查看数据
print("\n2. 数据基本信息:")
print(df.info())
print("\n前5行数据:")
print(df.head())

# 3. 检查缺失值
print("\n3. 检查缺失值:")
print(df.isnull().sum())

# 4. 计算几个最简单的特征
print("\n4. 计算基本特征...")

# 中间价
df['MidPrice'] = (df['BidPrice1'] + df['AskPrice1']) / 2

# 价差
df['Spread'] = df['AskPrice1'] - df['BidPrice1']

# 委托量差
df['BidAskImbalance'] = df['BidVolume1'] - df['AskVolume1']

# 订单流
df['OrderFlow'] = df['OrderBuyVolume'] - df['OrderSellVolume']

# 5. 保存结果
print("\n5. 保存处理后的数据...")
df.to_csv("processed_data/E_simple_processed.csv", index=False)

print(f"\n处理完成!")
print(f"原始列数: {len(list(pd.read_csv('data/E.csv').columns))}")
print(f"处理后列数: {len(df.columns)}")
print(f"新增列: {[col for col in df.columns if col not in pd.read_csv('data/E.csv').columns]}")

# 6. 显示结果
print("\n处理后的数据前5行:")
print(df[['Time', 'MidPrice', 'Spread', 'BidAskImbalance', 'OrderFlow', 'Return5min']].head())