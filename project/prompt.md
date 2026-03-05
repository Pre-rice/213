以下是一个量化交易场景的机器学习比赛，请仔细阅读并分析比赛要求。我已经写了一个初步的版本，你将基于此版本不断优化模型。

# 比赛题目说明
### 股票未来收益率预测
**板块效应**指的是股票市场中同一板块的个股因共同特点或关联题材，呈现协同涨跌的联动现象。因而我们可以利用板块效应来分析股票的未来收益率。
##### 比赛目标  
现有A,B,C,D,E五只股票，他们属于相同风格概念板块，会呈现板块联动效应。你将利用股票E的自身数据信息和板块内其他股票数据信息构建模型来**预测股票E的未来5分钟收益率(Return5min)**。你的模型将根据测试集数据最终进行评估。
##### 评估指标
股票E的未来5分钟预测收益率，与真实收益率的**N日平均IC值**。排名按测试集平均IC值从大到小排名，平均IC值高则排名高。

##### 比赛数据
训练集数据包含A,B,C,D,E五只股票5日Tick级别(500ms)数据，总共33个字段，包含快照类、委托类、成交类三种数据类型。

| 字段 | 类型 | 说明 |
|------|------|------|
| **快照类数据** | | **当前时刻的供需数据(存量)** |
| Time | int64 | 时间格式：HHMMSSmmm<br>每日93000000(9:30:00.000)至112959500(11:29:59.500)、130000000(13:00:00.000)至145000000(14:50:00.000)
| BidPrice1 | int64 | 买方第1档价格 |
| BidPrice2 | int64 | 买方第2档价格 |
| BidPrice3 | int64 | 买方第3档价格 |
| BidPrice4 | int64 | 买方第4档价格 |
| BidPrice5 | int64 | 买方第5档价格 |
| BidVolume1 | int64 | 买方第1档挂单量 |
| BidVolume2 | int64 | 买方第2档挂单量 |
| BidVolume3 | int64 | 买方第3档挂单量 |
| BidVolume4 | int64 | 买方第4档挂单量 |
| BidVolume5 | int64 | 买方第5档挂单量 |
| AskPrice1 | int64 | 卖方第1档价格 |
| AskPrice2 | int64 | 卖方第2档价格 |
| AskPrice3 | int64 | 卖方第3档价格 |
| AskPrice4 | int64 | 卖方第4档价格 |
| AskPrice5 | int64 | 卖方第5档价格 |
| AskVolume1 | int64 | 卖方第1档挂单量 |
| AskVolume2 | int64 | 卖方第2档挂单量 |
| AskVolume3 | int64 | 卖方第3档挂单量 |
| AskVolume4 | int64 | 卖方第4档挂单量 |
| AskVolume5 | int64 | 卖方第5档挂单量 |
| **委托类数据** | | **过去区间(500ms)内新增的订单流** |
| OrderBuyNum | int64 | 区间买单笔数 |
| OrderSellNum | int64 | 区间卖单笔数 |
| OrderBuyVolume | int64 | 区间买单量 |
| OrderSellVolume | int64 | 区间卖单量 |
| **成交类数据** | | **过去区间内已匹配完成的交易流** |
| TradeBuyNum | int64 | 区间主动买单成交笔数 |
| TradeSellNum | int64 | 区间主动卖单成交笔数 |
| TradeBuyVolume | int64 | 区间主动买单成交量 |
| TradeSellVolume | int64 | 区间主动卖单成交量 |
| TradeBuyAmount | Int64 | 区间主动买单成交额 |
| TradeSellAmount | Int64 | 区间主动卖单成交额 |
| LastPrice | int64 | 最新成交价 |
| Return5min | Float64 | 未来5分钟收益率(即比赛要预测的值) |

注：`Return5min`的计算方式为`MidPrice`(`BidPrice1`与`AskPrice1`的均值)五分钟后的相对增幅((`MidPrice`(t+5)-`MidPrice`t)/`MidPrice`t)计算`Return5min`

---

##### 文件说明
```
Example/
├── main.py          # 官方固定的预测结果生成文件(已给定)
├── MyModel.py       # 模型定义文件(必须包含 MyModel 类并实现相关接口)
├── utils.py         # 工具函数文件(已给定)
├── data/            # 数据目录(已给定)
├── output/          # 预测结果输出目录
├── requirements.txt # 依赖环境说明
└── other            # 其余文件
```

#### main.py
import pandas as pd
import numpy as np
from utils import *
from MyModel import MyModel

def run_test():
    #########Load Model
    model = MyModel()

    #########Load Day Data
    days = get_day_folders("./data")

    #########Online Predict
    for day in days:
        model.reset()
        
        day_data = load_day_data("./data", day)
        n_ticks = len(day_data['E'])

        ticktimes = day_data['E'].values.T[0, :]
        my_preds = np.zeros((n_ticks))

        for tick_index in range(n_ticks):
            ###########Get Tick Data(E and Sector)
            E_row_data = day_data['E'].iloc[tick_index]
            sector_row_datas = [
                day_data['A'].iloc[tick_index],
                day_data['B'].iloc[tick_index],
                day_data['C'].iloc[tick_index],
                day_data['D'].iloc[tick_index]
            ]

            ###########Predict
            my_preds[tick_index] = model.online_predict(E_row_data, sector_row_datas)

        ###########Save Data
        if os.path.exists("./output/"+day) is not True:
            os.makedirs("./output/"+day)
        out_frame = pd.DataFrame(np.vstack(([ticktimes, my_preds])).T)
        columns = ['Time', 'Predict']
        out_frame.columns = columns
        out_frame.to_csv("./output/"+day+"/E.csv", index=False)
        print ("Submit Day", day)
    
if __name__ == '__main__':
    run_test()

#### utils.py
import os
import pandas as pd
import numpy as np

def get_day_folders(data_path):
    folders = []
    for name in os.listdir(data_path):
        full_path = os.path.join(data_path, name)
        if os.path.isdir(full_path) and name.isdigit():
            folders.append(name)
    folders.sort(key=lambda x: int(x))
    return folders

def load_day_data(data_path, day_folder):
    day_path = os.path.join(data_path, day_folder)
    data = {}
    for stock in ['A', 'B', 'C', 'D', 'E']:
        csv_path = os.path.join(day_path, f'{stock}.csv')
        if os.path.exists(csv_path):
            data[stock] = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"Missing file: {csv_path}")
    return data


def clean_data(data):
    data = np.where(np.isnan(data), 0, data)
    data = np.where(np.isinf(data), 0, data)
    data = np.where(np.isinf(-data), 0, data)
    return data


def evaluate_ic(my_preds, ground_truth):
    data = np.vstack((my_preds, ground_truth))
    data = clean_data(data)
    cor = np.corrcoef(data)[0, 1]
    return cor
      
#### data格式
data目录下有12345五个文件夹代表五天数据，每个文件夹下有ABCDE五个.csv文件保存五只股票的数据，每个文件有27602行，例如data/1/E.csv的前几行如下：
Time,BidPrice1,BidPrice2,BidPrice3,BidPrice4,BidPrice5,BidVolume1,BidVolume2,BidVolume3,BidVolume4,BidVolume5,AskPrice1,AskPrice2,AskPrice3,AskPrice4,AskPrice5,AskVolume1,AskVolume2,AskVolume3,AskVolume4,AskVolume5,OrderBuyNum,OrderSellNum,OrderBuyVolume,OrderSellVolume,TradeBuyNum,TradeSellNum,TradeBuyVolume,TradeSellVolume,TradeBuyAmount,TradeSellAmount,LastPrice,Return5min
93000000,5454,5450,5448,5446,5445,3000,900,1400,1700,100,5455,5456,0,0,0,123700,1867732,0,0,0,2995,2025,6783200,4355432,866,1405,629400,1614300,3433377000,8806006500,5455,-0.05666408368849283
93000500,5355,5320,5311,5310,5306,200,2700,1300,600,700,5399,5400,5401,5404,5407,500,14000,2800,1100,100,621,940,1609000,1720832,1295,109,1461900,76200,7955343298,411804500,5399,-0.041448770094906065
93001000,5386,5300,5292,5291,5288,2400,5700,1000,100,2400,5426,5436,5438,5440,5441,1300,800,400,59673,600,299,542,370300,905685,364,44,265600,24800,1437979600,131694800,5436,-0.04736995059575705
93001500,5400,5380,5303,5300,5292,7300,200,600,5300,1000,5445,5446,5447,5448,5449,4673,37800,6800,37700,14300,302,390,304900,636900,334,24,248700,12200,1349522435,65545900,5445,-0.049042367962855486

#### 接口规范

`MyModel.py` 中的模型类 `MyModel` 必须实现以下两个方法(其余辅助方法的名称和实现完全自由)：

```python
class MyModel:
    def reset(self):
        """每个交易日开始时调用，重置模型状态"""
        pass
    
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
        pass
```

#### requirements.txt
numpy==1.26.4
pandas==3.0.0
matplotlib==3.8.2
scikit-learn==1.4.0
lightGBM==4.6.0

#### 重要提示

1. **Return5min 列不可见**：测试集将不包含该字段(需要预测，但是五分钟前的Return5min可以被计算)
2. **禁止访问未来数据**：`online_predict()` 只能使用当前及历史数据，传入一个tick(防止未来信息泄露)
3. **评估指标**：预测值与真实值的皮尔森相关系数(IC)，按日计算后取平均
4. **避免无效IC结果**: 避免直接使用**价格类序列**(如BidPrice1、AskPrice1等)做特征或者预测值产生的高IC无效情况，比赛会对预测值做交易信号验证，以确保IC结果的预测有效性；验证方法暂不对比赛人员开放
5. **时间连贯性说明**: 训练的5日数据在时间上是连贯的。但在线测试的n日数据在时间上是不连贯的，没有依赖关系，是完全独立的。且在线测试的n日数据不在时间上接着训练的5日数据。
注: 由于预测时每日独立，我们在训练时将不严格遵循日间时序关系，且基于历史窗口的特征在每日刚开始时，将不会使用过去一日的结尾数据，而是使用今日现有历史的平均值替代

---
##### 初步方案

data_processor.py进行数据预处理并计算特征，保存至train.csv
train_model.py使用train.csv训练模型，使用严格的五折交叉验证，每次使用一天做验证集，其余四天做训练集，输出五折IC结果、平均IC与ICIR，以评估模型效果，持续优化(五折IC结果、平均IC与ICIR是严格的评判标准)
待训练结果满意后，再保存模型，编写MyModel.py实现在线预测接口(这一步没有我的指令不要执行，不断优化模型是你的首要任务)

目前的版本使用了六个特征，初步训练了一个线性模型，以下是代码与输出结果:

# data_processor.py
import os
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

python train_model.py
开始 5 折交叉验证 ...
Fold   | Test Day | IC        
------------------------------
Fold 1  | Day 5    | 0.155542
Fold 2  | Day 4    | 0.121979
Fold 3  | Day 3    | 0.146799
Fold 4  | Day 2    | 0.283276
Fold 5  | Day 1    | 0.185924
------------------------------
平均 IC: 0.178704
ICIR: 3.1831