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

#### 重要提示

1. **Return5min 列不可见**：测试集将不包含该字段(需要预测，但是五分钟前的Return5min可以被计算)
2. **禁止访问未来数据**：`online_predict()` 只能使用当前及历史数据，传入一个tick(防止未来信息泄露)
3. **评估指标**：预测值与真实值的皮尔森相关系数(IC)，按日计算后取平均
4. **避免无效IC结果**: 避免直接使用**价格类序列**(如BidPrice1、AskPrice1等)做特征或者预测值产生的高IC无效情况，比赛会对预测值做交易信号验证，以确保IC结果的预测有效性；验证方法暂不对比赛人员开放
5. **时间连贯性说明**: 训练的5日数据在时间上是连贯的。但在线测试的n日数据在时间上是不连贯的，没有依赖关系，是完全独立的。且在线测试的n日数据不在时间上接着训练的5日数据。
注: 由于预测时每日独立，我们在训练时将不严格遵循日间时序关系，且基于历史窗口的特征在每日刚开始时，将不会使用过去一日的结尾数据，而是使用今日现有历史的平均值替代

---
##### 实现方案

data_processor.py进行数据预处理并计算特征，保存至train.csv
train_model.py使用train.csv训练模型，使用严格的五折交叉验证，每次使用一天做验证集，其余四天做训练集，输出五折IC结果、平均IC与ICIR，以评估模型效果，持续优化(五折IC结果、平均IC与ICIR是严格的评判标准)
待训练结果满意后，再保存模型，编写MyModel.py实现在线预测接口