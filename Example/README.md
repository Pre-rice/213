# 股票收益率预测比赛 - 参赛者指南

## 文件结构

```
Example/
├── main.py          # [必须提交] 预测结果生成文件
├── model.py         # [必须提交] 模型定义文件
├── utils.py         # [必须提交] 工具函数文件
├── README.md        # 本说明文件
├── data/            # 示例数据目录（不提交）
│   └── 1/
│       ├── A.csv
│       ├── B.csv
│       ├── C.csv
│       ├── D.csv
│       └── E.csv
└── output/          # 预测结果输出目录
```

## 注释与命名规范

 - 代码需符合语言标准，变量名采用下划线命名法（如train_data）。
 - 关键步骤添加注释说明，例如：
    - 数据预处理：去除缺失值（作者：李四）
    - des_data = src_data.dropna()

## 接口规范

### 固定接口（不可修改）

您的模型类 `MyModel` 必须实现以下两个方法：

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
            float: 预测股票 E 的`Return5min`
        """
        pass
```

### 自由部分（可自定义）

- `save_data()` 等辅助方法：名称和实现完全自由

---

## 本地测试提交

```bash
# 在 Example 目录下运行
python main.py
```

测试脚本会：
1. 遍历 `data/` 目录下每一天的数据
2. 调用 `model.reset()` 重置状态
3. 逐 tick 调用 `model.online_predict()` 获取预测
4. 计算并输出每日预测结果并在output目录下保存成预测结果文件

---

## 提交要求

| 文件 | 是否必须 | 说明 |
|------|---------|------|
| `main.py` | 必须 | 官方固定生成脚本 |
| `MyModel.py` | 必须 |包含 `MyModel` 类 |
| `utils.py` 等辅助文件 | 可选 | 自定义工具函数 |
| 模型权重文件 | 可选 | 如 `.pth`, `.pkl`, `.joblib` 等 |
| `data/` 目录 | 不提交 | 仅用于本地调试 |

---

## 重要提示

1. **Return5min 列不可见**：测试集将不包含该字段
2. **禁止访问未来数据**：`online_predict()` 只能使用当前及历史数据，传入一个tick`
3. **评估指标**：预测值与真实值的皮尔森相关系数（IC），按日计算后取平均
4. **避免无效IC结果**: 避免直接使用价格类序列（如BidPrice1、AskPrice1等）做特征或者预测值产生的高IC无效情况,比赛会对预测值做交易信号验证，以确保IC结果的预测有效性；验证方法暂不对比赛人员开放


---

## 依赖环境说明

为保证线上评测环境可复现并顺利运行，要求提交 `requirements.txt`，评测端会在运行前执行：

```bash
pip install -r requirements.txt
```

- 如果你使用了深度学习框架（如 PyTorch）、特征处理库等，请将对应包及版本写入 `requirements.txt`。
- 建议尽量固定版本号（例如 `torch==2.1.2`），避免不同环境版本差异导致结果不一致。
- 如无额外依赖，也请提交 `requirements.txt`（可仅包含 numpy/pandas）。
