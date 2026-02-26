import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# ===================== 全局配置（贴合比赛目录/规则，可按需微调）=====================
# 目录结构（原始数据/处理后数据路径
DATA_ROOT = os.path.join("data")    # 原始数据：data/1/A.csv ~ data/5/E.csv
OUTPUT_ROOT = os.path.join("output")# 输出数据：单股清洗/多股面板/指标数据
# 比赛固定参数
STOCK_CODES = ["A", "B", "C", "D", "E"]       # 板块5只股票，E为预测基准
TRADING_DAYS = ["1", "2", "3", "4", "5"]      # 5个交易日
TIME_INTERVAL = 500                           # 时间戳步长：500ms
# 交易时段规则（HHMMSSmmm格式，严格剔除11:30-13:00午休）
TRADING_RULES = {
    "morning_start": 93000000,    # 09:30:00.000
    "morning_end": 112959500,     # 11:29:59.500
    "afternoon_start": 130000000, # 13:00:00.000
    "afternoon_end": 145000000    # 14:50:00.000
}
# 核心字段定义（覆盖比赛原生字段，分类管理方便处理）
CORE_COLS = [
    "Time", "BidPrice1", "BidPrice2", "BidPrice3", "BidPrice4", "BidPrice5",
    "BidVolume1", "BidVolume2", "BidVolume3", "BidVolume4", "BidVolume5",
    "AskPrice1", "AskPrice2", "AskPrice3", "AskPrice4", "AskPrice5",
    "AskVolume1", "AskVolume2", "AskVolume3", "AskVolume4", "AskVolume5",
    "OrderBuyNum", "OrderSellNum", "OrderBuyVolume", "OrderSellVolume",
    "TradeBuyNum", "TradeSellNum", "TradeBuyVolume", "TradeSellVolume",
    "TradeBuyAmount", "TradeSellAmount", "LastPrice", "Return5min"
]
PRICE_COLS = [col for col in CORE_COLS if "Price" in col] + ["LastPrice"]
VOLUME_COLS = [col for col in CORE_COLS if "Volume" in col]
BID_VOL_COLS = [col for col in CORE_COLS if "BidVolume" in col]
AMOUNT_COLS = ["TradeBuyAmount", "TradeSellAmount"]
ORDER_NUM_COLS = [col for col in CORE_COLS if "Num" in col]

# ===================== 工具函数：目录初始化 =====================
def init_directories():
    """初始化输出目录，按交易日/单股/多股/指标分层"""
    for day in TRADING_DAYS:
        os.makedirs(os.path.join(OUTPUT_ROOT, day, "single_stock"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "multi_stock_panel"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "indicators"), exist_ok=True)
    print("✅ 目录初始化完成")

# ===================== 核心1：加载数据（适配比赛目录结构）=====================
def load_data(day: str, stock_code: str) -> pd.DataFrame:
    """
    加载单交易日单股票原始数据
    :param day: 交易日（1/2/3/4/5）
    :param stock_code: 股票代码（A/B/C/D/E）
    :return: 清洗字段后的原始DataFrame
    """
    file_path = os.path.join(DATA_ROOT, day, f"{stock_code}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"原始数据文件不存在：{file_path}")
    
    # 读取数据，指定字段类型，过滤冗余列
    df = pd.read_csv(
        file_path,
        usecols=CORE_COLS,
        dtype={
            "Time": np.int64,
            **{col: np.int64 for col in PRICE_COLS + VOLUME_COLS + ORDER_NUM_COLS},
            **{col: np.float64 for col in AMOUNT_COLS},
            "Return5min": np.float64
        },
        na_values=["", "NaN", "null", "-"]
    )
    print(f"📥 加载完成：{day}日-{stock_code}，原始行数：{df.shape[0]}")
    return df

# ===================== 核心2：单股处理（清洗+对齐+标准化）=====================
def process_single_stock(day: str, stock_code: str) -> pd.DataFrame:
    """
    单股完整处理流程：加载→去重排序→时段过滤→异常值清洗→500ms对齐→缺失值填充→时间标准化
    满足要求：无重复/乱序、剔除午休、仅ffill填充、新增datetime、无NaN/0填充
    """
    # 1. 加载原始数据
    df = load_data(day, stock_code)
    if df.empty:
        print(f" {day}日-{stock_code} 原始数据为空，跳过")
        return pd.DataFrame()
    
    # 2. 基础清洗：去重+时序排序
    df = df.drop_duplicates(subset=["Time"], keep="first")  # 去重重复时间戳
    df = df.sort_values(by="Time", ascending=True).reset_index(drop=True)  # 严格升序
    
    # 3. 时段过滤：完全剔除午休/盘前/盘后，仅保留合规交易时间
    df = df[
        ((df["Time"] >= TRADING_RULES["morning_start"]) & (df["Time"] <= TRADING_RULES["morning_end"])) |
        ((df["Time"] >= TRADING_RULES["afternoon_start"]) & (df["Time"] <= TRADING_RULES["afternoon_end"]))
    ].reset_index(drop=True)
    if df.empty:
        print(f"{day}日-{stock_code} 无合规交易时段数据，跳过")
        return pd.DataFrame()
    
    # 4. 异常值清洗：严格符合业务规则
    df = df[
        (df["BidPrice1"] < df["AskPrice1"]) &  # 买一价 < 卖一价
        (df[PRICE_COLS] > 0).all(axis=1) &     # 所有价格>0
        (df[VOLUME_COLS] >= 0).all(axis=1) &   # 所有量能非负
        (df[BID_VOL_COLS] % 100 == 0).all(axis=1)  # 买方挂单量为100整数倍
    ].reset_index(drop=True)
    if df.empty:
        print(f"⚠️ {day}日-{stock_code} 异常值清洗后无数据，跳过")
        return pd.DataFrame()
    
    # 5. 生成500ms基准时间轴（早盘+午盘，剔除午休）
    def gen_500ms_timeaxis(start: int, end: int) -> list:
        """生成HHMMSSmmm格式的500ms间隔时间轴"""
        time_list = []
        current = start
        while current <= end:
            time_list.append(current)
            current += TIME_INTERVAL
            # 处理时间进位（毫秒→秒→分→时）
            if current % 1000000 >= 60000:  # 秒进位（如xx:xx:59.500 → xx:xx+1:00.000）
                current += 40000
            if current % 100000000 >= 60000000:  # 分进位（如xx:59:59.500 → xx+1:00:00.000）
                current += 40000000
        return time_list
    # 合并早盘+午盘时间轴，无午休
    full_timeaxis = gen_500ms_timeaxis(TRADING_RULES["morning_start"], TRADING_RULES["morning_end"]) + \
                    gen_500ms_timeaxis(TRADING_RULES["afternoon_start"], TRADING_RULES["afternoon_end"])
    df_timeaxis = pd.DataFrame({"Time": full_timeaxis})
    
    # 6. 时序对齐：左连接基准时间轴，保证500ms严格递增
    df_aligned = pd.merge(df_timeaxis, df, on="Time", how="left")
    
    # 7. 缺失值处理：仅前向填充ffill（绝对禁止bfill，避免未来数据泄露）
    df_aligned = df_aligned.ffill()
    # 开盘首行缺失：用第一条有效数据填充（仅1次，不跨时段）
    df_aligned = df_aligned.bfill(limit=1)
    # 剔除极端缺失行
    df_aligned = df_aligned.dropna().reset_index(drop=True)
    
    # 8. 时间标准化：新增datetime字段（HHMMSSmmm→标准时间格式），不修改原始Time
    day_date = f"2024-01-0{day}"  # 虚拟日期，避免跨交易日时间冲突
    df_aligned["datetime"] = pd.to_datetime(
        day_date + " " + df_aligned["Time"].astype(str).str.zfill(9)
        .str.replace(r"(\d{2})(\d{2})(\d{2})(\d{3})", r"\1:\2:\3.\4", regex=True)
    )
    
    # 9. 数据类型二次校准，避免填充后类型异常
    df_aligned["Time"] = df_aligned["Time"].astype(np.int64)
    for col in PRICE_COLS + VOLUME_COLS + ORDER_NUM_COLS:
        df_aligned[col] = df_aligned[col].astype(np.int64)
    for col in AMOUNT_COLS:
        df_aligned[col] = df_aligned[col].astype(np.float64)
    
    # 10. 保存单股清洗后数据
    save_path = os.path.join(OUTPUT_ROOT, day, "single_stock", f"{stock_code}_cleaned.csv")
    df_aligned.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✅ 单股处理完成：{day}日-{stock_code}，对齐后行数：{df_aligned.shape[0]}")
    return df_aligned

# ===================== 核心3：多股处理（以E为基准对齐，生成面板数据）=====================
def process_multi_stock(day: str) -> pd.DataFrame:
    """
    单交易日多股对齐：以E股时间轴为唯一基准，合并A/B/C/D/E数据，字段加后缀避免冲突
    满足要求：一行一个时间戳、E原生字段、其他股加后缀、无NaN/无午休、无未来数据
    """
    # 1. 加载当日所有股票清洗后的数据
    stock_data = {}
    for code in STOCK_CODES:
        file_path = os.path.join(OUTPUT_ROOT, day, "single_stock", f"{code}_cleaned.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"单股清洗数据缺失：{file_path}")
        df = pd.read_csv(file_path, parse_dates=["datetime"])
        stock_data[code] = df
    
    # 2. 以E股为基准，E保留原生字段名，其他股票字段加【_代码】后缀
    df_e = stock_data["E"].copy()
    for code in ["A", "B", "C", "D"]:
        df_temp = stock_data[code].copy()
        # 仅对业务字段加后缀，时间字段保留原生名用于对齐
        rename_cols = {col: f"{col}_{code}" for col in df_temp.columns if col not in ["Time", "datetime"]}
        df_temp = df_temp.rename(columns=rename_cols)
        # 左连接：严格以E股时间轴为基准，保证时序完全一致
        df_e = pd.merge(df_e, df_temp, on=["Time", "datetime"], how="left")
    
    # 3. 最终填充：仅前向填充，确保无NaN
    df_panel = df_e.ffill().dropna().reset_index(drop=True)
    
    # 4. 保存多股面板数据
    save_path = os.path.join(OUTPUT_ROOT, "multi_stock_panel", f"day_{day}_panel.csv")
    df_panel.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✅ 多股对齐完成：{day}日，面板维度：{df_panel.shape[0]}行 × {df_panel.shape[1]}列")
    return df_panel

# ===================== 核心4：计算基本指标（中间价/价差/订单流）=====================
def calculate_basic_indicators(df: pd.DataFrame, is_single_stock: bool = True, stock_code: str = None) -> pd.DataFrame:
    """
    计算金融基础指标，支持单股数据/多股面板数据
    指标：中间价、绝对价差、相对价差、订单流、累计订单流
    :param df: 单股清洗后数据 / 多股面板数据
    :param is_single_stock: 是否为单股数据
    :param stock_code: 单股代码（A/B/C/D/E），多股时为None
    :return: 带指标的DataFrame
    """
    df_indicator = df.copy()
    prefix = "" if is_single_stock else f"{stock_code}_" if stock_code else ""
    
    # 计算核心指标（基于买一/卖一/最新价，最具代表性）
    bid1 = f"{prefix}BidPrice1"
    ask1 = f"{prefix}AskPrice1"
    last = f"{prefix}LastPrice"
    vol = f"{prefix}TradeBuyVolume" if is_single_stock else f"{prefix}TradeBuyVolume"
    
    # 1. 中间价 = (买一价 + 卖一价) / 2
    df_indicator[f"{prefix}mid_price"] = (df_indicator[bid1] + df_indicator[ask1]) / 2
    # 2. 绝对价差 = 卖一价 - 买一价
    df_indicator[f"{prefix}abs_spread"] = df_indicator[ask1] - df_indicator[bid1]
    # 3. 相对价差 = 绝对价差 / 中间价（避免除零，加极小值）
    df_indicator[f"{prefix}rel_spread"] = df_indicator[f"{prefix}abs_spread"] / (df_indicator[f"{prefix}mid_price"] + 1e-8)
    # 4. 订单流：主动买=正，主动卖=负（最新价≥中间价→主动买，反之主动卖）
    df_indicator[f"{prefix}order_flow"] = np.where(
        df_indicator[last] >= df_indicator[f"{prefix}mid_price"],
        df_indicator[vol],
        -df_indicator[vol]
    )
    # 5. 累计订单流（时序累计，反映资金趋势）
    df_indicator[f"{prefix}cum_order_flow"] = df_indicator[f"{prefix}order_flow"].cumsum()
    
    print(f" 指标计算完成：{('单股' if is_single_stock else '多股')}，新增{5}个基础指标")
    return df_indicator

# ===================== 主流程：一键执行全流程 =====================
def main():
    # 1. 初始化目录
    init_directories()
    # 2. 逐交易日处理：单股清洗 → 多股对齐 → 指标计算
    all_day_panel = []
    for day in tqdm(TRADING_DAYS, desc="全流程处理进度"):
        print(f"\n===== 开始处理【{day}日】数据 =====")
        # 2.1 单股批量处理
        for code in STOCK_CODES:
            df_single = process_single_stock(day, code)
            if not df_single.empty:
                # 单股指标计算并保存
                df_single_indicator = calculate_basic_indicators(df_single, is_single_stock=True, stock_code=code)
                save_path = os.path.join(OUTPUT_ROOT, day, "single_stock", f"{code}_cleaned_indicator.csv")
                df_single_indicator.to_csv(save_path, index=False, encoding="utf-8-sig")
        # 2.2 多股对齐
        df_panel = process_multi_stock(day)
        all_day_panel.append(df_panel)
        # 2.3 多股面板指标计算（E+A/B/C/D分别计算）
        df_panel_indicator = df_panel.copy()
        # E股指标（原生字段）
        df_panel_indicator = calculate_basic_indicators(df_panel_indicator, is_single_stock=True, stock_code="")
        # A/B/C/D股指标（加后缀）
        for code in ["A", "B", "C", "D"]:
            df_panel_indicator = calculate_basic_indicators(df_panel_indicator, is_single_stock=False, stock_code=code)
        # 保存多股指标面板
        save_path = os.path.join(OUTPUT_ROOT, "indicators", f"day_{day}_panel_indicator.csv")
        df_panel_indicator.to_csv(save_path, index=False, encoding="utf-8-sig")
    
    # 3. 合并所有交易日的多股面板数据（可直接用于模型训练）
    df_full_panel = pd.concat(all_day_panel, axis=0).sort_values(by="datetime").reset_index(drop=True)
    df_full_panel.to_csv(os.path.join(OUTPUT_ROOT, "multi_stock_panel", "all_days_full_panel.csv"), 
                         index=False, encoding="utf-8-sig")
    # 4. 合并所有交易日指标面板
    df_full_indicator = []
    for day in TRADING_DAYS:
        df = pd.read_csv(os.path.join(OUTPUT_ROOT, "indicators", f"day_{day}_panel_indicator.csv"), parse_dates=["datetime"])
        df_full_indicator.append(df)
    df_full_indicator = pd.concat(df_full_indicator, axis=0).sort_values(by="datetime").reset_index(drop=True)
    df_full_indicator.to_csv(os.path.join(OUTPUT_ROOT, "indicators", "all_days_full_indicator.csv"), 
                             index=False, encoding="utf-8-sig")
    
    print(f"\n===== 全流程处理完成 =====\n📁 输出根目录：{OUTPUT_ROOT}\n📈 全量面板行数：{df_full_panel.shape[0]}\n📊 全量指标面板行数：{df_full_indicator.shape[0]}")

# ===================== 执行入口 =====================
if __name__ == "__main__":
    main()