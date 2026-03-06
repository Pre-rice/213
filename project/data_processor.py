import numpy as np
import pandas as pd
from utils import get_day_folders, load_day_data

# 收益率剪裁阈值（防止极端值）
_RET_CLIP_LONG  = 0.1    # 长窗口（≥120 tick）收益率剪裁
_RET_CLIP_SHORT = 0.05   # 短窗口（30/60 tick）收益率剪裁

# 日内上午/下午分割阈值（tick 索引，约对应 11:25AM 前后）
_AM_SPLIT = 13800


def _imb(a, b):
    """通用失衡公式: (a - b) / (a + b + 1e-6)"""
    return (a - b) / (a + b + 1e-6)


def _sma(s, w):
    return pd.Series(s).rolling(w, min_periods=1).mean()


def _ema(s, span):
    return pd.Series(s).ewm(span=span, adjust=False).mean()


def process_day_data(day_data):
    """
    处理单日全部股票数据，生成 E 股票的特征表。
    由于各股票时间戳完全对齐，可直接按行对应，无需 merge。

    共生成 41 个特征，覆盖多个多空动态视角，供动态集成模型使用：

    ── E 自身基础信号 ──────────────────────────────────────────────────────────
      1.  TotalBidVol         - E 五档总买量 (市场深度)
      2.  TradeImb_600        - E 成交量失衡 600 窗口 SMA (长期趋势)
      3.  TradeImb_diff       - E 当前成交量失衡与长期均值偏差

    ── E 成交量失衡脉冲 (SMA/EMA) ──────────────────────────────────────────────
      4.  TradeImb_p15        - TI SMA15 - SMA600 (极短期脉冲)
      5.  TradeImb_p30        - TI SMA30 - SMA600
      6.  TradeImb_p40        - TI SMA40 - SMA600
      7.  TradeImb_p60        - TI SMA60 - SMA600 (中期脉冲)
      8.  TradeImb_ep60       - TI EMA60 - EMA600 (指数加权近期变化)
      9.  TradeImb_ep15       - TI EMA15 - EMA600 (更短期 EMA 脉冲)

    ── E 委托量失衡 (OVI) 脉冲 ─────────────────────────────────────────────────
     10.  OVI_p15             - OVI SMA15 - SMA600 (短期挂单动能)
     11.  OVI_p30             - OVI SMA30 - SMA600
     12.  OVI_p60             - OVI SMA60 - SMA600 (中期挂单动能)
     13.  OVI_ep15            - OVI EMA15 - EMA600
     14.  OVI_ep5             - OVI EMA5  - EMA600 (超短期，供 niche 模型使用)

    ── E 委托笔数失衡 (ONI) 脉冲 ────────────────────────────────────────────────
     15.  ONI_p15             - ONI SMA15 - SMA600
     16.  ONI_p30             - ONI SMA30 - SMA600

    ── E 成交笔数失衡 EMA 脉冲 ─────────────────────────────────────────────────
     17.  TNI_ep15            - TNI EMA15 - EMA600

    ── E 挂单深度失衡 (OBI) ────────────────────────────────────────────────────
     18.  OBI1_p15            - E 一档买卖量失衡 SMA15 脉冲 (即时流动性压力)
     19.  OBI1_ep15           - E 一档买卖量失衡 EMA15 脉冲
     20.  OBI_total_p15       - E 五档总买卖量失衡 SMA15 脉冲 (全深度压力)

    ── 板块均值信号 ─────────────────────────────────────────────────────────────
     21.  Sect_OBI1           - 板块 (ABCD) 一档委托失衡均值
     22.  E_TI_rel_600        - E 的长期 TI 相对板块偏差 (均值回复)
     23.  Sect_TI_p40         - 板块成交量失衡 SMA40 脉冲
     24.  Sect_OVI_p20        - 板块委托量失衡 SMA20 脉冲
     25.  Sect_ONI_p30        - 板块委托笔数失衡 SMA30 脉冲
     26.  Sect_OVI_ep5        - 板块委托量失衡 EMA5  - EMA600 (超短期板块动能)
     27.  Sect_OVI_ep15       - 板块委托量失衡 EMA15 - EMA600

    ── 日内时间特征 ─────────────────────────────────────────────────────────────
     28.  aft_13800           - tick_index > 13800 二值 (交易日后半段指示)
     29.  aft_12000           - tick_index > 12000 二值 (稍早分界，与 28 互补)

    ── 滞后已实现收益特征 ─────────────────────────────────────────────────────
     30.  sect_ret_lag        - 板块(ABCD)过去5分钟平均已实现收益（动量/反转）
     31.  e_ret_lag           - E 过去5分钟已实现收益（均值回复信号）
     32.  past_ret_30         - E 中间价过去 15 秒收益率 (30 ticks)
     33.  past_ret_60         - E 中间价过去 30 秒收益率 (60 ticks)
     34.  past_ret_120        - E 中间价过去 1 分钟收益率 (120 ticks)
     35.  past_ret_300        - E 中间价过去 2.5 分钟收益率 (300 ticks)
     36.  past_ret_600        - E 中间价过去 5 分钟收益率 (600 ticks)

     37.  vwap_dev            - E 中间价偏离日内 VWAP（均值回复，剪裁 ±1%）

    ── 信号一致性交叉特征 (Signal Coherence Interactions) ──────────────────────
     38.  ovi_ti_short        - OVI_p15 × TradeImb_p15（短期订单-成交同向强化）
     39.  ovi_ti_medium       - OVI_p60 × TradeImb_p60（中期信号一致性）
     40.  sect_ovi_ti         - Sect_OVI_p20 × Sect_TI_p40（板块信号一致性）
     41.  obi_ovi_s           - OBI1_p15 × OVI_p15（挂单深度与流量方向一致性）

    另存元数据列（非特征）：
      tick_idx  - 当日 tick 索引（0 起），用于上午/下午时段分割训练
    """
    e = day_data['E']

    # ── E 自身信号 ───────────────────────────────────────────────────────────────
    e_ti  = _imb(e['TradeBuyVolume'],  e['TradeSellVolume'])
    e_ovi = _imb(e['OrderBuyVolume'],  e['OrderSellVolume'])
    e_oni = _imb(e['OrderBuyNum'],     e['OrderSellNum'])
    e_tni = _imb(e['TradeBuyNum'],     e['TradeSellNum'])

    TotalBidVol = sum(e[f'BidVolume{i}'].values for i in range(1, 6))
    TotalAskVol = sum(e[f'AskVolume{i}'].values for i in range(1, 6))

    ti_600  = _sma(e_ti, 600).values
    ti_15   = _sma(e_ti,  15).values
    ti_30   = _sma(e_ti,  30).values
    ti_40   = _sma(e_ti,  40).values
    ti_60   = _sma(e_ti,  60).values
    ti_e15  = _ema(e_ti,  15).values
    ti_e60  = _ema(e_ti,  60).values
    ti_e600 = _ema(e_ti, 600).values

    ovi_600  = _sma(e_ovi, 600).values
    ovi_15   = _sma(e_ovi,  15).values
    ovi_30   = _sma(e_ovi,  30).values
    ovi_60   = _sma(e_ovi,  60).values
    ovi_e5   = _ema(e_ovi,   5).values
    ovi_e15  = _ema(e_ovi,  15).values
    ovi_e600 = _ema(e_ovi, 600).values

    oni_600 = _sma(e_oni, 600).values
    oni_15  = _sma(e_oni,  15).values
    oni_30  = _sma(e_oni,  30).values

    tni_e15  = _ema(e_tni,  15).values
    tni_e600 = _ema(e_tni, 600).values

    # ── E 挂单深度失衡 (Order Book Imbalance) ───────────────────────────────────
    e_obi1      = _imb(e['BidVolume1'], e['AskVolume1'])
    e_obi_total = _imb(pd.Series(TotalBidVol), pd.Series(TotalAskVol))
    obi1_15     = _sma(e_obi1,      15).values
    obi1_600    = _sma(e_obi1,     600).values
    obi1_e15    = _ema(e_obi1,      15).values
    obi1_e600   = _ema(e_obi1,     600).values
    obi_tot_15  = _sma(e_obi_total, 15).values
    obi_tot_600 = _sma(e_obi_total, 600).values

    feats = {
        'TotalBidVol':   TotalBidVol,
        'TradeImb_600':  ti_600,
        'TradeImb_diff': e_ti.values - ti_600,
        'TradeImb_p15':  ti_15  - ti_600,
        'TradeImb_p30':  ti_30  - ti_600,
        'TradeImb_p40':  ti_40  - ti_600,
        'TradeImb_p60':  ti_60  - ti_600,
        'TradeImb_ep60': ti_e60 - ti_e600,
        'TradeImb_ep15': ti_e15 - ti_e600,
        'OVI_p15':       ovi_15  - ovi_600,
        'OVI_p30':       ovi_30  - ovi_600,
        'OVI_p60':       ovi_60  - ovi_600,
        'OVI_ep15':      ovi_e15 - ovi_e600,
        'OVI_ep5':       ovi_e5  - ovi_e600,
        'ONI_p15':       oni_15  - oni_600,
        'ONI_p30':       oni_30  - oni_600,
        'TNI_ep15':      tni_e15 - tni_e600,
        'OBI1_p15':      obi1_15    - obi1_600,
        'OBI1_ep15':     obi1_e15   - obi1_e600,
        'OBI_total_p15': obi_tot_15 - obi_tot_600,
    }

    # ── 板块特征 (A B C D，时间戳完全对齐) ─────────────────────────────────────
    sect_obi1 = np.zeros(len(e))
    sect_ti   = np.zeros(len(e))
    sect_ovi  = np.zeros(len(e))
    sect_oni  = np.zeros(len(e))
    for s in ('A', 'B', 'C', 'D'):
        ds = day_data[s]
        sect_obi1 += _imb(ds['BidVolume1'],    ds['AskVolume1']).values
        sect_ti   += _imb(ds['TradeBuyVolume'], ds['TradeSellVolume']).values
        sect_ovi  += _imb(ds['OrderBuyVolume'], ds['OrderSellVolume']).values
        sect_oni  += _imb(ds['OrderBuyNum'],    ds['OrderSellNum']).values
    sect_obi1 /= 4.0
    sect_ti   /= 4.0
    sect_ovi  /= 4.0
    sect_oni  /= 4.0

    s_ti_600  = _sma(sect_ti,  600).values
    s_ti_40   = _sma(sect_ti,   40).values
    s_ovi_600 = _sma(sect_ovi, 600).values
    s_ovi_20  = _sma(sect_ovi,  20).values
    s_ovi_e5  = _ema(sect_ovi,   5).values
    s_ovi_e15 = _ema(sect_ovi,  15).values
    s_ovi_e600= _ema(sect_ovi, 600).values
    s_oni_600 = _sma(sect_oni, 600).values
    s_oni_30  = _sma(sect_oni,  30).values

    feats['Sect_OBI1']     = sect_obi1
    feats['E_TI_rel_600']  = ti_600   - s_ti_600
    feats['Sect_TI_p40']   = s_ti_40  - s_ti_600
    feats['Sect_OVI_p20']  = s_ovi_20 - s_ovi_600
    feats['Sect_ONI_p30']  = s_oni_30 - s_oni_600
    feats['Sect_OVI_ep5']  = s_ovi_e5  - s_ovi_e600
    feats['Sect_OVI_ep15'] = s_ovi_e15 - s_ovi_e600

    # ── 日内时间特征 ─────────────────────────────────────────────────────────────
    # 利用日内 tick 位置作为二值特征，捕捉上午/下午收益率系统性偏移：
    #   aft_13800: tick_index > 13800 (交易日约后半段)
    #   aft_12000: tick_index > 12000 (稍早的分界点，与 aft_13800 形成互补)
    tick_idx = np.arange(len(e), dtype=int)
    feats['aft_13800'] = (tick_idx > 13800).astype(float)
    feats['aft_12000'] = (tick_idx > 12000).astype(float)

    # ── 滞后已实现收益特征 ──────────────────────────────────────────────────────
    # 核心思路：Return5min(t) 在 t+600 时可知，因此 Return5min(t-600) 在 t 时
    # 已完全已知，无任何前视偏差。这些滞后收益携带市场动量/反转信息。
    #
    # sect_ret_lag: 板块(ABCD)过去5分钟平均已实现收益（各股取平均）
    # e_ret_lag:    E股自身过去5分钟已实现收益（均值回复信号）
    # past_ret_120/300/600: E股中间价在过去 1/2.5/5 分钟内的收益率
    sect_ret_arr = np.zeros(len(e))
    for s in ('A', 'B', 'C', 'D'):
        sect_ret_arr += (pd.Series(day_data[s]['Return5min'].values)
                         .shift(600).fillna(0.0).values / 4.0)
    feats['sect_ret_lag'] = np.clip(sect_ret_arr, -_RET_CLIP_LONG, _RET_CLIP_LONG)
    feats['e_ret_lag'] = np.clip(
        pd.Series(e['Return5min'].values).shift(600).fillna(0.0).values,
        -_RET_CLIP_LONG, _RET_CLIP_LONG)

    mid = (e['BidPrice1'].values + e['AskPrice1'].values) / 2.0
    mid_s = pd.Series(mid)
    for lag in (30, 60, 120, 300, 600):
        clip = _RET_CLIP_SHORT if lag <= 60 else _RET_CLIP_LONG
        ret = (mid_s - mid_s.shift(lag)).divide(mid_s.shift(lag) + 1e-9).fillna(0.0).values
        feats[f'past_ret_{lag}'] = np.clip(ret, -clip, clip)

    # ── 信号一致性交叉特征 (Signal Coherence Interactions) ──────────────────────
    # 核心思路：当订单流失衡 (OVI) 与成交量失衡 (TI) 方向一致时，信号更可信。
    # Ridge 无法从两个独立线性特征捕捉乘法交互，故显式添加乘积项。
    #   ovi_ti_short: OVI_p15 × TI_p15（短期订单与成交流同向强化）
    #   ovi_ti_medium: OVI_p60 × TI_p60（中期信号一致性）
    #   sect_ovi_ti: 板块 OVI_p20 × 板块 TI_p40（板块层面信号一致性）
    #   obi_ovi_s: OBI1_p15 × OVI_p15（挂单深度与流量方向一致性）
    # 注意：此处的计算逻辑在 MyModel.py 的 online_predict() 中有对应的在线版本。
    feats['ovi_ti_short']  = feats['OVI_p15']      * feats['TradeImb_p15']
    feats['ovi_ti_medium'] = feats['OVI_p60']      * feats['TradeImb_p60']
    feats['sect_ovi_ti']   = feats['Sect_OVI_p20'] * feats['Sect_TI_p40']
    feats['obi_ovi_s']     = feats['OBI1_p15']     * feats['OVI_p15']
    # 核心思路：当前中间价偏离日内 VWAP 越多，未来越倾向于均值回复
    # vwap_dev = (mid - VWAP) / VWAP，全时段无前视偏差（仅用 t 时刻前的累积成交量/价格）
    e_tvol = (e['TradeBuyVolume'] + e['TradeSellVolume']).values.astype(float)
    cum_val = np.cumsum(e_tvol * mid)
    cum_vol = np.cumsum(e_tvol) + 1e-9
    e_vwap  = cum_val / cum_vol
    feats['vwap_dev'] = np.clip((mid - e_vwap) / (e_vwap + 1e-9), -0.01, 0.01)

    # ── 清洗并组装 DataFrame ────────────────────────────────────────────────────
    feature_cols = [
        'TotalBidVol', 'TradeImb_600', 'TradeImb_diff',
        'TradeImb_p15', 'TradeImb_p30', 'TradeImb_p40', 'TradeImb_p60',
        'TradeImb_ep60', 'TradeImb_ep15',
        'OVI_p15', 'OVI_p30', 'OVI_p60', 'OVI_ep15', 'OVI_ep5',
        'ONI_p15', 'ONI_p30', 'TNI_ep15',
        'OBI1_p15', 'OBI1_ep15', 'OBI_total_p15',
        'Sect_OBI1', 'E_TI_rel_600', 'Sect_TI_p40', 'Sect_OVI_p20', 'Sect_ONI_p30',
        'Sect_OVI_ep5', 'Sect_OVI_ep15',
        'aft_13800', 'aft_12000',
        'sect_ret_lag', 'e_ret_lag',
        'past_ret_30', 'past_ret_60', 'past_ret_120', 'past_ret_300', 'past_ret_600',
        'vwap_dev',
        'ovi_ti_short', 'ovi_ti_medium', 'sect_ovi_ti', 'obi_ovi_s',
    ]

    df_out = pd.DataFrame({'Time': e['Time'].values})
    for col in feature_cols:
        arr = np.asarray(feats[col])
        df_out[col] = np.where(np.isfinite(arr), arr, 0.0)

    df_out['Return5min'] = e['Return5min'].values
    df_out['tick_idx']   = tick_idx          # 元数据：当日 tick 索引，用于 AM/PM 分割训练
    return df_out


def run_processor():
    print("开始预处理...")
    data_path = "./data"
    days = get_day_folders(data_path)
    all_data = []

    for d in days:
        print(f"Processing Day {d}...")
        day_data = load_day_data(data_path, d)
        day_df = process_day_data(day_data)
        day_df['Day'] = int(d)
        all_data.append(day_df)

    total = pd.concat(all_data, ignore_index=True)
    total.to_csv("train.csv", index=False)
    print(f"Done. Shape: {total.shape}, Features: {total.shape[1] - 4}")  # Time, Return5min, Day, tick_idx (41 features)


if __name__ == "__main__":
    run_processor()
