import numpy as np
import pandas as pd
from utils import get_day_folders, load_day_data

# 收益率剪裁阈值（防止极端值）
_RET_CLIP_LONG  = 0.1    # 长窗口（≥120 tick）收益率剪裁
_RET_CLIP_SHORT = 0.05   # 短窗口（30/60 tick）收益率剪裁


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

    共生成 42 个特征，覆盖多个多空动态视角，供动态集成模型使用：

    ── E 自身基础信号 ──────────────────────────────────────────────────────────
      1.  TotalBidVol         - E 五档总买量 (市场深度)
      2.  TradeImb_600        - E 成交量失衡 600 窗口 SMA (长期趋势)
      3.  TradeImb_diff       - E 当前成交量失衡与长期均值偏差

    ── E 成交量失衡脉冲 (SMA) ──────────────────────────────────────────────────
      4.  TradeImb_p15        - TI SMA15 - SMA600 (极短期脉冲)
      5.  TradeImb_p30        - TI SMA30 - SMA600
      6.  TradeImb_p40        - TI SMA40 - SMA600
      7.  TradeImb_p60        - TI SMA60 - SMA600 (中期脉冲)
      8.  TradeImb_ep60       - TI EMA60 - EMA600 (指数加权近期变化)

    ── E 委托量失衡 (OVI) 脉冲 ─────────────────────────────────────────────────
      9.  OVI_p15             - OVI SMA15 - SMA600 (短期挂单动能)
     10.  OVI_p30             - OVI SMA30 - SMA600
     11.  OVI_p60             - OVI SMA60 - SMA600 (中期挂单动能)
     12.  OVI_ep15            - OVI EMA15 - EMA600
     13.  OVI_ep5             - OVI EMA5  - EMA600 (超短期，供 niche 模型使用)

    ── E 委托笔数失衡 (ONI) 脉冲 ────────────────────────────────────────────────
     14.  ONI_p15             - ONI SMA15 - SMA600
     15.  ONI_p30             - ONI SMA30 - SMA600

    ── E 成交笔数失衡 EMA 脉冲 ─────────────────────────────────────────────────
     16.  TNI_ep15            - TNI EMA15 - EMA600

    ── 板块均值信号 ─────────────────────────────────────────────────────────────
     17.  Sect_OBI1           - 板块 (ABCD) 一档委托失衡均值
     18.  E_TI_rel_600        - E 的长期 TI 相对板块偏差 (均值回复)
     19.  Sect_TI_p40         - 板块成交量失衡 SMA40 脉冲
     20.  Sect_OVI_p20        - 板块委托量失衡 SMA20 脉冲
     21.  Sect_ONI_p30        - 板块委托笔数失衡 SMA30 脉冲
     22.  Sect_OVI_ep5        - 板块委托量失衡 EMA5  - EMA600 (超短期板块动能)
     23.  Sect_OVI_ep15       - 板块委托量失衡 EMA15 - EMA600

    ── 日内时间特征 ─────────────────────────────────────────────────────────────
     24.  aft_13800           - tick_index > 13800 二值 (交易日后半段指示)
     25.  aft_12000           - tick_index > 12000 二值 (稍早分界，与 24 互补)

    ── 滞后已实现收益特征 ─────────────────────────────────────────────────────
     26.  sect_ret_lag        - 板块(ABCD)过去5分钟平均已实现收益（动量/反转）
     27.  e_ret_lag           - E 过去5分钟已实现收益（均值回复信号）
     28.  past_ret_30         - E 中间价过去 15 秒收益率 (30 ticks)
     29.  past_ret_60         - E 中间价过去 30 秒收益率 (60 ticks)
     30.  past_ret_120        - E 中间价过去 1 分钟收益率 (120 ticks)
     31.  past_ret_300        - E 中间价过去 2.5 分钟收益率 (300 ticks)
     32.  past_ret_600        - E 中间价过去 5 分钟收益率 (600 ticks)

    ── 板块短期价格收益率 & 横截面相对收益（新增）────────────────────────────
     33.  sect_mid_ret_30     - 板块(ABCD)平均中间价过去 15 秒收益率 (订单流外的价格信息)
     34.  sect_mid_ret_120    - 板块(ABCD)平均中间价过去 1 分钟收益率
     35.  csm_ret_120         - sect_mid_ret_120 - past_ret_120（E 相对板块 1 分钟落后量）
     36.  e_spread_pulse      - E 相对买卖价差偏离 600-tick 均值（做市商不确定性信号）
     37.  lot_imb_15          - 15-tick 平均买入笔金额 vs 卖出笔金额相对偏差（大单方向）
                                在全部 5 日 IC 符号一致为正，弱信号日（1,5）尤为突出
     38.  sect_lot_imb_15     - 板块(ABCD)平均大单成交失衡（机构活跃度方向，全日一致正向）

    ── 深层委托簿失衡脉冲（新增）──────────────────────────────────────────────
     39.  obi_deep_p15        - E 股 2-5 档委托失衡 SMA15 - SMA600 脉冲
                                与 OVI（1 档）正交（相关性 0.23-0.26），捕捉深层买卖双方
                                "耐心资金"的相对强弱，在弱信号日（1/5）有额外正向贡献。

    ── 累计成交流量失衡（新增）──────────────────────────────────────────────────
     40.  cum_flow_imb        - E 股日内累计买卖成交量失衡，对 600-tick 均值去趋势
                                (cum_buy - cum_sell) / cum_total - SMA600 偏差
                                在 Day5 有 IC=+0.23（TI 偏相关 IC=+0.24），捕捉日内趋势方向。
     41.  sect_cum_flow_imb   - 板块(ABCD)平均累计成交量失衡去趋势
                                与 cum_flow_imb 正交（Day3 IC=+0.26，Day5 偏相关 IC=+0.14）
     42.  e_ret_lag2          - E 股 Return5min 900-tick 滞后（7.5 分钟前已实现收益，无前视）
                                Day5 IC=-0.25（偏相关 -0.18），Day2/3 IC=-0.18/-0.18
    """
    e = day_data['E']

    # ── E 自身信号 ───────────────────────────────────────────────────────────────
    e_ti  = _imb(e['TradeBuyVolume'],  e['TradeSellVolume'])
    e_ovi = _imb(e['OrderBuyVolume'],  e['OrderSellVolume'])
    e_oni = _imb(e['OrderBuyNum'],     e['OrderSellNum'])
    e_tni = _imb(e['TradeBuyNum'],     e['TradeSellNum'])

    TotalBidVol = sum(e[f'BidVolume{i}'].values for i in range(1, 6))

    ti_600  = _sma(e_ti, 600).values
    ti_15   = _sma(e_ti,  15).values
    ti_30   = _sma(e_ti,  30).values
    ti_40   = _sma(e_ti,  40).values
    ti_60   = _sma(e_ti,  60).values
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

    feats = {
        'TotalBidVol':   TotalBidVol,
        'TradeImb_600':  ti_600,
        'TradeImb_diff': e_ti.values - ti_600,
        'TradeImb_p15':  ti_15  - ti_600,
        'TradeImb_p30':  ti_30  - ti_600,
        'TradeImb_p40':  ti_40  - ti_600,
        'TradeImb_p60':  ti_60  - ti_600,
        'TradeImb_ep60': ti_e60 - ti_e600,
        'OVI_p15':       ovi_15  - ovi_600,
        'OVI_p30':       ovi_30  - ovi_600,
        'OVI_p60':       ovi_60  - ovi_600,
        'OVI_ep15':      ovi_e15 - ovi_e600,
        'OVI_ep5':       ovi_e5  - ovi_e600,
        'ONI_p15':       oni_15  - oni_600,
        'ONI_p30':       oni_30  - oni_600,
        'TNI_ep15':      tni_e15 - tni_e600,
    }

    # ── 板块特征 (A B C D，时间戳完全对齐) ─────────────────────────────────────
    sect_obi1 = np.zeros(len(e))
    sect_ti   = np.zeros(len(e))
    sect_ovi  = np.zeros(len(e))
    sect_oni  = np.zeros(len(e))
    sect_mid  = np.zeros(len(e))   # 板块平均中间价（用于短期价格收益率特征）
    for s in ('A', 'B', 'C', 'D'):
        ds = day_data[s]
        sect_obi1 += _imb(ds['BidVolume1'],    ds['AskVolume1']).values
        sect_ti   += _imb(ds['TradeBuyVolume'], ds['TradeSellVolume']).values
        sect_ovi  += _imb(ds['OrderBuyVolume'], ds['OrderSellVolume']).values
        sect_oni  += _imb(ds['OrderBuyNum'],    ds['OrderSellNum']).values
        sect_mid  += (ds['BidPrice1'].values + ds['AskPrice1'].values) / 2.0
    sect_obi1 /= 4.0
    sect_ti   /= 4.0
    sect_ovi  /= 4.0
    sect_oni  /= 4.0
    sect_mid  /= 4.0

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

    # ── 板块短期价格收益率 & 跨截面相对收益（新增）─────────────────────────────
    # 现有板块特征（Sect_TI_p40, Sect_OVI_p20 等）均基于订单流失衡，
    # 而板块价格近期涨跌（30/120 tick 收益率）提供了完全独立的信息维度：
    #   sect_mid_ret_30  — 板块平均中间价过去 15 秒涨跌（极短期板块价格动能）
    #   sect_mid_ret_120 — 板块平均中间价过去 1 分钟涨跌（短期板块价格动能）
    #   csm_ret_120      — sect_mid_ret_120 - past_ret_120_E：
    #                      "板块领先 E" 的横截面相对表现（均值回复/追涨）
    # 三个特征均使用当前及历史价格计算，无任何前视偏差。
    sect_mid_s = pd.Series(sect_mid)
    for lag, clip in ((30, _RET_CLIP_SHORT), (120, _RET_CLIP_LONG)):
        s_ret = (sect_mid_s - sect_mid_s.shift(lag)).divide(
            sect_mid_s.shift(lag) + 1e-9).fillna(0.0).values
        feats[f'sect_mid_ret_{lag}'] = np.clip(s_ret, -clip, clip)

    # csm_ret_120: 板块 1 分钟收益 – E 自身 1 分钟收益
    # 若 > 0：板块领先 E，E 可能追涨（动能跟随）或均值回复（反转）
    feats['csm_ret_120'] = np.clip(
        feats['sect_mid_ret_120'] - feats['past_ret_120'],
        -_RET_CLIP_LONG * 2, _RET_CLIP_LONG * 2)

    # ── E 股买卖价差脉冲（新增）─────────────────────────────────────────────────
    # e_spread_pulse: 当前相对价差偏离其 600-tick 均值
    # 价差扩大 → 做市商风险厌恶上升 → 往往先于价格较大波动
    e_spread = (e['AskPrice1'].values - e['BidPrice1'].values) / (mid + 1e-9)
    e_spread_sma600 = _sma(e_spread, 600).values
    feats['e_spread_pulse'] = e_spread - e_spread_sma600

    # ── 大单成交失衡（新增）────────────────────────────────────────────────────
    # lot_imb_15: 近 15 tick 平均买入笔金额 vs 平均卖出笔金额的相对偏差
    # 核心逻辑：大单（机构）买入时 avg_buy_size 高于历史均值，
    #           这与 OVI/TI（仅统计数量/总量）正交，捕捉资金大小的方向信息。
    # 实证：在本数据集 5 日中 IC 符号一致为正（0.02~0.07），尤其在弱信号日更强。
    buy_num_e  = np.maximum(e['TradeBuyNum'].values.astype(float),  1.0)
    sell_num_e = np.maximum(e['TradeSellNum'].values.astype(float), 1.0)
    avg_buy_size  = e['TradeBuyAmount'].values.astype(float)  / buy_num_e
    avg_sell_size = e['TradeSellAmount'].values.astype(float) / sell_num_e

    b_sma15  = _sma(avg_buy_size,  15).values
    b_sma600 = _sma(avg_buy_size, 600).values
    s_sma15  = _sma(avg_sell_size,  15).values
    s_sma600 = _sma(avg_sell_size, 600).values

    # 各自归一化后做差：(avg_buy_15/avg_buy_600 - 1) - (avg_sell_15/avg_sell_600 - 1)
    feats['lot_imb_15'] = np.clip(
        b_sma15 / (b_sma600 + 1e-9) - s_sma15 / (s_sma600 + 1e-9),
        -1.0, 1.0)

    # sect_lot_imb_15: 板块(ABCD)平均大单成交失衡
    # 全日 IC 方向一致为正（0.02~0.08），与 E 自身 lot_imb 互补但更稳定
    sect_lot_imb = np.zeros(len(e))
    for s in ('A', 'B', 'C', 'D'):
        ds = day_data[s]
        s_buy_n  = np.maximum(ds['TradeBuyNum'].values.astype(float),  1.0)
        s_sell_n = np.maximum(ds['TradeSellNum'].values.astype(float), 1.0)
        s_avg_b  = ds['TradeBuyAmount'].values.astype(float)  / s_buy_n
        s_avg_s  = ds['TradeSellAmount'].values.astype(float) / s_sell_n
        s_b15    = _sma(s_avg_b, 15).values
        s_b600   = _sma(s_avg_b, 600).values
        s_s15    = _sma(s_avg_s, 15).values
        s_s600   = _sma(s_avg_s, 600).values
        sect_lot_imb += np.clip(s_b15 / (s_b600 + 1e-9) - s_s15 / (s_s600 + 1e-9), -1.0, 1.0)
    sect_lot_imb /= 4.0
    feats['sect_lot_imb_15'] = sect_lot_imb

    # ── 深层委托簿失衡脉冲（新增）──────────────────────────────────────────────
    # obi_deep_p15: E 股 2-5 档累计买卖失衡的 SMA15 - SMA600 脉冲
    # 与 OVI（仅使用 1 档）正交（相关系数 0.23-0.26），捕捉"深度订单簿"
    # 方向信息（耐心资金 / 机构挂单方向）。
    # 实证：全 5 日 IC 一致为正（0.07-0.13），与现有 OVI 特征仅低度相关，
    # 在弱信号日（1/5）作为 OVI 的补充信号尤为有效。
    bid_vols = [e[f'BidVolume{i}'].values for i in range(2, 6)]
    ask_vols = [e[f'AskVolume{i}'].values for i in range(2, 6)]
    bid_deep = np.sum(bid_vols, axis=0)
    ask_deep = np.sum(ask_vols, axis=0)
    obi_deep     = _imb(bid_deep, ask_deep)
    obi_deep_15  = _sma(obi_deep, 15).values
    obi_deep_600 = _sma(obi_deep, 600).values
    feats['obi_deep_p15'] = obi_deep_15 - obi_deep_600

    # ── 累计成交流量失衡（新增）──────────────────────────────────────────────────
    # cum_flow_imb: (日内累计买量 - 累计卖量) / 累计总量 的 600-tick 去趋势版本
    # 与短期成交失衡（TI 系列）正交：TI 关注最近窗口的方向，
    # cum_flow_imb 关注"全天截至目前"的方向积累（类似 VPIN/order flow inventory），
    # 在 Day4/5（低波动、趋势日）尤为有效（偏相关 IC > 0.17）。
    cum_buy  = pd.Series(e['TradeBuyVolume'].values).cumsum().values
    cum_sell = pd.Series(e['TradeSellVolume'].values).cumsum().values
    cum_total = cum_buy + cum_sell + 1e-6
    cum_flow_raw = (cum_buy - cum_sell) / cum_total
    cum_flow_base = _sma(cum_flow_raw, 600).values
    feats['cum_flow_imb'] = np.clip(cum_flow_raw - cum_flow_base, -0.5, 0.5)

    # sect_cum_flow_imb: 板块(ABCD)累计成交量失衡去趋势均值
    # 与 E 自身 cum_flow_imb 正交（相关 < 0.4），Day3 IC=+0.26，Day5 偏相关 IC=+0.14。
    sect_cfb = np.zeros(len(e))
    sect_cfs = np.zeros(len(e))
    for s in ('A', 'B', 'C', 'D'):
        ds = day_data[s]
        sect_cfb += pd.Series(ds['TradeBuyVolume'].values).cumsum().values / 4.0
        sect_cfs += pd.Series(ds['TradeSellVolume'].values).cumsum().values / 4.0
    sect_cum_raw  = (sect_cfb - sect_cfs) / (sect_cfb + sect_cfs + 1e-6)
    sect_cum_base = _sma(sect_cum_raw, 600).values
    feats['sect_cum_flow_imb'] = np.clip(sect_cum_raw - sect_cum_base, -0.5, 0.5)

    # e_ret_lag2: E 股 Return5min 900-tick 滞后
    # 在 t 时刻完全已知（900-tick 前发生，600-tick 前即可读取）。
    # 提供比 e_ret_lag（600-tick）更长时间轴的反转信号：
    # Day5 IC=-0.25（偏相关 -0.18），Day2/3 IC=-0.18。
    feats['e_ret_lag2'] = np.clip(
        pd.Series(e['Return5min'].values).shift(900).fillna(0.0).values,
        -_RET_CLIP_LONG, _RET_CLIP_LONG)

    # ── 清洗并组装 DataFrame ────────────────────────────────────────────────────
    feature_cols = [
        'TotalBidVol', 'TradeImb_600', 'TradeImb_diff',
        'TradeImb_p15', 'TradeImb_p30', 'TradeImb_p40', 'TradeImb_p60', 'TradeImb_ep60',
        'OVI_p15', 'OVI_p30', 'OVI_p60', 'OVI_ep15', 'OVI_ep5',
        'ONI_p15', 'ONI_p30', 'TNI_ep15',
        'Sect_OBI1', 'E_TI_rel_600', 'Sect_TI_p40', 'Sect_OVI_p20', 'Sect_ONI_p30',
        'Sect_OVI_ep5', 'Sect_OVI_ep15',
        'aft_13800', 'aft_12000',
        'sect_ret_lag', 'e_ret_lag',
        'past_ret_30', 'past_ret_60', 'past_ret_120', 'past_ret_300', 'past_ret_600',
        # 新增：板块短期价格收益率 + 横截面相对收益 + E 价差脉冲 + 大单成交失衡
        'sect_mid_ret_30', 'sect_mid_ret_120', 'csm_ret_120', 'e_spread_pulse',
        'lot_imb_15', 'sect_lot_imb_15',
        # 新增：深层委托簿失衡脉冲（2-5 档）
        'obi_deep_p15',
        # 新增：累计成交流量失衡 + 900-tick 滞后收益
        'cum_flow_imb', 'sect_cum_flow_imb', 'e_ret_lag2',
    ]

    df_out = pd.DataFrame({'Time': e['Time'].values})
    for col in feature_cols:
        arr = np.asarray(feats[col])
        df_out[col] = np.where(np.isfinite(arr), arr, 0.0)

    df_out['Return5min'] = e['Return5min'].values
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
    print(f"Done. Shape: {total.shape}, Features: {total.shape[1] - 3}")


if __name__ == "__main__":
    run_processor()
