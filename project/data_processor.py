import numpy as np
import pandas as pd
from utils import get_day_folders, load_day_data


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

    共生成 20 个特征，覆盖多个多空动态视角，供动态集成模型使用：

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

    ── E 委托笔数失衡 (ONI) 脉冲 ────────────────────────────────────────────────
     13.  ONI_p15             - ONI SMA15 - SMA600
     14.  ONI_p30             - ONI SMA30 - SMA600

    ── E 成交笔数失衡 EMA 脉冲 ─────────────────────────────────────────────────
     15.  TNI_ep15            - TNI EMA15 - EMA600

    ── 板块均值信号 ─────────────────────────────────────────────────────────────
     16.  Sect_OBI1           - 板块 (ABCD) 一档委托失衡均值
     17.  E_TI_rel_600        - E 的长期 TI 相对板块偏差 (均值回复)
     18.  Sect_TI_p40         - 板块成交量失衡 SMA40 脉冲
     19.  Sect_OVI_p20        - 板块委托量失衡 SMA20 脉冲
     20.  Sect_ONI_p30        - 板块委托笔数失衡 SMA30 脉冲
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
        'ONI_p15':       oni_15  - oni_600,
        'ONI_p30':       oni_30  - oni_600,
        'TNI_ep15':      tni_e15 - tni_e600,
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
    s_oni_600 = _sma(sect_oni, 600).values
    s_oni_30  = _sma(sect_oni,  30).values

    feats['Sect_OBI1']    = sect_obi1
    feats['E_TI_rel_600'] = ti_600   - s_ti_600
    feats['Sect_TI_p40']  = s_ti_40  - s_ti_600
    feats['Sect_OVI_p20'] = s_ovi_20 - s_ovi_600
    feats['Sect_ONI_p30'] = s_oni_30 - s_oni_600

    # ── 清洗并组装 DataFrame ────────────────────────────────────────────────────
    feature_cols = [
        'TotalBidVol', 'TradeImb_600', 'TradeImb_diff',
        'TradeImb_p15', 'TradeImb_p30', 'TradeImb_p40', 'TradeImb_p60', 'TradeImb_ep60',
        'OVI_p15', 'OVI_p30', 'OVI_p60', 'OVI_ep15',
        'ONI_p15', 'ONI_p30', 'TNI_ep15',
        'Sect_OBI1', 'E_TI_rel_600', 'Sect_TI_p40', 'Sect_OVI_p20', 'Sect_ONI_p30',
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
