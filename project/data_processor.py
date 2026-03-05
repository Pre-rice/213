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

    最终特征 (14个，Ridge 线性模型)，通过严格的 5 折时序交叉验证确定：
      1.  TotalBidVol         - E 五档总买量 (原始量纲，体现市场深度)
      2.  TradeImb_600        - E 成交量失衡的 600 窗口 SMA (长期卖压指标)
      3.  TradeImb_diff       - E 当前成交量失衡与长期均值的偏差
      4.  Sect_OBI1           - 板块 (ABCD) 一档委托失衡均值
      5.  E_TI_rel_600        - E 的长期成交失衡相对板块的偏差 (均值回复信号)
      6.  TradeImb_p60        - E 成交量失衡 60 窗口 SMA 相对 600 窗口的脉冲
      7.  TradeImb_p40        - E 成交量失衡 40 窗口 SMA 相对 600 窗口的脉冲
      8.  Sect_TI_p40         - 板块成交量失衡 40 窗口 SMA 脉冲
      9.  OVI_p15             - E 委托量失衡 15 窗口 SMA 脉冲 (短期挂单动能)
     10.  OVI_p60             - E 委托量失衡 60 窗口 SMA 脉冲 (中期挂单动能)
     11.  Sect_OVI_p20        - 板块委托量失衡 20 窗口 SMA 脉冲
     12.  TradeImb_ep60       - E 成交量失衡的 EMA60 - EMA600 脉冲 (指数加权近期变化)
     13.  OVI_ep15            - E 委托量失衡的 EMA15 - EMA600 脉冲
     14.  TNI_ep15            - E 成交笔数失衡的 EMA15 - EMA600 脉冲
    """
    e = day_data['E']

    # ── E 自身特征 ──────────────────────────────────────────────────────────────
    e_ti  = _imb(e['TradeBuyVolume'],  e['TradeSellVolume'])   # 成交量失衡
    e_ovi = _imb(e['OrderBuyVolume'],  e['OrderSellVolume'])   # 委托量失衡
    e_tni = _imb(e['TradeBuyNum'],     e['TradeSellNum'])      # 成交笔数失衡

    TotalBidVol = sum(e[f'BidVolume{i}'].values for i in range(1, 6))

    ti_600 = _sma(e_ti, 600).values
    ti_40  = _sma(e_ti, 40).values
    ti_60  = _sma(e_ti, 60).values
    ti_e60 = _ema(e_ti, 60).values
    ti_e600= _ema(e_ti, 600).values

    ovi_600 = _sma(e_ovi, 600).values
    ovi_15  = _sma(e_ovi, 15).values
    ovi_60  = _sma(e_ovi, 60).values
    ovi_e15 = _ema(e_ovi, 15).values
    ovi_e600= _ema(e_ovi, 600).values

    tni_e15  = _ema(e_tni, 15).values
    tni_e600 = _ema(e_tni, 600).values

    feats = {
        'TotalBidVol':   TotalBidVol,
        'TradeImb_600':  ti_600,
        'TradeImb_diff': e_ti.values - ti_600,
        'TradeImb_p60':  ti_60  - ti_600,
        'TradeImb_p40':  ti_40  - ti_600,
        'TradeImb_ep60': ti_e60 - ti_e600,
        'OVI_p15':       ovi_15  - ovi_600,
        'OVI_p60':       ovi_60  - ovi_600,
        'OVI_ep15':      ovi_e15 - ovi_e600,
        'TNI_ep15':      tni_e15 - tni_e600,
    }

    # ── 板块特征 (A B C D，时间戳完全对齐) ─────────────────────────────────────
    sect_obi1 = np.zeros(len(e))
    sect_ti   = np.zeros(len(e))
    sect_ovi  = np.zeros(len(e))
    for s in ('A', 'B', 'C', 'D'):
        ds = day_data[s]
        sect_obi1 += _imb(ds['BidVolume1'], ds['AskVolume1']).values
        sect_ti   += _imb(ds['TradeBuyVolume'], ds['TradeSellVolume']).values
        sect_ovi  += _imb(ds['OrderBuyVolume'], ds['OrderSellVolume']).values
    sect_obi1 /= 4.0
    sect_ti   /= 4.0
    sect_ovi  /= 4.0

    s_ti_600 = _sma(sect_ti, 600).values
    s_ti_40  = _sma(sect_ti, 40).values

    s_ovi_600 = _sma(sect_ovi, 600).values
    s_ovi_20  = _sma(sect_ovi, 20).values

    feats['Sect_OBI1']     = sect_obi1
    feats['Sect_TI_p40']   = s_ti_40  - s_ti_600
    feats['Sect_OVI_p20']  = s_ovi_20 - s_ovi_600
    feats['E_TI_rel_600']  = ti_600   - s_ti_600

    # ── 清洗并组装 DataFrame ────────────────────────────────────────────────────
    feature_cols = [
        'TotalBidVol', 'TradeImb_600', 'TradeImb_diff',
        'Sect_OBI1', 'E_TI_rel_600',
        'TradeImb_p60', 'TradeImb_p40', 'Sect_TI_p40',
        'OVI_p15', 'OVI_p60', 'Sect_OVI_p20',
        'TradeImb_ep60', 'OVI_ep15', 'TNI_ep15',
    ]

    df_out = pd.DataFrame({'Time': e['Time'].values})
    for col in feature_cols:
        arr = np.asarray(feats[col])
        arr = np.where(np.isfinite(arr), arr, 0.0)
        df_out[col] = arr

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