# MyModel.py
"""
在线预测模型：将离线训练的 33 个 Ridge 子模型 + 动态集成逻辑移植为逐 tick 推理。

架构：
  1. __init__: 在全部5天训练数据上训练 33 个 Ridge 模型，保存系数向量。
  2. reset():  每日开始时重置日内运行状态（滑动窗口、累计量、lag缓冲区等）。
  3. online_predict(E_row, sector_rows):
       - 更新日内运行统计（SMA/EMA、累计量、lag缓冲区）
       - 计算全部 49 个特征
       - 对 33 个子模型做线性推理（w·x）
       - 用滚动 IC-EWMA 动态集成，得到最终预测值
       - 集成参数与离线 train_model.py 完全一致（Iter13优化值）

数据假设（参考 data_processor.py / main.py）：
  - 各股票（A/B/C/D/E）CSV 行按时间完全对齐，逐 tick 一一对应。
  - E_row_data: pandas Series，字段见 data/1/E.csv 表头。
  - sector_row_datas: 长度为 4 的列表，每个元素为 A/B/C/D 的一行 Series。
"""

import os
import warnings
import numpy as np
import pandas as pd
from collections import deque
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')


# ── 导入集成参数和模型定义（与 train_model.py 共享常量）──────────────────────
from train_model import (
    MODELS, MODEL_NAMES, MODEL_IS_NICHE,
    ENSEMBLE_WINDOW, ENSEMBLE_TEMP, ENSEMBLE_FLOOR,
    RETURN_DELAY, NICHE_INIT_WEIGHT,
    ENSEMBLE_UPDATE_FREQ, ENSEMBLE_EWMA_BETA, STABLE_PRIOR,
)
from data_processor import process_day_data
from utils import get_day_folders, load_day_data

# 特征裁剪阈值（与 data_processor.py 保持一致）
_RET_CLIP_LONG  = 0.1
_RET_CLIP_SHORT = 0.05


def _imb(a, b):
    return (a - b) / (a + b + 1e-6)


class _RunSMA:
    """增量式简单移动平均（固定窗口）"""
    __slots__ = ('window', '_buf', '_sum', '_count')

    def __init__(self, window: int):
        self.window  = window
        self._buf    = deque(maxlen=window)
        self._sum    = 0.0
        self._count  = 0

    def update(self, val: float) -> float:
        if self._count == self.window:
            self._sum -= self._buf[0]
        else:
            self._count += 1
        self._buf.append(val)
        self._sum += val
        return self._sum / self._count

    def reset(self):
        self._buf.clear()
        self._sum   = 0.0
        self._count = 0


class _RunEMA:
    """增量式指数移动平均（固定 span）"""
    __slots__ = ('alpha', 'value')

    def __init__(self, span: int):
        self.alpha = 2.0 / (span + 1.0)
        self.value: float | None = None

    def update(self, val: float) -> float:
        if self.value is None:
            self.value = val
        else:
            self.value = self.alpha * val + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


class _LagBuf:
    """固定延迟缓冲区：返回 lag 个 tick 前的值。"""
    __slots__ = ('lag', '_buf', '_default')

    def __init__(self, lag: int, default: float = 0.0):
        self.lag      = lag
        self._buf     = deque(maxlen=lag + 1)
        self._default = default

    def push_and_get(self, val: float) -> float:
        """压入当前值，返回 lag 前的值（若不足则返回 default）。"""
        self._buf.append(val)
        if len(self._buf) <= self.lag:
            return self._default
        return self._buf[0]

    def reset(self, default: float = 0.0):
        self._buf.clear()
        self._default = default


class _RollingIC:
    """
    滚动 Pearson IC（窗口大小 window）。
    在 window//4 个样本前返回 0（冷启动）。
    """
    __slots__ = ('window', 'min_p', '_x', '_y', '_n')

    def __init__(self, window: int):
        self.window = window
        self.min_p  = max(window // 4, 10)
        self._x     = deque(maxlen=window)
        self._y     = deque(maxlen=window)
        self._n     = 0

    def update(self, x: float, y: float) -> float:
        self._x.append(x)
        self._y.append(y)
        self._n += 1
        if self._n < self.min_p:
            return 0.0
        xa = np.asarray(self._x, dtype=float)
        ya = np.asarray(self._y, dtype=float)
        sx, sy = xa.std(), ya.std()
        if sx < 1e-9 or sy < 1e-9:
            return 0.0
        return float(np.corrcoef(xa, ya)[0, 1])

    def reset(self):
        self._x.clear()
        self._y.clear()
        self._n = 0


class MyModel:
    """
    在线预测模型。
    init 中训练全部 39 个 Ridge 子模型（在 train.csv 全量数据上）。
    online_predict 接受逐 tick 数据，返回该 tick 的 Return5min 预测值。
    """

    # ─────────────────────────────────────────────────────────────────────────
    def __init__(self):
        # ── 1. 训练全量模型 ────────────────────────────────────────────────────
        csv_path = os.path.join(os.path.dirname(__file__), 'train.csv')
        df = pd.read_csv(csv_path)
        y  = df['Return5min'].values

        self._coefs: dict[str, tuple[np.ndarray, float]] = {}  # name → (coef, intercept)
        for name, (feats, alpha, _) in MODELS.items():
            m = Ridge(alpha=alpha)
            m.fit(df[feats].values, y)
            self._coefs[name] = (m.coef_.copy(), float(m.intercept_))

        # ── 2. 初始化日内状态（首次调用 reset() 会真正重置）──────────────────
        self._tick_idx   = -1   # 当日已处理的 tick 计数（从 0 开始）
        self._init_state()

    # ─────────────────────────────────────────────────────────────────────────
    def reset(self):
        """每日开始前调用：重置所有日内状态。"""
        self._tick_idx = -1
        self._init_state()

    # ─────────────────────────────────────────────────────────────────────────
    def _init_state(self):
        """创建/重置所有增量计算状态。"""
        # ── E 股 SMA ──────────────────────────────────────────────────────────
        self._ti_s600  = _RunSMA(600);  self._ti_s15  = _RunSMA(15)
        self._ti_s30   = _RunSMA(30);   self._ti_s40  = _RunSMA(40)
        self._ti_s60   = _RunSMA(60)
        self._ti_e60   = _RunEMA(60);   self._ti_e600 = _RunEMA(600)

        self._ovi_s600 = _RunSMA(600);  self._ovi_s15  = _RunSMA(15)
        self._ovi_s30  = _RunSMA(30);   self._ovi_s60  = _RunSMA(60)
        self._ovi_e5   = _RunEMA(5);    self._ovi_e15  = _RunEMA(15)
        self._ovi_e600 = _RunEMA(600)

        self._oni_s600 = _RunSMA(600);  self._oni_s15  = _RunSMA(15)
        self._oni_s30  = _RunSMA(30)
        self._oni_e15  = _RunEMA(15);   self._oni_e600 = _RunEMA(600)

        self._tni_e15  = _RunEMA(15);   self._tni_e600 = _RunEMA(600)

        self._tbv_s600 = _RunSMA(600)   # TotalBidVol SMA600

        # ── 板块 SMA/EMA ──────────────────────────────────────────────────────
        self._sti_s600  = _RunSMA(600); self._sti_s40  = _RunSMA(40)
        self._sovi_s600 = _RunSMA(600); self._sovi_s20 = _RunSMA(20)
        self._sovi_e5   = _RunEMA(5);   self._sovi_e15  = _RunEMA(15)
        self._sovi_e600 = _RunEMA(600)
        self._soni_s600 = _RunSMA(600); self._soni_s30  = _RunSMA(30)

        # ── E 股买卖价差 ──────────────────────────────────────────────────────
        self._spd_s600  = _RunSMA(600)

        # ── 大单成交失衡（E 股）──────────────────────────────────────────────
        self._abs_s15   = _RunSMA(15);  self._abs_s600 = _RunSMA(600)
        self._ass_s15   = _RunSMA(15);  self._ass_s600 = _RunSMA(600)

        # ── 板块大单（4 股各一组 SMA）────────────────────────────────────────
        self._sec_abs_s15  = [_RunSMA(15)  for _ in range(4)]
        self._sec_abs_s600 = [_RunSMA(600) for _ in range(4)]
        self._sec_ass_s15  = [_RunSMA(15)  for _ in range(4)]
        self._sec_ass_s600 = [_RunSMA(600) for _ in range(4)]

        # ── 深层委托簿（2-5 档）─────────────────────────────────────────────
        self._deep_s15  = _RunSMA(15);  self._deep_s600 = _RunSMA(600)

        # ── 累计成交量失衡 ────────────────────────────────────────────────────
        self._cum_buy   = 0.0;  self._cum_sell  = 0.0
        self._cum_s600  = _RunSMA(600)
        self._scum_buy  = 0.0;  self._scum_sell = 0.0
        self._scum_s600 = _RunSMA(600)

        # ── 全档书压（Iter14: book_pres_pulse）────────────────────────────────
        self._bp_s15    = _RunSMA(15);   self._bp_s600   = _RunSMA(600)

        # ── 价格 lag 缓冲区（E 中间价）────────────────────────────────────────
        # 保存最近 901 个 tick 的中间价
        self._mid_buf  = deque(maxlen=901)
        self._smid_buf = deque(maxlen=121)  # 板块均值中间价，最长 lag=120

        # ── Return5min lag 缓冲区（用于 sect_ret_lag / e_ret_lag / e_ret_lag2）
        # Return5min(t) 在 t+600 tick 后可知，因此：
        #   e_ret_lag   = Return5min(t-600)
        #   sect_ret_lag= 各板块 Return5min(t-600) 均值
        #   e_ret_lag2  = Return5min(t-900)
        # 在日初 (<600 tick) 无历史，返回 0（中性，与在线预测一致）
        self._e_ret_buf    = deque(maxlen=901)  # E 的 Return5min lag 缓冲
        self._sec_ret_bufs = [deque(maxlen=601) for _ in range(4)]  # A/B/C/D

        # ── 动态集成状态 ─────────────────────────────────────────────────────
        n_models = len(MODEL_NAMES)
        # rolling IC：每个模型一个滚动 IC 计算器
        self._roll_ic      = [_RollingIC(ENSEMBLE_WINDOW) for _ in range(n_models)]
        # EWMA 平滑后的 IC（每模型一个标量）
        self._ewma_ic      = np.zeros(n_models, dtype=float)
        # 当前权重（初始化为预热期权重）
        self._weights      = self._warmup_weights()
        # 各模型预测缓冲区（用于 rolling IC 的预测侧）
        self._pred_bufs    = [deque(maxlen=ENSEMBLE_WINDOW) for _ in range(n_models)]
        # Return5min 缓冲区（用于 rolling IC 的真值侧，延迟 RETURN_DELAY ticks）
        self._ret_for_ic   = deque(maxlen=ENSEMBLE_WINDOW)
        # 预测缓冲区（等待 RETURN_DELAY ticks 后配对进入 rolling IC）
        self._preds_delay  = [deque(maxlen=RETURN_DELAY + 1) for _ in range(n_models)]
        # 上次权重更新的 tick 计数
        self._last_upd     = -1

    def _warmup_weights(self) -> np.ndarray:
        """预热期权重：稳定模型等权，niche 模型零权重（niche_init=0.0）。"""
        nf = np.array(MODEL_IS_NICHE, dtype=float)
        w  = np.where(nf, NICHE_INIT_WEIGHT, 1.0)
        return w / (w.sum() + 1e-12)

    # ─────────────────────────────────────────────────────────────────────────
    def online_predict(self, E_row_data: pd.Series,
                       sector_row_datas: list) -> float:
        """
        逐 tick 在线预测。

        参数：
          E_row_data        - E 股当前 tick 的 pandas Series（与 train.csv 字段对应）
          sector_row_datas  - 4 个 Series 组成的列表：[A_row, B_row, C_row, D_row]

        返回：
          float - E 股未来 5 分钟收益率预测值
        """
        self._tick_idx += 1
        t = self._tick_idx

        # ── 获取原始值 ─────────────────────────────────────────────────────
        e = E_row_data

        # E 股各信号
        e_ti  = _imb(float(e['TradeBuyVolume']),  float(e['TradeSellVolume']))
        e_ovi = _imb(float(e['OrderBuyVolume']),  float(e['OrderSellVolume']))
        e_oni = _imb(float(e['OrderBuyNum']),     float(e['OrderSellNum']))
        e_tni = _imb(float(e['TradeBuyNum']),     float(e['TradeSellNum']))

        bid1 = float(e['BidPrice1']);  ask1 = float(e['AskPrice1'])
        e_mid = (bid1 + ask1) / 2.0
        e_spread = (ask1 - bid1) / (e_mid + 1e-9)

        tbv = sum(float(e[f'BidVolume{i}']) for i in range(1, 6))
        bid_deep = sum(float(e[f'BidVolume{i}']) for i in range(2, 6))
        ask_deep = sum(float(e[f'AskVolume{i}']) for i in range(2, 6))
        obi_deep_val = _imb(bid_deep, ask_deep)

        # 全档书压（Iter14: book_pres_pulse = SMA15 - SMA600 of all-5-level book imbalance）
        ask_all = sum(float(e[f'AskVolume{i}']) for i in range(1, 6))
        book_pres_val = _imb(tbv, ask_all)

        buy_num_e   = max(float(e['TradeBuyNum']),  1.0)
        sell_num_e  = max(float(e['TradeSellNum']), 1.0)
        avg_buy_e   = float(e['TradeBuyAmount'])  / buy_num_e
        avg_sell_e  = float(e['TradeSellAmount']) / sell_num_e

        # E 股 Return5min（用于 rolling IC 的真值；在 +600 tick 后对外可知）
        e_ret_now = float(e['Return5min']) if 'Return5min' in e.index else 0.0

        # 板块信号
        s_obi1_sum = 0.0
        s_ti_sum   = 0.0
        s_ovi_sum  = 0.0
        s_oni_sum  = 0.0
        s_mid_sum  = 0.0
        s_cum_buy_delta  = 0.0
        s_cum_sell_delta = 0.0

        for si, sr in enumerate(sector_row_datas):
            s_obi1_sum += _imb(float(sr['BidVolume1']), float(sr['AskVolume1']))
            s_ti_sum   += _imb(float(sr['TradeBuyVolume']), float(sr['TradeSellVolume']))
            s_ovi_sum  += _imb(float(sr['OrderBuyVolume']), float(sr['OrderSellVolume']))
            s_oni_sum  += _imb(float(sr['OrderBuyNum']), float(sr['OrderSellNum']))
            s_mid_sum  += (float(sr['BidPrice1']) + float(sr['AskPrice1'])) / 2.0
            s_cum_buy_delta  += float(sr['TradeBuyVolume'])
            s_cum_sell_delta += float(sr['TradeSellVolume'])

            s_buy_n  = max(float(sr['TradeBuyNum']),  1.0)
            s_sell_n = max(float(sr['TradeSellNum']), 1.0)
            s_avg_b  = float(sr['TradeBuyAmount'])  / s_buy_n
            s_avg_s  = float(sr['TradeSellAmount']) / s_sell_n
            b15   = self._sec_abs_s15[si].update(s_avg_b)
            b600  = self._sec_abs_s600[si].update(s_avg_b)
            s15   = self._sec_ass_s15[si].update(s_avg_s)
            s600  = self._sec_ass_s600[si].update(s_avg_s)
            # 板块大单失衡（累加，最后除以4）
            if si == 0:
                sect_lot_sum = np.clip(b15 / (b600 + 1e-9) - s15 / (s600 + 1e-9), -1.0, 1.0)
            else:
                sect_lot_sum += np.clip(b15 / (b600 + 1e-9) - s15 / (s600 + 1e-9), -1.0, 1.0)

        sect_obi1 = s_obi1_sum / 4.0
        sect_ti   = s_ti_sum   / 4.0
        sect_ovi  = s_ovi_sum  / 4.0
        sect_oni  = s_oni_sum  / 4.0
        sect_mid  = s_mid_sum  / 4.0
        sect_lot_imb = sect_lot_sum / 4.0

        # ── 更新 SMA/EMA ───────────────────────────────────────────────────
        ti600  = self._ti_s600.update(e_ti);  ti15  = self._ti_s15.update(e_ti)
        ti30   = self._ti_s30.update(e_ti);   ti40  = self._ti_s40.update(e_ti)
        ti60   = self._ti_s60.update(e_ti)
        ti_e60  = self._ti_e60.update(e_ti);  ti_e600 = self._ti_e600.update(e_ti)

        ovi600  = self._ovi_s600.update(e_ovi); ovi15  = self._ovi_s15.update(e_ovi)
        ovi30   = self._ovi_s30.update(e_ovi);  ovi60  = self._ovi_s60.update(e_ovi)
        ovi_e5  = self._ovi_e5.update(e_ovi);   ovi_e15 = self._ovi_e15.update(e_ovi)
        ovi_e600= self._ovi_e600.update(e_ovi)

        oni600  = self._oni_s600.update(e_oni); oni15  = self._oni_s15.update(e_oni)
        oni30   = self._oni_s30.update(e_oni)
        oni_e15 = self._oni_e15.update(e_oni);  oni_e600 = self._oni_e600.update(e_oni)

        tni_e15  = self._tni_e15.update(e_tni); tni_e600 = self._tni_e600.update(e_tni)

        tbv600   = self._tbv_s600.update(tbv)

        s_ti600  = self._sti_s600.update(sect_ti);  s_ti40   = self._sti_s40.update(sect_ti)
        s_ovi600 = self._sovi_s600.update(sect_ovi); s_ovi20  = self._sovi_s20.update(sect_ovi)
        s_ovi_e5 = self._sovi_e5.update(sect_ovi);   s_ovi_e15= self._sovi_e15.update(sect_ovi)
        s_ovi_e600= self._sovi_e600.update(sect_ovi)
        s_oni600 = self._soni_s600.update(sect_oni); s_oni30  = self._soni_s30.update(sect_oni)

        spd600   = self._spd_s600.update(e_spread)

        ab_s15   = self._abs_s15.update(avg_buy_e);  ab_s600 = self._abs_s600.update(avg_buy_e)
        as_s15   = self._ass_s15.update(avg_sell_e); as_s600 = self._ass_s600.update(avg_sell_e)

        deep15   = self._deep_s15.update(obi_deep_val)
        deep600  = self._deep_s600.update(obi_deep_val)

        # 全档书压 SMA (Iter14)
        bp15  = self._bp_s15.update(book_pres_val)
        bp600 = self._bp_s600.update(book_pres_val)

        # 累计成交量失衡
        self._cum_buy  += float(e['TradeBuyVolume'])
        self._cum_sell += float(e['TradeSellVolume'])
        cum_total = self._cum_buy + self._cum_sell + 1e-6
        cum_raw   = (self._cum_buy - self._cum_sell) / cum_total
        cum_base  = self._cum_s600.update(cum_raw)

        self._scum_buy  += s_cum_buy_delta
        self._scum_sell += s_cum_sell_delta
        scum_total = self._scum_buy + self._scum_sell + 1e-6
        scum_raw   = (self._scum_buy - self._scum_sell) / scum_total
        scum_base  = self._scum_s600.update(scum_raw)

        # ── Return5min lag 缓冲区 ─────────────────────────────────────────
        self._e_ret_buf.append(e_ret_now)
        for si, sr in enumerate(sector_row_datas):
            sr_ret = float(sr['Return5min']) if 'Return5min' in sr.index else 0.0
            self._sec_ret_bufs[si].append(sr_ret)

        # 600-tick 前的已实现收益（冷启动期返回 0）
        e_ret_lag  = self._e_ret_buf[0]  if len(self._e_ret_buf) > 600 else 0.0
        e_ret_lag2 = self._e_ret_buf[0]  if len(self._e_ret_buf) > 900 else 0.0
        # 对 e_ret_lag2，需要 900 tick 前：buffer maxlen=901，需要 len>900
        if len(self._e_ret_buf) > 900:
            e_ret_lag2 = list(self._e_ret_buf)[0]  # 最旧的那个
        else:
            e_ret_lag2 = 0.0

        sect_ret_lag = 0.0
        for si in range(4):
            buf = self._sec_ret_bufs[si]
            sect_ret_lag += (buf[0] if len(buf) > 600 else 0.0) / 4.0

        # ── 价格 lag ────────────────────────────────────────────────────────
        self._mid_buf.append(e_mid)
        self._smid_buf.append(sect_mid)

        def _price_ret(buf, lag, clip):
            if len(buf) <= lag:
                return 0.0
            old = list(buf)[-(lag + 1)]  # lag 个 tick 之前的值（下标从0数）
            if old < 1e-9:
                return 0.0
            return float(np.clip((buf[-1] - old) / old, -clip, clip))

        pr30  = _price_ret(self._mid_buf,   30, _RET_CLIP_SHORT)
        pr60  = _price_ret(self._mid_buf,   60, _RET_CLIP_SHORT)
        pr120 = _price_ret(self._mid_buf,  120, _RET_CLIP_LONG)
        pr300 = _price_ret(self._mid_buf,  300, _RET_CLIP_LONG)
        pr600 = _price_ret(self._mid_buf,  600, _RET_CLIP_LONG)
        pr900 = _price_ret(self._mid_buf,  900, _RET_CLIP_LONG)

        smr30  = _price_ret(self._smid_buf,  30, _RET_CLIP_SHORT)
        smr120 = _price_ret(self._smid_buf, 120, _RET_CLIP_LONG)

        # ── 组装 52 个特征 ─────────────────────────────────────────────────
        ovi_p15 = ovi15 - ovi600

        feats_arr = {
            'TotalBidVol':        tbv,
            'TradeImb_600':       ti600,
            'TradeImb_diff':      e_ti - ti600,
            'TradeImb_p15':       ti15  - ti600,
            'TradeImb_p30':       ti30  - ti600,
            'TradeImb_p40':       ti40  - ti600,
            'TradeImb_p60':       ti60  - ti600,
            'TradeImb_ep60':      ti_e60 - ti_e600,
            'OVI_p15':            ovi_p15,
            'OVI_p30':            ovi30 - ovi600,
            'OVI_p60':            ovi60 - ovi600,
            'OVI_ep15':           ovi_e15 - ovi_e600,
            'OVI_ep5':            ovi_e5  - ovi_e600,
            'ONI_p15':            oni15 - oni600,
            'ONI_p30':            oni30 - oni600,
            'ONI_ep15':           oni_e15 - oni_e600,
            'TNI_ep15':           tni_e15 - tni_e600,
            'Sect_OBI1':          sect_obi1,
            'E_TI_rel_600':       ti600 - s_ti600,
            'Sect_TI_p40':        s_ti40 - s_ti600,
            'Sect_OVI_p20':       s_ovi20 - s_ovi600,
            'Sect_ONI_p30':       s_oni30 - s_oni600,
            'Sect_OVI_ep5':       s_ovi_e5 - s_ovi_e600,
            'Sect_OVI_ep15':      s_ovi_e15 - s_ovi_e600,
            'aft_13800':          1.0 if t > 13800 else 0.0,
            'aft_12000':          1.0 if t > 12000 else 0.0,
            'sect_ret_lag':       float(np.clip(sect_ret_lag, -_RET_CLIP_LONG, _RET_CLIP_LONG)),
            'e_ret_lag':          float(np.clip(e_ret_lag,    -_RET_CLIP_LONG, _RET_CLIP_LONG)),
            'past_ret_30':        pr30,
            'past_ret_60':        pr60,
            'past_ret_120':       pr120,
            'past_ret_300':       pr300,
            'past_ret_600':       pr600,
            'past_ret_900':       pr900,
            'sect_mid_ret_30':    smr30,
            'sect_mid_ret_120':   smr120,
            'csm_ret_120':        float(np.clip(smr120 - pr120,
                                                -_RET_CLIP_LONG * 2, _RET_CLIP_LONG * 2)),
            'e_spread_pulse':     e_spread - spd600,
            'lot_imb_15':         float(np.clip(
                                      ab_s15 / (ab_s600 + 1e-9) - as_s15 / (as_s600 + 1e-9),
                                      -1.0, 1.0)),
            'sect_lot_imb_15':    sect_lot_imb,
            'obi_deep_p15':       deep15 - deep600,
            'cum_flow_imb':       float(np.clip(cum_raw  - cum_base,  -0.5, 0.5)),
            'sect_cum_flow_imb':  float(np.clip(scum_raw - scum_base, -0.5, 0.5)),
            'e_ret_lag2':         float(np.clip(e_ret_lag2, -_RET_CLIP_LONG, _RET_CLIP_LONG)),
            # 交互特征
            'ovi_x_abs_ret':      float(np.clip(ovi_p15 * abs(pr600) * 20, -0.5, 0.5)),
            'tbv_x_ovi':          float(np.clip(tbv / (tbv600 + 1e-9) * ovi_p15 * 5, -0.5, 0.5)),
            'srl_x_ovi':          float(np.clip(
                                      np.clip(sect_ret_lag, -_RET_CLIP_LONG, _RET_CLIP_LONG)
                                      * ovi_p15 * 20, -0.5, 0.5)),
            'ret_x_cum':          float(np.clip(pr600 * float(np.clip(cum_raw - cum_base, -0.5, 0.5))
                                                * 20, -0.5, 0.5)),
            'oni_x_ovi':          float(np.clip((oni15 - oni600) * ovi_p15 * 5, -0.5, 0.5)),
            # Iter14 新增
            'book_pres_pulse':    float(np.clip(bp15 - bp600, -0.5, 0.5)),
            'ret_x_ti600':        float(np.clip(pr600 * ti600 * 20, -0.5, 0.5)),
            # Iter15 新增
            'ret_accel':          float(np.clip(pr300 - pr600, -0.1, 0.1)),
        }

        # NaN/Inf 清洗
        for k in feats_arr:
            v = feats_arr[k]
            if not np.isfinite(v):
                feats_arr[k] = 0.0

        # ── 各模型线性推理 ──────────────────────────────────────────────────
        model_preds = np.empty(len(MODEL_NAMES), dtype=float)
        for mi, name in enumerate(MODEL_NAMES):
            feats_list, _, _ = MODELS[name]
            coef, intercept = self._coefs[name]
            x = np.array([feats_arr[f] for f in feats_list], dtype=float)
            model_preds[mi] = float(np.dot(coef, x)) + intercept

        # ── 动态集成 ────────────────────────────────────────────────────────
        n_models = len(MODEL_NAMES)

        # 将当前预测压入延迟缓冲区
        for mi in range(n_models):
            self._preds_delay[mi].append(model_preds[mi])

        # RETURN_DELAY tick 前的预测和对应真实收益（若已可知）
        has_delayed = (len(self._preds_delay[0]) > RETURN_DELAY and
                       len(self._e_ret_buf) > RETURN_DELAY)

        if has_delayed:
            delayed_ret = list(self._e_ret_buf)[-(RETURN_DELAY + 1)]
            for mi in range(n_models):
                delayed_pred = list(self._preds_delay[mi])[-(RETURN_DELAY + 1)]
                ic_val = self._roll_ic[mi].update(delayed_pred, delayed_ret)
                # EWMA 平滑
                self._ewma_ic[mi] = (ENSEMBLE_EWMA_BETA * ic_val
                                     + (1.0 - ENSEMBLE_EWMA_BETA) * self._ewma_ic[mi])

        # 权重更新（按 update_freq 频率）
        warmup = RETURN_DELAY + ENSEMBLE_WINDOW // 4
        if t < warmup:
            weights = self._warmup_weights()
        elif t - self._last_upd >= ENSEMBLE_UPDATE_FREQ:
            # softmax on EWMA-smoothed IC
            ic_scaled = np.clip(self._ewma_ic * ENSEMBLE_TEMP, -10.0, 10.0)
            exp_ic    = np.exp(ic_scaled)
            weights   = exp_ic / (exp_ic.sum() + 1e-12)
            if ENSEMBLE_FLOOR > 0.0:
                weights = np.maximum(weights, ENSEMBLE_FLOOR)
                weights /= weights.sum()
            self._weights    = weights
            self._last_upd   = t
        else:
            weights = self._weights

        return float(np.dot(weights, model_preds))
