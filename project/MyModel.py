# MyModel.py
# 在线预测模型：8 个时间感知 Ridge 子模型 + 基于滚动 IC 的动态集成权重

from collections import deque
import os
import numpy as np


# ════════════════════════════════════════════════════════════════════════════════
# 内联辅助类（避免额外文件依赖）
# ════════════════════════════════════════════════════════════════════════════════

def _imb(a, b):
    return (a - b) / (a + b + 1e-6)


class _SMA:
    """滑动窗口均值（O(1) 更新）"""
    def __init__(self, w):
        self.w = w
        self.buf = deque(maxlen=w)
        self.s = 0.0

    def update(self, x):
        if len(self.buf) == self.w:
            self.s -= self.buf[0]
        self.buf.append(x)
        self.s += x

    def value(self):
        return self.s / len(self.buf) if self.buf else 0.0


class _EMA:
    """指数移动平均（pandas ewm(span, adjust=False) 兼容）"""
    def __init__(self, span):
        self.alpha = 2.0 / (span + 1.0)
        self.v = 0.0
        self.init = False

    def update(self, x):
        if not self.init:
            self.v = x
            self.init = True
        else:
            self.v = self.alpha * x + (1.0 - self.alpha) * self.v

    def value(self):
        return self.v


class _RollingIC:
    """
    O(1) 在线滚动 Pearson 相关系数（Welford 式更新）。
    维护 sum_x, sum_y, sum_x2, sum_y2, sum_xy 的滑动窗口统计量。
    """
    def __init__(self, window, min_periods=None):
        self.w = window
        self.min_p = min_periods if min_periods else max(window // 4, 10)
        self.buf_x = deque(maxlen=window)
        self.buf_y = deque(maxlen=window)
        self.n = 0
        self.sx = 0.0;  self.sy  = 0.0
        self.sx2 = 0.0; self.sy2 = 0.0; self.sxy = 0.0

    def update(self, x, y):
        if len(self.buf_x) == self.w:
            ox, oy = self.buf_x[0], self.buf_y[0]
            self.n  -= 1
            self.sx  -= ox;   self.sy  -= oy
            self.sx2 -= ox*ox; self.sy2 -= oy*oy; self.sxy -= ox*oy
        self.buf_x.append(x);  self.buf_y.append(y)
        self.n  += 1
        self.sx  += x;   self.sy  += y
        self.sx2 += x*x; self.sy2 += y*y; self.sxy += x*y

    def value(self):
        n = self.n
        if n < self.min_p:
            return 0.0
        dx = n * self.sx2 - self.sx * self.sx
        dy = n * self.sy2 - self.sy * self.sy
        if dx <= 0.0 or dy <= 0.0:
            return 0.0
        return float(np.clip(
            (n * self.sxy - self.sx * self.sy) / (dx**0.5 * dy**0.5),
            -1.0, 1.0))


# ════════════════════════════════════════════════════════════════════════════════
# 模型特征定义（与 train_model.py 保持完全一致）
# ════════════════════════════════════════════════════════════════════════════════

_BASE14 = [
    'TotalBidVol', 'TradeImb_600', 'TradeImb_diff',
    'Sect_OBI1', 'E_TI_rel_600',
    'TradeImb_p60', 'TradeImb_p40', 'Sect_TI_p40',
    'OVI_p15', 'OVI_p60', 'Sect_OVI_p20',
    'TradeImb_ep60', 'OVI_ep15', 'TNI_ep15',
]
_MC9 = [
    'TradeImb_600', 'TradeImb_diff',
    'TradeImb_p60', 'TradeImb_p40', 'TradeImb_ep60',
    'E_TI_rel_600', 'Sect_TI_p40',
    'TradeImb_p30', 'TradeImb_p15',
]
_ME8 = [
    'TotalBidVol', 'OVI_p15', 'OVI_p60', 'OVI_ep15',
    'ONI_p15', 'Sect_OBI1', 'Sect_OVI_p20', 'TNI_ep15',
]
_MD16 = _BASE14 + ['Sect_ONI_p30', 'OVI_p30']

_MODELS_DEF = {
    'MA':    (_BASE14,                         150),
    'MD':    (_MD16,                           150),
    'MT':    (_BASE14 + ['aft_13800'],         150),
    'MTC':   (_MC9   + ['aft_13800'],          200),
    'MTD':   (_MD16  + ['aft_13800'],          150),
    'MTE':   (_ME8   + ['aft_13800'],           80),
    'MT12':  (_BASE14 + ['aft_12000'],         150),
    'MTD12': (_MD16  + ['aft_12000'],          150),
}

# 动态集成超参数
_DELAY  = 600    # Return5min 可知延迟（ticks）
_WINDOW = 900    # 滚动 IC 窗口（ticks）
_TEMP   = 8.0    # softmax 温度
_FLOOR  = 0.0    # 权重下限

# 日内时间特征阈值
_T_13800 = 13800
_T_12000 = 12000


# ════════════════════════════════════════════════════════════════════════════════
# MyModel 主类
# ════════════════════════════════════════════════════════════════════════════════

class MyModel:
    """
    在线预测模型。

    流程：
      1. __init__: 从 train.csv 训练 8 个 Ridge 子模型（全量数据）。
      2. reset():  每个新测试日调用，重置所有 EMA/SMA 状态与滚动 IC 跟踪器。
      3. online_predict(): 每 tick 调用一次：
         a. 增量更新 EMA/SMA 状态
         b. 组装特征向量，各子模型预测
         c. 用历史中间价反算 Return5min（延迟 600 tick），更新各模型滚动 IC
         d. softmax(rolling_IC × TEMP) 计算集成权重
         e. 返回加权预测值
    """

    def __init__(self):
        self._coefs: dict = {}
        self._intercepts: dict = {}
        self._feat_lists: dict = {}
        self._model_names: list = []
        self._train()
        self.reset()

    # ──────────────────────────────────────────────────────────────────────────
    # 训练（全量 5 日数据）
    # ──────────────────────────────────────────────────────────────────────────

    def _train(self):
        import pandas as pd
        from sklearn.linear_model import Ridge

        here = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(here, 'train.csv')
        df = pd.read_csv(csv_path)
        y = df['Return5min'].values

        for name, (feats, alpha) in _MODELS_DEF.items():
            X = df[feats].values
            m = Ridge(alpha=alpha)
            m.fit(X, y)
            self._coefs[name]      = m.coef_.copy()
            self._intercepts[name] = float(m.intercept_)
            self._feat_lists[name] = feats

        self._model_names = list(_MODELS_DEF.keys())

    # ──────────────────────────────────────────────────────────────────────────
    # 重置（每个新测试日调用一次）
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self):
        self._tick = 0

        # E 股 SMA（各窗口独立维护，O(1) 更新）
        self._ti_sma  = {w: _SMA(w) for w in (15, 30, 40, 60, 600)}
        self._ovi_sma = {w: _SMA(w) for w in (15, 30, 60, 600)}
        self._oni_sma = {w: _SMA(w) for w in (15, 30, 600)}

        # E 股 EMA
        self._ti_ema  = {s: _EMA(s) for s in (60, 600)}
        self._ovi_ema = {s: _EMA(s) for s in (15, 600)}
        self._tni_ema = {s: _EMA(s) for s in (15, 600)}

        # 板块 SMA（A/B/C/D 四股均值流）
        self._s_ti_sma  = {w: _SMA(w) for w in (40, 600)}
        self._s_ovi_sma = {w: _SMA(w) for w in (20, 600)}
        self._s_oni_sma = {w: _SMA(w) for w in (30, 600)}

        # 中间价历史缓冲（用于反算 Return5min）
        # maxlen = DELAY + 1：buf[0] = mid(t-DELAY), buf[-1] = mid(t)
        self._mid_buf: deque = deque(maxlen=_DELAY + 1)

        # 各子模型预测缓冲（存储最近 DELAY 个预测，用于延迟匹配）
        # 初始化为 0；deque[0] 始终是 DELAY tick 前的预测
        self._pred_buf: dict = {
            n: deque([0.0] * _DELAY, maxlen=_DELAY)
            for n in self._model_names
        }

        # 各子模型滚动 IC 跟踪器
        self._ric: dict = {
            n: _RollingIC(_WINDOW) for n in self._model_names
        }

        # 当前集成权重（等权初始化）
        nm = len(self._model_names)
        self._weights = np.ones(nm) / nm

    # ──────────────────────────────────────────────────────────────────────────
    # 在线预测
    # ──────────────────────────────────────────────────────────────────────────

    def online_predict(self, E_row, sector_rows) -> float:
        """
        参数
        ----
        E_row       : pd.Series，E 股当前 tick 全量行情字段
        sector_rows : list[pd.Series]，A/B/C/D 四只股票当前 tick 行情

        返回
        ----
        float，当前 tick 的集成预测收益率
        """
        t = self._tick   # 0-indexed tick within current day

        # ── 1. 提取 E 股原始失衡信号 ─────────────────────────────────────────
        e_ti  = _imb(float(E_row['TradeBuyVolume']),  float(E_row['TradeSellVolume']))
        e_ovi = _imb(float(E_row['OrderBuyVolume']),  float(E_row['OrderSellVolume']))
        e_oni = _imb(float(E_row['OrderBuyNum']),     float(E_row['OrderSellNum']))
        e_tni = _imb(float(E_row['TradeBuyNum']),     float(E_row['TradeSellNum']))
        tbv   = sum(float(E_row[f'BidVolume{i}']) for i in range(1, 6))

        # ── 2. 提取板块均值信号 ────────────────────────────────────────────────
        s_obi1 = s_ti = s_ovi = s_oni = 0.0
        for sr in sector_rows:
            s_obi1 += _imb(float(sr['BidVolume1']),    float(sr['AskVolume1']))
            s_ti   += _imb(float(sr['TradeBuyVolume']), float(sr['TradeSellVolume']))
            s_ovi  += _imb(float(sr['OrderBuyVolume']), float(sr['OrderSellVolume']))
            s_oni  += _imb(float(sr['OrderBuyNum']),    float(sr['OrderSellNum']))
        s_obi1 /= 4.0;  s_ti /= 4.0;  s_ovi /= 4.0;  s_oni /= 4.0

        # ── 3. 增量更新 EMA/SMA 状态 ────────────────────────────────────────
        for sma in self._ti_sma.values():    sma.update(e_ti)
        for ema in self._ti_ema.values():    ema.update(e_ti)
        for sma in self._ovi_sma.values():   sma.update(e_ovi)
        for ema in self._ovi_ema.values():   ema.update(e_ovi)
        for sma in self._oni_sma.values():   sma.update(e_oni)
        for ema in self._tni_ema.values():   ema.update(e_tni)
        for sma in self._s_ti_sma.values():  sma.update(s_ti)
        for sma in self._s_ovi_sma.values(): sma.update(s_ovi)
        for sma in self._s_oni_sma.values(): sma.update(s_oni)

        # 记录中间价（供延迟真实收益计算）
        mid = (float(E_row['BidPrice1']) + float(E_row['AskPrice1'])) / 2.0
        self._mid_buf.append(mid)

        # ── 4. 组装特征字典 ────────────────────────────────────────────────────
        t600   = self._ti_sma[600].value()
        ov600  = self._ovi_sma[600].value()
        on600  = self._oni_sma[600].value()
        sti600 = self._s_ti_sma[600].value()
        sov600 = self._s_ovi_sma[600].value()
        son600 = self._s_oni_sma[600].value()

        fv = {
            'TotalBidVol':   tbv,
            'TradeImb_600':  t600,
            'TradeImb_diff': e_ti - t600,
            'TradeImb_p15':  self._ti_sma[15].value()  - t600,
            'TradeImb_p30':  self._ti_sma[30].value()  - t600,
            'TradeImb_p40':  self._ti_sma[40].value()  - t600,
            'TradeImb_p60':  self._ti_sma[60].value()  - t600,
            'TradeImb_ep60': self._ti_ema[60].value()  - self._ti_ema[600].value(),
            'OVI_p15':       self._ovi_sma[15].value() - ov600,
            'OVI_p30':       self._ovi_sma[30].value() - ov600,
            'OVI_p60':       self._ovi_sma[60].value() - ov600,
            'OVI_ep15':      self._ovi_ema[15].value() - self._ovi_ema[600].value(),
            'ONI_p15':       self._oni_sma[15].value() - on600,
            'ONI_p30':       self._oni_sma[30].value() - on600,
            'TNI_ep15':      self._tni_ema[15].value() - self._tni_ema[600].value(),
            'Sect_OBI1':     s_obi1,
            'E_TI_rel_600':  t600 - sti600,
            'Sect_TI_p40':   self._s_ti_sma[40].value()  - sti600,
            'Sect_OVI_p20':  self._s_ovi_sma[20].value() - sov600,
            'Sect_ONI_p30':  self._s_oni_sma[30].value() - son600,
            'aft_13800':     1.0 if t > _T_13800 else 0.0,
            'aft_12000':     1.0 if t > _T_12000 else 0.0,
        }

        # ── 5. 各子模型预测 ────────────────────────────────────────────────────
        model_preds: dict = {}
        for name in self._model_names:
            x = np.array([fv[f] for f in self._feat_lists[name]])
            model_preds[name] = float(
                np.dot(self._coefs[name], x) + self._intercepts[name])

        # ── 6. 用延迟真实收益更新滚动 IC ───────────────────────────────────────
        # 在 tick t，可知 tick (t-DELAY) 的真实收益：
        # Return5min[t-DELAY] = (mid[t] - mid[t-DELAY]) / mid[t-DELAY]
        # _pred_buf[name][0] = DELAY tick 前的预测（deque 最旧元素）
        if t >= _DELAY and len(self._mid_buf) == _DELAY + 1:
            mid_now  = self._mid_buf[-1]
            mid_past = self._mid_buf[0]
            if abs(mid_past) > 1e-9:
                y_true = (mid_now - mid_past) / mid_past
                for name in self._model_names:
                    self._ric[name].update(self._pred_buf[name][0], y_true)

        # 将当前预测追加到缓冲（追加在 IC 更新之后）
        for name in self._model_names:
            self._pred_buf[name].append(model_preds[name])

        # ── 7. 更新集成权重 ────────────────────────────────────────────────────
        warmup = _DELAY + _WINDOW // 4   # 825 tick 前使用等权
        if t >= warmup:
            ic_arr = np.array([self._ric[n].value() for n in self._model_names])
            exp_ic = np.exp(np.clip(ic_arr * _TEMP, -10.0, 10.0))
            w = exp_ic / (exp_ic.sum() + 1e-12)
            if _FLOOR > 0.0:
                w = np.maximum(w, _FLOOR)
                w /= w.sum()
            self._weights = w

        # ── 8. 加权集成并返回 ─────────────────────────────────────────────────
        self._tick += 1
        return float(sum(self._weights[i] * model_preds[n]
                         for i, n in enumerate(self._model_names)))
