# MyModel.py
# 在线预测模型：12 个时间感知 Ridge 子模型 + 基于滚动 IC 的动态集成权重

from collections import deque
import os
import numpy as np


# ════════════════════════════════════════════════════════════════════════════════
# 内联辅助类
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
    """O(1) 在线滚动 Pearson 相关系数（Welford 式更新）"""
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
_SR = ['sect_ret_lag', 'e_ret_lag']
_PR = ['past_ret_120', 'past_ret_300', 'past_ret_600']

_MODELS_DEF = {
    'MA':       (_BASE14,                             150),
    'MD':       (_MD16,                               150),
    'MT':       (_BASE14 + ['aft_13800'],             150),
    'MTC':      (_MC9   + ['aft_13800'],              200),
    'MTD':      (_MD16  + ['aft_13800'],              150),
    'MTE':      (_ME8   + ['aft_13800'],               80),
    'MT12':     (_BASE14 + ['aft_12000'],             150),
    'MTD12':    (_MD16  + ['aft_12000'],              150),
    'MTsr12':   (_BASE14 + ['aft_12000'] + _SR,       150),
    'MTpr12':   (_BASE14 + ['aft_12000'] + _PR,       150),
    'MTsrpr12': (_BASE14 + ['aft_12000'] + _SR + _PR, 150),
    'MTDsrpr12':(_MD16  + ['aft_12000'] + _SR + _PR,  150),
}

# 动态集成超参数
_DELAY  = 600    # Return5min 可知延迟（ticks）
_WINDOW = 900    # 滚动 IC 窗口（ticks）
_TEMP   = 10.0   # softmax 温度
_FLOOR  = 0.0    # 权重下限

# 日内时间特征阈值
_T_13800 = 13800
_T_12000 = 12000

# 滞后收益剪裁范围
_RET_CLIP = 0.1


# ════════════════════════════════════════════════════════════════════════════════
# MyModel 主类
# ════════════════════════════════════════════════════════════════════════════════

class MyModel:
    """
    在线预测模型。

    训练：从 train.csv 训练 12 个 Ridge 子模型（全量数据）。
    重置：每个新测试日调用 reset()，清除所有状态。
    预测：online_predict() 每 tick 调用一次，增量更新 EMA/SMA 状态并返回集成预测。

    新增特征（无前视偏差）：
      sect_ret_lag  = 板块(ABCD)过去5分钟平均已实现收益
      e_ret_lag     = E 过去5分钟已实现收益（≈ past_ret_600 的最后价版本）
      past_ret_120  = E 中间价过去 1 分钟收益率
      past_ret_300  = E 中间价过去 2.5 分钟收益率
      past_ret_600  = E 中间价过去 5 分钟收益率
    在线计算方法：维护 E 和各板块股票的中间价/最终价 601 元素滑动缓冲区。
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

        # E 股 SMA
        self._ti_sma  = {w: _SMA(w) for w in (15, 30, 40, 60, 600)}
        self._ovi_sma = {w: _SMA(w) for w in (15, 30, 60, 600)}
        self._oni_sma = {w: _SMA(w) for w in (15, 30, 600)}

        # E 股 EMA
        self._ti_ema  = {s: _EMA(s) for s in (60, 600)}
        self._ovi_ema = {s: _EMA(s) for s in (15, 600)}
        self._tni_ema = {s: _EMA(s) for s in (15, 600)}

        # 板块 SMA
        self._s_ti_sma  = {w: _SMA(w) for w in (40, 600)}
        self._s_ovi_sma = {w: _SMA(w) for w in (20, 600)}
        self._s_oni_sma = {w: _SMA(w) for w in (30, 600)}

        # 中间价历史缓冲（同时用于 Return5min 延迟计算 和 past_ret 计算）
        # maxlen = DELAY + 1 = 601，buf[0]=mid(t-600), buf[-1]=mid(t)
        self._mid_buf: deque = deque(maxlen=_DELAY + 1)

        # 各板块中间价缓冲（用于 sect_ret_lag 在线计算）
        self._sect_mid_buf: list = [deque(maxlen=_DELAY + 1) for _ in range(4)]

        # 各子模型预测缓冲（DELAY tick 延迟，用于滚动 IC 更新）
        self._pred_buf: dict = {
            n: deque([0.0] * _DELAY, maxlen=_DELAY)
            for n in self._model_names
        }

        # 各子模型滚动 IC 跟踪器
        self._ric: dict = {
            n: _RollingIC(_WINDOW) for n in self._model_names
        }

        # 等权初始化
        nm = len(self._model_names)
        self._weights = np.ones(nm) / nm

    # ──────────────────────────────────────────────────────────────────────────
    # 在线预测
    # ──────────────────────────────────────────────────────────────────────────

    def online_predict(self, E_row, sector_rows) -> float:
        """
        参数：E_row = E 股当前 tick 行情 (pd.Series)
              sector_rows = [A, B, C, D] 四只板块股票当前 tick 行情
        返回：float，当前 tick 集成预测收益率
        """
        t = self._tick

        # ── 1. E 股原始失衡信号 ───────────────────────────────────────────────
        e_ti  = _imb(float(E_row['TradeBuyVolume']),  float(E_row['TradeSellVolume']))
        e_ovi = _imb(float(E_row['OrderBuyVolume']),  float(E_row['OrderSellVolume']))
        e_oni = _imb(float(E_row['OrderBuyNum']),     float(E_row['OrderSellNum']))
        e_tni = _imb(float(E_row['TradeBuyNum']),     float(E_row['TradeSellNum']))
        tbv   = sum(float(E_row[f'BidVolume{i}']) for i in range(1, 6))

        # ── 2. 板块均值信号 ───────────────────────────────────────────────────
        s_obi1 = s_ti = s_ovi = s_oni = 0.0
        for sr in sector_rows:
            s_obi1 += _imb(float(sr['BidVolume1']),    float(sr['AskVolume1']))
            s_ti   += _imb(float(sr['TradeBuyVolume']), float(sr['TradeSellVolume']))
            s_ovi  += _imb(float(sr['OrderBuyVolume']), float(sr['OrderSellVolume']))
            s_oni  += _imb(float(sr['OrderBuyNum']),    float(sr['OrderSellNum']))
        s_obi1 /= 4.0;  s_ti /= 4.0;  s_ovi /= 4.0;  s_oni /= 4.0

        # ── 3. 增量更新 EMA/SMA 状态 ─────────────────────────────────────────
        for sma in self._ti_sma.values():    sma.update(e_ti)
        for ema in self._ti_ema.values():    ema.update(e_ti)
        for sma in self._ovi_sma.values():   sma.update(e_ovi)
        for ema in self._ovi_ema.values():   ema.update(e_ovi)
        for sma in self._oni_sma.values():   sma.update(e_oni)
        for ema in self._tni_ema.values():   ema.update(e_tni)
        for sma in self._s_ti_sma.values():  sma.update(s_ti)
        for sma in self._s_ovi_sma.values(): sma.update(s_ovi)
        for sma in self._s_oni_sma.values(): sma.update(s_oni)

        # ── 4. 更新价格历史缓冲 ───────────────────────────────────────────────
        mid = (float(E_row['BidPrice1']) + float(E_row['AskPrice1'])) / 2.0
        self._mid_buf.append(mid)

        for i, sr in enumerate(sector_rows):
            s_mid = (float(sr['BidPrice1']) + float(sr['AskPrice1'])) / 2.0
            self._sect_mid_buf[i].append(s_mid)

        # ── 5. 计算滞后收益特征 ───────────────────────────────────────────────
        # 只有缓冲区填满 601 个元素后才有有效值
        buf_full = len(self._mid_buf) == _DELAY + 1

        # E 股中间价过去 N tick 收益率
        def _past_ret(buf, lag):
            if len(buf) < lag + 1:
                return 0.0
            buf_list = list(buf)
            p_now  = buf_list[-1]
            p_past = buf_list[-1 - lag]
            if abs(p_past) < 1e-9:
                return 0.0
            return float(np.clip((p_now - p_past) / p_past, -_RET_CLIP, _RET_CLIP))

        past_ret_600 = _past_ret(self._mid_buf, 600)
        past_ret_300 = _past_ret(self._mid_buf, 300)
        past_ret_120 = _past_ret(self._mid_buf, 120)

        # sect_ret_lag: 板块平均过去5分钟收益（用各板块中间价缓冲计算）
        sect_ret_lag = 0.0
        for sbuf in self._sect_mid_buf:
            sect_ret_lag += _past_ret(sbuf, 600) / 4.0
        sect_ret_lag = float(np.clip(sect_ret_lag, -_RET_CLIP, _RET_CLIP))

        # e_ret_lag: E 股过去5分钟已实现收益（≈ past_ret_600，用于与训练一致）
        e_ret_lag = float(np.clip(past_ret_600, -_RET_CLIP, _RET_CLIP))

        # ── 6. 组装特征字典 ───────────────────────────────────────────────────
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
            'sect_ret_lag':  sect_ret_lag,
            'e_ret_lag':     e_ret_lag,
            'past_ret_120':  past_ret_120,
            'past_ret_300':  past_ret_300,
            'past_ret_600':  past_ret_600,
        }

        # ── 7. 各子模型预测 ───────────────────────────────────────────────────
        model_preds: dict = {}
        for name in self._model_names:
            x = np.array([fv[f] for f in self._feat_lists[name]])
            model_preds[name] = float(
                np.dot(self._coefs[name], x) + self._intercepts[name])

        # ── 8. 用延迟真实收益更新滚动 IC ──────────────────────────────────────
        # 在 tick t，可知 tick (t-DELAY) 的真实收益：
        # Return5min[t-DELAY] = (mid[t] - mid[t-DELAY]) / mid[t-DELAY]
        if buf_full:
            mid_now  = self._mid_buf[-1]
            mid_past = self._mid_buf[0]
            if abs(mid_past) > 1e-9:
                y_true = (mid_now - mid_past) / mid_past
                for name in self._model_names:
                    self._ric[name].update(self._pred_buf[name][0], y_true)

        # 追加预测到缓冲
        for name in self._model_names:
            self._pred_buf[name].append(model_preds[name])

        # ── 9. 更新集成权重 ───────────────────────────────────────────────────
        warmup = _DELAY + _WINDOW // 4   # 825 tick 预热期使用等权
        if t >= warmup:
            ic_arr = np.array([self._ric[n].value() for n in self._model_names])
            exp_ic = np.exp(np.clip(ic_arr * _TEMP, -10.0, 10.0))
            w = exp_ic / (exp_ic.sum() + 1e-12)
            if _FLOOR > 0.0:
                w = np.maximum(w, _FLOOR)
                w /= w.sum()
            self._weights = w

        # ── 10. 加权集成并返回 ────────────────────────────────────────────────
        self._tick += 1
        return float(sum(self._weights[i] * model_preds[n]
                         for i, n in enumerate(self._model_names)))
