# MyModel.py
"""
在线预测模型：加载离线训练好的 28 个子模型（Ridge）+ 动态集成逻辑，逐 tick 推理。

架构：
  1. __init__: 从 models.pkl 加载预训练的 28 个模型系数向量。
  2. reset():  每日开始时重置日内运行状态（滑动窗口、累计量、lag 缓冲区等）。
  3. online_predict(E_row, sector_rows):
       - 更新日内运行统计（SMA/EMA、累计量、lag 缓冲区）
       - 计算全部 59 个特征
       - 对 28 个子模型做线性推理（w·x）
       - 用滚动 IC-EWMA 动态集成，得到最终预测值

数据假设：
  - 各股票（A/B/C/D/E）CSV 行按时间完全对齐，逐 tick 一一对应。
  - E_row_data: pandas Series，字段见 data/1/E.csv 表头。
  - sector_row_datas: 长度为 4 的列表，每个元素为 A/B/C/D 的一行 Series。
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from collections import deque

warnings.filterwarnings('ignore')


# ── 特征裁剪阈值 ──────────────────────────────────────────────────────────────
_RET_CLIP_LONG  = 0.1   # 长周期收益率裁剪上限
_RET_CLIP_SHORT = 0.05  # 短周期收益率裁剪上限

# ── 子模型特征列表 ────────────────────────────────────────────────────────────
# 基础成交失衡特征组合
_MC9 = [
    'TradeImb_600', 'TradeImb_diff',
    'TradeImb_p60', 'TradeImb_p40', 'TradeImb_ep60',
    'E_TI_rel_600', 'Sect_TI_p40',
    'TradeImb_p30', 'TradeImb_p15',
]
# 委托量失衡核心特征（8/9 个）
_ME8 = [
    'TotalBidVol', 'OVI_p15', 'OVI_p60', 'OVI_ep15',
    'ONI_p15', 'Sect_OBI1', 'Sect_OVI_p20', 'TNI_ep15',
]
_ME9 = _ME8 + ['ONI_ep15']  # ME8 + EMA 委托笔数失衡

# 滞后已实现收益（无前视偏差）
_SR  = ['sect_ret_lag', 'e_ret_lag']
_PR2 = ['past_ret_30', 'past_ret_60', 'past_ret_120', 'past_ret_300', 'past_ret_600']
_PR3 = _PR2 + ['past_ret_900']
_LAG2 = ['e_ret_lag2']  # 900-tick 滞后收益

# 超短期 OVI EMA 脉冲（高方差，适合 niche 模型）
_OVI5 = ['OVI_ep5', 'Sect_OVI_ep5']

# 板块短期价格收益率 & 横截面相对收益
_SMR = ['sect_mid_ret_30', 'sect_mid_ret_120', 'csm_ret_120']

# 大单成交失衡
_LOT = ['lot_imb_15', 'sect_lot_imb_15']

# 深层委托簿失衡脉冲（2-5 档）
_DEEP = ['obi_deep_p15']

# 累计成交流量失衡（去趋势）
_CUM  = ['cum_flow_imb']
_CUM2 = ['cum_flow_imb', 'sect_cum_flow_imb']

# OVI 交互特征（非线性信号增强）
_IXN3 = ['ovi_x_abs_ret', 'tbv_x_ovi', 'srl_x_ovi', 'ret_x_cum']
_IXN4 = _IXN3 + ['oni_x_ovi']

# 全档书压脉冲 & 价格×成交流量交互
_BP   = ['book_pres_pulse']
_RXTI = ['ret_x_ti600']

# 动量加速度
_RACCEL = ['ret_accel']

# 波动率条件化 OVI & 特异性 OVI
_VCOVI = ['vol_cond_ovi']
_IOVI  = ['idio_ovi']

# 截面相对深层书压 & 委托笔数失衡加速度
_ESOBG  = ['e_sect_obi_gap']
_OACCEL = ['oni_accel']

# 价差加权 OVI & OVI 非线性幅度 & 截面反转
_SWOVI = ['spread_wt_ovi']
_OVISQ = ['ovi_sq']
_ESLG  = ['e_sect_lag_gap']

# ── 子模型定义：(feature_list, ridge_alpha, is_niche) ─────────────────────────
# is_niche=True 的模型在预热期权重为 0，由滚动 IC 机制动态发现价值。
# 共 28 个子模型：1 个稳定模型（MTC）+ 27 个 niche Ridge 模型。
MODELS = {
    # 稳定模型（预热期 100% 权重）
    'MTC':              (_MC9   + ['aft_13800'],                                               200, False),
    # 累计流量族
    'N_cum_ME2':        (_ME8 + _OVI5 + ['aft_12000'] + _SR + _LAG2 + _PR2 + _LOT + _CUM,    20,  True),
    'N_cum_ME_T':       (_ME8 + _OVI5 + _SR + _PR2 + _LOT + _CUM,                            20,  True),
    'N_both_ME_T':      (_ME8 + _OVI5 + _SR + _PR2 + _LOT + _CUM2,                           20,  True),
    # IXN 交互族
    'N_IXN4_SMR_ME2':   (_ME8 + _OVI5 + ['aft_12000'] + _SR + _LAG2 + _PR2 + _LOT + _CUM + _SMR + _IXN4,  15, True),
    'N_IXN4_cum_ME2':   (_ME8 + _OVI5 + ['aft_12000'] + _SR + _LAG2 + _PR2 + _LOT + _CUM + _IXN4,         15, True),
    'N_IXN4_cum_T':     (_ME8 + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN4,                               15, True),
    'N_IXN3_cum_T':     (_ME8 + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN3,                               15, True),
    'N_IXN3_cum_ME2':   (_ME8 + _OVI5 + ['aft_12000'] + _SR + _LAG2 + _PR2 + _LOT + _CUM + _IXN3,        15, True),
    'N_IXN4_cum_ME2_D': (_ME8 + _OVI5 + ['aft_12000'] + _SR + _LAG2 + _PR2 + _LOT + _CUM + _IXN4 + _DEEP, 15, True),
    # ONI/PR900 族
    'N_oni9_ME2':       (_ME9 + _OVI5 + ['aft_12000'] + _SR + _LAG2 + _PR3 + _LOT + _CUM,    20,  True),
    # 书压 + IXN 族
    'N_bp_IXN3':        (_ME8 + _BP + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN3,             15,  True),
    'N_rxti_T':         (_ME8 + _RXTI + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN3,           15,  True),
    'N_bp_deep_T':      (_ME8 + _BP + _DEEP + _OVI5 + _SR + _PR2 + _LOT + _CUM2,             15,  True),
    # 动量加速族
    'N_raccel_ME2':     (_ME9 + _RACCEL + _OVI5 + ['aft_12000'] + _SR + _LAG2 + _PR3 + _LOT + _CUM, 20, True),
    'N_raccel_T':       (_ME8 + _RACCEL + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN3,         15,  True),
    'N_raccel_ME9T':    (_ME9 + _RACCEL + _OVI5 + _SR + _PR3 + _LOT + _CUM2 + _IXN3,         15,  True),
    # 条件 OVI 族
    'N_vcovi_T':        (_ME8 + _VCOVI + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN3,          15,  True),
    'N_idio_ME2':       (_ME9 + _IOVI + _OVI5 + ['aft_12000'] + _SR + _LAG2 + _PR3 + _LOT + _CUM, 20, True),
    'N_vcidio_T':       (_ME9 + _VCOVI + _IOVI + _OVI5 + _SR + _PR3 + _LOT + _CUM2 + _IXN3,  15,  True),
    # 截面书压/加速度族
    'N_esobg_T':        (_ME8 + _ESOBG + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN3,          15,  True),
    'N_esobg_ME2':      (_ME9 + _ESOBG + _OVI5 + ['aft_12000'] + _SR + _LAG2 + _PR3 + _LOT + _CUM, 20, True),
    'N_oaccel_T':       (_ME8 + _OACCEL + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN3,         15,  True),
    # 非线性 OVI 族
    'N_swovi_T':        (_ME8 + _SWOVI + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN3,          15,  True),
    'N_ovisq_T':        (_ME8 + _OVISQ + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN3,          15,  True),
    'N_ovisq_ME2':      (_ME9 + _OVISQ + _OVI5 + ['aft_12000'] + _SR + _LAG2 + _PR3 + _LOT + _CUM, 20, True),
    # 截面反转族
    'N_eslg_T':         (_ME8 + _ESLG + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN3,           15,  True),
    'N_eslg_ME2':       (_ME9 + _ESLG + _OVI5 + ['aft_12000'] + _SR + _LAG2 + _PR3 + _LOT + _CUM, 20, True),
}

MODEL_NAMES    = list(MODELS.keys())
MODEL_IS_NICHE = [v[2] for v in MODELS.values()]

# ── 动态集成超参数 ────────────────────────────────────────────────────────────
ENSEMBLE_WINDOW      = 600    # 滚动 IC 窗口（tick 数）
ENSEMBLE_TEMP        = 17     # softmax 温度（控制权重集中程度）
ENSEMBLE_FLOOR       = 0.0    # 权重下限
RETURN_DELAY         = 600    # Return5min 可知延迟（5 分钟 / 0.5s = 600 ticks）
NICHE_INIT_WEIGHT    = 0.0    # niche 模型预热期权重（0 = 等待滚动 IC 发现价值）
ENSEMBLE_UPDATE_FREQ = 15     # 权重更新频率（tick 数）
ENSEMBLE_EWMA_BETA   = 0.007  # IC 的 EWMA 平滑系数（α = 0.007）
STABLE_PRIOR         = 0.0    # 稳定模型权重下限


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
    __init__ 从 models.pkl 加载预训练的子模型系数（离线在全量数据上训练）。
    online_predict 接受逐 tick 数据，返回该 tick 的 Return5min 预测值。
    """

    # ─────────────────────────────────────────────────────────────────────────
    def __init__(self):
        # ── 1. 加载预训练模型系数 ─────────────────────────────────────────────
        pkl_path = os.path.join(os.path.dirname(__file__), 'models.pkl')
        try:
            with open(pkl_path, 'rb') as f:
                self._coefs: dict[str, tuple[np.ndarray, float]] = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"找不到预训练模型文件: {pkl_path}\n"
                "请先运行 train_model.py 生成 models.pkl："
                "  cd project && python train_model.py"
            )
        except Exception as e:
            raise RuntimeError(
                f"加载模型文件失败: {pkl_path}\n"
                f"原始错误: {e}\n"
                "请重新运行 train_model.py 生成 models.pkl。"
            ) from e

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

        # ── 全档书压（book_pres_pulse = 全5档委托失衡 SMA15-SMA600 脉冲）────────
        self._bp_s15    = _RunSMA(15);   self._bp_s600   = _RunSMA(600)

        # ── 波动率滚动窗口（vol_cond_ovi 所需，past_ret_30 的短/长期滚动标准差）
        self._pr30_buf120 = deque(maxlen=120)
        self._pr30_buf600 = deque(maxlen=600)

        # ── 价格 lag 缓冲区（E 中间价）────────────────────────────────────────
        # 保存最近 901 个 tick 的中间价
        self._mid_buf  = deque(maxlen=901)
        self._smid_buf = deque(maxlen=121)  # 板块均值中间价，最长 lag=120

        # ── Return5min lag 缓冲区 ─────────────────────────────────────────────
        # Return5min(t) 在 t+600 tick 后可知：
        #   e_ret_lag    = Return5min(t-600)，sect_ret_lag = 各板块均值
        #   e_ret_lag2   = Return5min(t-900)
        # 日初不足延迟长度时返回 0（中性）
        self._e_ret_buf    = deque(maxlen=901)  # E 的 Return5min lag 缓冲
        self._sec_ret_bufs = [deque(maxlen=601) for _ in range(4)]  # A/B/C/D

        # ── 动态集成状态 ─────────────────────────────────────────────────────
        n_models = len(MODEL_NAMES)
        self._roll_ic      = [_RollingIC(ENSEMBLE_WINDOW) for _ in range(n_models)]
        self._ewma_ic      = np.zeros(n_models, dtype=float)
        self._weights      = self._warmup_weights()
        self._pred_bufs    = [deque(maxlen=ENSEMBLE_WINDOW) for _ in range(n_models)]
        self._ret_for_ic   = deque(maxlen=ENSEMBLE_WINDOW)
        self._preds_delay  = [deque(maxlen=RETURN_DELAY + 1) for _ in range(n_models)]
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

        # 全档书压：全 5 档买卖委托失衡（与深层委托簿 2-5 档信号互补）
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

        # 全档书压 SMA
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
        # e_ret_lag2：需要 900-tick 前的收益（buffer maxlen=901，取最旧元素）
        if len(self._e_ret_buf) > 900:
            e_ret_lag2 = list(self._e_ret_buf)[0]
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
            """计算 lag 个 tick 前的中间价收益率，裁剪至 [-clip, clip]。"""
            if len(buf) <= lag:
                return 0.0
            old = list(buf)[-(lag + 1)]
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

        # ── 组装 59 个特征 ─────────────────────────────────────────────────
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
            # OVI 非线性交互特征
            'ovi_x_abs_ret':      float(np.clip(ovi_p15 * abs(pr600) * 20, -0.5, 0.5)),
            'tbv_x_ovi':          float(np.clip(tbv / (tbv600 + 1e-9) * ovi_p15 * 5, -0.5, 0.5)),
            'srl_x_ovi':          float(np.clip(
                                      np.clip(sect_ret_lag, -_RET_CLIP_LONG, _RET_CLIP_LONG)
                                      * ovi_p15 * 20, -0.5, 0.5)),
            'ret_x_cum':          float(np.clip(pr600 * float(np.clip(cum_raw - cum_base, -0.5, 0.5))
                                                * 20, -0.5, 0.5)),
            'oni_x_ovi':          float(np.clip((oni15 - oni600) * ovi_p15 * 5, -0.5, 0.5)),
            # 书压 & 动量特征
            'book_pres_pulse':    float(np.clip(bp15 - bp600, -0.5, 0.5)),
            'ret_x_ti600':        float(np.clip(pr600 * ti600 * 20, -0.5, 0.5)),
            'ret_accel':          float(np.clip(pr300 - pr600, -0.1, 0.1)),
            # 条件化 OVI 特征（vol_cond_ovi 占位，后续填入）
            'vol_cond_ovi':       0.0,
            'idio_ovi':           (ovi_e15 - ovi_e600) - (s_ovi_e15 - s_ovi_e600),
            # 截面书压 & 委托加速度
            'e_sect_obi_gap':     float(np.clip((deep15 - deep600) - sect_obi1, -1.0, 1.0)),
            'oni_accel':          (oni15 - oni600) - (oni30 - oni600),
            # 非线性 OVI & 截面反转
            'spread_wt_ovi':      float(np.clip(
                                      (1 + (e_spread - spd600) * 10) * ovi_p15,
                                      -0.5, 0.5)),
            'ovi_sq':             float(np.clip(
                                      ovi_p15**2 * np.sign(ovi_p15) * 10,
                                      -0.5, 0.5)),
            'e_sect_lag_gap':     float(np.clip(sect_ret_lag - e_ret_lag, -0.1, 0.1)),
        }

        # ── 计算 vol_cond_ovi（短期/长期波动率比值条件化 OVI_p15）──────────────
        self._pr30_buf120.append(pr30)
        self._pr30_buf600.append(pr30)
        if len(self._pr30_buf120) >= 10:
            vol_120 = float(np.std(list(self._pr30_buf120)))
        else:
            vol_120 = 0.0
        if len(self._pr30_buf600) >= 60:
            vol_600 = float(np.std(list(self._pr30_buf600)))
        else:
            vol_600 = 0.0
        vol_ratio = vol_120 / (vol_600 + 1e-9)
        feats_arr['vol_cond_ovi'] = float(np.clip(ovi_p15 * vol_ratio * 3, -0.5, 0.5))

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
                # EWMA 平滑 IC 估计：α*IC(t) + (1-α)*EWMA(t-1)
                self._ewma_ic[mi] = (ENSEMBLE_EWMA_BETA * ic_val
                                     + (1.0 - ENSEMBLE_EWMA_BETA) * self._ewma_ic[mi])

        # 权重更新（按 update_freq 频率）
        warmup = RETURN_DELAY + ENSEMBLE_WINDOW // 4
        if t < warmup:
            weights = self._warmup_weights()
        elif t - self._last_upd >= ENSEMBLE_UPDATE_FREQ:
            # softmax 权重：以 EWMA-IC × 温度 为 logit
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
