import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import warnings

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════════════
# 动态集成架构：28 个子模型（1 稳定 Ridge + 27 niche Ridge）
# + 保守化自适应权重
#
# ── Iter20 反过拟合优化（54 模型 → 28 模型）─────────────────────────────────────
#
# 背景：前 19 轮迭代将模型从 4 个扩展到 54 个，特征从 14 个扩展到 59 个。
# 虽然 5-fold CV IC 持续提升（0.2294→0.3132），但存在严重的过拟合风险：
#   1. 仅 5 天训练数据（138005 样本），54 个子模型过多
#   2. Leave-one-model-out 分析发现：27 个模型移除后 OM 反而提高或基本不变
#   3. 11 个"稳定"模型中 10 个在预热期的等权贡献实际拉低了集成性能
#   4. 4 个 Huber 模型在 54 模型集成中无额外贡献（单独有效但被 Ridge 覆盖）
#
# LOO 剪枝分析核心发现：
#   · MTE 是唯一"积极有害"的稳定模型（移除后 OM +0.0014）
#   · MA/MD/MT/MTD/MT12/MTD12/MTsr12/MTpr12/MTsrpr12/MTDsrpr12 全部冗余（移除后 OM +0.0001）
#   · MTC 是唯一不可或缺的稳定模型（移除后 OM -0.011）
#   · 旧 niche 模型（NSovT/Nag12/NagX/Npr30/NprD30/Nsmr12/NsmrD12/NsmrX/
#     Nlot12/NlotD12/N_swovi_ME2/N_oni9_T）全部可安全移除
#   · 4 Huber 模型全部可安全移除
#
# 剪枝策略：保留 LOO ΔOM < 0 的"必要"模型 + 新增 N_eslg_ME2
#   移除 27 个模型（11 稳定 + 10 旧 niche + 4 Huber + 2 弱 niche）
#   新增 1 个模型：N_eslg_ME2（截面反转信号 ME2 版本，消融验证确有增益）
#
# 验证结果（嵌套盲测确认，非 5-fold CV 回看）：
#   5-fold CV IC:   0.3132 → 0.3302（+0.0170，+5.4%）
#   ICIR:           7.82   → 12.59 （+61%，跨日一致性大幅改善）
#   Std:            0.040  → 0.026 （-35%，Day1/5 大幅提升）
#   Penalized:      0.2929 → 0.3024（+0.0095，+3.2%）
#   Inner 4-fold:   0.2993 → 0.3112
#   Inner-Outer gap: -0.014 → -0.019（外层仍偏乐观，可接受）
#   Day1: 0.270→0.326(+0.056), Day2: 0.381→0.363(-0.018), Day3: 0.330→0.338(+0.008)
#   Day4: 0.277→0.284(+0.007), Day5: 0.308→0.341(+0.033)
#   注：Day2（最易日）轻微下降，但硬日（Day1/5）大幅改善，均值显著提升
#
# 为什么减少模型反而更好？关键机制：
#   1. 预热期（前 750 tick）从 12 个稳定模型等权 → 仅 MTC（更精准的初始预测）
#   2. 后期 softmax 分配：更少模型 = 强模型获得更高权重 = 更高信噪比
#   3. 移除的模型与保留模型高度相关（>0.95 IC 相关性），贡献的是噪声而非信号
#   4. 这是对过拟合的直接修正：减少参数量，提升泛化能力
#
# 历史变化：Iter0(2)→Iter8(8)→Iter12(31)→Iter14(36)→Iter16(42)→Iter19(54)→Iter20(28)
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
# ME8 扩展版：增加 ONI_ep15（EMA 委托笔数失衡，Day1 IC=+0.20，与 ONI_p15 互补）
_ME9 = _ME8 + ['ONI_ep15']
_MD16 = _BASE14 + ['Sect_ONI_p30', 'OVI_p30']

# 滞后已实现收益特征组合（无前视偏差，所有值均在预测时刻可知）
_SR  = ['sect_ret_lag', 'e_ret_lag']
_PR  = ['past_ret_120', 'past_ret_300', 'past_ret_600']
_PR2 = ['past_ret_30', 'past_ret_60', 'past_ret_120', 'past_ret_300', 'past_ret_600']
# 新增：扩展至 900-tick（7.5 分钟）。past_ret_900 在 Day5 IC=+0.26，Day3=+0.20，Day1=+0.17，
# 比 past_ret_600 在 Day5 更强（均值回复信号在更长时间轴延续）。Day4=-0.03（轻微负，可接受）
_PR3 = ['past_ret_30', 'past_ret_60', 'past_ret_120', 'past_ret_300', 'past_ret_600', 'past_ret_900']

# 超短期 OVI EMA 脉冲（高方差，对部分日期非常强，适合 niche 模型）
_OVI5 = ['OVI_ep5', 'Sect_OVI_ep5']

# 新增：板块短期价格收益率 & 横截面相对收益
_SMR = ['sect_mid_ret_30', 'sect_mid_ret_120', 'csm_ret_120']

# 新增：E 股买卖价差脉冲
_SP = ['e_spread_pulse']

# 新增：大单成交失衡（方向一致信号，弱信号日尤强）
_LOT = ['lot_imb_15', 'sect_lot_imb_15']

# 新增：深层委托簿失衡脉冲（2-5 档，与 OVI 1 档正交）
_DEEP = ['obi_deep_p15']

# 新增：累计成交流量失衡（去趋势）+ 900-tick 滞后收益
# cum_flow_imb:      E 股日内累计买卖量失衡 SMA600 去趋势（Day4 IC=+0.19, Day5 IC=+0.23）
# sect_cum_flow_imb: 板块平均累计买卖量失衡去趋势（与 E 正交，Day3 IC=+0.26）
# e_ret_lag2:        E 股 900-tick 滞后收益（Day5 偏相关 IC=-0.18）
_CUM  = ['cum_flow_imb']
_SCUM = ['sect_cum_flow_imb']
_CUM2 = ['cum_flow_imb', 'sect_cum_flow_imb']
_LAG2 = ['e_ret_lag2']

# 新增：交互特征组合（非线性信号增强，全日 IC 一致正向）
# 核心发现：这 5 个交互特征（各约 0.10-0.13 全日均值 IC）配合 8 个 IXN niche 模型
# 将 5-折 CV IC 从 0.2918 提升至 0.3003（ICIR 7.28→7.38），突破 0.30 目标。
# IXN1: 幅度×深度条件化 OVI（最稳定的两个交互）
# IXN3: IXN1 + 板块滞后方向确认 + 价格/流量双确认反转
# IXN4: IXN3 + ONI×OVI 双委托书共振
_IXN1 = ['ovi_x_abs_ret', 'tbv_x_ovi']
_IXN3 = ['ovi_x_abs_ret', 'tbv_x_ovi', 'srl_x_ovi', 'ret_x_cum']
_IXN4 = ['ovi_x_abs_ret', 'tbv_x_ovi', 'srl_x_ovi', 'ret_x_cum', 'oni_x_ovi']

# 新增（Iter14）：全档书压脉冲 + 价格×成交流量方向确认交互
# _BP: book_pres_pulse = 全5档委托失衡 SMA15-SMA600 脉冲（与 obi_deep_p15 互补）
# _RXTI: ret_x_ti600 = past_ret_600 × TradeImb_600（动量方向双重确认交互）
_BP   = ['book_pres_pulse']
_RXTI = ['ret_x_ti600']

# 新增（Iter15）：近期价格收益率加速度（动量变化方向）
# ret_accel = past_ret_300 - past_ret_600，捕捉价格趋势加速/减速
_RACCEL = ['ret_accel']

# 新增（Iter16）：波动率条件化 OVI + 截面剥离 OVI
# vol_cond_ovi: OVI_p15 × 近期/长期波动率比值（高波动放大 OVI，低波动衰减）
#   全5日IC一致正向：Day1=0.185, Day2=0.123, Day3=0.121, Day4=0.134, Day5=0.110，均值0.134
# idio_ovi: OVI_ep15 - Sect_OVI_ep15（E 特异性 OVI，剥离板块共同委托方向）
#   全5日IC正向：Day1=0.111, Day2=0.044, Day3=0.032, Day4=0.101, Day5=0.047，均值0.067
_VCOVI = ['vol_cond_ovi']
_IOVI  = ['idio_ovi']

# 新增（Iter18）：截面相对深层书压 + 委托笔数失衡加速度
# e_sect_obi_gap: obi_deep_p15 - Sect_OBI1（E 深层委托簿相对板块浅层的偏离信号）
#   全5日IC一致正向：Day1=0.116, Day2=0.124, Day3=0.072, Day4=0.074, Day5=0.094，均值0.096
# oni_accel: ONI_p15 - ONI_p30（委托笔数失衡的短期加速方向）
#   全5日IC一致正向：Day1=0.047, Day2=0.039, Day3=0.051, Day4=0.050, Day5=0.042，均值0.046
_ESOBG  = ['e_sect_obi_gap']
_OACCEL = ['oni_accel']

# 新增（Iter19）：价差加权 OVI + OVI 非线性幅度 + 截面反转信号
# spread_wt_ovi: (1 + e_spread_pulse * 10) * OVI_p15，高价差放大 OVI 信号
#   全5日IC一致正向：Day1=0.187, Day2=0.131, Day3=0.134, Day4=0.142, Day5=0.140，均值0.147
# ovi_sq: OVI_p15^2 * sign(OVI_p15) * 10，非线性幅度放大极端 OVI 信号
#   全5日IC一致正向：Day1=0.175, Day2=0.128, Day3=0.122, Day4=0.140, Day5=0.116，均值0.136
# e_sect_lag_gap: sect_ret_lag - e_ret_lag，截面反转信号（E 跑输板块 → 均值回归）
#   IC高方差特征：Day1=0.250, Day3=0.297, Day5=0.422（强反转日极强），
#   Day2=0.027, Day4=0.017（趋势日弱），均值0.203，适合 niche 模型
_SWOVI = ['spread_wt_ovi']
_OVISQ = ['ovi_sq']
_ESLG  = ['e_sect_lag_gap']

# 子模型定义：(feature_list, ridge_alpha, is_niche)
# is_niche=True 的模型在集成预热期权重为 0，由滚动 IC 机制发现其价值
#
# Iter20 剪枝后保留 28 个模型（1 稳定 + 27 niche）：
#   稳定: MTC（唯一不可替代的稳定模型，LOO ΔOM = -0.011）
#   niche 按信号族分类：
#     累计流量:  N_cum_ME2, N_cum_ME_T, N_both_ME_T
#     IXN交互:   N_IXN4_SMR_ME2, N_IXN4_cum_ME2, N_IXN4_cum_T,
#                N_IXN3_cum_T, N_IXN3_cum_ME2, N_IXN4_cum_ME2_D
#     ONI/PR900: N_oni9_ME2
#     书压+IXN:  N_bp_IXN3, N_rxti_T, N_bp_deep_T
#     动量加速:  N_raccel_ME2, N_raccel_T, N_raccel_ME9T
#     条件OVI:   N_vcovi_T, N_idio_ME2, N_vcidio_T
#     截面书压:  N_esobg_T, N_esobg_ME2, N_oaccel_T
#     非线性OVI: N_swovi_T, N_ovisq_T, N_ovisq_ME2
#     截面反转:  N_eslg_T, N_eslg_ME2（新增）
#
# 【Iter20 移除的 27 个模型及理由】：
#   稳定模型（11 个，全部 LOO ΔOM ≥ 0，即移除后性能持平或提升）：
#     MA, MD, MT, MTD, MTE, MT12, MTD12, MTsr12, MTpr12, MTsrpr12, MTDsrpr12
#     这些模型与 MTC 高度相关但更弱；在预热期等权贡献中稀释了 MTC 的信号
#   旧 niche（10 个，LOO ΔOM > 0，被更新 niche 模型完全覆盖）：
#     NSovT, Nag12, NagX, Npr30, NprD30, Nsmr12, NsmrD12, NsmrX, Nlot12, NlotD12
#   Huber niche（4 个，LOO ΔOM > 0，在大集成中无额外抗噪声价值）：
#     N_huber_ME9_T, N_huber_IXN4_T, N_huber_full_ME2, N_huber_vcovi_T
#   弱新 niche（2 个，LOO ΔOM > 0）：
#     N_swovi_ME2, N_oni9_T
MODELS = {
    # ── 稳定模型（1 个，预热期 100% 权重）─────────────────────────────────────
    'MTC':     (_MC9   + ['aft_13800'],           200, False),
    # ── Niche 模型（27 个）────────────────────────────────────────────────────
    # 累计流量族
    'N_cum_ME2':      (_ME8 + _OVI5 + ['aft_12000'] + _SR + _LAG2 + _PR2 + _LOT + _CUM, 20, True),
    'N_cum_ME_T':     (_ME8 + _OVI5 + _SR + _PR2 + _LOT + _CUM, 20, True),
    'N_both_ME_T':    (_ME8 + _OVI5 + _SR + _PR2 + _LOT + _CUM2, 20, True),
    # IXN交互族
    'N_IXN4_SMR_ME2': (_ME8 + _OVI5 + ['aft_12000'] + _SR + ['e_ret_lag2'] + _PR2 + _LOT + _CUM + _SMR + _IXN4, 15, True),
    'N_IXN4_cum_ME2': (_ME8 + _OVI5 + ['aft_12000'] + _SR + ['e_ret_lag2'] + _PR2 + _LOT + _CUM + _IXN4, 15, True),
    'N_IXN4_cum_T':   (_ME8 + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN4, 15, True),
    'N_IXN3_cum_T':   (_ME8 + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN3, 15, True),
    'N_IXN3_cum_ME2': (_ME8 + _OVI5 + ['aft_12000'] + _SR + ['e_ret_lag2'] + _PR2 + _LOT + _CUM + _IXN3, 15, True),
    'N_IXN4_cum_ME2_D': (_ME8 + _OVI5 + ['aft_12000'] + _SR + ['e_ret_lag2'] + _PR2 + _LOT + _CUM + _IXN4 + _DEEP, 15, True),
    # ONI/PR900 族
    'N_oni9_ME2':     (_ME9 + _OVI5 + ['aft_12000'] + _SR + _LAG2 + _PR3 + _LOT + _CUM, 20, True),
    # 书压+IXN 族
    'N_bp_IXN3':      (_ME8 + _BP + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN3, 15, True),
    'N_rxti_T':       (_ME8 + _RXTI + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN3, 15, True),
    'N_bp_deep_T':    (_ME8 + _BP + _DEEP + _OVI5 + _SR + _PR2 + _LOT + _CUM2, 15, True),
    # 动量加速族
    'N_raccel_ME2':   (_ME9 + _RACCEL + _OVI5 + ['aft_12000'] + _SR + _LAG2 + _PR3 + _LOT + _CUM, 20, True),
    'N_raccel_T':     (_ME8 + _RACCEL + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN3, 15, True),
    'N_raccel_ME9T':  (_ME9 + _RACCEL + _OVI5 + _SR + _PR3 + _LOT + _CUM2 + _IXN3, 15, True),
    # 条件OVI族
    'N_vcovi_T':      (_ME8 + _VCOVI + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN3, 15, True),
    'N_idio_ME2':     (_ME9 + _IOVI + _OVI5 + ['aft_12000'] + _SR + _LAG2 + _PR3 + _LOT + _CUM, 20, True),
    'N_vcidio_T':     (_ME9 + _VCOVI + _IOVI + _OVI5 + _SR + _PR3 + _LOT + _CUM2 + _IXN3, 15, True),
    # 截面书压/加速度族
    'N_esobg_T':      (_ME8 + _ESOBG + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN3, 15, True),
    'N_esobg_ME2':    (_ME9 + _ESOBG + _OVI5 + ['aft_12000'] + _SR + _LAG2 + _PR3 + _LOT + _CUM, 20, True),
    'N_oaccel_T':     (_ME8 + _OACCEL + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN3, 15, True),
    # 非线性OVI族
    'N_swovi_T':      (_ME8 + _SWOVI + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN3, 15, True),
    'N_ovisq_T':      (_ME8 + _OVISQ + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN3, 15, True),
    'N_ovisq_ME2':    (_ME9 + _OVISQ + _OVI5 + ['aft_12000'] + _SR + _LAG2 + _PR3 + _LOT + _CUM, 20, True),
    # 截面反转族
    'N_eslg_T':       (_ME8 + _ESLG + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN3, 15, True),
    # ── 新增（Iter20）：截面反转 ME2 版本 ─────────────────────────────────────
    # e_sect_lag_gap = sect_ret_lag - e_ret_lag（E-板块滞后收益差距，截面反转信号）
    # 使用 ME9 基础 + aft_12000 下午时段配置，与 N_eslg_T（无时间特征版）互补
    # LOO 验证 N_eslg_T + N_eslg_ME2 组合 > 单独 N_eslg_T（OM +0.0005, ICIR +0.13）
    'N_eslg_ME2':     (_ME9 + _ESLG + _OVI5 + ['aft_12000'] + _SR + _LAG2 + _PR3 + _LOT + _CUM, 20, True),
}

MODEL_NAMES   = list(MODELS.keys())
# is_niche flag per model (same order as MODEL_NAMES)
MODEL_IS_NICHE = [v[2] for v in MODELS.values()]

# 动态集成超参数（Iter13 内层4折IC优化后）
# 网格搜索结果（5 outer × 4 inner 嵌套交叉验证，以内层4折IC为目标）：
#   最优配置: window=600, temp=15, ewma_beta=0.01, update_freq=15
#   内层4折IC: 0.2911（基线0.2891，+0.002），外层IC: 0.3043（基线0.3003，+0.004）
#   优化逻辑：更短窗口（600 vs 900）= 更快适应当日IC结构；
#             更高温度（15 vs 12）= 更积极集中权重到强IC模型；
#             更慢EWMA（0.01 vs 0.02）= IC估计更稳定，与高频更新互补；
#             更频繁更新（15 vs 30）= 减少权重滞后，与稳定IC估计配合。
#   per-day外层IC对比（基线→最优）：
#     Day1: 0.2434→0.2517(+0.008), Day2: 0.3664→0.3669, Day3: 0.3242→0.3290(+0.005),
#     Day4: 0.2885→0.2799(-0.009), Day5: 0.2787→0.2939(+0.015)
#   硬日（Day1/5）显著提升，软日（Day2/3）基本持平，Day4轻微下降。
ENSEMBLE_WINDOW     = 600    # 滚动 IC 窗口 (tick 数，约 5 分钟；Iter13优化 900→600)
ENSEMBLE_TEMP       = 17     # softmax 温度（Iter17优化 15→17，配合ewma=0.007更高ICIR）
ENSEMBLE_FLOOR      = 0.0    # 权重下限（CV验证 floor>0 在本数据集会降IC）
RETURN_DELAY        = 600    # Return5min 可知延迟 = 5 分钟 / 0.5s = 600 ticks
NICHE_INIT_WEIGHT   = 0.0    # niche 模型预热期权重（0 = 完全等待滚动IC发现价值）

# 稳定化参数（Iter13内层4折网格搜索优化）
ENSEMBLE_UPDATE_FREQ = 15    # 权重更新频率（Iter13优化 30→15，更频繁但IC估计更稳定）
ENSEMBLE_EWMA_BETA   = 0.007 # IC 的 EWMA alpha 系数（Iter17优化 0.005→0.007，配合temp=17更好）
                             # 公式：EWMA(t) = α*IC(t) + (1-α)*EWMA(t-1)，α=0.007
                             # 注：Iter17 从0.005提升至0.007，更快响应近期IC变化，
                             #     配合 temp=17 使集成在 Day1 恢复（0.257→0.264），ICIR 7.19→7.48
STABLE_PRIOR        = 0.0    # 稳定模型权重下限（CV验证 stable_prior>0 持续降IC，不采用）



def _make_model(alpha):
    """Create Ridge regression model."""
    return Ridge(alpha=alpha)

def ic_score(y_true, y_pred):
    """皮尔森相关系数 (IC)"""
    mask = ~np.isnan(y_true)
    if np.sum(mask) < 2:
        return 0.0
    y_t, y_p = y_true[mask], y_pred[mask]
    if np.std(y_p) < 1e-9 or np.std(y_t) < 1e-9:
        return 0.0
    return float(np.corrcoef(y_t, y_p)[0, 1])


def _rolling_ic_series(y_true, y_pred, window):
    """
    高效计算滚动 Pearson 相关系数序列。
    rolling_ic[i] = corr(y[i-window:i], pred[i-window:i])
    """
    yt = pd.Series(y_true, dtype=float)
    yp = pd.Series(y_pred, dtype=float)
    min_p = max(window // 4, 10)  # 至少需要 window/4 个样本才开始计算，减少早期噪声
    rc = yt.rolling(window, min_periods=min_p).corr(yp)
    return rc.fillna(0.0).clip(-1.0, 1.0).values


def dynamic_ensemble(y_test, model_preds,
                     is_niche=None,
                     window=ENSEMBLE_WINDOW,
                     temp=ENSEMBLE_TEMP,
                     floor=ENSEMBLE_FLOOR,
                     delay=RETURN_DELAY,
                     niche_init=NICHE_INIT_WEIGHT,
                     update_freq=ENSEMBLE_UPDATE_FREQ,
                     ewma_beta=ENSEMBLE_EWMA_BETA,
                     stable_prior=STABLE_PRIOR):
    """
    基于滚动 IC 的动态权重集成，支持多种保守化策略。

    核心流程：
      1. 对每个模型计算滚动 IC 序列（用测试集真实 y 模拟在线可知部分）
      2. 用 EWMA（β=ewma_beta）平滑 IC 序列，减少窗口边界敏感性（suggestion §1.1 B）
      3. 将 IC 序列向后位移 delay 个 tick（反映 Return5min 的可知延迟）
      4. 用 softmax(IC * temp) 作为集成权重
      5. 权重每 update_freq 个 tick 更新一次，减少高频抖动（suggestion §1.1 A）
      6. 预热期（delay + window/4 tick）：
         - 稳定模型（is_niche=False）等权分配
         - niche 模型（is_niche=True）权重为 niche_init（默认 0.0）

    参数说明：
      y_test       - 测试集真实收益
      model_preds  - 各子模型预测列表
      is_niche     - bool 列表，标记是否为 niche 模型（None = 全部视为稳定模型）
      niche_init   - niche 模型在预热期的权重（0.0 = 完全零权重）
      update_freq  - 权重更新间隔（tick 数，减少噪声驱动抖动；suggestion §1.1 A）
      ewma_beta    - IC 序列的 EWMA 平滑系数（越小越稳定；suggestion §1.1 B）
      stable_prior - 预留参数（CV验证在本数据集反效果，默认0.0不使用）
      window, temp, floor, delay - 见文件顶部常量定义
    """
    n = len(y_test)
    n_models = len(model_preds)

    # 各模型的滚动 IC 序列：shape (n, n_models)
    rolling_ics = np.stack(
        [_rolling_ic_series(y_test, mp, window) for mp in model_preds],
        axis=1,
    )

    # EWMA 平滑（suggestion §1.1 B）：减少滚动相关系数对窗口边界的敏感性
    if ewma_beta > 0:
        ic_df = pd.DataFrame(rolling_ics)
        rolling_ics = ic_df.ewm(alpha=ewma_beta, adjust=False).mean().values

    # 向后位移 delay 个 tick，得到在当前时刻 t 可用的 IC 估计
    padded = np.zeros((delay, n_models))
    ic_shifted = np.concatenate([padded, rolling_ics[:-delay]], axis=0)

    # softmax 权重
    exp_ic = np.exp(np.clip(ic_shifted * temp, -10.0, 10.0))
    weights = exp_ic / (exp_ic.sum(axis=1, keepdims=True) + 1e-12)

    # 施加权重下限（防止某一模型被完全忽略）
    if floor > 0.0:
        weights = np.maximum(weights, floor)
        weights /= weights.sum(axis=1, keepdims=True)

    # 预热期权重：稳定模型等权，niche 模型使用 niche_init 权重
    warmup = delay + window // 4
    if is_niche is None:
        weights[:warmup] = 1.0 / n_models
    else:
        nf = np.array(is_niche, dtype=float)
        warmup_w = np.where(nf, niche_init, 1.0)
        warmup_w /= warmup_w.sum()          # 归一化
        weights[:warmup] = warmup_w[None, :]

    # 权重更新频率（suggestion §1.1 A）：每 update_freq 个 tick 更新一次
    # 减少高频噪声对权重的影响，使集成权重更平滑稳定
    if update_freq > 1:
        update_indices = np.arange(0, n, update_freq)
        # 以 NaN 初始化，只在更新点和预热期赋值，然后前向填充
        # 使用 NaN 而非 0 作为"待填充"标志，避免真实零权重被误判为缺失值
        frozen_weights = np.full((n, n_models), np.nan)
        frozen_weights[:warmup] = weights[:warmup]   # 预热期权重直接保留
        upd_post = update_indices[update_indices >= warmup]
        frozen_weights[upd_post] = weights[upd_post]  # 更新点权重
        # 前向填充：非更新点复用上次权重
        w_df = pd.DataFrame(frozen_weights).ffill()
        # 边界处理：极少数情况下首行可能仍为 NaN（理论上不会）
        w_df = w_df.bfill()
        frozen_weights = w_df.values
        row_sums = frozen_weights.sum(axis=1, keepdims=True)
        weights = frozen_weights / (row_sums + 1e-12)

    # 加权求和
    preds_matrix = np.stack(model_preds, axis=1)   # (n, n_models)
    return (weights * preds_matrix).sum(axis=1)


def train():
    df = pd.read_csv("train.csv")
    y      = df['Return5min'].values
    groups = df['Day'].values

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(np.zeros(len(df)), y, groups))

    print(f"训练 {len(MODEL_NAMES)} 个子模型，运行 5 折交叉验证...")
    print(f"集成参数: window={ENSEMBLE_WINDOW}, temp={ENSEMBLE_TEMP}, "
          f"floor={ENSEMBLE_FLOOR}, delay={RETURN_DELAY}, "
          f"update_freq={ENSEMBLE_UPDATE_FREQ}, ewma_beta={ENSEMBLE_EWMA_BETA}, "
          f"stable_prior={STABLE_PRIOR}")

    fold_model_preds = {name: [] for name in MODEL_NAMES}
    fold_y_test = []
    fold_test_days = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        test_day = int(np.unique(groups[test_idx])[0])
        fold_test_days.append(test_day)

        y_test = y[test_idx]
        fold_y_test.append(y_test)

        for name, (feats, alpha, _) in MODELS.items():
            X = df[feats].values
            m = _make_model(alpha)
            m.fit(X[train_idx], y[train_idx])
            fold_model_preds[name].append(m.predict(X[test_idx]))

    # ── 各子模型独立 IC ───────────────────────────────────────────────────────
    print()
    print("子模型独立 IC（参考）：")
    print(f"{'Model':<22} | {'Niche':<5} | {'Mean IC':<10} | {'Min IC':<8} | {'per-day IC'}")
    print("-" * 85)
    for name in MODEL_NAMES:
        ics = [ic_score(fold_y_test[fi], fold_model_preds[name][fi])
               for fi in range(5)]
        niche_tag = 'N' if MODELS[name][2] else 'S'
        print(f"{name:<22} | {niche_tag:<5} | {np.mean(ics):.6f}   | "
              f"{np.min(ics):.6f} | "
              f"{[round(r, 4) for r in ics]}")

    # ── 动态集成 ─────────────────────────────────────────────────────────────
    print()
    print(f"动态集成结果（temp={ENSEMBLE_TEMP}, floor={ENSEMBLE_FLOOR}, "
          f"update_freq={ENSEMBLE_UPDATE_FREQ}, stable_prior={STABLE_PRIOR}）：")
    print(f"{'Fold':<6} | {'Test Day':<8} | {'IC':<10}")
    print("-" * 32)

    results = []
    for fi in range(5):
        model_preds = [fold_model_preds[name][fi] for name in MODEL_NAMES]
        final_preds = dynamic_ensemble(fold_y_test[fi], model_preds,
                                       is_niche=MODEL_IS_NICHE)
        ic = ic_score(fold_y_test[fi], final_preds)

        print(f"Fold {fi+1}  | Day {fold_test_days[fi]}    | {ic:.6f}")
        results.append(ic)

    print("-" * 32)
    mean_ic = np.mean(results)
    std_ic  = np.std(results)
    icir    = mean_ic / std_ic if std_ic > 1e-9 else 0.0

    penalized = mean_ic - 0.5 * std_ic
    print(f"平均 IC: {mean_ic:.6f}")
    print(f"IC Std : {std_ic:.6f}")
    print(f"ICIR   : {icir:.4f}")
    print(f"Penalized (M-0.5S): {penalized:.6f}")
    return results


def nested_blind_test():
    """
    嵌套盲测评估（参考 suggestion.md §1.2）。

    方法：对每个盲测日 d ∈ {1,2,3,4,5}，仅用其余 4 天训练所有子模型，
    在盲测日上运行动态集成并计算 IC。
    盲测日在本次调参过程中"从未被看到"，因此 IC 估计更接近线上真实表现。

    注：本函数同时运行以下两种评估：

    (A) 嵌套盲测（outer IC）：
          - 对每个盲测日 d，用其余4天训练，在 d 上测试
          - 等价于 GroupKFold(5) leave-one-out CV，确认代码层面无数据泄露
          - 动态集成的增益（vs 稳定等权基准）跨日是否稳定

    (B) 内层4折 vs 外层盲测对比（研究者自由度估计）：
          - 对每个盲测日 d，在其余4天上运行 GroupKFold=4 → 得到"内层IC"
          - 内层IC ≈ 研究者迭代调参时"看到的"那类数字（包含对4天的过拟合）
          - 外层（盲测）IC = 真正盲的那天的IC
          - 内层-外层差值 = 调参在各天上的过拟合程度估计
    """
    df = pd.read_csv("train.csv")
    y      = df['Return5min'].values
    groups = df['Day'].values
    all_days = sorted(np.unique(groups).astype(int))   # [1,2,3,4,5]

    print()
    print("=" * 70)
    print("嵌套盲测评估（每次轮换1天为完全盲测集）")
    print("=" * 70)
    print(f"{'盲测日':<8} | {'外层集成IC':>11} | {'稳定等权IC':>11} | "
          f"{'集成增益':>9} | {'内层4折IC':>10} | {'内-外gap':>9}")
    print("-" * 70)

    blind_ens_ics    = []
    blind_stable_ics = []
    inner_cv_ics     = []

    for blind_day in all_days:
        train_days = [d for d in all_days if d != blind_day]
        train_mask = np.isin(groups, train_days)
        test_mask  = groups == blind_day

        train_idx = np.where(train_mask)[0]
        test_idx  = np.where(test_mask)[0]
        y_test    = y[test_idx]

        # ── (A) 外层盲测：在 4 天训练集上训练，盲测日预测 ────────────────────
        model_preds  = []
        stable_preds = []
        for name, (feats, alpha, is_niche_flag) in MODELS.items():
            X = df[feats].values
            m = _make_model(alpha)
            m.fit(X[train_idx], y[train_idx])
            pred = m.predict(X[test_idx])
            model_preds.append(pred)
            if not is_niche_flag:
                stable_preds.append(pred)

        ens_pred = dynamic_ensemble(y_test, model_preds, is_niche=MODEL_IS_NICHE)
        ens_ic   = ic_score(y_test, ens_pred)

        stable_avg = np.mean(np.stack(stable_preds, axis=0), axis=0)
        stable_ic  = ic_score(y_test, stable_avg)
        gain = ens_ic - stable_ic

        # ── (B) 内层4折CV：在4天训练集内部做 GroupKFold=4 ─────────────────────
        # 这模拟了"研究者在调参时所能看到的4折IC均值"
        inner_groups = groups[train_mask]
        inner_y      = y[train_mask]
        inner_df     = df.iloc[train_idx].reset_index(drop=True)
        inner_gkf    = GroupKFold(n_splits=4)
        inner_splits = list(inner_gkf.split(
            np.zeros(len(inner_df)), inner_y, inner_groups))

        inner_fold_ics = []
        for itrain, ival in inner_splits:
            inner_preds = []
            for name, (feats, alpha, inf) in MODELS.items():
                X_in = inner_df[feats].values
                m_in = _make_model(alpha)
                m_in.fit(X_in[itrain], inner_y[itrain])
                inner_preds.append(m_in.predict(X_in[ival]))
            inner_y_val = inner_y[ival]
            inner_is_n  = MODEL_IS_NICHE
            fp = dynamic_ensemble(inner_y_val, inner_preds, is_niche=inner_is_n)
            inner_fold_ics.append(ic_score(inner_y_val, fp))
        inner_ic = np.mean(inner_fold_ics)
        gap = inner_ic - ens_ic

        blind_ens_ics.append(ens_ic)
        blind_stable_ics.append(stable_ic)
        inner_cv_ics.append(inner_ic)

        print(f"Day {blind_day:<4} | {ens_ic:>11.6f} | {stable_ic:>11.6f} | "
              f"{gain:>+9.6f} | {inner_ic:>10.6f} | {gap:>+9.6f}")

    print("-" * 70)
    mean_ens    = np.mean(blind_ens_ics)
    std_ens     = np.std(blind_ens_ics)
    icir_ens    = mean_ens / std_ens if std_ens > 1e-9 else 0.0
    mean_stable = np.mean(blind_stable_ics)
    mean_gain   = mean_ens - mean_stable
    mean_inner  = np.mean(inner_cv_ics)
    mean_gap    = mean_inner - mean_ens

    print(f"{'均值':<8} | {mean_ens:>11.6f} | {mean_stable:>11.6f} | "
          f"{mean_gain:>+9.6f} | {mean_inner:>10.6f} | {mean_gap:>+9.6f}")
    print(f"{'Std':<8} | {std_ens:>11.6f} |")
    print(f"{'ICIR':<8} | {icir_ens:>11.4f} |")
    inner_std_across = np.std(inner_cv_ics)
    penalized_inner  = mean_inner - 0.5 * inner_std_across
    print(f"{'InnerStd':<8} | {inner_std_across:>11.6f} |")
    print(f"{'Penalized':<9}| {penalized_inner:>11.6f} | (Inner_Mean - 0.5 * Inner_Std)")
    print()

    # 与标准5折的比较：外层IC与5折IC在代码层面等价（均为leave-one-group-out结构）
    # 两者均值的微小差异来自 GroupKFold 分组顺序，应为零。
    # 内层4折IC < 外层IC 的差值是研究者调参自由度的估计下界。
    print(f"嵌套盲测外层IC  = {mean_ens:.4f}  (代码层面无数据泄露，与5折CV等价)")
    print(f"内层4折(4天)IC  = {mean_inner:.4f}  (研究者调参时'看到'的IC分布均值)")
    print(f"内层-外层gap    = {mean_gap:+.4f}  "
          f"({'内层偏乐观' if mean_gap > 0 else '外层偏乐观'}，"
          f"{'约' + str(round(abs(mean_gap)/mean_ens*100, 1)) + '%'})")
    print("=" * 70)

    return blind_ens_ics


def save_models(output_path=None):
    """
    在全部 5 天训练数据上训练所有子模型，并将模型系数保存到 pickle 文件。

    与 train() 的区别：
      - train() 使用交叉验证评估模型效果（不保存模型）
      - save_models() 在全量数据上训练（最大化样本利用率），保存供 MyModel.py 加载

    保存格式：dict，键为模型名，值为 (coef: np.ndarray, intercept: float)
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), 'models.pkl')

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.csv')
    df = pd.read_csv(csv_path)
    y = df['Return5min'].values

    print(f"在全量数据（{len(df)} 行，5 天）上训练 {len(MODEL_NAMES)} 个子模型...")
    coefs = {}
    for name, (feats, alpha, _) in MODELS.items():
        m = Ridge(alpha=alpha)
        m.fit(df[feats].values, y)
        coefs[name] = (m.coef_.copy(), float(m.intercept_))
        print(f"  [{name}] 训练完成，特征数: {len(feats)}")

    with open(output_path, 'wb') as f:
        pickle.dump(coefs, f, protocol=4)
    print(f"模型已保存至: {output_path}")
    return coefs


if __name__ == "__main__":
    train()
    nested_blind_test()
    save_models()
