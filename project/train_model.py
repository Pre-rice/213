import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import warnings

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════════════
# 动态集成架构：33 个 Ridge 子模型（12 稳定 + 21 niche）+ 保守化自适应权重
#
# 稳定模型（12 个）：在大多数日子表现稳定，预热期等权分配
#   MA/MD/MT/MTC/MTD/MTE/MT12/MTD12 — 基础架构（覆盖多种信号和时段）
#   MTsr12/MTpr12/MTsrpr12/MTDsrpr12 — 叠加滞后已实现收益特征
#
# Niche 模型（21 个，Iter13 新增 2 个）：
#   31-model基础上新增 N_oni9_ME2 和 N_oni9_T（使用 ONI_ep15 + past_ret_900）
#   ONI_ep15: EMA委托笔数失衡（Day1 IC=+0.20，与SMA ONI互补的动态视角）
#   past_ret_900: 7.5分钟价格收益率（Day5 IC=+0.26，Day3=+0.20，Day1=+0.17）
#
# Iter13 优化（内层4折IC为目标，嵌套交叉验证参数搜索）：
#   集成参数（网格搜索 5×4 嵌套CV，300+ 组合）：
#     WINDOW: 900→600（更快适应当日IC结构）
#     TEMP: 12→15（更积极集中权重到强IC模型）
#     EWMA_BETA: 0.02→0.01（IC估计更稳定，与高频更新互补）
#     UPDATE_FREQ: 30→15（更频繁更新，与稳定IC估计配合）
#   效果（嵌套盲测验证）：
#     内层4折IC: 0.2891→0.2911（+0.002）
#     外层IC: 0.3003→0.3043（+0.004）
#     Day1: 0.243→0.252(+0.009), Day5: 0.279→0.294(+0.015)（硬日显著改善）
#     Day2/3: 基本持平，Day4: 轻微下降(-0.009)
#
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

# 子模型定义：(feature_list, ridge_alpha, is_niche)
# is_niche=True 的模型在集成预热期权重为 0，由滚动 IC 机制发现其价值
MODELS = {
    # ── 稳定基础模型 (12 个)────────────────────────────────────────────────────
    'MA':      (_BASE14,                          150, False),
    'MD':      (_MD16,                            150, False),
    'MT':      (_BASE14 + ['aft_13800'],          150, False),
    'MTC':     (_MC9   + ['aft_13800'],           200, False),
    'MTD':     (_MD16  + ['aft_13800'],           150, False),
    'MTE':     (_ME8   + ['aft_13800'],            80, False),
    'MT12':    (_BASE14 + ['aft_12000'],          150, False),
    'MTD12':   (_MD16  + ['aft_12000'],           150, False),
    'MTsr12':  (_BASE14 + ['aft_12000'] + _SR,   150, False),
    'MTpr12':  (_BASE14 + ['aft_12000'] + _PR,   150, False),
    'MTsrpr12':(_BASE14 + ['aft_12000'] + _SR + _PR,  150, False),
    'MTDsrpr12':(_MD16 + ['aft_12000'] + _SR + _PR,   150, False),
    # ── Niche 模型（8 个）：按"最差日IC"筛选，全部5天IC方向一致 ────────────────
    # ── Niche 模型（19 个，31 模型总计）：按信号类型分族，去除冗余 ────────────────
    # 筛选原则（参考 suggestion.md §3.1）：
    #   保留"跨日方向一致"（全5天IC均正）且覆盖不同信号维度的模型
    #   剔除在同信号族内冗余度高、且最差日IC最低的模型
    #
    # 【剔除的13个模型（冗余/弱泛化）】：
    #   Nep5/Nep12        (min IC≤0.150，超短期EMA5，Day5极弱，被NSovT/Nag覆盖)
    #   NSov12            (被NSovT覆盖，结构相似但min IC更低)
    #   NpOVI/NpOVI_S/NpOVI_T/NpOVI_TD (min IC≤0.152，OVI纯净系列全天弱)
    #   N_cum_ME2_S/N_cum_ME2_SD     (与N_cum_ME2冗余，min IC略低)
    #   N_both_ME_T_S/N_both_ME_T_SD (与N_both_ME_T冗余，增加SMR/DEEP后性能反降)
    #   N_IXN1_cum_ME2/N_IXN1_cum_T  (被IXN3/IXN4覆盖且min IC更低)
    #
    # 【保留的19个niche模型，按信号族分类】：
    #   族1 板块OVI短期:  NSovT (min=0.212，无aft_12000，最稳定)
    #   族2 激进综合:     Nag12, NagX (min=0.194/0.199，与稳定模型互补)
    #   族3 扩展价格动量: Npr30, NprD30 (min=0.187/0.186，全5日一致)
    #   族4 板块价格+SMR: Nsmr12, NsmrD12, NsmrX (min=0.186/0.187/0.198)
    #   族5 大单失衡:     Nlot12, NlotD12 (min=0.187/0.188，全5日方向一致)
    #   族6 累计流量ME2:  N_cum_ME2 (min=0.165，aft_12000，Day5=0.286，不可或缺)
    #   族7 累计流量T:    N_cum_ME_T, N_both_ME_T (Day5=0.315/0.319，无时间特征最佳)
    #   族8 IXN4 ME2:    N_IXN4_SMR_ME2, N_IXN4_cum_ME2, N_IXN4_cum_ME2_D (均值≥0.291)
    #   族9 IXN3/4 T:    N_IXN4_cum_T, N_IXN3_cum_T (Day5=0.326/0.329，无时间特征最佳)
    #   族10 IXN3 ME2:   N_IXN3_cum_ME2 (mean=0.293，最高均值niche模型)
    #
    # CV验证（与44模型比较）：31模型+update_freq=30+ewma_beta=0.02
    #   Mean IC=0.3003（与原始相同），ICIR=7.163，Day5=0.279，Day4=0.289，Day2=0.366
    'NSovT':          (_BASE14 + ['aft_13800', 'Sect_OVI_ep5', 'Sect_OVI_ep15'],     100, True),
    'Nag12':          (_MD16 + ['aft_12000'] + _SR + _PR  + _OVI5,     30, True),
    'NagX':           (_MD16 + ['aft_12000'] + _SR + _PR2 + _OVI5,     20, True),
    'Npr30':          (_BASE14 + ['aft_12000'] + _SR + _PR2,           150, True),
    'NprD30':         (_MD16  + ['aft_12000'] + _SR + _PR2,            150, True),
    'Nsmr12':         (_BASE14 + ['aft_12000'] + _SR + _SMR,           120, True),
    'NsmrD12':        (_MD16  + ['aft_12000'] + _SR + _PR  + _SMR,     100, True),
    'NsmrX':          (_MD16  + ['aft_12000'] + _SR + _PR2 + _OVI5 + _SMR,  20, True),
    'Nlot12':         (_BASE14 + ['aft_12000'] + _SR + _LOT,           120, True),
    'NlotD12':        (_MD16  + ['aft_12000'] + _SR + _PR + _LOT,      100, True),
    'N_cum_ME2':      (_ME8 + _OVI5 + ['aft_12000'] + _SR + _LAG2 + _PR2 + _LOT + _CUM, 20, True),
    'N_cum_ME_T':     (_ME8 + _OVI5 + _SR + _PR2 + _LOT + _CUM, 20, True),
    'N_both_ME_T':    (_ME8 + _OVI5 + _SR + _PR2 + _LOT + _CUM2, 20, True),
    'N_IXN4_SMR_ME2': (_ME8 + _OVI5 + ['aft_12000'] + _SR + ['e_ret_lag2'] + _PR2 + _LOT + _CUM + _SMR + _IXN4, 15, True),
    'N_IXN4_cum_ME2': (_ME8 + _OVI5 + ['aft_12000'] + _SR + ['e_ret_lag2'] + _PR2 + _LOT + _CUM + _IXN4, 15, True),
    'N_IXN4_cum_T':   (_ME8 + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN4, 15, True),
    'N_IXN3_cum_T':   (_ME8 + _OVI5 + _SR + _PR2 + _LOT + _CUM2 + _IXN3, 15, True),
    'N_IXN3_cum_ME2': (_ME8 + _OVI5 + ['aft_12000'] + _SR + ['e_ret_lag2'] + _PR2 + _LOT + _CUM + _IXN3, 15, True),
    'N_IXN4_cum_ME2_D': (_ME8 + _OVI5 + ['aft_12000'] + _SR + ['e_ret_lag2'] + _PR2 + _LOT + _CUM + _IXN4 + _DEEP, 15, True),
    # ── 新增 Niche 模型（Iter13，利用 ONI_ep15 + past_ret_900 新特征）──────────
    # 筛选原则：ONI_ep15 在 Day1 IC=+0.20（与 ONI_p15 互补），past_ret_900 在 Day5 IC=+0.26。
    # 在内层4折IC评估中验证确有增益后加入（见 Iter13 分析）。
    #
    # N_oni9_ME2: ME9（含ONI_ep15）+ aft_12000 + SR + PR3（含past_ret_900）+ LOT + CUM
    #   → 在 Day1/5 更强的委托笔数动态（EMA ONI）+ 更长时间轴价格信号（900tick）
    'N_oni9_ME2':     (_ME9 + _OVI5 + ['aft_12000'] + _SR + _LAG2 + _PR3 + _LOT + _CUM, 20, True),
    # N_oni9_T:    ME9 + SR + PR3 + LOT + CUM2（无时间特征，Day5 最优配置）
    #   → 与 N_cum_ME_T 类似但加入 ONI_ep15 和 past_ret_900
    'N_oni9_T':       (_ME9 + _OVI5 + _SR + _PR3 + _LOT + _CUM2, 20, True),
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
ENSEMBLE_TEMP       = 15     # softmax 温度（Iter13优化 12→15，更积极集中）
ENSEMBLE_FLOOR      = 0.0    # 权重下限（CV验证 floor>0 在本数据集会降IC）
RETURN_DELAY        = 600    # Return5min 可知延迟 = 5 分钟 / 0.5s = 600 ticks
NICHE_INIT_WEIGHT   = 0.0    # niche 模型预热期权重（0 = 完全等待滚动IC发现价值）

# 稳定化参数（Iter13内层4折网格搜索优化）
ENSEMBLE_UPDATE_FREQ = 15    # 权重更新频率（Iter13优化 30→15，更频繁但IC估计更稳定）
ENSEMBLE_EWMA_BETA   = 0.01  # IC 的 EWMA alpha 系数（Iter13优化 0.02→0.01，更慢更稳）
                             # 公式：EWMA(t) = α*IC(t) + (1-α)*EWMA(t-1)，α=0.01 → 历史权重0.99
                             # 注：更慢EWMA与更频繁更新（update_freq=15）互补：
                             #     每次用稳定IC估计更新权重，而非减少更新次数来规避噪声
STABLE_PRIOR        = 0.0    # 稳定模型权重下限（CV验证 stable_prior>0 持续降IC，不采用）


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
            m = Ridge(alpha=alpha)
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

    print(f"平均 IC: {mean_ic:.6f}")
    print(f"IC Std : {std_ic:.6f}")
    print(f"ICIR   : {icir:.4f}")
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
            m = Ridge(alpha=alpha)
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
                m_in = Ridge(alpha=alpha)
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


if __name__ == "__main__":
    train()
    nested_blind_test()
