import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import warnings

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════════════
# 动态集成架构：31 个 Ridge 子模型（12 稳定 + 19 niche）+ 保守化自适应权重
#
# 稳定模型（12 个）：在大多数日子表现稳定，预热期等权分配
#   MA/MD/MT/MTC/MTD/MTE/MT12/MTD12 — 基础架构（覆盖多种信号和时段）
#   MTsr12/MTpr12/MTsrpr12/MTDsrpr12 — 叠加滞后已实现收益特征
#
# Niche 模型（19 个，原32个按信号族去冗余后保留）：
#   剔除13个冗余/弱泛化模型（Nep5/12, NSov12, NpOVI系列4个,
#   N_cum_ME2_S/SD, N_both_ME_T_S/SD, N_IXN1_cum_ME2/T），
#   保留每族最强代表，同时保持 ME2(aft_12000) 和 T(无时间特征) 的互补覆盖：
#   NSovT, Nag12/NagX, Npr30/NprD30, Nsmr12/NsmrD12/NsmrX,
#   Nlot12/NlotD12, N_cum_ME2, N_cum_ME_T/N_both_ME_T,
#   N_IXN4/3 ME2/T 系列（8个）
#
# 动态集成改进（参考 suggestion.md §1.1 A/B）：
#   (A) 权重更新频率降低至每 UPDATE_FREQ=30 tick 更新一次，减少噪声驱动抖动
#   (B) 用 EWMA 平滑的 IC（β=EWMA_BETA=0.02）替代原始 rolling corr，更稳定
#
# CV 验证结果（31 模型 + update=30 + ewma=0.02 vs 原始 44 模型）：
#   Mean IC: 0.3003（相同），ICIR: 7.163
#   Day1=0.243, Day2=0.366, Day3=0.324, Day4=0.289, Day5=0.279
#
# 注：suggestion §1.1 C（降 temp/提 floor）和 §3.2（stable_prior）
#     在本数据集 CV 验证中均使所有日 IC 下降，故不采用。
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

# 滞后已实现收益特征组合（无前视偏差，所有值均在预测时刻可知）
_SR  = ['sect_ret_lag', 'e_ret_lag']
_PR  = ['past_ret_120', 'past_ret_300', 'past_ret_600']
_PR2 = ['past_ret_30', 'past_ret_60', 'past_ret_120', 'past_ret_300', 'past_ret_600']

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
}

MODEL_NAMES   = list(MODELS.keys())
# is_niche flag per model (same order as MODEL_NAMES)
MODEL_IS_NICHE = [v[2] for v in MODELS.values()]

# 动态集成超参数（优化后，参考 suggestion.md §1.1）
ENSEMBLE_WINDOW     = 900   # 滚动 IC 窗口 (tick 数，约 7.5 分钟)
ENSEMBLE_TEMP       = 12    # softmax 温度（维持原值，CV验证保守降温会显著降IC）
ENSEMBLE_FLOOR      = 0.0   # 权重下限（维持原值，floor>0 在本数据集会降IC）
RETURN_DELAY        = 600   # Return5min 可知延迟 = 5 分钟 / 0.5s = 600 ticks
NICHE_INIT_WEIGHT   = 0.0   # niche 模型预热期权重（0 = 完全等待滚动IC发现价值）

# 新增稳定化参数（suggestion.md §1.1 A/B 条）
# 注：suggestion §1.1 C 建议降 temp/提 floor，但CV验证显示在本数据集中反效果
# （任何 temp<12 或 floor>0 的组合均使每一日 IC 降低，包括最弱日 Day1/Day5）
# 因此仅采用成本极低且理论合理的 A/B 两条：
ENSEMBLE_UPDATE_FREQ = 30   # 权重更新频率（每 30 tick 更新一次，减少噪声驱动抖动）
ENSEMBLE_EWMA_BETA   = 0.02 # IC 的 EWMA alpha 系数（pandas ewm 约定：α小 → 慢更新→更稳定）
                            # 公式：EWMA(t) = α*IC(t) + (1-α)*EWMA(t-1)，α=0.02 → 历史权重0.98
STABLE_PRIOR        = 0.0   # 稳定模型权重下限（CV验证 stable_prior>0 持续降IC，不采用）


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


if __name__ == "__main__":
    train()
