import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import warnings

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════════════
# 动态集成架构：31 个 Ridge 子模型（12 稳定 + 8 niche + 11 时段/OBI/VWAP/交叉专用）
#            + 基于滚动 IC 的自适应权重
#
# 稳定模型（12 个）：在大多数日子表现稳定，预热期等权分配
#   MA/MD/MT/MTC/MTD/MTE/MT12/MTD12 — 基础架构（覆盖多种信号和时段）
#   MTsr12/MTpr12/MTsrpr12/MTDsrpr12 — 叠加滞后已实现收益特征
#
# Niche 模型（8 个）：整体不稳定但某些时段/机制下表现突出
#   预热期权重为 0（NICHE_INIT_WEIGHT=0.0），由滚动 IC 机制发现其优势窗口
#   Nep5/Nep12   — 超短期 OVI EMA5 脉冲（瞬时订单动能）
#   NSov12/NSovT — 板块 OVI 超短期 EMA（板块流动性领先信号）
#   Nag12/NagX   — 激进综合型（低正则化，高风险高收益）
#   Npr30/NprD30 — 扩展短期价格动量（含 30/60 tick 极短收益率）
#
# 上午时段专用 Niche 模型（3 个）：仅用上午数据训练，自然聚焦于上午规律
#   AM_D/AM_E/AM_F — 训练集仅含 tick_idx <= 13800 的上午行情
#
# OBI 深度失衡 Niche 模型（2 个）：利用新增 OBI 委托深度失衡特征
#   MOBI/MOBI_D — 全时段 OBI 特征模型
#
# VWAP 均值回复 Niche 模型（3 个）：利用日内 VWAP 偏差信号
#   NVol12/NVol_D/NVol_DX — vwap_dev 提供与动量信号互补的均值回复视角
#
# 动态集成逻辑：
#   Return5min(t) 在 t+600 可知，因此在 tick t 可用 [t-DELAY-WINDOW, t-DELAY]
#   区间内的真实收益评估各模型近期质量，以 softmax(IC * TEMP) 作为集成权重。
#   niche/时段专用模型在预热期权重为 0，确保不在数据不足时引入噪声，
#   预热后由滚动 IC 自动发现其价值并动态提升权重。
#
# 模型元组格式：(feature_list, ridge_alpha, is_niche, segment)
#   segment: 'ALL'=全时段训练, 'AM'=仅上午训练
# ════════════════════════════════════════════════════════════════════════════════

# 上午/下午时段分割阈值（与 data_processor.py 保持一致）
_AM_SPLIT = 13800

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

# 新增：委托深度失衡特征（OBI）+ TradeImb EMA15 脉冲
_OBI_NEW = ['OBI1_p15', 'OBI1_ep15', 'OBI_total_p15', 'TradeImb_ep15']

# 新增：日内 VWAP 偏差（均值回复信号）
_VWAP = ['vwap_dev']

# 新增：信号一致性交叉特征（OVI×TI 乘积项）
_INTER = ['ovi_ti_short', 'ovi_ti_medium', 'sect_ovi_ti', 'obi_ovi_s']

# 子模型定义：(feature_list, ridge_alpha, is_niche, segment)
# is_niche=True 的模型在集成预热期权重为 0，由滚动 IC 机制发现其价值
# segment: 'ALL'=全量训练, 'AM'=仅用 tick_idx<=_AM_SPLIT 训练, 'PM'=仅用 tick_idx>_AM_SPLIT 训练
MODELS = {
    # ── 稳定基础模型 (12 个)────────────────────────────────────────────────────
    'MA':        (_BASE14,                              150, False, 'ALL'),
    'MD':        (_MD16,                                150, False, 'ALL'),
    'MT':        (_BASE14 + ['aft_13800'],              150, False, 'ALL'),
    'MTC':       (_MC9   + ['aft_13800'],               200, False, 'ALL'),
    'MTD':       (_MD16  + ['aft_13800'],               150, False, 'ALL'),
    'MTE':       (_ME8   + ['aft_13800'],                80, False, 'ALL'),
    'MT12':      (_BASE14 + ['aft_12000'],              150, False, 'ALL'),
    'MTD12':     (_MD16  + ['aft_12000'],               150, False, 'ALL'),
    'MTsr12':    (_BASE14 + ['aft_12000'] + _SR,        150, False, 'ALL'),
    'MTpr12':    (_BASE14 + ['aft_12000'] + _PR,        150, False, 'ALL'),
    'MTsrpr12':  (_BASE14 + ['aft_12000'] + _SR + _PR,  150, False, 'ALL'),
    'MTDsrpr12': (_MD16  + ['aft_12000'] + _SR + _PR,   150, False, 'ALL'),
    # ── Niche 模型 (8 个)：初始权重 0，利用滚动 IC 发现其优势窗口 ────────────
    'Nep5':   (_ME8  + ['aft_13800'] + _OVI5,                        80, True, 'ALL'),
    'Nep12':  (_ME8  + ['aft_12000'] + _OVI5,                        80, True, 'ALL'),
    'NSov12': (_MD16 + ['aft_12000', 'Sect_OVI_ep5', 'Sect_OVI_ep15'] + _SR, 100, True, 'ALL'),
    'NSovT':  (_BASE14 + ['aft_13800', 'Sect_OVI_ep5', 'Sect_OVI_ep15'],     100, True, 'ALL'),
    'Nag12':  (_MD16 + ['aft_12000'] + _SR + _PR  + _OVI5,           30, True, 'ALL'),
    'NagX':   (_MD16 + ['aft_12000'] + _SR + _PR2 + _OVI5,           20, True, 'ALL'),
    'Npr30':  (_BASE14 + ['aft_12000'] + _SR + _PR2,                 150, True, 'ALL'),
    'NprD30': (_MD16  + ['aft_12000'] + _SR + _PR2,                  150, True, 'ALL'),
    # ── 上午时段专用 Niche 模型 (3 个)：仅用 AM 数据训练 ───────────────────────
    # 上午行情中 aft_12000 仍有区分度（覆盖早盘尾段），但 aft_13800 几乎全为 0
    'AM_D':  (_MD16  + ['aft_12000'] + _SR,                120, True, 'AM'),
    'AM_E':  (_ME8   + ['aft_12000'] + _OVI5,               80, True, 'AM'),
    'AM_F':  (_MD16  + ['aft_12000'] + _SR + _PR + _OVI5,  100, True, 'AM'),
    # ── OBI 深度失衡特征模型 (2 个)：使用新增 OBI 特征 ──────────────────────────
    'MOBI':   (_BASE14 + _OBI_NEW,                          120, True, 'ALL'),
    'MOBI_D': (_MD16   + _SR + _OBI_NEW,                    120, True, 'ALL'),
    # ── VWAP 均值回复模型 (3 个)：利用日内 VWAP 偏差信号 ──────────────────────────
    # vwap_dev 捕捉 OVI/TI 信号之外的均值回复动能，与动量信号形成互补
    'NVol12':   (_BASE14 + ['aft_12000'] + _SR + _VWAP,           150, True, 'ALL'),
    'NVol_D':   (_MD16  + ['aft_12000'] + _SR + _PR + _VWAP,      120, True, 'ALL'),
    'NVol_DX':  (_MD16  + ['aft_12000'] + _SR + _PR2 + _OVI5 + _VWAP, 80, True, 'ALL'),
    # ── 信号一致性交叉特征模型 (3 个)：使用 OVI×TI 乘积项 ────────────────────────
    # 当 OVI 与 TI 方向一致时，信号更可信，乘积项捕捉此非线性增强效应
    'NInter12': (_BASE14 + ['aft_12000'] + _SR + _INTER,          150, True, 'ALL'),
    'NInter_D': (_MD16  + ['aft_12000'] + _SR + _PR + _INTER,     120, True, 'ALL'),
    'NInter_DX':(_MD16  + ['aft_12000'] + _SR + _PR2 + _OVI5 + _INTER + _VWAP, 80, True, 'ALL'),
}

MODEL_NAMES    = list(MODELS.keys())
MODEL_IS_NICHE = [v[2] for v in MODELS.values()]
MODEL_SEGMENTS = [v[3] for v in MODELS.values()]

# 动态集成超参数（通过5折交叉验证搜索确定）
ENSEMBLE_WINDOW     = 900   # 滚动 IC 窗口 (tick 数，约 7.5 分钟)
ENSEMBLE_TEMP       = 12    # softmax 温度（越大越偏向最优模型）
ENSEMBLE_FLOOR      = 0.0   # 权重下限（0 = 不限制，允许最优模型独占权重）
RETURN_DELAY        = 600   # Return5min 可知延迟 = 5 分钟 / 0.5s = 600 ticks
NICHE_INIT_WEIGHT   = 0.0   # niche 模型预热期权重（0 = 完全等待滚动IC发现价值）


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
                     niche_init=NICHE_INIT_WEIGHT):
    """
    基于滚动 IC 的动态权重集成，支持 niche 模型零初始权重策略。

    核心流程：
      1. 对每个模型计算滚动 IC 序列（用测试集真实 y 模拟在线可知部分）
      2. 将 IC 序列向后位移 delay 个 tick（反映 Return5min 的可知延迟）
      3. 用 softmax(IC * temp) 作为当前 tick 的集成权重
      4. 预热期（delay + window/4 tick）：
         - 稳定模型（is_niche=False）等权分配
         - niche 模型（is_niche=True）权重为 niche_init（默认 0.0）
         预热期后，所有模型由滚动 IC 动态决定权重

    参数说明：
      y_test      - 测试集真实收益
      model_preds - 各子模型预测列表
      is_niche    - bool 列表，与 model_preds 等长，标记是否为 niche 模型
                    None = 全部视为稳定模型（等权预热）
      niche_init  - niche 模型在预热期的权重（0.0 = 完全零权重）
      window, temp, floor, delay - 见文件顶部常量定义
    """
    n = len(y_test)
    n_models = len(model_preds)

    # 各模型的滚动 IC 序列：shape (n, n_models)
    rolling_ics = np.stack(
        [_rolling_ic_series(y_test, mp, window) for mp in model_preds],
        axis=1,
    )

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

    # 加权求和
    preds_matrix = np.stack(model_preds, axis=1)   # (n, n_models)
    return (weights * preds_matrix).sum(axis=1)


def train():
    df = pd.read_csv("train.csv")
    y      = df['Return5min'].values
    groups = df['Day'].values

    # 当日 tick 索引（0 起），用于时段专用模型的训练数据切分
    if 'tick_idx' in df.columns:
        tick_idx_arr = df['tick_idx'].values
    else:
        # 兼容旧版 train.csv（无 tick_idx 列）：按 Day 分组内行序重建
        # 假设同一 Day 内的行按时间升序排列（data_processor.py 保证此顺序）
        tick_idx_arr = df.groupby('Day').cumcount().values

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(np.zeros(len(df)), y, groups))

    print(f"训练 {len(MODEL_NAMES)} 个子模型，运行 5 折交叉验证...")
    print(f"集成参数: window={ENSEMBLE_WINDOW}, temp={ENSEMBLE_TEMP}, "
          f"floor={ENSEMBLE_FLOOR}, delay={RETURN_DELAY}")
    print(f"时段专用模型分割阈值: AM tick_idx <= {_AM_SPLIT} / PM tick_idx > {_AM_SPLIT}")

    fold_model_preds = {name: [] for name in MODEL_NAMES}
    fold_y_test = []
    fold_test_days = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        test_day = int(np.unique(groups[test_idx])[0])
        fold_test_days.append(test_day)

        y_test = y[test_idx]
        fold_y_test.append(y_test)

        # 时段掩码：在训练集内再切分 AM / PM
        am_mask = tick_idx_arr[train_idx] <= _AM_SPLIT
        pm_mask = tick_idx_arr[train_idx] >  _AM_SPLIT
        train_am_idx = train_idx[am_mask]
        train_pm_idx = train_idx[pm_mask]

        for name, (feats, alpha, _, segment) in MODELS.items():
            X = df[feats].values
            m = Ridge(alpha=alpha)
            if segment == 'AM':
                m.fit(X[train_am_idx], y[train_am_idx])
            elif segment == 'PM':
                m.fit(X[train_pm_idx], y[train_pm_idx])
            else:
                m.fit(X[train_idx], y[train_idx])
            fold_model_preds[name].append(m.predict(X[test_idx]))

    # ── 各子模型独立 IC ───────────────────────────────────────────────────────
    n_folds = len(splits)
    print()
    print("子模型独立 IC（参考）：")
    print(f"{'Model':<14} | {'Tag':<6} | {'Mean IC':<10} | {'per-day IC'}")
    print("-" * 80)
    for name in MODEL_NAMES:
        ics = [ic_score(fold_y_test[fi], fold_model_preds[name][fi])
               for fi in range(n_folds)]
        seg  = MODELS[name][3]
        niche_tag = ('N-' + seg) if MODELS[name][2] else 'S'
        print(f"{name:<14} | {niche_tag:<6} | {np.mean(ics):.6f}   | "
              f"{[round(r, 4) for r in ics]}")

    # ── 动态集成（5 折交叉验证）──────────────────────────────────────────────
    print()
    print(f"════ 5 折交叉验证：动态集成（niche/时段 模型预热期权重={NICHE_INIT_WEIGHT}）════")
    print(f"{'Fold':<6} | {'Test Day':<8} | {'IC':<12} | {'备注'}")
    print("-" * 50)

    results = []
    for fi in range(n_folds):
        model_preds = [fold_model_preds[name][fi] for name in MODEL_NAMES]
        final_preds = dynamic_ensemble(fold_y_test[fi], model_preds,
                                       is_niche=MODEL_IS_NICHE)
        ic = ic_score(fold_y_test[fi], final_preds)
        print(f"Fold {fi+1}  | Day {fold_test_days[fi]:<5}  | {ic:.6f}     |")
        results.append(ic)

    print("-" * 50)
    mean_ic = np.mean(results)
    std_ic  = np.std(results)
    icir    = mean_ic / std_ic if std_ic > 1e-9 else 0.0

    print(f"平均 IC : {mean_ic:.6f}")
    print(f"IC Std  : {std_ic:.6f}")
    print(f"ICIR    : {icir:.4f}")
    return results


if __name__ == "__main__":
    train()
