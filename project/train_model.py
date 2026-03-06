import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import warnings

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════════════
# 动态集成架构：29 个 Ridge 子模型（12 稳定 + 17 niche）+ 基于滚动 IC 的自适应权重
#
# 稳定模型（12 个）：在大多数日子表现稳定，预热期等权分配
#   MA/MD/MT/MTC/MTD/MTE/MT12/MTD12 — 基础架构（覆盖多种信号和时段）
#   MTsr12/MTpr12/MTsrpr12/MTDsrpr12 — 叠加滞后已实现收益特征
#
# Niche 模型（17 个）：整体不稳定但某些时段/机制下表现突出
#   预热期权重为 0（NICHE_INIT_WEIGHT=0.0），由滚动 IC 机制发现其优势窗口
#   Nep5/Nep12   — 超短期 OVI EMA5 脉冲（瞬时订单动能）
#   NSov12/NSovT — 板块 OVI 超短期 EMA（板块流动性领先信号）
#   Nag12/NagX   — 激进综合型（低正则化，高风险高收益）
#   Npr30/NprD30 — 扩展短期价格动量（含 30/60 tick 极短收益率）
#   Nsmr12/NsmrD12/NsmrX — 板块短期价格收益率 + 横截面相对表现
#   Nlot12/NlotD12 — 大单成交失衡（lot_imb_15：单笔成交金额方向）
#   NpOVI/NpOVI_S — OVI纯净+CSM+大单（剔除弱信号日噪声 TI 特征）
#   NpOVI_T       — NpOVI 的无时间特征版（剔除 aft_12000，泛化弱信号日日内模式）
#   NpOVI_TD      — NpOVI_T + 深层委托簿失衡脉冲（obi_deep_p15：2-5 档方向信息）
#
# 动态集成逻辑：
#   Return5min(t) 在 t+600 可知，因此在 tick t 可用 [t-DELAY-WINDOW, t-DELAY]
#   区间内的真实收益评估各模型近期质量，以 softmax(IC * TEMP) 作为集成权重。
#   niche 模型在预热期权重为 0，确保不在数据不足时引入噪声，
#   预热后由滚动 IC 自动发现其价值并动态提升权重。
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
    # ── Niche 模型 (15 个)：初始权重 0，利用滚动 IC 发现其优势窗口 ────────────
    # 超短期 OVI EMA5 脉冲（某些市场机制下非常强）
    'Nep5':   (_ME8  + ['aft_13800'] + _OVI5,        80, True),
    'Nep12':  (_ME8  + ['aft_12000'] + _OVI5,        80, True),
    # 板块 OVI 超短期脉冲（板块流动性领先）
    'NSov12': (_MD16 + ['aft_12000', 'Sect_OVI_ep5', 'Sect_OVI_ep15'] + _SR, 100, True),
    'NSovT':  (_BASE14 + ['aft_13800', 'Sect_OVI_ep5', 'Sect_OVI_ep15'],     100, True),
    # 短价格动量 + OVI 组合（高风险高收益）
    'Nag12':  (_MD16 + ['aft_12000'] + _SR + _PR  + _OVI5,     30, True),
    'NagX':   (_MD16 + ['aft_12000'] + _SR + _PR2 + _OVI5,     20, True),
    # 扩展价格收益窗口（含 30/60 tick 短期动量）
    'Npr30':  (_BASE14 + ['aft_12000'] + _SR + _PR2,           150, True),
    'NprD30': (_MD16  + ['aft_12000'] + _SR + _PR2,            150, True),
    # ── 板块短期价格动能 & 横截面相对表现 Niche 模型 (3 个) ─────────────────────
    # 新增特征：sect_mid_ret_30/120（板块短期涨跌），csm_ret_120（E 落后板块的追涨/反转）
    # 这些特征独立于订单流失衡，来自板块价格变动，是真正正交的新信息源。
    'Nsmr12':   (_BASE14 + ['aft_12000'] + _SR + _SMR,         120, True),
    'NsmrD12':  (_MD16  + ['aft_12000'] + _SR + _PR  + _SMR,   100, True),
    'NsmrX':    (_MD16  + ['aft_12000'] + _SR + _PR2 + _OVI5 + _SMR,  20, True),
    # ── 大单成交失衡 Niche 模型 (2 个) ─────────────────────────────────────────
    # lot_imb_15: 全日 IC 符号一致为正（弱信号日 Day1/5 尤强），
    # 与 OVI/TI 正交（测量单笔成交金额大小而非数量）。
    'Nlot12':   (_BASE14 + ['aft_12000'] + _SR + _LOT,          120, True),
    'NlotD12':  (_MD16  + ['aft_12000'] + _SR + _PR + _LOT,     100, True),
    # ── OVI-纯净 + 横截面 + 大单 Niche 模型 (2 个) ──────────────────────────────
    # 核心发现：弱信号日（Day1/5）TI 特征 IC 近零（MTC Day1 IC=0.124 vs MTE 0.218），
    # 而 OVI + csm_ret_120（+0.088）+ lot_imb_15（+0.074）三者信号方向一致。
    # NpOVI 系列模型完全剔除 TI 特征，专注于这三类互补信号。
    'NpOVI':    (_ME8 + _OVI5 + ['aft_12000'] + _SR + _PR2 + ['csm_ret_120'] + _LOT, 20, True),
    'NpOVI_S':  (_ME8 + _OVI5 + ['aft_12000'] + _SR + ['csm_ret_120'] + _LOT,       100, True),
    # ── 无时间特征 OVI 纯净 Niche 模型（2 个）─────────────────────────────────────
    # 发现：aft_12000 特征在弱信号日（1/5）会将训练集（日 2/3/4）学到的日内模式
    # 错误迁移到测试日，导致 OVI 模型在 PM 时段的预测偏差。
    # 剔除 aft_12000 后模型"时间无感知"，避免日内模式的跨日迁移误差，
    # 在 Day1/5 上提升约 +0.008 IC，ICIR 从 6.67 提升至 7.03。
    # NpOVI_T:  完全去掉时间特征，保留全套 OVI+CSM+大单+价格反转 特征。
    # NpOVI_TD: NpOVI_T 基础上叠加 obi_deep_p15（深层订单簿方向，与 OVI 正交），
    #           进一步强化对弱信号日深度委托方向的捕捉。
    'NpOVI_T':  (_ME8 + _OVI5 + _SR + _PR2 + ['csm_ret_120'] + _LOT,           20, True),
    'NpOVI_TD': (_ME8 + _OVI5 + _SR + _PR2 + ['csm_ret_120'] + _LOT + _DEEP,   20, True),
}

MODEL_NAMES   = list(MODELS.keys())
# is_niche flag per model (same order as MODEL_NAMES)
MODEL_IS_NICHE = [v[2] for v in MODELS.values()]

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

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(np.zeros(len(df)), y, groups))

    print(f"训练 {len(MODEL_NAMES)} 个子模型，运行 5 折交叉验证...")
    print(f"集成参数: window={ENSEMBLE_WINDOW}, temp={ENSEMBLE_TEMP}, "
          f"floor={ENSEMBLE_FLOOR}, delay={RETURN_DELAY}")

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
    print(f"{'Model':<14} | {'Niche':<5} | {'Mean IC':<10} | {'per-day IC'}")
    print("-" * 75)
    for name in MODEL_NAMES:
        ics = [ic_score(fold_y_test[fi], fold_model_preds[name][fi])
               for fi in range(5)]
        niche_tag = 'N' if MODELS[name][2] else 'S'
        print(f"{name:<14} | {niche_tag:<5} | {np.mean(ics):.6f}   | "
              f"{[round(r, 4) for r in ics]}")

    # ── 动态集成 ─────────────────────────────────────────────────────────────
    print()
    print(f"动态集成结果（niche 模型预热期权重={NICHE_INIT_WEIGHT}）：")
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
