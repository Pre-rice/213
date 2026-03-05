import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import warnings

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════════════
# 动态集成架构：四个多样化 Ridge 子模型 + 基于滚动 IC 的自适应权重
#
# 设计原则：每个子模型聚焦不同的市场驱动因子，在不同"市场机制日"各有优势：
#
#   MA (balanced)   — 综合型：成交流 + 委托流，整体最强，大多数日子表现稳定
#   MC (trend)      — 趋势型：纯 TI 长期信号，趋势延续日（如 Day 1 午后）最优
#   MD (extended)   — 扩展综合型：MA + Sect_ONI_p30 + OVI_p30，特征最完整
#   ME (pure OVI)   — 极简委托型：仅 OVI/ONI/TotalBidVol，与 MA/MD 互补
#
# 动态集成逻辑（模拟在线预测中的自适应调权）：
#   由于 Return5min(t) = (MidPrice(t+600) - MidPrice(t)) / MidPrice(t)，
#   在 tick t+600 时可计算出 tick t 的真实收益率。
#   因此在在线预测的 tick t 处，可以用 [t-DELAY-WINDOW, t-DELAY] 区间内
#   已知的真实收益率评估各模型的近期预测质量（rolling IC），并以
#   softmax(rolling_IC * TEMP) 作为当前 tick 的集成权重。
# ════════════════════════════════════════════════════════════════════════════════

# 子模型定义：(feature_list, ridge_alpha)
MODELS = {
    'MA': (
        ['TotalBidVol', 'TradeImb_600', 'TradeImb_diff',
         'Sect_OBI1', 'E_TI_rel_600',
         'TradeImb_p60', 'TradeImb_p40', 'Sect_TI_p40',
         'OVI_p15', 'OVI_p60', 'Sect_OVI_p20',
         'TradeImb_ep60', 'OVI_ep15', 'TNI_ep15'],
        150,
    ),
    'MC': (
        ['TradeImb_600', 'TradeImb_diff',
         'TradeImb_p60', 'TradeImb_p40', 'TradeImb_ep60',
         'E_TI_rel_600', 'Sect_TI_p40',
         'TradeImb_p30', 'TradeImb_p15'],
        200,
    ),
    'MD': (
        ['TotalBidVol', 'TradeImb_600', 'TradeImb_diff',
         'Sect_OBI1', 'E_TI_rel_600',
         'TradeImb_p60', 'TradeImb_p40', 'Sect_TI_p40',
         'OVI_p15', 'OVI_p60', 'Sect_OVI_p20',
         'TradeImb_ep60', 'OVI_ep15', 'TNI_ep15',
         'Sect_ONI_p30', 'OVI_p30'],
        150,
    ),
    'ME': (
        ['TotalBidVol', 'OVI_p15', 'OVI_p60', 'OVI_ep15',
         'ONI_p15', 'Sect_OBI1', 'Sect_OVI_p20', 'TNI_ep15'],
        80,
    ),
}

MODEL_NAMES = list(MODELS.keys())

# 动态集成超参数（通过5折交叉验证搜索确定）
ENSEMBLE_WINDOW = 700   # 滚动 IC 窗口 (tick 数，约 5.8 分钟)
ENSEMBLE_TEMP   = 5     # softmax 温度（越大越偏向最优模型）
ENSEMBLE_FLOOR  = 0.05  # 每个模型的最小权重下限（防止极端权重）
RETURN_DELAY    = 600   # Return5min 可知延迟 = 5 分钟 / 0.5s = 600 ticks


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
                     window=ENSEMBLE_WINDOW,
                     temp=ENSEMBLE_TEMP,
                     floor=ENSEMBLE_FLOOR,
                     delay=RETURN_DELAY):
    """
    基于滚动 IC 的动态权重集成。

    核心流程：
      1. 对每个模型计算滚动 IC 序列（用测试集真实 y 模拟在线可知部分）
      2. 将 IC 序列向后位移 delay 个 tick（反映 Return5min 的可知延迟）
      3. 用 softmax(IC * temp) 作为当前 tick 的集成权重（施加 floor 防极端）
      4. 前 delay + window/4 tick 历史不足，改用等权

    参数说明：
      y_test      - 测试集真实收益（在 CV 中完整已知；在线预测中延迟 delay 可知）
      model_preds - 各子模型预测列表
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

    # 预热期使用等权
    warmup = delay + window // 4
    weights[:warmup] = 1.0 / n_models

    # 加权求和
    preds_matrix = np.stack(model_preds, axis=1)   # (n, n_models)
    return (weights * preds_matrix).sum(axis=1)


def train():
    df = pd.read_csv("train.csv")
    y      = df['Return5min'].values
    groups = df['Day'].values

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(np.zeros(len(df)), y, groups))

    print("训练 4 个子模型，运行 5 折交叉验证...")
    print(f"子模型: {[f'{n}({len(MODELS[n][0])}feat,α={MODELS[n][1]})' for n in MODEL_NAMES]}")
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

        for name, (feats, alpha) in MODELS.items():
            X = df[feats].values
            m = Ridge(alpha=alpha)
            m.fit(X[train_idx], y[train_idx])
            fold_model_preds[name].append(m.predict(X[test_idx]))

    # ── 各子模型独立 IC ───────────────────────────────────────────────────────
    print()
    print("子模型独立 IC（参考）：")
    print(f"{'Model':<6} | {'Mean IC':<10} | {'per-day IC'}")
    print("-" * 65)
    for name in MODEL_NAMES:
        ics = [ic_score(fold_y_test[fi], fold_model_preds[name][fi])
               for fi in range(5)]
        print(f"{name:<6} | {np.mean(ics):.6f}   | "
              f"{[round(r, 4) for r in ics]}")

    # ── 动态集成 ─────────────────────────────────────────────────────────────
    print()
    print(f"动态集成结果：")
    print(f"{'Fold':<6} | {'Test Day':<8} | {'IC':<10}")
    print("-" * 32)

    results = []
    for fi in range(5):
        model_preds = [fold_model_preds[name][fi] for name in MODEL_NAMES]
        final_preds = dynamic_ensemble(fold_y_test[fi], model_preds)
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
