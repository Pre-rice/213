"""
generate_figures.py
生成报告中所需的五幅 matplotlib 图表，输出至 figures/ 目录。

图表列表：
  fig1_ic_evolution.png   — 模型 IC 迭代演进曲线（Iter0~Iter20）
  fig2_final_cv.png       — Iter20 最终模型五折 CV 逐日 IC 条形图
  fig3_feature_categories.png — 特征体系分类饼图
  fig4_iter_comparison.png    — Iter1 vs Iter20 逐日 IC 对比条形图
  fig5_ensemble_gain.png      — 嵌套盲测动态集成 vs 稳定等权基准增益图
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 中文字体配置 ─────────────────────────────────────────────────────────────
# 优先使用系统可用的 CJK 字体；若均不可用则退出并给出提示。
def _setup_chinese_font():
    import matplotlib.font_manager as fm

    # 首先尝试直接添加已知路径（TTC / TTF / OTF 均支持）
    known_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    ]
    for path in known_paths:
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
            except Exception:
                pass

    cjk_candidates = [
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
        'Noto Sans CJK SC', 'Noto Serif CJK SC',
        'Source Han Sans CN', 'Source Han Serif CN',
        'SimHei', 'SimSun', 'Microsoft YaHei', 'PingFang SC',
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in cjk_candidates:
        if font in available:
            matplotlib.rcParams['font.sans-serif'] = [font] + matplotlib.rcParams['font.sans-serif']
            matplotlib.rcParams['axes.unicode_minus'] = False
            print(f"[fonts] Using: {font}")
            return font
    # 扫描系统字体目录（支持 ttf/otf/ttc）
    for path in fm.findSystemFonts(fontext='ttf') + fm.findSystemFonts(fontext='otf'):
        if any(k in path.lower() for k in ['noto', 'cjk', 'wenquan', 'simhei', 'simsun']):
            try:
                fm.fontManager.addfont(path)
                fname = fm.FontProperties(fname=path).get_name()
                matplotlib.rcParams['font.sans-serif'] = [fname] + matplotlib.rcParams['font.sans-serif']
                matplotlib.rcParams['axes.unicode_minus'] = False
                print(f"[fonts] Using path: {path}, name: {fname}")
                return fname
            except Exception:
                continue
    print("[fonts] Warning: No CJK font found. Labels may not render correctly.")
    matplotlib.rcParams['axes.unicode_minus'] = False
    return None

_setup_chinese_font()

OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 数据（来自 project/log.txt 记录的各迭代实验结果）
# ══════════════════════════════════════════════════════════════════════════════

# Iter0: 简单线性基线（2特征，非标准CV，平均约0.147）→ 0.147
# Iter1~Iter20: 严格五折 Leave-One-Day-Out CV 均值 IC
ITER_IC = {
    0:  0.147,   # Iter0 (非标准评估，5天简单平均)
    1:  0.1775,  # 6特征，单线性模型
    2:  0.2236,  # 14特征，Ridge baseline
    3:  0.2224,  # 4模型动态集成（初步引入集成，temp偏低）
    4:  0.2715,  # 8模型集成 + 日内时间特征（aft_12000/13800）
    5:  0.2744,  # 12模型 + 滞后收益特征
    6:  0.2766,  # 20模型 + OVI_ep5/Sect_OVI_ep5 + niche零预热
    7:  0.2769,  # 27模型 + 板块短期价格收益 + 大单失衡
    8:  0.2786,  # 29模型 + 深层委托簿失衡 (obi_deep_p15)
    9:  0.2918,  # 36模型 + 累计流量失衡 + 5个交互特征
    10: 0.3003,  # 44模型 + IXN交互族（突破IC=0.30）
    11: 0.3003,  # 31模型 + 集成稳定化（EWMA/update_freq精简模型）
    12: 0.3003,  # 嵌套盲测验证（代码无泄露确认）
    13: 0.3055,  # 33模型 + ONI_ep15 + past_ret_900 + 集成参数优化
    14: 0.3076,  # 36模型 + book_pres_pulse + ret_x_ti600
    15: 0.3101,  # 39模型 + ret_accel + ewma_beta 0.01→0.005
    16: 0.3110,  # 46模型 + vol_cond_ovi + idio_ovi + Huber模型
    17: 0.3114,  # 集成参数调优 temp=17, ewma=0.007（ICIR历史最高）
    18: 0.3120,  # 49模型 + e_sect_obi_gap + oni_accel
    19: 0.3132,  # 54模型 + spread_wt_ovi + ovi_sq + e_sect_lag_gap
    20: 0.3302,  # 28模型（反过拟合LOO剪枝 + N_eslg_ME2）
}

# 最终模型（Iter20）五折CV逐日 IC
ITER20_DAY_IC = {
    'Day1': 0.326,
    'Day2': 0.363,
    'Day3': 0.338,
    'Day4': 0.284,
    'Day5': 0.341,
}
ITER20_MEAN = 0.3302
ITER20_ICIR = 12.59

# Iter1 基线（6特征线性模型）逐日 IC
ITER1_DAY_IC = {
    'Day1': 0.189,
    'Day2': 0.281,
    'Day3': 0.144,
    'Day4': 0.124,
    'Day5': 0.151,
}

# 嵌套盲测（Iter19，五折外层集成IC vs 稳定等权IC）
# 使用 Iter19 数据，因其完整记录了稳定等权基准对比
NESTED_DAYS    = ['Day1', 'Day2', 'Day3', 'Day4', 'Day5']
NESTED_ENS_IC  = [0.270,  0.381,  0.330,  0.277,  0.308]
NESTED_STABLE  = [0.213,  0.350,  0.301,  0.245,  0.204]

# 59个特征的类别分布
FEATURE_CATEGORIES = {
    '成交量失衡\n(TI)脉冲': 8,     # TotalBidVol, TradeImb_600, TradeImb_diff, TradeImb_p15/30/40/60, TradeImb_ep60
    '委托量失衡\n(OVI)脉冲': 5,    # OVI_p15/30/60, OVI_ep15, OVI_ep5
    '委托/成交\n笔数失衡': 4,       # ONI_p15/30, ONI_ep15, TNI_ep15
    '板块联动\n因子': 7,            # Sect_OBI1, E_TI_rel_600, Sect_TI_p40, Sect_OVI_p20, Sect_ONI_p30, Sect_OVI_ep5, Sect_OVI_ep15
    '日内时间\n特征': 2,            # aft_13800, aft_12000
    '滞后价格\n收益特征': 12,       # sect_ret_lag, e_ret_lag, past_ret_30/60/120/300/600/900, sect_mid_ret_30/120, csm_ret_120, e_ret_lag2
    '委托簿深度\n与流量': 7,        # e_spread_pulse, lot_imb_15, sect_lot_imb_15, obi_deep_p15, cum_flow_imb, sect_cum_flow_imb, book_pres_pulse
    '交互与\n高级信号': 14,         # ovi_x_abs_ret, tbv_x_ovi, srl_x_ovi, ret_x_cum, oni_x_ovi, ret_x_ti600, ret_accel, vol_cond_ovi, idio_ovi, e_sect_obi_gap, oni_accel, spread_wt_ovi, ovi_sq, e_sect_lag_gap
}
assert sum(FEATURE_CATEGORIES.values()) == 59, f"Feature count mismatch: {sum(FEATURE_CATEGORIES.values())}"

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: IC 迭代演进曲线
# ══════════════════════════════════════════════════════════════════════════════
def plot_ic_evolution():
    iters = sorted(ITER_IC.keys())
    ics   = [ITER_IC[i] for i in iters]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(iters, ics, marker='o', color='#2563EB', linewidth=2,
            markersize=6, markerfacecolor='white', markeredgewidth=2, zorder=3)

    # 标注关键里程碑
    milestones = {
        4:  ('引入集成\n+日内时间', 'below'),
        9:  ('累计流量\n+交互特征', 'above'),
        10: ('突破IC=0.30', 'above'),
        20: ('反过拟合\nLOO剪枝', 'above'),
    }
    for it, (label, pos) in milestones.items():
        y  = ITER_IC[it]
        yd = -0.022 if pos == 'below' else 0.010
        ax.annotate(
            label,
            xy=(it, y),
            xytext=(it, y + yd),
            fontsize=8,
            ha='center',
            arrowprops=dict(arrowstyle='->', color='#6B7280', lw=1.2),
            color='#374151',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#F9FAFB', edgecolor='#D1D5DB', alpha=0.8),
        )

    # IC=0.30 参考线
    ax.axhline(0.30, color='#EF4444', linestyle='--', linewidth=1.2, alpha=0.7, label='IC = 0.30 目标线')

    ax.set_xlabel('迭代轮次 (Iter)', fontsize=12)
    ax.set_ylabel('五折 CV 均值 IC', fontsize=12)
    ax.set_title('模型 IC 迭代演进曲线（Iter0 → Iter20）', fontsize=13, fontweight='bold')
    ax.set_xticks(iters)
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(0.10, 0.38)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=10)

    # 最终结果标注
    ax.annotate(
        f'IC = {ITER_IC[20]:.4f}\nICIR = {ITER20_ICIR:.2f}',
        xy=(20, ITER_IC[20]),
        xytext=(17.5, 0.345),
        fontsize=9,
        color='#1D4ED8',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#EFF6FF', edgecolor='#3B82F6'),
        arrowprops=dict(arrowstyle='->', color='#1D4ED8', lw=1.5),
    )

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig1_ic_evolution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Iter20 五折 CV 逐日 IC 条形图
# ══════════════════════════════════════════════════════════════════════════════
def plot_final_cv():
    days = list(ITER20_DAY_IC.keys())
    ics  = list(ITER20_DAY_IC.values())

    # 颜色：Day2 最高（深蓝），Day4 最低（浅蓝），其余中等蓝
    palette = ['#60A5FA', '#1D4ED8', '#3B82F6', '#93C5FD', '#2563EB']
    sorted_idx = np.argsort(ics)
    colors = ['#93C5FD'] * 5
    colors[sorted_idx[-1]] = '#1D4ED8'   # 最高 → 深蓝
    colors[sorted_idx[0]]  = '#93C5FD'   # 最低 → 浅蓝

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(days, ics, color=colors, width=0.5, edgecolor='white', linewidth=1.5)

    for bar, ic in zip(bars, ics):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{ic:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 均值虚线
    ax.axhline(ITER20_MEAN, color='#EF4444', linestyle='--', linewidth=1.5,
               label=f'均值 IC = {ITER20_MEAN:.4f}')

    ax.set_xlabel('测试天', fontsize=12)
    ax.set_ylabel('IC（皮尔森相关系数）', fontsize=12)
    ax.set_title(f'Iter20 最终模型逐日 IC（五折 CV，ICIR = {ITER20_ICIR}）',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 0.43)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig2_final_cv.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: 特征体系分类饼图
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_categories():
    labels = list(FEATURE_CATEGORIES.keys())
    sizes  = list(FEATURE_CATEGORIES.values())

    colors = [
        '#3B82F6', '#60A5FA', '#93C5FD', '#1D4ED8',
        '#BFDBFE', '#2563EB', '#DBEAFE', '#1E40AF',
    ]
    explode = [0.03] * len(labels)

    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct='%1.1f%%',
        colors=colors, explode=explode,
        startangle=140, pctdistance=0.80,
        wedgeprops=dict(edgecolor='white', linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(9)

    # 图例
    legend_labels = [f'{l.replace(chr(10), "")} ({s}个)' for l, s in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc='lower right', fontsize=9,
              bbox_to_anchor=(1.35, 0.0), frameon=True)

    ax.set_title('特征体系分类分布（共 59 个特征）', fontsize=13, fontweight='bold',
                 pad=15)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig3_feature_categories.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Iter1 vs Iter20 逐日 IC 对比条形图
# ══════════════════════════════════════════════════════════════════════════════
def plot_iteration_comparison():
    days   = ['Day1', 'Day2', 'Day3', 'Day4', 'Day5']
    ic1    = [ITER1_DAY_IC[d]    for d in days]
    ic20   = [ITER20_DAY_IC[d]   for d in days]
    mean1  = sum(ic1) / len(ic1)
    mean20 = ITER20_MEAN

    x      = np.arange(len(days))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, ic1,  width, label=f'Iter1  基线（均值 {mean1:.4f}）',
                   color='#93C5FD', edgecolor='white', linewidth=1.2)
    bars2 = ax.bar(x + width / 2, ic20, width, label=f'Iter20 最终（均值 {mean20:.4f}）',
                   color='#1D4ED8', edgecolor='white', linewidth=1.2)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    # 提升幅度标注
    for i, (d, y1, y20) in enumerate(zip(days, ic1, ic20)):
        gain = (y20 - y1) / y1 * 100
        ax.annotate(
            f'+{gain:.0f}%',
            xy=(x[i], max(y1, y20) + 0.025),
            ha='center', fontsize=8.5, color='#16A34A', fontweight='bold',
        )

    ax.set_xlabel('测试天', fontsize=12)
    ax.set_ylabel('IC（皮尔森相关系数）', fontsize=12)
    ax.set_title('Iter1 基线 vs Iter20 最终模型逐日 IC 对比', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(days)
    ax.set_ylim(0, 0.50)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig4_iter_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5: 嵌套盲测 – 动态集成 vs 等权基准 逐日对比（Iter19）
# ══════════════════════════════════════════════════════════════════════════════
def plot_ensemble_gain():
    days   = NESTED_DAYS
    ens    = NESTED_ENS_IC
    stable = NESTED_STABLE
    gain   = [e - s for e, s in zip(ens, stable)]

    x     = np.arange(len(days))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图: 集成 IC vs 稳定等权 IC
    ax = axes[0]
    ax.bar(x - width / 2, stable, width, label='稳定等权基准',
           color='#93C5FD', edgecolor='white')
    ax.bar(x + width / 2, ens,    width, label='动态集成预测',
           color='#1D4ED8', edgecolor='white')
    for xi, (s, e) in enumerate(zip(stable, ens)):
        ax.text(xi - width / 2, s + 0.005, f'{s:.3f}', ha='center', fontsize=8)
        ax.text(xi + width / 2, e + 0.005, f'{e:.3f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_xlabel('测试天', fontsize=11)
    ax.set_ylabel('IC', fontsize=11)
    ax.set_title('嵌套盲测：动态集成 vs 稳定等权基准', fontsize=11, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(days)
    ax.set_ylim(0, 0.45)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # 右图: 集成增益（差值）
    ax2 = axes[1]
    bar_colors = ['#16A34A' if g > 0 else '#DC2626' for g in gain]
    bars = ax2.bar(days, gain, color=bar_colors, edgecolor='white', width=0.5)
    for bar, g in zip(bars, gain):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.002,
                 f'+{g:.3f}', ha='center', fontsize=10, fontweight='bold',
                 color='#16A34A')
    ax2.axhline(0, color='#374151', linewidth=1)
    ax2.axhline(sum(gain) / len(gain), color='#EF4444', linestyle='--', linewidth=1.5,
                label=f'平均增益 = {sum(gain)/len(gain):.3f}')
    ax2.set_xlabel('测试天', fontsize=11)
    ax2.set_ylabel('IC 增益', fontsize=11)
    ax2.set_title('动态集成相对稳定等权基准的增益', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 0.12)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    plt.suptitle('动态滚动 IC-Softmax 集成机制有效性验证（Iter19 嵌套盲测）',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig5_ensemble_gain.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating figures...")
    plot_ic_evolution()
    plot_final_cv()
    plot_feature_categories()
    plot_iteration_comparison()
    plot_ensemble_gain()
    print("All figures generated successfully.")
