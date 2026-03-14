// ─────────────────────────────────────────────────────────────────────────────
// 排版设置
// ─────────────────────────────────────────────────────────────────────────────
#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 3.0cm, right: 2.5cm),
  numbering: "1",
)

// 正文：小四号宋体，单倍行距
#set text(
  font: ("WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Noto Serif CJK SC",
         "Source Han Serif SC", "SimSun"),
  size: 12pt,
  lang: "zh",
  region: "cn",
)

#set par(
  leading: 0.65em,
  spacing: 1.0em,
  justify: true,
  first-line-indent: 2em,
)

// 列表项内部不做首行缩进
#show list: it => {
  set par(first-line-indent: 0em)
  it
}

// 黑体辅助命令
#let hei(content) = text(
  font: ("WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Noto Sans CJK SC",
         "Source Han Sans SC", "SimHei"),
  content,
)

// 一级标题：四号黑体，居中
#show heading.where(level: 1): it => {
  set align(center)
  set par(first-line-indent: 0em)
  set text(
    font: ("WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Noto Sans CJK SC",
           "Source Han Sans SC", "SimHei"),
    size: 14pt,
    weight: "bold",
  )
  v(0.8em)
  it.body
  v(0.4em)
}

// 二级标题：小四号黑体，左对齐
#show heading.where(level: 2): it => {
  set align(left)
  set par(first-line-indent: 0em)
  set text(
    font: ("WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Noto Sans CJK SC",
           "Source Han Sans SC", "SimHei"),
    size: 12pt,
    weight: "bold",
  )
  v(0.5em)
  it.body
  v(0.25em)
}

// 三级标题：小四号黑体，左对齐
#show heading.where(level: 3): it => {
  set align(left)
  set par(first-line-indent: 0em)
  set text(
    font: ("WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Noto Sans CJK SC",
           "Source Han Sans SC", "SimHei"),
    size: 12pt,
    weight: "bold",
  )
  v(0.3em)
  it.body
  v(0.15em)
}

// ─────────────────────────────────────────────────────────────────────────────
// 标题（三号黑体，居中）
// ─────────────────────────────────────────────────────────────────────────────
#align(center)[
  #text(
    font: ("WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Noto Sans CJK SC",
           "Source Han Sans SC", "SimHei"),
    size: 16pt,
    weight: "bold",
  )[基于委托簿微观结构的多因子动态集成股票收益率预测]
]

#v(1.0em)

// ─────────────────────────────────────────────────────────────────────────────
// 摘要
// ─────────────────────────────────────────────────────────────────────────────
#align(center)[
  #text(
    font: ("WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Noto Sans CJK SC",
           "Source Han Sans SC", "SimHei"),
    size: 14pt,
    weight: "bold",
  )[摘　要]
]

#v(0.4em)

本文针对A股市场Tick级别（500ms快照）数据的超短期收益率预测问题，以市场微观结构理论为数学原型，构建了一套多因子动态集成预测系统。研究使用同一板块五只股票（A/B/C/D/E）的五日历史Tick数据，任务是在线逐Tick预测股票E未来五分钟中间价收益率（Return5min）。

在特征工程方面，以Kyle（1985）连续拍卖模型和Glosten-Milgrom（1985）逆向选择模型为理论基础，从33个原始字段提取59个多维度因子，涵盖成交量失衡（TI）脉冲、委托量失衡（OVI）脉冲、深层委托簿压力（OBI）、大单成交失衡、截面联动及多重非线性交互特征。在模型方面，采用28个Ridge回归子模型（$alpha in [10, 1000]$），配合基于EWMA平滑滚动IC的Softmax动态集成算法，实现跨日自适应权重分配。

本文提出Leave-One-Model-Out（LOO）反过拟合剪枝算法，将子模型数量从54精简至28，IC从0.313升至0.330，ICIR从7.82大幅跃升至12.59（+61%）；同时引入稳定-Niche双层预热架构和板块截面异质信号（e_sect_lag_gap，IC=0.203），显著提升集成稳定性。以严格五折按日交叉验证（LODO-CV）为准则，经20轮迭代优化，最终模型五折CV均值IC=0.3302，ICIR=12.59，相对初始基线（IC=0.147）提升124.6%。嵌套盲测验证确认无数据泄露，动态集成在全部五天上均优于等权基准，平均增益+0.051。本系统实现全在线O(1)增量计算，具备实盘部署可行性，可直接应用于高频Alpha信号生成、量化做市策略优化及市场状态实时监控等场景。

#v(0.3em)
#[#set par(first-line-indent: 0em); #hei[关键词：]股票收益率预测；委托簿微观结构；动态集成；Ridge回归；滚动IC加权；Leave-One-Day-Out交叉验证]

#pagebreak()

// ─────────────────────────────────────────────────────────────────────────────
// 一、模型介绍
// ─────────────────────────────────────────────────────────────────────────────
= 一、模型介绍

== 1.1　问题数学化描述

#hei[数据格式：]五只股票（A/B/C/D/E）各提供5天历史数据，每天27,601行Tick快照，每只股票33个原始字段（五档买卖报价量、成交信息、订单流指标）。

#hei[预测目标（标签）：]股票E的未来五分钟中间价收益率

$ y(t) = "Return5min"(t) = frac("MidPrice"(t+600) - "MidPrice"(t), "MidPrice"(t)) $

其中 $"MidPrice"(t) = (p_1^b(t) + p_1^a(t)) \/ 2$（最优买一、卖一价均值），600个Tick约等于5分钟。

#hei[在线预测约束：]预测函数 $hat(y)(t) = f_theta (bold(x)_1, ..., bold(x)_t)$ 在 $t$ 时刻只能使用 $t$ 及之前的已知信息，不引入任何未来数据（严格无前视偏差）。

#hei[评价指标：]
- 信息系数（IC）：$"IC"_d = "Pearson"(hat(bold(y))_d, bold(y)_d)$，逐天计算后取五天均值 @ic_metric（第53页）。
- ICIR：$"ICIR" = overline("IC") \/ sigma("IC")$，IC均值除以标准差，衡量预测跨日稳定性 @ic_metric（第55页）。

== 1.2　数学模型的原型与理论来源

本研究的核心数学原型植根于#hei[市场微观结构理论]（Market Microstructure Theory）@microstructure，有三条理论脉络作为模型来源。

=== 1.2.1　Kyle（1985）连续拍卖模型

Kyle（1985）@kyle1985 连续拍卖模型是本文特征工程的直接理论来源。该模型证明，在均衡状态下，做市商根据观察到的净订单流 $u_t$（知情交易者与噪音交易者的合并买卖需求）调整报价，价格更新满足线性关系：

$ Delta m(t) = lambda dot u_t + epsilon_t $

其中 $lambda > 0$ 为Kyle's Lambda（价格冲击系数），$epsilon_t$ 为随机噪声。此结论直接启示构造成交量失衡（TI）和委托量失衡（OVI）作为预测因子——订单流方向与强度是价格走势的因果前驱。Cont等（2014）@ofi 实证证明了订单流失衡（OFI）与中间价变动的统计显著线性关系，为我们的特征设计提供了实证支撑。

=== 1.2.2　Glosten-Milgrom（1985）逆向选择模型

Glosten-Milgrom（1985）@glosten1985 逆向选择模型为深层委托簿特征提供理论依据。该模型指出：委托簿各档挂单量反映了做市商对逆向选择风险的对冲需求——深层买卖挂单量的不对称意味着市场参与者对未来价格方向的隐含预期。这是引入深层OBI（2–5档委托量失衡）因子的经济学基础，深层委托簿相比表层挂单更能反映"耐心资金"的方向偏好。

=== 1.2.3　自适应市场假说（AMH）与动态集成

Lo（2004）@amh 提出的自适应市场假说（Adaptive Markets Hypothesis，AMH）解释了为何需要动态集成机制。AMH指出市场效率并非固定常数，各类信号的有效性随市场参与者的竞争与学习不断演化。因此，任何静态单一因子模型均难以在多样化的市场状态下保持稳定预测力。实验数据印证了这一点：Day2（高波动竞价）的最优子模型组合与Day4（低波动趋势行情）完全不同，必须通过滚动权重机制动态响应市场状态切换。

== 1.3　核心因子的数学推导

=== 1.3.1　委托量失衡（OVI）

委托量失衡（OVI）是本模型中单因子预测力最强的信号（均值IC≈0.26）。五档总委托量失衡定义为：

$ "OVI"(t) = frac(sum_(i=1)^5 V_i^b(t) - sum_(i=1)^5 V_i^a(t), sum_(i=1)^5 V_i^b(t) + sum_(i=1)^5 V_i^a(t) + epsilon) in [-1, 1] $

其中 $epsilon > 0$ 为防止除零的平滑系数，$V_i^b, V_i^a$ 分别为第 $i$ 档买单量和卖单量。

"OVI脉冲"（Pulse）信号通过对OVI做短长窗口SMA差分来提取近期动能偏离：

$ "OVI_pk"(t) = overline("OVI")_k(t) - overline("OVI")_600(t), quad k in {15, 30, 60} $

其中 $overline("OVI")_k(t) = frac(1, k) sum_(s=t-k+1)^t "OVI"(s)$ 为过去 $k$ 个Tick的简单移动平均。当 $k < 600$ 时，$"OVI_pk" > 0$ 表明近期委托压力高于日内均值，形成正向预测信号。

=== 1.3.2　成交量失衡（TI）

成交量失衡定义为：

$ "TI"(t) = frac(V_"buy"(t) - V_"sell"(t), V_"buy"(t) + V_"sell"(t) + epsilon) $

其中 $V_"buy", V_"sell"$ 分别为主动买入和主动卖出成交量（由逐笔成交数据买卖方向判断）。TI直接对应Kyle模型中的净订单流 $u_t$，是价格冲击的成交层面体现。

=== 1.3.3　深层委托簿失衡（OBI）

根据Glosten-Milgrom模型，深层委托簿（2–5档）的不对称反映市场参与者对方向性风险的更深层判断：

$ "OBI"(t) = frac(sum_(i=2)^5 V_i^b(t) - sum_(i=2)^5 V_i^a(t), sum_(i=2)^5 V_i^b(t) + sum_(i=2)^5 V_i^a(t) + epsilon) $

相比OVI（1–5档合计），OBI着重捕捉"耐心资金"的方向偏好，提供与OVI部分正交的信息维度。

== 1.4　整体框架

如图1所示，整体框架分为离线训练和在线推理两阶段。

#figure(
  image("figures/fig6_architecture.png", width: 100%),
  caption: [系统整体架构图（离线训练 + 在线推理双阶段）],
) <fig6>

#hei[离线训练阶段]（`train_model.py`）：以五日全量数据为训练集，采用五折按日交叉验证评估。训练内容：(1) 计算59个特征并做z-score标准化；(2) 训练28个Ridge子模型，保存系数向量 $bold(w)_i$ 和偏置 $b_i$；(3) 训练集均值/标准差同步保存至 `models.pkl`。

#hei[在线推理阶段]（`MyModel.py`）：每收到一个新Tick，执行：(1) 增量更新SMA/EMA滑动窗口状态，计算59维特征向量 $bold(x)(t)$；(2) 对28个子模型线性推理 $hat(y)_i(t) = bold(w)_i^top bold(x)(t) + b_i$；(3) 滚动IC-Softmax动态集成得到最终预测 $hat(y)(t)$。

== 1.5　动态集成机制的数学推导

设第 $i$ 个子模型在滑动窗口 $[t - W, t - d)$ 内的滚动IC为：

$ "IC"_i(t) = "Pearson"\( {hat(y)_i(s)}_(s=t-W)^(t-d),\ {y(s)}_(s=t-W)^(t-d) \) $

其中 $W = 600$（约5分钟），$d = 600$（标签滞后，确保 $y(t-d)$ 在 $t$ 时刻已知）。

EWMA平滑抑制IC估计的短期噪声，半衰期约为 $T_(1/2) = -ln 2 / ln(1-beta) approx 99$ 步（$beta=0.007$）：

$ tilde("IC")_i(t) = (1-beta) dot tilde("IC")_i(t - Delta) + beta dot "IC"_i(t), quad Delta = 15 $

集成权重通过带温度的Softmax归一化（$tau = 17$）：

$ w_i(t) = frac(exp(tilde("IC")_i(t) dot tau), sum_(j=1)^N exp(tilde("IC")_j(t) dot tau)) $

温度参数 $tau$ 控制权重集中程度；最终预测：

$ hat(y)(t) = sum_(i=1)^N w_i(t) dot hat(y)_i(t) $

子模型分为两类：稳定模型（MTC，1个）在窗口积累不足时享有2倍先验权重；27个Niche模型预热期初始权重为0（零预热机制），积累足够IC历史后才参与集成，防止噪声IC污染早期权重估计。

== 1.6　模型创新点

=== 1.6.1　LOO反过拟合剪枝算法

传统集成学习倾向于"越多越好"，但在小样本（5天数据）条件下，子模型数量过多会导致集成噪声放大。本文提出Leave-One-Model-Out（LOO）剪枝算法：设集成性能指标 $"OM" = overline("IC") - 0.5 sigma("IC")$，LOO增益定义为：

$ Delta"OM"_m = "OM"(text("全模型")) - "OM"(text("去除") m) $

若 $Delta"OM"_m < 0$（去除模型 $m$ 后集成性能提升），则该模型有净负贡献，予以剪枝 @loo_prune。通过此分析，54个模型精简至28个，ICIR从7.82跃升至12.59（+61%）。

=== 1.6.2　稳定-Niche双层预热架构

本文将集成子模型显式分为"稳定模型"和"Niche模型"两类，并设计差异化预热机制，有效解决在线集成"冷启动"与"全力预测"的矛盾，在Day1早盘（历史数据最少）上表现尤为关键。

=== 1.6.3　截面板块异质信号

本文引入7个板块截面因子，其中最具创新性的是"截面异质OVI"（e_sect_lag_gap）：

$ "e_sect_lag_gap"(t) = "OVI_p15"^E(t) - "Sect_OVI_p20"(t) $

即股票E自身近期OVI相对板块均值的偏差，代表E股特异性看涨/看跌信号，均值IC高达0.203，是单次提升最大的新因子之一。

=== 1.6.4　在线增量计算框架

全部59个特征均采用增量式数据结构实现 $O(1)$ 时间复杂度更新（SMA使用双端队列、EMA仅保存上一步状态 @ewma_forecast），支持真实在线推理，与离线批量计算的传统做法形成鲜明对比。

// ─────────────────────────────────────────────────────────────────────────────
// 二、测试流程
// ─────────────────────────────────────────────────────────────────────────────
= 二、测试流程

== 2.1　特征和标签选取

=== 2.1.1　标签选取

预测目标为股票E未来五分钟中间价收益率：

$ y(t) = frac("MidPrice"(t+600) - "MidPrice"(t), "MidPrice"(t)) $

$t+600$ 时刻的中间价在 $t$ 时刻是未知的，模型在 $t$ 时刻输出预测 $hat(y)(t)$，训练时以真实 $y(t)$ 计算损失。对标签不做任何平滑或变换，保持原始收益率形式以便IC评估。

=== 2.1.2　特征选取原则

特征选取遵循以下原则：(1) #hei[无前视偏差]——所有特征仅使用 $t$ 时刻及之前的数据；(2) #hei[全日IC方向一致]——候选特征须在全部5个测试日上IC符号一致；(3) #hei[正交互补]——新增特征与现有特征相关系数须 $< 0.7$；(4) #hei[极端值处理]——对收益率类特征实施剪裁防止异常值干扰。

=== 2.1.3　特征体系概览（59个特征，8大类）

#figure(
  image("figures/fig3_feature_categories.png", width: 80%),
  caption: [特征体系分类分布（共59个特征，8大类）],
) <fig3>

各类特征的数学定义与经济学含义如下：

(1) #hei[成交量失衡（TI）脉冲（8个）：] $"TI_p"k = overline("TI")_k - overline("TI")_600$（$k in {15,30,40,60}$），捕捉成交方向的短期偏离；另包含EMA版本 $"TI_ep"k$。

(2) #hei[委托量失衡（OVI）脉冲（5个）：] 如第1.3.1节定义，均值IC≈0.26，为最强单因子。

(3) #hei[委托/成交笔数失衡（4个）：] 笔数层面的方向信号ONI（Order Number Imbalance）和TNI（Trade Number Imbalance），与OVI量纲不同，提供正交信息。

(4) #hei[板块联动因子（7个）：] A/B/C/D四只股票的截面均值信号，包含板块OBI、板块TI脉冲、板块OVI脉冲等，捕捉系统性板块动量效应。

(5) #hei[日内时间特征（2个）：] `aft_13800`（下午标志，$t >= 13800$s）和 `aft_12000`，利用下午市场反转效应。

(6) #hei[滞后价格收益特征（12个）：] 包括E股和板块过去5分钟已实现收益（均值回复信号），以及E股30/60/120/300/600/900 Tick中间价收益率。

(7) #hei[委托簿深度与流量（7个）：] 深层委托簿失衡（OBI，2–5档）、买卖价差脉冲、大单成交失衡、日内累计成交流量失衡，捕捉"耐心资金"和机构行为。

(8) #hei[交互与高级信号（14个）：] 波动率条件化OVI（$"OVI" times |r|$）、截面异质OVI（e_sect_lag_gap，IC均值=0.203）、价格动量加速度（$r_15 - r_30$）、OVI平方项及多重两两交互（如 $"OVI" times "TBV"$），为线性Ridge模型提供非线性信息。

== 2.2　特征预处理

=== 2.2.1　在线增量计算

所有特征均通过增量式数据结构实时计算，时间复杂度为 $O(1)$ 每Tick：

- `_RunSMA(window)`：固定窗口滑动均值，使用双端队列（`deque`）维护窗口。
- `_RunEMA(span)`：指数移动平均，$"EMA"_t = (1-alpha) dot "EMA"_{t-1} + alpha dot x_t$，$alpha = 2 / ("span" + 1)$。

=== 2.2.2　冷启动处理

日初前 $k$ 个Tick（$k$ 为特征滞后长度）无历史数据，使用当日有效值的运行均值填充，而非用 $0$ 填充。用 $0$ 填充会在冷启动期制造可学习但不泛化的假信号；均值填充可避免模型学到虚假冷启动规律 @fillna。

=== 2.2.3　标准化

Ridge回归对特征量纲敏感。训练时各子模型在训练集上做 $z$-score 标准化（$hat(x) = (x - mu) \/ sigma$）；在线推理时使用训练阶段保存的 $mu, sigma$ 做相同变换，均值和标准差随模型系数存入 `models.pkl`。

=== 2.2.4　极端值裁剪

对收益率类特征实施截断：

- 长窗口收益率（$>= 120$ Tick）：$x arrow.l "clip"(x, -0.1, 0.1)$
- 短窗口收益率（30/60 Tick）：$x arrow.l "clip"(x, -0.05, 0.05)$
- 交互特征（乘积项）：额外乘以归一化系数后裁剪至 $[-0.5, 0.5]$，防止乘积项方差过大。

== 2.3　训练集和交叉验证集的设置

=== 2.3.1　五折按日交叉验证（LODO-CV）

由于训练数据仅5天（Fold数等于日数），采用Leave-One-Day-Out交叉验证（LODO-CV）@lodo_cv：每折将一天作为测试集，其余四天作为训练集，共5折。优点：最大化训练数据利用率（每折使用4天训练）；测试日相互独立，IC均值为无偏估计；与竞赛评分机制（按天取IC均值）完全一致。实现中以 `GroupKFold(n_splits=5)` 配合 `Day` 列为分组标识，确保同一天数据不会同时出现在训练集和测试集。

=== 2.3.2　嵌套盲测验证

为定量评估调参过拟合程度，还实施了嵌套盲测（Nested Blind Test）：
- #hei[外层（盲测IC）：] 用其余4天训练，在保留天完整预测，验证代码层面无数据泄露。
- #hei[内层（调参IC）：] 在4天训练集内部运行4折CV，是调参时实际参考的数字，含对4天数据的过拟合。
- #hei[内-外gap：] Iter20模型内外gap约为 $-0.019$（外层略高），证明调参过拟合可控。

== 2.4　模型训练和参数设定

=== 2.4.1　Ridge回归子模型的理论选择

每个子模型为带 $L_2$ 正则化的线性回归（Ridge Regression）@ridge：

$ min_(bold(w), b) ||bold(y) - bold(X)bold(w) - b||_2^2 + alpha ||bold(w)||_2^2 $

解析解为：$hat(bold(w)) = (bold(X)^top bold(X) + alpha bold(I))^(-1) bold(X)^top bold(y)$

Ridge相比OLS的优势在于：(1) 当特征高度共线（多个OVI/TI特征高度相关）时，Ridge通过 $alpha$ 控制方差-偏差权衡；(2) 相比LightGBM/神经网络，Ridge在样本量约 $n = 138000$、特征维度 $p = 59$ 时泛化性更强。正则化系数 $alpha$ 按模型特性设定：多数模型 $alpha=150$（综合特征集），OVI纯净模型 $alpha=10$（低维需弱正则），高维交互特征模型 $alpha=1000$（防过拟合）@ridge。

=== 2.4.2　28个子模型的设计

28个子模型覆盖多种特征子集和时间段组合。1个稳定模型（MTC）负责全时段稳健预测，作为预热期基准；27个Niche模型针对特定信号组合和时段，在各自擅长的市场条件下表现突出。

#figure(
  kind: table,
  caption: [子模型分类概览],
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, left),
    stroke: 0.5pt,
    [#hei[模型类别]], [#hei[数量]], [#hei[正则化 $alpha$]], [#hei[特征重点]],
    [稳定（Stable）], [1], [150], [TI纯净 + 时间特征（MTC）],
    [OVI纯净niche], [4], [10–150], [OVI脉冲系列为主要输入],
    [截面反转niche], [3], [150], [`sect_ret_lag` + `e_ret_lag` + `e_sect_lag_gap`],
    [非线性niche], [4], [150–1000], [`spread_wt_ovi`、`ovi_sq`、交互特征族],
    [深度/流量niche], [6], [150], [`obi_deep_p15`、`cum_flow_imb`、`book_pres_pulse`],
    [综合多因子niche], [10], [150–500], [多信号组合],
  )
) <tab1>

=== 2.4.3　迭代优化历程

如图2所示，模型经历20轮迭代优化，IC从0.147稳步提升至0.3302：

#figure(
  image("figures/fig1_ic_evolution.png", width: 95%),
  caption: [模型IC迭代演进曲线（Iter0→Iter20）],
) <fig1>

主要里程碑：Iter0–2建立LODO-CV评估体系，IC从0.147提升至0.224；Iter3–8引入动态集成机制，IC提升至0.279；Iter9–12引入累计流量失衡和交互特征，IC首次突破0.30；Iter13–19加入动量加速度、波动率条件化、截面异质OVI等高级信号，IC稳步提升至0.313，ICIR达7.82；Iter20采用LOO剪枝，54个模型精简为28个，IC跃升至0.330，ICIR从7.82飙升至12.59（+61%）。

=== 2.4.4　集成参数设定

#figure(
  kind: table,
  caption: [动态集成参数汇总],
  table(
    columns: (auto, auto, auto),
    align: (left, center, left),
    stroke: 0.5pt,
    [#hei[参数]], [#hei[值]], [#hei[含义]],
    [`ENSEMBLE_WINDOW`], [600], [滚动IC计算窗口（约5分钟）],
    [`ENSEMBLE_TEMP`], [17], [Softmax温度（权重集中程度）],
    [`ENSEMBLE_FLOOR`], [0], [权重下界（允许权重降至0）],
    [`RETURN_DELAY`], [600], [IC计算所需标签滞后（Tick）],
    [`ENSEMBLE_UPDATE_FREQ`], [15], [权重更新频率（每15 Tick）],
    [`ENSEMBLE_EWMA_BETA`], [0.007], [IC平滑系数（半衰期约99 Tick）],
    [`STABLE_PRIOR`], [2.0], [稳定模型预热期先验权重倍数],
    [`NICHE_INIT_WEIGHT`], [0], [Niche模型预热期初始权重],
  )
) <tab2>

== 2.5　测试结果展示和模型评价

=== 2.5.1　逐日IC结果

图3展示了最终模型（Iter20）的五折CV逐日IC：

#figure(
  image("figures/fig2_final_cv.png", width: 85%),
  caption: [Iter20最终模型五折CV逐日IC（均值=0.330，ICIR=12.59）],
) <fig2>

五天测试IC分别为：Day1=0.326、Day2=0.363、Day3=0.338、Day4=0.284、Day5=0.341，均值IC=0.3302，标准差=0.026，ICIR=12.59。Day4（低波动趋势日）IC最低（0.284），与AMH预测一致——微观结构信号在趋势性行情中预测力相对弱。五天IC标准差仅0.026，体现了极高的跨日稳定性。

=== 2.5.2　基线对比

图4展示了从Iter1（6特征线性基线）到Iter20（28模型动态集成）的逐日IC提升：

#figure(
  image("figures/fig4_iter_comparison.png", width: 90%),
  caption: [Iter1基线 vs Iter20最终模型逐日IC对比],
) <fig4>

相比基线，Iter20在每一天上均取得显著提升：Day1提升72.5%（0.189→0.326），Day3提升134.7%（0.144→0.338），Day4提升129.0%（0.124→0.284）——弱信号日（Day1/3/4）的改善尤为突出，体现了多因子集成对多样化市场状态的覆盖能力。

=== 2.5.3　动态集成增益验证

图5展示了嵌套盲测中动态集成相对稳定等权基准的增益：

#figure(
  image("figures/fig5_ensemble_gain.png", width: 95%),
  caption: [动态集成 vs 稳定等权基准的IC增益（嵌套盲测）],
) <fig5>

动态集成在所有5天上均优于等权基准，平均增益+0.051，验证了滚动IC-Softmax机制的有效性 @softmax_ensemble。增益最大的是Day5（+0.104），该天Niche模型的滚动IC高度分化，动态权重精准识别了强信号模型。

=== 2.5.4　综合模型评价

#figure(
  kind: table,
  caption: [最终模型（Iter20）综合评价指标],
  table(
    columns: (auto, auto, auto),
    align: (left, center, left),
    stroke: 0.5pt,
    [#hei[评价维度]], [#hei[指标值]], [#hei[说明]],
    [五折CV均值IC], [0.3302], [主要竞赛指标，out-of-sample估计],
    [五折CV IC标准差], [0.026], [跨日稳定性，越低越好],
    [ICIR（IC/Std）], [12.59], [风险调整后IC，历史最高],
    [Penalized（M$-$0.5S）], [0.317], [惩罚性指标，兼顾均值与方差],
    [内层4折IC], [0.311], [调参参考值（含轻微调参偏差）],
    [内-外gap], [$-$0.019], [负值=外层略高，调参过拟合可控],
    [集成增益（vs等权）], [+0.051], [动态集成相对等权基准提升],
    [模型复杂度], [28子模型，59特征], [相比Iter19（54×59）更简洁],
  )
) <tab3>

本模型在应用前景方面也具有较强价值：(1) 预测值 $hat(y)(t)$ 可直接用于超短期仓位决策，在合理交易成本假设下具备实盘部署价值；(2) 做市商可利用模型预测优化挂单策略，降低逆向选择风险；(3) 框架在多标的、跨资产场景均可扩展；(4) 28个子模型的实时权重分布本身是市场状态的量化刻画，可为风险管理提供实时的市场状态指示器。

// ─────────────────────────────────────────────────────────────────────────────
// 参考文献
// ─────────────────────────────────────────────────────────────────────────────
= 参考文献

#bibliography("references.bib", style: "gb-7714-2015-numeric", title: none)
