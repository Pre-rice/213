"""
评估多天的IC值 - 按天匹配预测和真实数据
"""

import numpy as np
import pandas as pd
import os
import glob

def find_matching_days():
    """查找有预测和真实数据的对应天数"""
    print("🔍 查找匹配的天数...")
    
    # 获取所有有预测的天数
    pred_days = []
    for item in os.listdir("./output"):
        pred_path = os.path.join("./output", item, "E.csv")
        if os.path.exists(pred_path):
            pred_days.append(item)
    
    print(f"预测天数: {sorted(pred_days)}")
    
    # 获取所有有真实数据的天数
    true_days = []
    for item in os.listdir("./data"):
        true_path = os.path.join("./data", item, "E.csv")
        if os.path.exists(true_path):
            true_days.append(item)
    
    print(f"真实数据天数: {sorted(true_days)}")
    
    # 找出共同的天数
    common_days = sorted(set(pred_days) & set(true_days))
    print(f"匹配的天数: {common_days}")
    
    return common_days

def evaluate_day(day):
    """评估单天的数据"""
    print(f"\n📅 评估第 {day} 天...")
    
    # 加载预测数据
    pred_path = f"./output/{day}/E.csv"
    df_pred = pd.read_csv(pred_path)
    predictions = df_pred['Predict'].values
    
    print(f"  预测文件: {pred_path}")
    print(f"  预测样本数: {len(predictions)}")
    
    # 加载真实数据
    true_path = f"./data/{day}/E.csv"
    df_true = pd.read_csv(true_path)
    
    if 'Return5min' not in df_true.columns:
        print(f"❌ 第 {day} 天真实数据没有Return5min列")
        return None, None, None
    
    true_returns = df_true['Return5min'].values
    print(f"  真实文件: {true_path}")
    print(f"  真实样本数: {len(true_returns)}")
    
    # 对齐数据
    min_len = min(len(predictions), len(true_returns))
    pred_aligned = predictions[:min_len]
    true_aligned = true_returns[:min_len]
    
    print(f"  对齐后样本数: {min_len}")
    
    # 计算IC值
    def evaluate_ic(pred, true):
        def clean_data(data):
            data = np.where(np.isnan(data), 0, data)
            data = np.where(np.isinf(data), 0, data)
            data = np.where(np.isinf(-data), 0, data)
            return data
        
        data = np.vstack((pred, true))
        data = clean_data(data)
        return np.corrcoef(data)[0, 1]
    
    ic = evaluate_ic(pred_aligned, true_aligned)
    print(f"  IC值: {ic:.6f}")
    
    return ic, pred_aligned, true_aligned

def main():
    """主函数"""
    print("="*60)
    print("多天IC值评估工具")
    print("="*60)
    
    # 1. 查找匹配的天数
    common_days = find_matching_days()
    
    if not common_days:
        print("❌ 没有找到匹配的天数！")
        print("请检查目录结构：")
        print("  ./output/<day>/E.csv 应该存在")
        print("  ./data/<day>/E.csv 应该存在且包含Return5min列")
        return
    
    # 2. 按天评估
    daily_results = []
    all_predictions = []
    all_true_returns = []
    
    for day in common_days:
        ic, preds, trues = evaluate_day(day)
        if ic is not None:
            daily_results.append({
                'day': day,
                'ic': ic,
                'samples': len(preds),
                'pred_mean': np.mean(preds),
                'pred_std': np.std(preds),
                'true_mean': np.mean(trues),
                'true_std': np.std(trues)
            })
            all_predictions.extend(preds)
            all_true_returns.extend(trues)
    
    if not daily_results:
        print("❌ 没有有效的评估结果")
        return
    
    # 3. 计算总体IC
    def evaluate_ic(pred, true):
        def clean_data(data):
            data = np.where(np.isnan(data), 0, data)
            data = np.where(np.isinf(data), 0, data)
            data = np.where(np.isinf(-data), 0, data)
            return data
        
        data = np.vstack((pred, true))
        data = clean_data(data)
        return np.corrcoef(data)[0, 1]
    
    overall_ic = evaluate_ic(np.array(all_predictions), np.array(all_true_returns))
    avg_daily_ic = np.mean([r['ic'] for r in daily_results])
    
    # 4. 显示结果
    print("\n" + "="*60)
    print("评估结果汇总")
    print("="*60)
    
    print("\n📊 每日结果:")
    for result in daily_results:
        print(f"  第 {result['day']} 天: IC = {result['ic']:.6f}, 样本数 = {result['samples']}")
    
    print(f"\n📈 总体结果:")
    print(f"  总天数: {len(daily_results)}")
    print(f"  总样本数: {len(all_predictions)}")
    print(f"  日平均IC: {avg_daily_ic:.6f}")
    print(f"  总体IC: {overall_ic:.6f}")
    
    # 5. 分析
    print("\n📋 结果分析:")
    if abs(overall_ic) < 0.001:
        print("  IC值接近0，模型几乎没有预测能力")
        print("  可能原因：")
        print("  1. 特征与标签相关性太弱")
        print("  2. 模型过于简单")
        print("  3. 需要更好的特征工程")
    elif overall_ic > 0.01:
        print(f"  🎉 IC = {overall_ic:.4f}，有一定预测能力！")
        if overall_ic > 0.02:
            print("  👍 表现不错！IC > 0.02 通常被认为是有效的")
    elif overall_ic > 0:
        print(f"  📈 IC = {overall_ic:.4f}，有轻微的正相关性")
        print("  可以尝试改进模型以获得更好的结果")
    else:
        print(f"  📉 IC = {overall_ic:.4f}，预测方向与真实方向相反")
        print("  可能原因：")
        print("  1. 特征与标签负相关")
        print("  2. 模型参数需要调整")
    
    # 6. 保存结果
    eval_dir = "./evaluation/"
    os.makedirs(eval_dir, exist_ok=True)
    
    # 保存汇总结果
    summary_df = pd.DataFrame(daily_results)
    summary_path = os.path.join(eval_dir, "daily_ic_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # 保存所有数据
    all_data_df = pd.DataFrame({
        '预测值': all_predictions,
        '真实值': all_true_returns,
        '天数': np.repeat([r['day'] for r in daily_results], [r['samples'] for r in daily_results])
    })
    all_data_path = os.path.join(eval_dir, "all_evaluation_data.csv")
    all_data_df.to_csv(all_data_path, index=False)
    
    print(f"\n✅ 结果已保存:")
    print(f"  每日汇总: {summary_path}")
    print(f"  所有数据: {all_data_path}")
    
    # 7. 可视化
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(14, 10))
        
        # 1. 每日IC值柱状图
        plt.subplot(2, 3, 1)
        days = [r['day'] for r in daily_results]
        ics = [r['ic'] for r in daily_results]
        bars = plt.bar(range(len(ics)), ics)
        
        # 根据IC值着色
        for i, bar in enumerate(bars):
            if ics[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        plt.axhline(y=avg_daily_ic, color='blue', linestyle='--', label=f'平均IC: {avg_daily_ic:.4f}')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xticks(range(len(days)), days, rotation=45)
        plt.xlabel('天数')
        plt.ylabel('IC值')
        plt.title('每日IC值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 预测vs真实散点图
        plt.subplot(2, 3, 2)
        plt.scatter(all_true_returns, all_predictions, alpha=0.2, s=10)
        plt.xlabel('真实收益率')
        plt.ylabel('预测值')
        plt.title(f'预测vs真实 (总体IC={overall_ic:.4f})')
        plt.grid(True, alpha=0.3)
        
        # 添加回归线
        if len(all_predictions) > 1:
            z = np.polyfit(all_true_returns, all_predictions, 1)
            p = np.poly1d(z)
            plt.plot(np.sort(all_true_returns), p(np.sort(all_true_returns)), "r--", alpha=0.8)
        
        # 3. 时间序列对比（前300个样本）
        plt.subplot(2, 3, 3)
        sample_limit = min(300, len(all_predictions))
        plt.plot(all_predictions[:sample_limit], label='预测', alpha=0.7, linewidth=1)
        plt.plot(all_true_returns[:sample_limit], label='真实', alpha=0.7, linewidth=1)
        plt.xlabel('样本索引')
        plt.ylabel('值')
        plt.title('时间序列对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 分布对比
        plt.subplot(2, 3, 4)
        plt.hist(all_predictions, bins=50, alpha=0.5, label='预测', density=True)
        plt.hist(all_true_returns, bins=50, alpha=0.5, label='真实', density=True)
        plt.xlabel('值')
        plt.ylabel('密度')
        plt.title('预测值与真实值分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. IC值分布
        plt.subplot(2, 3, 5)
        plt.hist(ics, bins=10, edgecolor='black', alpha=0.7)
        plt.axvline(x=avg_daily_ic, color='red', linestyle='--', label=f'平均IC: {avg_daily_ic:.4f}')
        plt.xlabel('IC值')
        plt.ylabel('天数')
        plt.title('IC值分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. 累计值对比
        plt.subplot(2, 3, 6)
        cum_pred = np.cumsum(all_predictions)
        cum_true = np.cumsum(all_true_returns)
        plt.plot(cum_pred, label='预测累计', alpha=0.7)
        plt.plot(cum_true, label='真实累计', alpha=0.7)
        plt.xlabel('样本数')
        plt.ylabel('累计值')
        plt.title('累计值对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = os.path.join(eval_dir, "multi_day_evaluation.png")
        plt.savefig(chart_path, dpi=100, bbox_inches='tight')
        print(f"✅ 图表保存到: {chart_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"⚠️  可视化失败: {e}")

if __name__ == "__main__":
    main()