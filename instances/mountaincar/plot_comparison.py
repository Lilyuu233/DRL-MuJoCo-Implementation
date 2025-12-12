# 文件名: plot_comparison.py
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_comparison(dir_before, dir_after):
    """
    加载两个实验的日志并绘制奖励对比图。
    """
    path_before = os.path.join(dir_before, 'progress.csv')
    path_after = os.path.join(dir_after, 'progress.csv')

    # 检查文件是否存在
    if not os.path.exists(path_before) or not os.path.exists(path_after):
        print("错误: 找不到 progress.csv 文件。")
        if not os.path.exists(path_before):
            print(f" - 在此路径下未找到: {path_before}")
        if not os.path.exists(path_after):
            print(f" - 在此路径下未找到: {path_after}")
        print("请确保您已经成功运行了两个训练脚本，并且日志目录路径正确。")
        return
        
    try:
        # 读取两个CSV文件
        df_before = pd.read_csv(path_before)
        df_after = pd.read_csv(path_after)
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return

    # --- 开始绘图 ---
    # 使用 'ggplot' 风格
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 8))

    # 绘制 "修改前" (Original) 的曲线
    plt.plot(df_before['misc/epoch'], df_before['eprewmean'], label='TRPO (Original CG)', color='orangered', linewidth=2)
    
    # 绘制 "修改后" (Warm Start) 的曲线
    plt.plot(df_after['misc/epoch'], df_after['eprewmean'], label='TRPO (Warm Start CG)', color='dodgerblue', linewidth=2)

    # 设置图表信息
    plt.title('Performance Comparison: Original vs. Warm Start Conjugate Gradient')
    plt.xlabel('Number of Policy Iterations')
    plt.ylabel('Mean Episode Reward')
    plt.legend() # 显示图例
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='比较两种TRPO实现的训练奖励。')
    parser.add_argument('dir_before', type=str, help='修改前 (try1.py) 的日志目录路径。')
    parser.add_argument('dir_after', type=str, help='修改后 (mountaincar_after.py) 的日志目录路径。')
    args = parser.parse_args()
    
    plot_comparison(args.dir_before, args.dir_after)