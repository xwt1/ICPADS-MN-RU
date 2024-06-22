import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main(root_path):
    # CSV文件路径和保存图片的路径
    csv_file = root_path + '/output/figure_2/unreachable_points_phenomenon_extream.csv'
    output_path = root_path + '/output/figure_2/figure_extream_recall.png'

    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 每隔10次选择一次数据点
    df_sampled = df[::10]

    # 全局设置字体属性
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'

    # 设置图表尺寸和DPI
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 绘制折线图，使用较小的圆点
    ax.plot(df['iteration_number'], df['recall'], linestyle='-', color='lightblue', label='RECALL')
    ax.plot(df_sampled['iteration_number'], df_sampled['recall'], marker='o', linestyle='None', color='lightblue', markersize=3)

    # 设置图表标签和标题
    ax.set_xlabel('Iteration Number')
    ax.set_ylabel('RECALL')
    ax.set_title('RECALL Over Iterations')
    ax.legend(prop={'size': 12, 'weight': 'bold'})

    # 设置刻度标签字体加粗
    ax.tick_params(axis='both', which='major', labelsize=12, width=1, length=5)

    # 添加网格线
    ax.grid(True, linestyle='--', color='grey', linewidth=0.5)

    # 保存图表到指定路径
    fig.savefig(output_path, bbox_inches='tight')

    # 显示图表
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate line chart from CSV data.')
    parser.add_argument('root_path', type=str, help='Root path for input CSV and output image')
    args = parser.parse_args()
    main(args.root_path)
