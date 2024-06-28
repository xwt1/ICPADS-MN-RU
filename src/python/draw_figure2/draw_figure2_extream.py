import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main(root_path):
    # CSV文件路径和保存图片的路径
    csv_file = root_path + '/output/figure_2/unreachable_points_phenomenon_extream.csv'
    output_path = root_path + '/output/figure_2/figure_extream.png'

    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 确保unreachable_points_number列为正数
    df['unreachable_points_number'] = df['unreachable_points_number'].abs()

    # 每隔10次选择一次数据点
    df_sampled = df[::10]

    # 全局设置字体属性
    plt.rcParams['font.size'] = 20

    # 设置图表尺寸和DPI
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 绘制折线图，使用较小的圆点
    ax.plot(df['iteration_number'], df['unreachable_points_number'], linestyle='-', color='pink', label='Unreachable Points')
    ax.plot(df_sampled['iteration_number'], df_sampled['unreachable_points_number'], marker='o', linestyle='None', color='pink', markersize=3)

    # 设置图表标签和标题
    ax.set_xlabel('Iteration Number')
    ax.set_ylabel('Unreachable Points Number')
    ax.set_title('Unreachable Points Number Over Iterations')
    ax.legend(prop={'size': 20})

    # 设置刻度标签字体大小
    ax.tick_params(axis='both', which='major', labelsize=20, width=1, length=5)

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
