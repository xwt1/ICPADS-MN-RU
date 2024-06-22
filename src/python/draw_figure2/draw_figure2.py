import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main(root_path):
    # CSV文件路径和保存图片的路径
    csv_file = root_path + '/output/figure_2/unreachable_points_phenomenon.csv'
    output_path = root_path + '/output/figure_2/figure.png'

    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 确保unreachable_points_number列为正数
    df['unreachable_points_number'] = df['unreachable_points_number'].abs()

    # 每隔10次选择一次数据点
    df_sampled = df[::10]

    # 设置图表尺寸
    plt.figure(figsize=(10, 6))

    # 绘制折线图，使用较小的圆点
    plt.plot(df['iteration_number'], df['unreachable_points_number'], linestyle='-', color='b', label='Unreachable Points')
    plt.plot(df_sampled['iteration_number'], df_sampled['unreachable_points_number'], marker='o', linestyle='None', color='b', markersize=3)

    # 设置图表标签和标题
    plt.xlabel('Iteration Number')
    plt.ylabel('Unreachable Points Number')
    plt.title('Unreachable Points Number Over Iterations')
    plt.legend()

    # 保存图表到指定路径
    plt.savefig(output_path)

    # 显示图表
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate line chart from CSV data.')
    parser.add_argument('root_path', type=str, help='Root path for input CSV and output image')
    args = parser.parse_args()
    main(args.root_path)



# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import argparse
#
# def main(root_path):
#     # CSV文件路径和保存图片的路径
#     csv_file = root_path + '/output/figure_2/unreachable_points_phenomenon.csv'
#     output_path = root_path + '/output/figure_2/figure.png'
#
#     # 读取CSV文件
#     df = pd.read_csv(csv_file)
#
#     # 确保unreachable_points_number列为正数
#     df['unreachable_points_number'] = df['unreachable_points_number'].abs()
#
#     # 每隔固定间隔选择一次数据点
#     interval = 10
#     df_sampled = df[::interval]
#
#     # 设置图表尺寸
#     plt.figure(figsize=(10, 6))
#
#     # 绘制折线图，使用较小的圆点
#     plt.plot(df['iteration_number'], df['unreachable_points_number'], linestyle='-', color='b', label='Unreachable Points')
#     plt.plot(df_sampled['iteration_number'], df_sampled['unreachable_points_number'], marker='o', linestyle='None', color='b', markersize=3)
#
#     # 添加竖线和横线
#     for i in range(0, len(df), interval):
#         plt.axvline(x=df['iteration_number'][i], color='gray', linestyle='--', linewidth=0.5)
#         plt.axhline(y=df['unreachable_points_number'][i], color='gray', linestyle='--', linewidth=0.5)
#
#     # 设置图表标签和标题
#     plt.xlabel('Iteration Number')
#     plt.ylabel('Unreachable Points Number')
#     plt.title('Unreachable Points Number Over Iterations')
#     plt.legend()
#
#     # 保存图表到指定路径
#     plt.savefig(output_path)
#
#     # 显示图表
#     plt.show()
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Generate line chart from CSV data.')
#     parser.add_argument('root_path', type=str, help='Root path for input CSV and output image')
#     args = parser.parse_args()
#     main(args.root_path)
