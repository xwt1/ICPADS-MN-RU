import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_recall_vs_query_time(csv_files, colors, markers, labels, output_path):
    # 设置图表尺寸和DPI
    plt.rcParams['font.size'] = 28
    # plt.rcParams['font.weight'] = 'bold'

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    for csv_file, color, marker, label in zip(csv_files, colors, markers, labels):
        # 检查CSV文件是否存在
        if not os.path.exists(csv_file):
            print(f"CSV file '{csv_file}' not found.")
            continue

        # 读取CSV文件
        df = pd.read_csv(csv_file)

        df['query_time'] *= 1000
        # 将recall转为百分比计数
        df['recall'] *= 100

        ax.plot(df['query_time'], df['recall'], linestyle='-', color=color, marker=marker, label=label)

    # 设置图表标签和标题
    ax.set_xlabel('Query Time (ms)')
    ax.set_ylabel('Recall (%)')
    # ax.set_title('Recall vs Query Time')
    # ax.legend(prop={'size': 10, 'weight': 'bold'})
    ax.grid(True, linestyle='--', color='grey', linewidth=0.5)
    # 设置图例并放置到图表下方
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, prop={'size': 28})
    # 保存图表
    fig.savefig(output_path, bbox_inches='tight')

    # 显示图表
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate line chart from CSV data for recall vs query time.')
    parser.add_argument('root_path', type=str, help='Root path for input CSV files and output images')
    args = parser.parse_args()

    # 定义CSV文件的根目录和图像的输出路径
    csv_dir = os.path.join(args.root_path, 'output', 'full_coverage', 'imageNet')
    output_path = os.path.join(args.root_path, 'output', 'full_coverage', 'imageNet', 'full_coverage_imageNet_2M_end_recall_10.pdf')

    # 手动编码的CSV文件名列表
    csv_files = [
        'edge_connected_replaced_update7_end_recall.csv',
        'edge_connected_replaced_update8_end_recall.csv',
        'edge_connected_replaced_update9_end_recall.csv',
        'edge_connected_replaced_update10_end_recall.csv',
        'replaced_update_end_recall.csv',
        "faiss_end_recall_imageNet_2M.csv"
    ]
    # 手动编码的颜色和点的形状
    colors = ['r', 'g', 'b', 'm', 'c', 'y']  # 添加了一个额外的颜色'y'（黄色）
    markers = ['o', 's', 'D', '^', '*', 'p']  # 添加了一个额外的标记'p'（六边形）
    # 手动编码的图例标签
    labels = [
        'MN-RU α',
        'MN-RU β',
        'MN-RU γ',
        'MN-THN-RU',
        'HNSW-RU',
        'IVF-FLAT'
    ]
    # # 手动编码的CSV文件名列表
    # csv_files = [
    #     'edge_connected_replaced_update7_end_recall.csv',
    #     'edge_connected_replaced_update8_end_recall.csv',
    #     'edge_connected_replaced_update9_end_recall.csv',
    #     'edge_connected_replaced_update10_end_recall.csv',
    #     'replaced_update_end_recall.csv',
    #     # "faiss_end_recall_imageNet_2M.csv"
    # ]
    # # 手动编码的颜色和点的形状
    # colors = ['r', 'g', 'b', 'm', 'c']  # 添加了一个额外的颜色'y'（黄色）
    # markers = ['o', 's', 'D', '^', '*']  # 添加了一个额外的标记'p'（六边形）
    # # 手动编码的图例标签
    # labels = [
    #     'MN-RU α',
    #     'MN-RU β',
    #     'MN-RU γ',
    #     'MN-THN-RU',
    #     'HNSW-RU',
    # ]


    csv_file_paths = [os.path.join(csv_dir, csv_file) for csv_file in csv_files]

    plot_recall_vs_query_time(csv_file_paths, colors, markers, labels, output_path)
