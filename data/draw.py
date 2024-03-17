# import sys
#
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# import os
#
# if len(sys.argv) < 2:
#     print("Usage: python plot_script.py <project_root_path>")
#     sys.exit(1)
#
# project_root_path = sys.argv[1]
#
# # 读取数据
# points_df = pd.read_csv(project_root_path + '/data/Points.csv')
# query_df = pd.read_csv(project_root_path + '/data/query.csv')
# ans_df = pd.read_csv(project_root_path + '/data/ans.csv')
#
# # 创建3D绘图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制query中的点
# ax.scatter(query_df.iloc[:, 0], query_df.iloc[:, 1], query_df.iloc[:, 2], color='red',  label='Query Point')
#
# # 绘制ans中的点
# ax.scatter(ans_df.iloc[:, 0], ans_df.iloc[:, 1], ans_df.iloc[:, 2], color='green', label='Answer Points')
#
# # 绘制Points中的点
# ax.scatter(points_df.iloc[:, 0], points_df.iloc[:, 1], points_df.iloc[:, 2], color='blue', label='Points')
#
#
#
# # 设置图例
# ax.legend()
#
# # 设置坐标轴标签
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
#
# # 保存图像到文件
# plt.savefig(project_root_path + '/data/plot.png')
#
# # 显示图形
# plt.show()
#
#


# import sys
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import os
#
# if len(sys.argv) < 2:
#     print("Usage: python plot_script.py <project_root_path>")
#     sys.exit(1)
#
# project_root_path = sys.argv[1]
#
# # 读取数据
# points_df = pd.read_csv(project_root_path + '/data/Points.csv')
# query_df = pd.read_csv(project_root_path + '/data/query.csv')
# ans_df = pd.read_csv(project_root_path + '/data/ans.csv')
#
# # 创建3D绘图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 首先绘制优先级最低的points_df中的点
# ax.scatter(points_df.iloc[:, 0], points_df.iloc[:, 1], points_df.iloc[:, 2], color='blue', alpha=0.5, label='Points')
#
# # 接着绘制次之优先级的ans_df中的点
# ax.scatter(ans_df.iloc[:, 0], ans_df.iloc[:, 1], ans_df.iloc[:, 2], color='green', s=60, label='Answer Points')
#
# # 最后绘制优先级最高的query_df中的点
# ax.scatter(query_df.iloc[:, 0], query_df.iloc[:, 1], query_df.iloc[:, 2], color='red', s=100, label='Query Point')
#
# # 设置图例
# ax.legend()
#
# # 设置坐标轴标签
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# # 保存图像到文件
# plt.savefig(project_root_path + '/data/plot.png')
#
# # 显示图形
# plt.show()
#


import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

if len(sys.argv) < 2:
    print("Usage: python plot_script.py <project_root_path>")
    sys.exit(1)

project_root_path = sys.argv[1]

# 读取数据
points_df = pd.read_csv(project_root_path + '/data/Points.csv')
query_df = pd.read_csv(project_root_path + '/data/query.csv')
ans_df = pd.read_csv(project_root_path + '/data/ans.csv')

# 创建2D绘图
fig, ax = plt.subplots()  # 注意这里的改变，直接使用subplots创建二维绘图

# 首先绘制优先级最低的points_df中的点
ax.scatter(points_df.iloc[:, 0], points_df.iloc[:, 1], color='blue', alpha=0.5, label='Points')

# 接着绘制次之优先级的ans_df中的点
ax.scatter(ans_df.iloc[:, 0], ans_df.iloc[:, 1], color='green', s=60, label='Answer Points')

# 最后绘制优先级最高的query_df中的点
ax.scatter(query_df.iloc[:, 0], query_df.iloc[:, 1], color='red', s=100, label='Query Point')

# 设置图例
ax.legend()

# 设置坐标轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')

# 保存图像到文件
plt.savefig(project_root_path + '/data/plot_2d.png')

# 显示图形
plt.show()