import matplotlib.pyplot as plt
import os

def read_nodes(filename):
    nodes_per_level = {}
    with open(filename, 'r') as file:
        current_level = -1
        for line in file:
            if line.startswith("Level"):
                current_level = int(line.split()[1])
                nodes_per_level[current_level] = []
            else:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                node_id = int(parts[0])
                label = parts[1]
                nodes_per_level[current_level].append((node_id, label))
    return nodes_per_level

def plot_and_save_nodes(nodes_per_level, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for level, nodes in nodes_per_level.items():
        plt.figure(figsize=(10, 10))
        x = [node_id % 1000 for node_id, _ in nodes]  # Dummy x-coordinates
        y = [node_id // 1000 for node_id, _ in nodes]  # Dummy y-coordinates
        plt.scatter(x, y, s=10, label=f'Level {level}')
        plt.title(f"Graph Level {level}")
        plt.legend()
        output_path = os.path.join(output_dir, f"graph_level_{level}.png")
        plt.savefig(output_path)
        plt.close()

if __name__ == "__main__":
    filename = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/random/graph.txt"
    output_dir = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/retrieval-diversity-enhancement/data/random/delete_index/graph"
    nodes_per_level = read_nodes(filename)
    plot_and_save_nodes(nodes_per_level, output_dir)

