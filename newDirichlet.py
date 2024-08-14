import pandas as pd
import numpy as np
import os
import shutil

# 加载数据
def load_data(file_path='trainAll.csv'):
    return pd.read_csv(file_path, header=0)

# 使用狄利克雷分布生成每个标签在每个客户端的比例
def generate_dirichlet_distribution(alpha, num_clients=6, num_labels=8):
    distributions = np.random.dirichlet(alpha=alpha*np.ones(num_clients), size=num_labels)
    return distributions

# 根据生成的比例选择数据
def distribute_data(data, distributions, label_ranges, num_clients):
    client_data = {i: [] for i in range(1, num_clients+1)}
    client_indices = {i: [] for i in range(1, num_clients+1)}
    for label_idx, (start, end) in enumerate(label_ranges):
        label_data = data.iloc[start-1:end]
        proportions = np.round(distributions[label_idx] * len(label_data)).astype(int)
        proportions[-1] = len(label_data) - np.sum(proportions[:-1])
        
        indices = np.random.permutation(len(label_data))
        current_index = 0
        for client_idx, count in enumerate(proportions):
            idx = indices[current_index:current_index+count]
            client_data[client_idx+1].append(label_data.iloc[idx])
            client_indices[client_idx+1].extend(label_data.index[idx].tolist())
            current_index += count

    # 将数据组合并转化为DataFrame
    for client_idx in client_data:
        client_data[client_idx] = pd.concat(client_data[client_idx])

    return client_data, client_indices

def move_and_rename_file(src_file='testAll.csv', dest_dir='DirichletDistribution0.3', new_name='test.csv'):
    # 创建目标目录（如果不存在）
    os.makedirs(dest_dir, exist_ok=True)
    
    # 构建目标文件的完整路径
    dest_file = os.path.join(dest_dir, new_name)
    
    # 移动并重命名文件
    shutil.copy(src_file, dest_file)
    print(f"File '{src_file}' moved to '{dest_file}'")

# 保存到CSV
def save_data(client_data, base_dir='DirichletDistribution0.3'):
    os.makedirs(base_dir, exist_ok=True)
    for client_idx, data in client_data.items():
        data.to_csv(f'{base_dir}/train{client_idx}.csv', index=False)

def summarize_details(base_dir='DirichletDistribution0.3', output_file='detail.txt'):
    client_files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]
    with open(os.path.join(base_dir, output_file), 'w') as f:
        for file in client_files:
            file_path = os.path.join(base_dir, file)
            df = pd.read_csv(file_path)
            # 从1到8排序
            counts = df.iloc[:, 1].value_counts(normalize=True).reindex(range(0, 8), fill_value=0)
            f.write(f'Details for {file}:\n')
            f.write(f'Total rows: {len(df)}\n')
            f.write(f'Label distribution:\n')
            for label, count in counts.items():
                f.write(f'Label {label}: {count*100:.2f}%\n')
            f.write('\n')
            
def save_distributions(distributions, base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    file_path = os.path.join(base_dir, 'distributions.csv')
    df = pd.DataFrame(distributions)
    df.to_csv(file_path, index=False)
    print(f"Distributions saved to {file_path}")
    
# 主函数
def main():
    base_dir = 'DirichletDistribution0.3'
    data = load_data()
    num_clients = 6
    label_ranges = [(1, 401), (402, 801), (802, 1202), (1203, 1517), (1518, 1918), (1919, 2318), (2319, 2719), (2720, 3119)]
    distributions = generate_dirichlet_distribution(alpha=0.3, num_clients=num_clients, num_labels=len(label_ranges))
    print("Generated distributions (each row corresponds to a label and each column to a client):")
    print(distributions)
    save_distributions(distributions, base_dir)
    client_data, client_indices = distribute_data(data, distributions, label_ranges, num_clients)
    save_data(client_data)
    move_and_rename_file()
    summarize_details()

if __name__ == "__main__":
    main()
