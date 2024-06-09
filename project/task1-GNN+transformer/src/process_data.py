import os
import pickle as pkl
import torch
import copy
import random

from tqdm import tqdm

# 定义数据路径
data_dir = 'project_data/'

# 获取目录下的所有文件列表
files = [os.path.join(data_dir, file)
         for file in os.listdir(data_dir) if file.endswith('.pkl')]
# num_samples = 10000
# files = random.sample(files, num_samples)
print('Data number:', len(files))

# 遍历文件并输出 state 和 label
test_data = []

for file in tqdm(files, desc="Processing files"):
    with open(file, 'rb') as pkl_file:
        data = pkl.load(pkl_file)
        states = data['input']
        labels = data['target']
        for state, label in zip(states, labels):
            initial_aig, actions = state.split('_')
            actions = torch.tensor([int(i)
                                    for i in actions], dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
            sample = {}
            sample['initial_aig'] = initial_aig + '.aig'
            sample['actions'] = actions
            sample['label'] = label
            new_sample = copy.deepcopy(sample)
            test_data.append(new_sample)

for i in range(10):
    print(test_data[i]['actions'])

# 保存处理后的数据
data_path = 'data/train_data.pt'
torch.save(test_data, data_path)
