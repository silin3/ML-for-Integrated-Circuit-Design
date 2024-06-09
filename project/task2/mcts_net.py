import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from transformers import BertModel, BertTokenizer

# Define necessary paths
result_file = 'results1.json'
data_path = "./aig_tensor.pt"
all_aig_tensor_data = torch.load(data_path)


initial_aig = all_aig_tensor_data['alu4.aig']

# Load AIG tensor data
all_aig_tensor_data = torch.load(data_path)

# Define operation dictionary
synthesisOpToPosDic = {
    0: "refactor",
    1: "refactor -z",
    2: "rewrite",
    3: "rewrite -z",
    4: "resub",
    5: "resub -z",
    6: "balance"
}

# Define model and device
class AIGModel(nn.Module):
    def __init__(self):
        super(AIGModel, self).__init__()
        self.gcn1 = GCNConv(2, 32)
        self.gcn2 = GCNConv(32, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.transformer = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(832, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, graphs, device):
        x, edge_index, batch, action_sequence = graphs.x, graphs.edge_index, graphs.batch, graphs.actions
        x = x.float()
        action_sequence = action_sequence.reshape(-1, 10).long()
        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        aig_embedding = torch.cat((mean_pool, max_pool), dim=1)
        action_sequence[action_sequence == -1] = self.tokenizer.pad_token_id
        attention_mask = (action_sequence != self.tokenizer.pad_token_id).long()
        sequence_outputs = self.transformer(input_ids=action_sequence, attention_mask=attention_mask)
        sequence_embedding = sequence_outputs['pooler_output']
        combined_embedding = torch.cat((aig_embedding, sequence_embedding), dim=1)
        x = self.fc1(combined_embedding)
        x = self.fc2(x)
        output = self.fc3(x)
        return output

def predict(initial_aig, actions, model, device):
    actions = F.pad(actions, (0, 10 - actions.size(0)), "constant", -1)
    node_type = initial_aig['node_type'].unsqueeze(1)
    num_inverted_predecessors = initial_aig['num_inverted_predecessors'].unsqueeze(1)
    edge_index = initial_aig['edge_index']
    nodes = initial_aig['nodes'].unsqueeze(0)
    graph = Data(x=torch.cat([node_type, num_inverted_predecessors], dim=1),
                 edge_index=edge_index,
                 num_nodes=nodes,
                 actions=actions,
                 y=0.0)
    model.eval()
    with torch.no_grad():
        output = model(graph, device).to(device)
    return output

device = torch.device('cpu')
aig_model = AIGModel()
aig_model.load_state_dict(torch.load('model_mae.pth', map_location=device))
aig_model.to(device)

class MCTSNode:
    def __init__(self, initial_aig, parent=None, action=None):
        self.initial_aig = initial_aig
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = -float('inf')
        self.action_sequence = parent.action_sequence + [action] if parent else []

    def add_child(self, child_node):
        self.children.append(child_node)

    def is_fully_expanded(self):
        return len(self.children) == len(synthesisOpToPosDic)

    def best_child(self, exploration_weight=1.0):
        weights = []
        for child in self.children:
            if child.visits > 0:
                weight = (child.score / child.visits) + exploration_weight * np.sqrt(2 * np.log(self.visits) / child.visits)
            else:
                weight = float('inf')
            weights.append(weight)
        return self.children[np.argmax(weights)]

def mcts(initial_aig, iterations=50000, steps=10):
    root = MCTSNode(initial_aig)
    
    for _ in range(iterations):
        node = root
        path = [node]  # 记录当前路径

        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
            path.append(node)
            if len(path) >= steps:  # 如果路径长度达到steps，停止选择
                break

        # Expansion
        for _ in range(steps - len(path)):  # 在一次迭代中尽可能扩展多个节点
            if not node.is_fully_expanded():
                expanded = False
                possible_actions = list(range(len(synthesisOpToPosDic)))
                random.shuffle(possible_actions)  # 随机打乱动作顺序
                for action in possible_actions:
                    if any(child.action_sequence[-1] == action for child in node.children):
                        continue
                    child_node = MCTSNode(initial_aig, node, action)
                    node.add_child(child_node)
                    node = child_node  # 更新当前节点为新扩展的子节点
                    path.append(node)
                    expanded = True
                    break  # 扩展一个子节点后，继续下一步扩展
                if not expanded:
                    break  # 如果没有扩展，退出扩展循环

        # Simulation
        eval_node = path[-1]  # 使用路径中的最后一个节点进行评估
        actions = torch.tensor(eval_node.action_sequence, dtype=torch.long)
        try:
            score = predict(initial_aig, actions, aig_model, device).item()
        except Exception as e:
            score = -float('inf')
            print(f"Prediction failed for actions {eval_node.action_sequence}: {e}")

        # Backpropagation
        while node:
            node.visits += 1
            node.score = max(node.score, score)
            node = node.parent

    # 获取最佳操作序列
    best_sequence = []
    node = root
    while node.children:
        best_child = node.best_child(exploration_weight=0)
        best_sequence.append(best_child)
        node = best_child

    best_action_sequence = [child.action_sequence[-1] for child in best_sequence if child.action_sequence]
    best_score = root.score

    return best_action_sequence, best_score

test_files = [
    'alu4.aig', 'apex1.aig', 'apex2.aig', 'apex4.aig', 'b9.aig', 
    'bar.aig', 'c880.aig', 'c7552.aig', 'cavlc.aig', 'div.aig', 
    'i9.aig', 'm4.aig', 'max1024.aig', 'mem_ctrl.aig', 'pair.aig', 
    'prom1.aig', 'router.aig', 'sqrt.aig', 'square.aig', 'voter.aig'
]

results = {}

for test_file in test_files:
    initial_aig = all_aig_tensor_data[test_file]
    best_action_sequence, best_score = mcts(initial_aig)
    results[test_file] = {
        'best_action_sequence': best_action_sequence,
        'best_score': best_score
    }
    
    with open(result_file, 'w') as outfile:
        json.dump(results, outfile, indent=4)
    
    print(f"File: {test_file}, Best Action Sequence: {best_action_sequence}, Best Score: {best_score}")
