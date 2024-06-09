import os
import re
import json
import subprocess
import numpy as np
import random
from math import sqrt, log

# 定义必要的路径
initial_aig_path = './InitialAIG/test/'
lib_file = './lib/7nm/7nm.lib'
log_file = 'temp.log'
result_file = 'results1.json'

# 定义操作字典
synthesisOpToPosDic = {
    0: "refactor",
    1: "refactor -z",
    2: "rewrite",
    3: "rewrite -z",
    4: "resub",
    5: "resub -z",
    6: "balance"
}

yosys_abc_path = '/usr/local/bin/yosys-abc'

def calculate_baseline(aig_file):
    RESYN2_CMD = "balance; rewrite; refactor; balance; rewrite -z; balance; refactor -z; rewrite -z; balance;"
    abcRunCmd = f"{yosys_abc_path} -c \"read {aig_file}; {RESYN2_CMD} read_lib {lib_file}; map; topo; stime\" > {log_file}"
    result = subprocess.run(abcRunCmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"yosys-abc with resyn2 failed with return code {result.returncode}")
    
    with open(log_file) as f:
        lines = f.readlines()
        if not lines:
            raise RuntimeError("Log file is empty after resyn2, yosys-abc might have failed")
        area_information = re.findall('[a-zA-Z0-9.]+', lines[-1])
        if len(area_information) < 10:
            raise RuntimeError("Unexpected log file format after resyn2")
    
    baseline = float(area_information[-9]) * float(area_information[-4])
    return baseline

def evaluation(aig_file, baseline):
    # 运行初始评估命令
    abcRunCmd = f"{yosys_abc_path} -c \"read {aig_file}; read_lib {lib_file}; map; topo; stime\" > {log_file}"
    result = subprocess.run(abcRunCmd, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"yosys-abc failed with return code {result.returncode}")
    
    with open(log_file) as f:
        lines = f.readlines()
        if not lines:
            raise RuntimeError("Log file is empty, yosys-abc might have failed")
        area_information = re.findall('[a-zA-Z0-9.]+', lines[-1])
        if len(area_information) < 10:
            raise RuntimeError("Unexpected log file format")
    
    eval_value = float(area_information[-9]) * float(area_information[-4])
    eval_normalized = 1 - eval_value / baseline

    return eval_normalized

class MCTSNode:
    def __init__(self, aig_file, parent=None):
        self.aig_file = aig_file
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = -float('inf')

    def is_fully_expanded(self):
        return len(self.children) == len(synthesisOpToPosDic)

    def add_child(self, child_node):
        self.children.append(child_node)

    def best_child(self, exploration_weight=1.414):
        choices_weights = [
            (child.score / (child.visits + 1e-8)) + exploration_weight * ((2 * np.log(self.visits + 1) / (child.visits + 1e-8))**0.5)
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

def mcts(aig_file, baseline, iterations=500, steps=10):
    root = MCTSNode(aig_file)
    
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
        for _ in range(steps - len(path)):
            if not node.is_fully_expanded():
                available_actions = list(range(len(synthesisOpToPosDic)))
                random.shuffle(available_actions)  # 随机打乱动作顺序
                expanded = False
                for action in available_actions:
                    if any(child.aig_file.endswith(f'_{action}.aig') for child in node.children):
                        continue
                    child_file = f"{node.aig_file.split('.')[0]}_{action}.aig"
                    action_cmd = synthesisOpToPosDic[action]
                    abcRunCmd = f"{yosys_abc_path} -c \"read {node.aig_file}; {action_cmd}; read_lib {lib_file}; write {child_file}; print_stats\" > {log_file}"
                    result = subprocess.run(abcRunCmd, shell=True)
                    if result.returncode != 0:
                        continue
                    child_node = MCTSNode(child_file, node)
                    node.add_child(child_node)
                    node = child_node  # 更新当前节点为新扩展的子节点
                    path.append(node)
                    expanded = True
                    break  # 扩展一个子节点后，继续向下扩展
            else:
                break  # 如果当前节点已经完全扩展，停止扩展

        # Simulation
        eval_node = path[-1]  # 使用路径中的最后一个节点进行评估
        try:
            score = evaluation(eval_node.aig_file, baseline)
        except Exception as e:
            score = -float('inf')
            print(f"Evaluation failed for file {eval_node.aig_file}: {e}")

        # Backpropagation
        while node:
            node.visits += 1
            node.score = max(node.score, score)  # Ensure score is the best found
            node = node.parent

    # 获取最佳操作序列
    best_sequence = []
    node = root
    while node.children:
        best_child = node.best_child(exploration_weight=0)
        best_sequence.append(best_child)
        node = best_child

    best_action_sequence = [int(child.aig_file.split('_')[-1].split('.')[0]) for child in best_sequence]
    best_score = best_sequence[-1].score if best_sequence else root.score  # The best score at the root after MCTS

    return best_action_sequence, best_score




# Process all test files
test_files = [f for f in os.listdir(initial_aig_path) if f.endswith('.aig')]
results = {}

for test_file in test_files:
    test_file_path = os.path.join(initial_aig_path, test_file)
    baseline = calculate_baseline(test_file_path)
    best_action_sequence, best_score = mcts(test_file_path, baseline)
    results[test_file] = {
        'best_action_sequence': [int(action) for action in best_action_sequence],
        'best_score': best_score
    }
    '''
    # Save results to JSON file
    with open(result_file, 'w') as outfile:
        json.dump(results, outfile, indent=4)
    ''' 
    # Print results
    print(f"File: {test_file}, Best Action Sequence: {best_action_sequence}, Best Score: {best_score}")


