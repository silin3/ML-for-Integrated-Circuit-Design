import os
import abc_py
import numpy as np
import torch


def get_gragh_data(state):
    _abc = abc_py.AbcInterface()
    _abc.start()
    _abc.read(state)

    data = {}
    numNodes = _abc.numNodes()
    data['node_type'] = np.zeros(numNodes, dtype=int)
    data['num_inverted_predecessors'] = np.zeros(numNodes, dtype=int)
    edge_src_index = []
    edge_target_index = []

    for nodeIdx in range(numNodes):
        aigNode = _abc.aigNode(nodeIdx)
        nodeType = aigNode.nodeType()
        data['num_inverted_predecessors'][nodeIdx] = 0
        if nodeType == 0 or nodeType == 2:
            data['node_type'][nodeIdx] = 0
        elif nodeType == 1:
            data['node_type'][nodeIdx] = 1
        else:
            data['node_type'][nodeIdx] = 2
            if nodeType == 4:
                data['num_inverted_predecessors'][nodeIdx] = 1
            if nodeType == 5:
                data['num_inverted_predecessors'][nodeIdx] = 2
        if (aigNode.hasFanin0()):
            fanin = aigNode.fanin0()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
        if (aigNode.hasFanin1()):
            fanin = aigNode.fanin1()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)

    data['edge_index'] = torch.tensor(
        [edge_src_index, edge_target_index], dtype=torch.long)
    data['node_type'] = torch.tensor(data['node_type'])
    data['num_inverted_predecessors'] = torch.tensor(
        data['num_inverted_predecessors'])
    data['nodes'] = torch.tensor(numNodes)

    return data


if __name__ == "__main__":
    aig_tensor = {}
    initial_aig_list = [os.path.join('InitialAIG/train/', file) for file in os.listdir('InitialAIG/train/') if file.endswith('.aig')] + \
        [os.path.join('InitialAIG/test/', file)
         for file in os.listdir('InitialAIG/test/') if file.endswith('.aig')]

    for initial_aig in initial_aig_list:
        data = get_gragh_data(initial_aig)
        aig_tensor[os.path.basename(initial_aig)] = data
    print(aig_tensor.keys())
    print(aig_tensor['alu2.aig'])
    data_path = 'data/aig_tensor.pt'
    torch.save(aig_tensor, data_path)
