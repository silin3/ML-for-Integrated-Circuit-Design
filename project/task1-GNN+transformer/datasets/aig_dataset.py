import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch_geometric.data import Data


class AIGDataset(Dataset):
    def __init__(self, aig_tensor, data):
        self.aig_tensor = aig_tensor
        self.data = data
        self.action_max_len = 10

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        actions = item['actions']
        actions = F.pad(actions, (0, self.action_max_len -
                        actions.size(0)), "constant", -1)

        label = item['label']
        initial_aig = self.aig_tensor[item['initial_aig']]

        node_type = initial_aig['node_type'].unsqueeze(1)
        num_inverted_predecessors = initial_aig['num_inverted_predecessors'].unsqueeze(
            1)
        edge_index = initial_aig['edge_index']
        nodes = initial_aig['nodes'].unsqueeze(0)

        graph = Data(x=torch.cat([node_type, num_inverted_predecessors], dim=1),
                     edge_index=edge_index,
                     num_nodes=nodes,
                     actions=actions,
                     y=label
                     )

        return graph


if __name__ == '__main__':
    aig_tensor = torch.load('data/origin/aig_tensor.pt')
    data = torch.load('data/alu2.pt')
    dataset = AIGDataset(aig_tensor, data)
    print(len(dataset))

    graph = dataset.__getitem__(5)
    print(graph.x.shape)
    print(graph.edge_index.shape)
    print(graph.num_nodes.shape)
    print(graph.actions.shape)
    print(graph.y)
