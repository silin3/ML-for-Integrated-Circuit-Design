import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from transformers import BertModel, BertTokenizer


class AIGModel(nn.Module):
    def __init__(self):
        super(AIGModel, self).__init__()

        # Define GCN layers
        self.gcn1 = GCNConv(2, 32)
        self.gcn2 = GCNConv(32, 32)

        # Define batch normalization layers
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)

        # Define Transformer encoder
        self.transformer = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Define fully connected layers
        self.fc1 = nn.Linear(832, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        # Batch Normalization for FC layers
        # self.bn_fc1 = nn.BatchNorm1d(256)
        # self.bn_fc2 = nn.BatchNorm1d(256)

        # Define other layers
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, graphs, device='cuda'):
        # print("device: ", device)
        x, edge_index, batch, action_sequence = graphs.x, graphs.edge_index, graphs.batch, graphs.actions
        # 转化成torch.float32
        x = x.float()
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)
        action_sequence = action_sequence.to(device)
        action_sequence = action_sequence.reshape(-1, 10).long()

        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = self.leaky_relu(x)

        x = self.gcn2(x, edge_index)
        x = self.bn2(x)

        # Pooling
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        aig_embedding = torch.cat((mean_pool, max_pool), dim=1)

        # Prepare action_sequence for BERT
        action_sequence[action_sequence == -1] = self.tokenizer.pad_token_id
        attention_mask = (action_sequence !=
                          self.tokenizer.pad_token_id).long()

        # Transformer encoder forward pass
        sequence_outputs = self.transformer(
            input_ids=action_sequence, attention_mask=attention_mask)
        sequence_embedding = sequence_outputs['pooler_output']

        # Concatenate embeddings
        combined_embedding = torch.cat(
            (aig_embedding, sequence_embedding), dim=1)

        # Fully connected layers
        x = self.fc1(combined_embedding)
        # x = self.bn_fc1(x)
        # x = self.leaky_relu(x)

        x = self.fc2(x)
        # x = self.bn_fc2(x)
        # x = self.leaky_relu(x)

        output = self.fc3(x)

        return output


if __name__ == '__main__':
    model = AIGModel()
    print(model)
