import torch
import torch.nn as nn
from torch.utils.data import Dataset


class DeepONet(nn.Module):
    def __init__(self, branch_input=100, trunk_input=1, num_p=40,
                 num_hidden_layer=2, num_neuron=40):
        super(DeepONet, self).__init__()
        self.branch_input = branch_input
        self.trunk_input = trunk_input
        self.branch_input_layer = nn.Linear(branch_input, num_neuron)
        self.branch_hidden_layer = nn.ModuleList([nn.Linear(num_neuron, num_neuron) for _ in range(num_hidden_layer)])
        self.branch_output_layer = nn.Linear(num_neuron, num_p)
        self.trunk_input_layer = nn.Linear(trunk_input, num_neuron)
        self.trunk_hidden_layer = nn.ModuleList([nn.Linear(num_neuron,num_neuron) for _ in range(num_hidden_layer)])
        self.trunk_output_layer = nn.Linear(num_neuron, num_p)
        self.activate = nn.ReLU()

    def forward(self, x):
        # branch_net
        x_branch = x[:, :self.branch_input]
        x_trunk = x[:, self.branch_input:]
        out1 = self.activate(self.branch_input_layer(x_branch))
        for layer in self.branch_hidden_layer:
            out1 = self.activate(layer(out1))
        out1 = self.branch_output_layer(out1)
        # trunk_net
        out2 = self.activate(self.trunk_input_layer(x_trunk))
        for layer in self.trunk_hidden_layer:
            out2 = self.activate(layer(out2))
        out2 = self.trunk_output_layer(out2)
        # output
        res = torch.sum(out1*out2, dim=-1, keepdim=True)
        return res


class CustomizedDataset(Dataset):
    def __init__(self, data):
        super(CustomizedDataset, self).__init__()
        self.data = data[0]
        self.label = data[1]

    def __getitem__(self, index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)

