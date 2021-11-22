import torch
import torch.nn as nn
from torch.nn import GRU, Dropout, ReLU
from torch.autograd import Variable as Var
from torch.nn.utils.rnn import pad_packed_sequence
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool
from torch.nn.utils.rnn import pack_sequence


class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, dropout1):
        # in_dim is the input dim and mem_dim is the output dim
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.drop = nn.Dropout(dropout1)

    def node_forward(self, inputs, child_c, child_h):
        inputs = torch.unsqueeze(inputs, 0)
        child_h_sum = torch.sum(child_h, dim=0)
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(self.fh(child_h) + self.fx(inputs).repeat(len(child_h), 1))
        fc = torch.mul(f, child_c)
        c = torch.mul(i, u) + torch.sum(fc, dim=0)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def forward(self, data):
        tree = data[0]
        inputs = data[1]
        # The inputs here are the tree structure built from class Tree and the input is a list of values with the
        # node ids as the key to store the tree values
        _ = [self.forward([tree.children[idx], inputs]) for idx in range(tree.num_children)]

        if tree.num_children == 0:
            child_c = Var(inputs[tree.id].data.new(1, self.mem_dim).fill_(0.))
            child_h = Var(inputs[tree.id].data.new(1, self.mem_dim).fill_(0.))
        else:
            child_c, child_h = zip(*map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
        tree.state = self.node_forward(inputs[tree.id], child_c, child_h)
        return tree.state


class Vulnerability(torch.nn.Module):
    def __init__(self, h_size, num_node_feature, num_classes, feature_representation_size, drop_out_rate, num_conv_layers):
        super(Vulnerability, self).__init__()
        self.h_size = h_size
        self.num_node_feature = num_node_feature
        self.num_classes = num_classes
        self.feature_representation_size = feature_representation_size
        self.drop_out_rate = drop_out_rate
        self.layer_num = num_conv_layers
        # The 1th feature input (tree)
        self.tree_lstm = ChildSumTreeLSTM(self.feature_representation_size, self.h_size, self.drop_out_rate)
        # The 2th feature input (sequence)
        self.gru_1 = GRU(input_size=self.feature_representation_size, hidden_size=self.h_size, batch_first=True)
        # The 3th feature input (sequence)
        self.gru_2 = GRU(input_size=self.feature_representation_size, hidden_size=self.h_size, batch_first=True)
        # The 4th feature input (sequence)
        self.gru_3 = GRU(input_size=self.feature_representation_size, hidden_size=self.h_size, batch_first=True)
        # The 5th feature input (sequence)
        self.gru_4 = GRU(input_size=self.feature_representation_size, hidden_size=self.h_size, batch_first=True)
        # This layer is the bi-directional GRU layer
        self.gru_combine = GRU(input_size=self.h_size, hidden_size=self.h_size, bidirectional=True, batch_first=True)
        self.dropout = Dropout(self.drop_out_rate)
        # This layer is the GCN Model layer
        # h_size : in channels.  self.num_classes: out channels(2)
        self.connect = nn.Linear(self.h_size * self.num_node_feature * 2, self.h_size)
        for i in range(self.layer_num):
            if i < self.layer_num - 1:
                exec('self.conv_{} = GCNConv(self.h_size, self.h_size)'.format(i))
            if i == self.layer_num - 1:
                exec('self.conv_{} = GCNConv(self.h_size, self.num_node_feature)'.format(i))
        self.relu = ReLU(inplace=True)

    def forward(self, my_data, edge_index):
        # Input data format: a list that contains main graph, feature 1, ..., feature 5. The feature 1 is tree
        # structured and features 2-5 are sequences.
        feature_1 = my_data[1]
        feature_2 = my_data[0]
        feature_3 = my_data[2]
        feature_4 = my_data[3]
        feature_5 = my_data[4]
        feature_vec1 = None
        # for every statement, get its AST subtree
        for i in range(len(feature_1)):
            if i == 0:
                _, feature_vec1 = self.tree_lstm(feature_1[i])
            else:
                _, feature_vec_temp = self.tree_lstm(feature_1[i])
                feature_vec1 = torch.cat((feature_vec1, feature_vec_temp), 0)
        feature_vec1 = torch.reshape(feature_vec1, (-1, 1, self.h_size))

        # pack for feature 2-5 is done in gen_graphs
        feature_2 = pack_sequence(feature_2, enforce_sorted=False)
        feature_2, _ = self.gru_1(feature_2.float())
        feature_2, out_len = pad_packed_sequence(feature_2, batch_first=True)

        feature_3 = pack_sequence(feature_3, enforce_sorted=False)
        feature_3, _ = self.gru_2(feature_3.float())
        feature_3, out_len = pad_packed_sequence(feature_3, batch_first=True)

        feature_4 = pack_sequence(feature_4, enforce_sorted=False)
        feature_4, _ = self.gru_3(feature_4.float())
        feature_4, out_len = pad_packed_sequence(feature_4, batch_first=True)

        feature_5 = pack_sequence(feature_5, enforce_sorted=False)
        feature_5, _ = self.gru_4(feature_5.float())
        feature_5, out_len = pad_packed_sequence(feature_5, batch_first=True)

        feature_input = torch.cat(
            (feature_vec1, feature_2[:, -1:, :], feature_3[:, -1:, :], feature_4[:, -1:, :], feature_5[:, -1:, :]), 1)

        feature_vec, _ = self.gru_combine(feature_input)
        feature_vec = self.dropout(feature_vec)
        feature_vec = torch.flatten(feature_vec, 1)
        feature_vec = self.connect(feature_vec)
        for i in range(self.layer_num):
            if i < self.layer_num - 1:
                feature_vec = eval('self.conv_{}(feature_vec, edge_index)'.format(i))
                feature_vec = self.relu(feature_vec)
            if i == self.layer_num - 1:
                conv_output = eval('self.conv_{}(feature_vec, edge_index)'.format(i))
        pooled = global_max_pool(conv_output, torch.zeros(conv_output.shape[0], dtype=int, device=conv_output.device))
        pooled = nn.Softmax(dim=1)(pooled)
        return pooled
