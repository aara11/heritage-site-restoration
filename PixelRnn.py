import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

hidden_dim = 40
num_layers = 7
batch_size = 16


class RowLSTM(nn.Module):
    def __init__(self, input_size, input_dim, batch_first=False, bias=True, return_all_layers=False):
        super(RowLSTM, self).__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.batch_first = batch_first
        self.bias = bias
        cell_list = []
        self.row_conv_c = nn.Conv2d(hidden_dim, hidden_dim, [1, 3], stride=1, padding=[0, 1])
        self.row_conv_h = nn.Conv2d(hidden_dim, hidden_dim, [1, 3], stride=1, padding=[0, 1])
        for i in range(0, num_layers):
            cur_input_dim = self.input_dim if i == 0 else hidden_dim
            cell_list.append(RowLSTMCell(input_size=(self.height, self.width),
                                         input_dim=cur_input_dim,
                                         bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_out = nn.Conv2d(hidden_dim, 256, 1, stride=1, padding=0)
        self.conv_i_s = nn.Conv2d(1, 2 * hidden_dim, 7, stride=1, padding=3)

    def forward(self, input_tensor, target_tensor):
        seq_len = input_tensor.size(2)
        i_s_conv = conv_2d(7, 1, 2 * hidden_dim)
        i_s_target = i_s_conv(target_tensor)
        i_s_input = self.conv_i_s(input_tensor)
        i_s = torch.cat([i_s_input, i_s_target], dim=1)
        h = torch.zeros(batch_size, hidden_dim, 1, self.width)
        c = torch.zeros(batch_size, hidden_dim, 1, self.width)
        hidden_states = []
        for t in range(seq_len):
            i_s_curr = i_s[:, :, t:t + 1, :]
            h = self.row_conv_h(h)
            c = self.row_conv_c(c)
            for layer_idx in range(num_layers):
                h, c = self.cell_list[layer_idx](i_s_curr, [h, c])
            hidden_states.append(h)
        hidden_tensor = torch.cat(hidden_states, dim=2)
        output = self.conv_out(hidden_tensor)
        output = F.relu(output)
        return output

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class RowLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, bias):
        super(RowLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.height, self.width = input_size
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=1, padding=0, stride=1,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        h_conv = self.conv(h_cur)
        combined_conv = input_tensor + h_conv  # torch.cat([input_tensor, h_conv], dim=1)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, hidden_dim, self.height, self.width)),
                Variable(torch.zeros(batch_size, hidden_dim, self.height, self.width)))


class conv_2d(nn.Module):
    def __init__(self, kernel_size, dim_in, dim_out, type_='A'):
        super(conv_2d, self).__init__()
        self.weight = torch.tensor(torch.rand(dim_out, dim_in, kernel_size, kernel_size, requires_grad=True))
        self.bias = torch.tensor(torch.ones(dim_out), requires_grad=True)
        self.padding = kernel_size // 2
        self.type_ = type_

    def forward(self, input_tensor):
        self.weight[:, :, self.padding + 1:, :] = 0
        self.weight[:, :, self.padding, self.padding + 1:] = 0
        if self.type_ == 'A':
            self.weight[:, :, self.padding, self.padding] = 0
        out = F.conv2d(input=input_tensor, weight=self.weight, bias=self.bias, stride=1, padding=self.padding)
        return out
