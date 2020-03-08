import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import IPython
import time

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from IPython.core.debugger import set_trace

def get_mask_from_input(x):
    return ~torch.isnan(x).any(-1)

def Linear(input_dim, output_dim, act_fn='leaky_relu', init_weight_uniform=True):
    gain = torch.nn.init.calculate_gain(act_fn)
    fc = torch.nn.Linear(input_dim, output_dim)
    if init_weight_uniform:
        nn.init.xavier_uniform_(fc.weight, gain=gain)
    else:
        nn.init.xavier_normal_(fc.weight, gain=gain)
    nn.init.constant_(fc.bias, 0.00)
    return fc
    
class DDRQN(nn.Module):

    def __init__(self, input_dim, output_dim, mlp_layer_size=32, rnn_layer_num=1, rnn_h_size=256, GRU=False, **kwargs):
        super(DDRQN, self).__init__()

        self.fc1 = Linear(input_dim, mlp_layer_size)
        self.fc2 = Linear(mlp_layer_size, rnn_h_size)
        if GRU:
            self.rnn = nn.GRU(rnn_h_size, hidden_size=rnn_h_size, num_layers=rnn_layer_num, batch_first=True)
        else:
            self.rnn = nn.LSTM(rnn_h_size, hidden_size=rnn_h_size, num_layers=rnn_layer_num, batch_first=True)
        self.fc3 = Linear(rnn_h_size, mlp_layer_size)
        self.fc4 = Linear(mlp_layer_size, output_dim, act_fn='linear')
       
    def forward(self, x, h=None):
        xx = pad_sequence(x, padding_value=torch.tensor(float('nan')), batch_first=True)
        mask = get_mask_from_input(xx)
        x = pad_sequence(x, padding_value=torch.tensor(0.0), batch_first=True)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        x = pack_padded_sequence(x, mask.sum(1), batch_first=True, enforce_sorted=False)
        x, h = self.rnn(x, h)
        x = pad_packed_sequence(x, padding_value=torch.tensor(0.0), batch_first=True)[0]

        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        x = x[mask]

        return x, h

class Cen_DDRQN(nn.Module):

    def __init__(self, input_dim, output_dim, shared_RNN=None, **kwargs):
        super(Cen_DDRQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = Linear(input_dim, shared_RNN.input_dim)
        self.lstm = shared_RNN
        self.fc2 = Linear(shared_RNN.rnn_h_size, 128)
        self.fc3 = Linear(128, output_dim)

    def forward(self, x, h=None):
        # split, pad and then pack sequence
        x = pad_sequence(x, padding_value=torch.tensor(float('nan')), batch_first=True)
        batch_size, trace_len, _ = x.shape
        mask = get_mask_from_input(x)
        x = F.relu(self.fc1(x[mask]))
        
        shape = batch_size, trace_len, self.lstm.input_dim
        xx = torch.full(shape, torch.tensor(float('nan')))
        xx[mask] = x
        x = xx

        x = pack_padded_sequence(x, mask.sum(1), batch_first=True)

        x, h = self.lstm(x, h)
        # unpack padded sequence 
        x = pad_packed_sequence(x, padding_value=torch.tensor(float('nan')), batch_first=True)[0]
        # remove 'nan' tensor
        x = x[mask]
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x, h

class Shared_RNN(nn.Module):

    def __init__(self, rnn_input_dim=None, rnn_layer_num=1, rnn_h_size=256, **kwargs):
        super(Shared_RNN, self).__init__()
        self.input_dim = rnn_input_dim
        self.rnn_h_size = rnn_h_size

        self.lstm = nn.LSTM(rnn_input_dim, hidden_size=rnn_h_size, num_layers=rnn_layer_num, batch_first=True)

    def forward(self, x, h):
        x, h = self.lstm(x, h)
        return x, h

class DDRQN_compare(nn.Module):

    def __init__(self, input_dim, output_dim, rnn_layer_num=1, rnn_h_size=64, **kwargs):
        super(DDRQN_compare, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn_h_size = rnn_h_size

        self.fc1 = Linear(input_dim, 32)
        self.fc2 = Linear(32, rnn_h_size)

        self.lstm = nn.LSTM(rnn_h_size, hidden_size=rnn_h_size, num_layers=rnn_layer_num, batch_first=True)

        self.fc3 = Linear(rnn_h_size, 32)
        self.fc4 = Linear(32, output_dim)
       
    def forward(self, x, h=None):
        # pad and then pack sequence
        x = pad_sequence(x, padding_value=torch.tensor(float('nan')), batch_first=True)
        batch_size, trace_len, _ = x.shape
        mask = get_mask_from_input(x)
        x = x[mask]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # change the shape of x to match (batch_size, trace_len, -1)
        shape = batch_size, trace_len, self.rnn_h_size
        xx = torch.full(shape, torch.tensor(float('nan')))
        xx[mask] = x
        x = xx

        x = pack_padded_sequence(x, mask.sum(1), batch_first=True)
        x, h = self.lstm(x, h)
        # unpack padded sequence 
        x = pad_packed_sequence(x, padding_value=torch.tensor(float('nan')), batch_first=True)[0]
        # remove 'nan' tensor
        x = x[mask]
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x, h

class DDQN(nn.Module):

    def __init__(self, input_dim, output_dim, **kwargs):
        super(DDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = Linear(input_dim, 16)
        #self.fc2 = Linear(16, 64)

        #self.fc3 = Linear(64, 32)
        self.fc4 = Linear(16, output_dim, act_fn='linear')
       
    def forward(self, x, h=None):
        # pad and then pack sequence
        x = pad_sequence(x, padding_value=torch.tensor(float('nan')), batch_first=True)
        mask = get_mask_from_input(x)

        x = x[mask]
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

class BN_DDRQN(nn.Module):
    
    def __init__(self, input_dim, output_dim, rnn_layer_num=1, rnn_h_size=256, GRU=False, **kwargs):
        super(BN_DDRQN, self).__init__()

        self.fc1 = Linear(input_dim, 32)
        self.fc2 = Linear(32, rnn_h_size)
        if GRU:
            self.rnn = nn.GRU(rnn_h_size, hidden_size=rnn_h_size, num_layers=rnn_layer_num, batch_first=True)
        else:
            self.rnn = nn.LSTM(rnn_h_size, hidden_size=rnn_h_size, num_layers=rnn_layer_num, batch_first=True)
        self.fc3 = Linear(rnn_h_size, 32)
        self.fc4 = Linear(32, output_dim, act_fn='linear')


        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(rnn_h_size)
        self.bn3 = nn.BatchNorm1d(rnn_h_size)
        self.bn4 = nn.BatchNorm1d(32)
       
    def forward(self, x, h=None):
        xx = pad_sequence(x, padding_value=torch.tensor(float('nan')), batch_first=True)
        mask = get_mask_from_input(xx)
        x = pad_sequence(x, padding_value=torch.tensor(0.0), batch_first=True)

        x = F.leaky_relu(self.bn1(self.fc1(x).permute(0,2,1)).permute(0,2,1))
        x = F.leaky_relu(self.bn2(self.fc2(x).permute(0,2,1)).permute(0,2,1))

        x = pack_padded_sequence(x, mask.sum(1), batch_first=True, enforce_sorted=False)
        x, h = self.rnn(x, h)
        x = pad_packed_sequence(x, padding_value=torch.tensor(0.0), batch_first=True)[0]

        x = F.leaky_relu(self.bn4(self.fc3(self.bn3(x.permute(0,2,1)).permute(0,2,1)).permute(0,2,1)).permute(0,2,1))
        x = self.fc4(x)
        x = x[mask]

        return x, h






