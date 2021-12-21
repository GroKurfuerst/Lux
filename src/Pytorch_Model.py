import torch.nn as nn
from torch.nn import functional as f
from torch import as_tensor, cat, argmax, ones, tensor, mean as pt_mean, stack, log, zeros, no_grad, load, save
from torch.optim import Adam
from torch.distributions import Categorical
import torch


class NN(nn.Module):
    def __init__(self,
                 input_size=9,
                 out_channels=32,
                 kernel_size=2,
                 hidden_layers=10,  # 这个感觉太多了
                 conv_output=3200,
                 w_output_dim=7,
                 ct_output_dim=2,
                 kt_output_dim=2,
                 ):
        super().__init__()
        self.out_channels = out_channels
        self.w_output_dim = w_output_dim
        self.ct_output_dim = ct_output_dim
        self.kt_output_dim = kt_output_dim

        # self.linear = nn.Linear()

        self.to_conv2d = nn.Conv2d(in_channels=input_size,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size)
        self.conv2d = nn.Conv2d(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size)

        self.conv2ds = nn.ModuleList(
            [self.conv2d for _ in range(hidden_layers)])
        # self.shape_helper = ones((1, 32, 32))

        self.helper = conv_output

        self.to_w = nn.Linear(self.helper, 64)
        self.w_output = nn.Linear(64, self.w_output_dim)

        self.to_ct = nn.Linear(self.helper, 64)
        self.ct_output = nn.Linear(64, self.ct_output_dim)

        self.to_kt = nn.Linear(self.helper, 64)
        self.kt_output = nn.Linear(64, self.kt_output_dim)

    def forward(self, x):
        """Forward/Predict"""
        step_size = x.shape[0]
        x = f.selu(self.to_conv2d(x))

        for conv2d in self.conv2ds:
            x = f.max_pool2d(f.selu(conv2d(x)), kernel_size=2, stride=1)

        # TODO 看 12， 16， 24， 32这里怎么改
        x = x.reshape(step_size, x.shape[1] * x.shape[2] * x.shape[3])

        x_w = self.to_w(x)
        logits = self.w_output(x_w).reshape(step_size, -1)
        w_probs = f.softmax(logits, dim=-1).squeeze()
        w_action = Categorical(w_probs).sample()

        x_ct = self.to_ct(x)
        logits = self.ct_output(x_ct).reshape(step_size, -1)
        ct_probs = f.softmax(logits, dim=-1).squeeze()
        ct_action = Categorical(ct_probs).sample()

        return w_probs, w_action, ct_probs, ct_action


class DQN(nn.Module):
    def __init__(self, size, w_output_dim=7, ct_output_dim=2, kt_output_dim=2):
        super().__init__()
        self.squeeze_dim = nn.Conv2d(9, 3, kernel_size=(1, 1))
        self.convs1 = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=(1, 1))
        self.convs2 = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=(1, 1))

        feature_size = (size - 2)**2 * 3
        self.to_w = nn.Linear(feature_size, 64)
        self.w_output = nn.Linear(64, self.w_output_dim)

        self.to_ct = nn.Linear(feature_size, 64)
        self.ct_output = nn.Linear(64, self.ct_output_dim)

        self.to_kt = nn.Linear(feature_size, 64)
        self.kt_output = nn.Linear(64, self.kt_output_dim)

    def forward(self, inputs):

        x = inputs
        x = self.squeeze_dim(x)
        x = f.relu(self.convs1(x))
        x = f.max_pool2d(x, kernel_size=2, stride=1)
        x = f.relu(self.convs2(x))
        x = f.max_pool2d(x, kernel_size=2, stride=1)

        x = f.relu(self.mlp(x))

        w_qvalue = self.w_output(self.to_w(x))
        c_qvalue = self.ct_output(self.to_ct(x))
        kt_qvalue = self.kt_output(self.to_kt(x))

        return w_qvalue, c_qvalue, kt_qvalue

