import torch

class Swp1d(torch.nn.Module):
    '''

    '''

    def __init__(self, bs, in_features, out_features, bias=False):
        super(Swp1d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(bs, in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)

        torch.nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, input):
        y = torch.bmm(input, self.weight)
        return y