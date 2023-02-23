import torch.nn as nn
from collections import OrderedDict


class LeNetOutput(nn.Module):
    def __init__(self):
        super(LeNetOutput, self).__init__()
        act = nn.Sigmoid()
        self.body = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('layer', nn.Conv2d(1, 12, kernel_size=5, padding=2, stride=2, bias=False)),
                ('act', act)
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=2, bias=False)),
                ('act', act)
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=1, bias=False)),
                ('act', act)
            ])),
            nn.Sequential(OrderedDict([
                ('layer', nn.Linear(588, 1, bias=False)),
                ('act', nn.Identity())
            ]))
        ])

    def forward(self, x):
        input = []
        for layer in self.body:
            input.append(x)
            if isinstance(layer.layer, nn.Linear):
                x = x.flatten(1)
            x = layer(x)
        return x, input

    @staticmethod
    def name():
        return 'LeNet'
