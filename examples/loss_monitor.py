#!/usr/bin/env python

"""
The example below is taken from http://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-custom-nn-modules
where you can find in-depth comment about the standard PyTorch code.

We will focus here on the monitoring instead to illustrate the `StatMonitor`
class. Note however that the design of the `StatMonitor` was to be integrated
to a `ModelInspector`.

The analysis consist in a summary (average + std and first + last)
of the loss values
"""

import torch
from torch.autograd import Variable

from pt_inspector import StatMonitor

N, D_in, H, D_out = 64, 1000, 100, 10


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


if __name__ == '__main__':
    model = TwoLayerNet(D_in, H, D_out)
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    # Create a StatMonitor for the loss. It can work on non-in-place
    # Variable by re-registering the Variable with the same name at each
    # iteration.
    monitor = StatMonitor()

    for epoch in range(5):
        for iteration in range(100):
            # Create batch
            x = Variable(torch.randn(N, D_in))
            y = Variable(torch.randn(N, D_out), requires_grad=False)

            y_pred = model(x)
            loss = criterion(y_pred, y)

            # (Re-)register the Variable with a fix name to keep track
            # of how it evolves between iteraitons
            monitor.register(loss, "TwoLayerNet.loss")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print the analyzis
        print("### Analysis at epoch", epoch)
        monitor.analyze()
        print("-" * 80, "\n\n")






