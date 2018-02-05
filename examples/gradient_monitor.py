#!/usr/bin/env python

"""
The example below is taken from http://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-custom-nn-modules
where you can find in-depth comment about the standard PyTorch code.

We will focus here on the monitoring instead to illustrate the `GradientMonitor`
class. Note however that the design of the `GradientMonitor` was to be integrated
to a `ModelInspector`.

The analysis consist in a per layer summary (average + std and first + last)
since the last analysis of the square partial derivatives of the weights.
"""

import torch
from torch.autograd import Variable

from pt_inspector import GradientMonitor

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

    # Create weight monitor and register model
    # This monitor only works on "in place" tensors, such as those from
    # a module.
    monitor = GradientMonitor().register(model, "TwoLayerNet > gradient")

    for epoch in range(5):
        for iteration in range(100):
            # Create batch
            x = Variable(torch.randn(N, D_in))
            y = Variable(torch.randn(N, D_out), requires_grad=False)

            y_pred = model(x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print analysis. In the case of WeightMonitor, it will print a summary
        # (average + std) per layer of the L2 distance of the weights since
        # the last analysis.
        # It will also print the smallest and largest weight (in absolute value)
        # for each layer.
        # As a consequence of working with "in place" tensors, there is no
        # need to actively track the tensors by re-registering it, so that
        # the monitoring code is minimally invasive
        print("### Analysis at epoch", epoch)
        monitor.analyze()
        print("-" * 80, "\n\n")




