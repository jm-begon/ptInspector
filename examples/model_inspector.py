#!/usr/bin/env python

"""
The example below is taken from http://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-custom-nn-modules
where you can find in-depth comment about the standard PyTorch code.

We will focus here on the monitoring instead to illustrate the `ModelInspector`
class.

You should check gradient_monitor.py, loss_monitor.py and weight_monitor.py
first.

The `ModelInspector` relies on a pseudo-singleton pattern which allows to get
an given instance at a different place in the code without keeping a global
variable.
It tracks the weights and gradients and can also monitor the loss
"""

import torch
from torch.autograd import Variable

from pt_inspector import ModelInspector

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


def backward(model, x, y, criterion, optimizer):
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # (Re-)register the loss value
    ModelInspector.get("TwoLayerNet").loss_monitor(loss, "mse")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if __name__ == '__main__':
    model = TwoLayerNet(D_in, H, D_out)
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    # Use the pseudo-singleton form to create and get the ModelInspector
    # and resgister the model to monitor weights and gradients
    ModelInspector.get("TwoLayerNet").register_model(model)

    for epoch in range(5):
        for iteration in range(100):
            # Create batch
            x = Variable(torch.randn(N, D_in))
            y = Variable(torch.randn(N, D_out), requires_grad=False)

            backward(model, x, y, criterion, optimizer)

        # Print the whole analysis
        print("### Analysis at epoch", epoch)
        ModelInspector.get("TwoLayerNet").analyze()
        print()





