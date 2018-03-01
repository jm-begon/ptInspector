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
import datetime
import time
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from pt_inspector import ProgressTracker

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


class RandomDataset(Dataset):

    def __len__(self):
        return N * 100

    def __getitem__(self, item):
        x = torch.randn(N, D_in)
        y = torch.randn(N, D_out)
        return x, y


if __name__ == '__main__':
    model = TwoLayerNet(D_in, H, D_out)
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    data_loader = DataLoader(RandomDataset(), batch_size=N)

    # Create a ProgressTracker to monitor learning
    data_loader = ProgressTracker(data_loader)
    for epoch in range(5):
        start = time.time()
        data_loader.set_label("Epoch {}".format(epoch))
        for x, y in data_loader:
            x = Variable(x)
            y = Variable(y, requires_grad=False)

            y_pred = model(x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch", epoch, "ad hoc duration:",
              datetime.timedelta(seconds=time.time()-start))








