import datetime
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torch.nn import Module


def var2np(variable):
    return variable.cpu().data.numpy()


class StreamingStat(object):
    """
    RunningStat
    ===========
    Computes the mean and std of some quantity given by batches of tensors
    """
    def __init__(self):
        self.size = 0
        self.first_mean = 0
        self.first_var = 0
        self.last_mean = 0
        self.last_var = 0
        self.running_mean = 0
        self.running_square_mean = 0
        self.running_var = 0  # Running within "addition" variance

    def add(self, np_tensor):
        # New values
        mean = np_tensor.mean()
        mean_sq = mean ** 2
        var = np_tensor.var()
        # Save last
        self.last_mean = mean
        self.last_var = var

        size = np.prod(np_tensor.shape)
        if self.size == 0:
            # First capture
            self.first_mean = self.last_mean
            self.first_var = self.last_var

        # Running stuff
        size_ratio_correction = self.size / float(self.size + size)
        size_ratio = size / float(self.size + size)
        self.size += size

        self.running_mean = mean * size_ratio \
                + self.running_mean * size_ratio_correction
        self.running_square_mean = mean_sq * size_ratio \
                + self.running_square_mean * size_ratio_correction
        self.running_var = var * size_ratio \
                + self.running_var * size_ratio_correction

    def get_running(self):
        btw_var = self.running_square_mean - self.running_mean**2
        wth_var = self.running_var
        return self.running_mean, np.sqrt(btw_var + wth_var)

    def get_first(self):
        return self.first_mean, np.sqrt(self.first_var)

    def get_last(self):
        return self.last_mean, np.sqrt(self.last_var)

    def reset(self):
        self.size = 0


# ================================== ANALYZER ================================ #
class Analyzer(object, metaclass=ABCMeta):
    def analyze(self, last=True):
        if self.empty():
            if last:
                self.footline()
            return
        self.headline()
        self.title()
        self._analyze()
        if last:
            self.line()
        else:
            self.footline()

    @abstractmethod
    def _analyze(self):
        """
        Produce and print the analysis
        """
        pass

    def line(self):
        print("+", "-" * 78, "+", sep="")

    def headline(self):
        print("+", "-" * 78, "+", sep="")
        pass

    def footline(self):
        print("|", " "*78, "|", sep="")

    def title(self):
        pass

    def empty(self):
        """If empty return True, it means there is nothing to analyze"""
        return False

    def print(self, line):
        print("| {:<77}|".format(line))


# ================================== MONITOR ================================= #
class Monitor(Analyzer, metaclass=ABCMeta):
    """
    Monitor
    =======
    Base class for `Monitor`. A monitor keeps track of some network-related
    quantity (change of weights, gradients after backward passes, loss values,
    etc.).

    The variable(s) (or module(s)) must first be registered. Then a report
    is printed on the standard output each time the :meth:`analyze` is used.

    The span of activity is specific to each `Monitor`.
    """
    def __call__(self, module_or_variable, name=None):
        return self.register(module_or_variable, name)

    @abstractmethod
    def register(self, to_be_registered, label):
        """
        Register the thing of interest
        """
        pass


class PseudoMonitor(Monitor):
    # Easy way to deactivate monitoring
    def register(self, to_be_registered, label):
        pass

    def analyze(self):
        pass

    def _analyze(self):
        pass


# =============================== MONITOR KINDS ============================== #
class ModelMonitor(Monitor, metaclass=ABCMeta):
    """
    `ModelMonitor`
    --------------
    Base class for registering models (instances of torch.nn.Module)
    """
    @abstractmethod
    def register(self, to_be_registered: Module, label):
        pass


class VariableMonitor(Monitor, metaclass=ABCMeta):

    @abstractmethod
    def register(self, to_be_registered: torch.autograd.Variable, label):
        pass

    def register_model(self, model: Module):
        for label, parameter in model.named_parameters():
            self.register(parameter, label)
        return self


class Datafeed(Monitor, metaclass=ABCMeta):

    def __init__(self):
        self._registered = {}

    def empty(self):
        return len(self._registered) == 0

    def register(self, data_loader, label=None):
        """
        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
            The loader containing the data
        label: str (Default: None)
            The label of the loader. If None, default name will be provided
        """
        if label is None:
            label = str(id(data_loader))

        self._registered[label] = data_loader
        return self


# =============================== BASE MONITOR =============================== #
class MetricMonitor(Monitor):
    """
    `MetricMonitor`
    ===============
    The `to_be_registered` argument must be tuple (loss, accuracy, size of data)
    """
    def __init__(self):
        super().__init__()
        self.scalars = {}

    def register(self, to_be_registered, label):
        self.scalars[label] = to_be_registered
        return self

    def _analyze(self):
        mask = "{{:<15}}{0:13}{{:^15}}{0:13}{{:^15}}".format(" ")
        self.print(mask.format("Label", "Accuracy", "Avg. cross entropy"))
        for label, (loss, accuracy, size) in self.scalars.items():
            correct = int(accuracy*size)
            self.print(mask.format(label,
                                   "{}/{} ({:.2f}%)".format(correct,
                                                            size,
                                                            accuracy),
                                   "{:.2E}".format(loss)))

    def title(self):
        self.print("Loss and accuracy")
        self.line()

    def empty(self):
        return len(self.scalars) == 0


# ============================= VARIABLE MONITOR ============================= #
class WeightMonitor(VariableMonitor):
    """
    WeightMonitor
    =============
    Monitor the evolution of weights:
    - How much the weights have changed since the last analysis (i.e. L2
    distance of weights between 2 analyses, avg + std per layer)
    - Magnitude of the weights (i.e. min/max absolute value of the weight per
    layers)
    """
    def __init__(self):
        super().__init__()
        self._var_and_weight = {}  # name --> tuples (t_variable, np_weight)

    def register(self, variable: torch.autograd.Variable, label):
        d_entry = self._var_and_weight.get(label)
        if d_entry is None:
            np_weight = var2np(variable).copy()
        else:
            _, np_weight = d_entry
        self._var_and_weight[label] = variable, np_weight
        return self

    def title(self):
        self.print("Weights: L2 distance from previous and smallest/largest |w|")
        self.line()

    def _analyze(self):
        mask = "{{:<19}}{{:^23}}{0:7}{{:^8}}{0:7}{{:^8}}".format(" ")
        self.print(mask.format("Var. name", "L2 Dist", "Smallest",
                               "Largest"))
        for name, (variable, weight) in self._var_and_weight.items():
            current_weight = var2np(variable)
            dist = (current_weight - weight) ** 2
            abs_weight = np.abs(current_weight)
            self.print(mask.format(name,
                                   "{:.2E}  +/- {:.2E}".format(dist.mean(),
                                                               dist.std()),
                                   "{:.2E}".format(abs_weight.min()),
                                   "{:.2E}".format(abs_weight.max())))

    def empty(self):
        return len(self._var_and_weight) == 0


class StatMonitor(VariableMonitor):
    """
    StatMonitor
    ===========
    Generic class to monitor some variables. Reuses the :meth:`register` method
    to track the state of variables. Useful in combination with the iterative
    nature of network training
    """
    def __init__(self):
        super().__init__()
        self._running_stats = defaultdict(StreamingStat)

    def register(self, variable: torch.autograd.Variable, label):
        self._running_stats[label].add(var2np(variable))
        return self

    def _analyze(self):
        mask = "{{:<19}}{{:^23}}{0:7}{{:^8}}{0:7}{{:^8}}".format(" ")
        self.print(mask.format("Var. name", "On average", "First it.",
                               "Last it."))
        for name, running_stat in self._running_stats.items():
            avg_mean, avg_std = running_stat.get_running()
            avg_first, _ = running_stat.get_first()
            avg_last, _ = running_stat.get_last()
            self.print(mask.format(name,
                                   "{:.2E}  +/- {:.2E}".format(avg_mean,
                                                               avg_std),
                                   "{:.2E}".format(avg_first),
                                   "{:.2E}".format(avg_last)))

    def title(self):
        self.print("Statistic monitoring (Mean/[Std])")
        self.line()


class GradientMonitor(StatMonitor):
    """
    GradientMonitor
    ===============
    Monitor average square partial derivative. Use the hook mechanism of Pytorch
    """
    def __init__(self):
        super().__init__()
        self._running_stats = defaultdict(StreamingStat)

    def create_hook(self, name):
        def magnitude_gradient_hook(variable):
            self._running_stats[name].add(var2np(variable)**2)
        return magnitude_gradient_hook

    def register(self, variable: torch.autograd.Variable, label):
        variable.register_hook(self.create_hook(label))
        return self

    def title(self, duration="average"):
        print("| Mean gradient magnitude ({})"
              "".format(duration).ljust(79),
              "|", sep="")
        self.line()

    def empty(self):
        return len(self._running_stats) == 0


# ============================= DATAFEED MONITOR ============================= #
class MetricDatafeed(Datafeed):
    """
    Constructor parameters
    ----------------------
    model: torch.NN.module
        the model is expected to output raw predictions (no softmax layer)

    hook: callable (loss, accuracy) --> Nothing
    """
    def __init__(self, model, use_cuda=False, hook=None):
        super().__init__()
        self.model = model
        self.use_cuda = use_cuda
        self.hook = hook

    def title(self):
        self.print("Loss and accuracy")
        self.line()

    def _compute_loss_acc(self, data_loader):
        correct = 0
        loss = 0
        size = 0
        for data, target in data_loader:
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            size += data.size(0)
            output = self.model(data)
            # sum up batch loss
            loss += F.cross_entropy(output, target, size_average=False).data[0]
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        loss /= size
        accuracy = correct / size
        return loss, accuracy

    def _analyze(self):
        mask = "{{:<30}}{0:10}{{:^10}}{0:10}{{:^10}}}".format(" ")
        self.print(mask.format("Label", "Accuracy [%]", "Avg. cross entropy"))
        training = self.model.training
        try:
            self.model.eval()
            for label, data_loader in self._registered.items():
                loss, accuracy = self._compute_loss_acc(data_loader)
                if self.hook:
                    self.hook(label, loss, accuracy)
                self.print(mask.format(label, accuracy, loss))

        finally:
            if training:
                self.model.train()
            else:
                self.model.eval()


# ================================= PROGRESS ================================= #
class ProgressTracker(DataLoader):

    def __init__(self, data_loader, label="", update_rate=0.1,
                 eta_decay_rate=.9):
        self.data_loader = data_loader
        self.label = label
        self.update_rate = update_rate
        self.decay_rate = eta_decay_rate
        self.dataset_size = len(data_loader.dataset)
        self.n_batches = len(data_loader)

    def set_label(self, label):
        self.label = label

    @property
    def dataset(self):
        return self.data_loader.dataset

    @property
    def batch_size(self):
        return self.data_loader.batch_size

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        print_it = False
        last_print_iteration = 0
        start = time.time()
        end = start
        average_speed = None
        for i, x in enumerate(self.data_loader):
            yield x

            i += 1
            # Should we print ?
            if (i-last_print_iteration)/self.n_batches >= self.update_rate:
                print_it = True

            # Do print if needed
            if print_it:
                # Compute estimated time of arrival
                now = time.time()
                duration = now - end
                current_speed = (i - last_print_iteration) / duration
                if average_speed is None:
                    average_speed = current_speed
                average_speed = self.decay_rate * average_speed \
                                + (1 - self.decay_rate) * current_speed
                eta = (self.n_batches - i) / average_speed

                # Print info
                self.print(i, i / self.n_batches, now-start, eta)
                
                # Reset everything
                last_print_iteration = i
                print_it = False
                end = now

        end = time.time()
        print(self.label, "| Duration:",
              datetime.timedelta(seconds=end - start))

    def print(self, iteration, ratio, elapsed, eta):
        mask = "{{}}    {:<20} Elapsed {{:>10}} | ETA {{:<13}}" \
               "".format("[{}/{} ({:.0f}%)]".format(iteration * self.batch_size,
                                                    self.dataset_size,
                                                    ratio * 100))
        line = mask.format(self.label,
                           str(datetime.timedelta(seconds=int(elapsed))),
                           str(datetime.timedelta(seconds=int(eta))))

        print("{:<80}".format(line))


# ================================= INSPECTOR ================================ #
class ModelInspector(Analyzer):
    """
    ModelInspector
    ==============
    Custom model inspector. Monitor the weights, the gradients and possibily
    the loss function.

    The `ModelInspector` relies on a pseudo-singleton pattern which allows to get
    an given instance at a different place in the code without keeping a global
    variable.
    """
    __instances = {}  # name -> ModelInspector

    @classmethod
    def get(cls, name):
        inspector = cls.__instances.get(name)
        if inspector is None:
            inspector = ModelInspector(name)
            cls.__instances[name] = inspector
        return inspector

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.weight_monitor = WeightMonitor()
        self.gradient_monitor = GradientMonitor()
        self.loss_monitor = StatMonitor()
        self.metric_monitor = MetricMonitor()
        self.monitors = [self.weight_monitor, self.gradient_monitor,
                         self.loss_monitor, self.metric_monitor]

    def register_model(self, model: Module):
        for monitor in self.weight_monitor, self.gradient_monitor:
            monitor.register_model(model)
        return self

    def monitor_loss(self, variable: torch.autograd.Variable, name):
        self.loss_monitor.register(variable, name)
        return self

    def monitor_metrics(self, label, loss, accuracy, size):
        self.metric_monitor.register((loss, accuracy, size), label)
        return self

    def headline(self):
        print("/", "=" * 30, " Model inspection ", "=" * 30, "\\", sep="")

    def title(self):
        print("|", self.name.center(78), "|", sep="")

    def footline(self):
        print("\\", "=" * 78, "/", sep="")

    def _analyze(self):
        for monitor in self.monitors:
            monitor.analyze(last=False)



