from abc import ABCMeta, abstractmethod
from collections import defaultdict

import numpy as np
import torch


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


class Monitor(object, metaclass=ABCMeta):
    """
    Monitor
    =======
    Base class for `Monitor`. A monitor keeps track of some network-related
    quantity (change of weights, gradients after backward passes, loss values,
    etc.).

    The variable(s) (or module(s)) must first be registered. Then a report
    is printed on the standard output each time the :meth:`analyze` is used.

    The span of activity is specific to each `Monitor`
    """
    # Dispatching
    def register(self, module_or_variable, name=None):
        if isinstance(module_or_variable, torch.autograd.Variable):
            # Variable or torch.nn.Parameter
            if name is None:
                raise ValueError("In the case of 'torch.autograd.Variable'"
                                 " a name must be supplied in order to track"
                                 " the parameter")
            # Storing values
            self._store_variable(module_or_variable, name)

        elif hasattr(module_or_variable, "named_parameters"):
            # torch.nn.Module
            for name, parameter in module_or_variable.named_parameters():
                self._store_variable(parameter, name)
        else:
            raise TypeError("Can only monitor 'torch.autograd.Variable'"
                            " or 'torch.nn.Module'")
        return self

    def __call__(self, module_or_variable, name=None):
        return self.register(module_or_variable, name)

    @abstractmethod
    def _store_variable(self, variable: torch.autograd.Variable, name):
        """
        Actually store the variable and all quantity of interest
        """
        pass

    def analyze(self):
        self.headline()
        self.title()
        self._analyze()
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

    def print(self, line):
        print("| {:<77}|".format(line))


class PseudoMonitor(Monitor):
    # Easy way to deactivate monitoring
    def _store_variable(self, variable: torch.autograd.Variable, name):
        pass

    def analyze(self):
        pass

    def _analyze(self):
        pass


class WeightMonitor(Monitor):
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

    def _store_variable(self, variable: torch.autograd.Variable, name):
        d_entry = self._var_and_weight.get(name)
        if d_entry is None:
            np_weight = var2np(variable).copy()
        else:
            _, np_weight = d_entry
        self._var_and_weight[name] = variable, np_weight

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


class StatMonitor(Monitor):
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

    def _store_variable(self, variable: torch.autograd.Variable, name):
        self._running_stats[name].add(var2np(variable))

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

    def _store_variable(self, variable: torch.autograd.Variable, name):
        variable.register_hook(self.create_hook(name))

    def title(self, duration="average"):
        print("| Mean gradient magnitude ({})"
              "".format(duration).ljust(79),
              "|", sep="")
        self.line()


class ModelInspector(Monitor):
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
        self.monitors = [WeightMonitor(), GradientMonitor()]
        self.loss_monitor = StatMonitor()

    def _store_variable(self, variable: torch.autograd.Variable, name):
        for monitor in self.monitors:
            monitor._store_variable(variable, name)

    def update_loss(self, variable: torch.autograd.Variable, name):
        self.loss_monitor._store_variable(variable, name)
        return self

    def headline(self):
        print("/", "=" * 30, " Model inspection ", "=" * 30, "\\", sep="")

    def title(self):
        print("|", self.name.center(78), "|", sep="")

    def footline(self):
        print("\\", "=" * 78, "/", sep="")

    def _analyze(self):
        for monitor in self.monitors:
            monitor.analyze()
        self.loss_monitor.analyze()



