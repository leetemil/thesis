import math
from collections.abc import Iterable

import torch
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_, uniform_

def variational(module, parameter_names = ["weight", "bias"]):
    if isinstance(parameter_names, str):
        Variational.apply(module, parameter_names)
    elif isinstance(parameter_names, Iterable):
        for parameter_name in parameter_names:
            Variational.apply(module, parameter_name)
    else:
        raise NotImplementedError
    return module

class Variational:
    last_fan_in = 0
    def __init__(self, name):
        self.name = name

    @staticmethod
    def apply(module, parameter_name):
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, Variational) and hook.name == parameter_name:
                raise RuntimeError(f"Cannot register two variational hooks on the same parameter {parameter_name}")

        hook = Variational(parameter_name)
        parameter = getattr(module, parameter_name)

        del module._parameters[parameter_name]

        mean_parameter = Parameter(torch.zeros_like(parameter))
        logvar_parameter = Parameter(torch.zeros_like(parameter))

        with torch.no_grad():
            # Initialize
            if len(parameter.shape) == 1:
                bound = 1 / math.sqrt(Variational.last_fan_in)
                variance = ((bound * 2) ** 2) / 12
                logvar_parameter[:] = math.log(variance)
            elif len(parameter.shape) >= 2:
                Variational.last_fan_in = mean_parameter.size(1)
                variance = 2 / (parameter.size(0) + parameter.size(1))
                logvar_parameter[:] = math.log(variance)

        module.register_parameter(parameter_name + "_mean", mean_parameter)
        module.register_parameter(parameter_name + "_logvar", logvar_parameter)
        setattr(module, parameter_name, hook.rsample(module))

        module.register_forward_pre_hook(hook)
        return hook

    def rsample(self, module):
        mean = getattr(module, self.name + '_mean')
        logvar = getattr(module, self.name + '_logvar')
        distribution = Normal(mean, logvar.mul(0.5).exp())
        return distribution.rsample()

    def rsample_new(self, module):
        setattr(module, self.name, self.rsample(module))

    def __call__(self, module, inputs):
        if module.training:
            self.rsample_new(module)
