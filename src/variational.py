import math
from collections.abc import Iterable

import torch
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_, uniform_

def variational(module, parameter_names = "weight_and_bias"):
    if isinstance(parameter_names, str):
        Variational.apply(module, parameter_names)
    elif isinstance(parameter_names, Iterable):
        for parameter_name in parameter_names:
            Variational.apply(module, parameter_name)
    else:
        raise NotImplementedError
    return module

class Variational:
    def __init__(self, name):
        self.name = name

    @staticmethod
    def apply_linear(module):
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, Variational) and hook.name == "weight" or hook.name == "bias":
                raise RuntimeError(f"Cannot register two variational hooks on the same parameter.")

        hook_weight = Variational("weight")
        hook_bias = Variational("bias")

        weight = getattr(module, "weight")
        bias = getattr(module, "bias")

        del module._parameters["weight"]
        del module._parameters["bias"]

        weight_mean_parameter = Parameter(torch.zeros_like(weight))
        weight_logvar_parameter = Parameter(torch.zeros_like(weight))

        bias_mean_parameter = Parameter(torch.zeros_like(bias))
        bias_logvar_parameter = Parameter(torch.zeros_like(bias))

        with torch.no_grad():
            variance = 2 / (weight.size(0) + weight.size(1))
            weight_logvar_parameter[:] = math.log(variance)

            bound = 1 / math.sqrt(weight.size(1))
            variance = ((bound * 2) ** 2) / 12
            bias_logvar_parameter[:] = math.log(variance)

        module.register_parameter("weight_mean", weight_mean_parameter)
        module.register_parameter("weight_logvar", weight_logvar_parameter)
        module.register_parameter("bias_mean", bias_mean_parameter)
        module.register_parameter("bias_logvar", bias_logvar_parameter)
        setattr(module, "weight", hook_weight.rsample(module))
        setattr(module, "bias", hook_bias.rsample(module))
        setattr(module, "sample_new_weight", lambda: hook_weight.rsample_new(module))
        setattr(module, "sample_new_bias", lambda: hook_bias.rsample_new(module))

        module.register_forward_pre_hook(hook_weight)
        module.register_forward_pre_hook(hook_bias)
        return hook_weight, hook_bias

    @staticmethod
    def apply(module, parameter_name):
        if parameter_name == "weight_and_bias":
            return Variational.apply_linear(module)

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
            variance = 2 / (sum(parameter.shape))
            logvar_parameter[:] = math.log(variance)

        module.register_parameter(parameter_name + "_mean", mean_parameter)
        module.register_parameter(parameter_name + "_logvar", logvar_parameter)
        setattr(module, parameter_name, hook.rsample(module))
        setattr(module, "sample_new_" + parameter_name, lambda: hook.rsample_new(module))

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
