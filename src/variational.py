from collections.abc import Iterable

import torch
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter

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

        module.register_parameter(parameter_name + "_mean", mean_parameter)
        module.register_parameter(parameter_name + "_logvar", logvar_parameter)
        setattr(module, parameter_name, hook.sample(module))

        module.register_forward_pre_hook(hook)
        return hook

    def sample(self, module):
        mean = getattr(module, self.name + '_mean')
        logvar = getattr(module, self.name + '_logvar')
        distribution = Normal(mean, logvar.mul(0.5).exp())
        return distribution.rsample()

    def __call__(self, module, inputs):
        setattr(module, self.name, self.sample(module))

if __name__ == "__main__":
    l = torch.nn.Linear(10, 20)
    vl = variational(l, ["weight", "bias"])

    inputs = torch.randn(10)
    result = vl(inputs)
    loss = result.sum()

    breakpoint()
