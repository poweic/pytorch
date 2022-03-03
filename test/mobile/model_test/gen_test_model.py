import torch
from math_ops import MathOpsModule
from nn_ops import NNOpsModule
from sampling_ops import SamplingOpsModule
from tensor_ops import TensorOpsModule


def scriptAndSave(module, name):
    script_module = torch.jit.script(module)
    script_module._save_for_lite_interpreter(name)
    script_module()
    print("model saved.")
    ops = torch.jit.export_opnames(script_module)
    return ops


ops = [
    scriptAndSave(MathOpsModule(), "ios/TestApp/models/math_ops.ptl"),
    scriptAndSave(TensorOpsModule(), "ios/TestApp/models/tensor_ops.ptl"),
    scriptAndSave(NNOpsModule(), "ios/TestApp/models/nn_ops.ptl"),
    scriptAndSave(SamplingOpsModule(), "ios/TestApp/models/sampling_ops.ptl"),
]
print(set().union(*ops))
