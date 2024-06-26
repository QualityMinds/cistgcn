import torch
from fvcore.nn import FlopCountAnalysis


def compute_flops_pytorch_model(model, input_sz=(1, 10, 22, 3)):
    flops = FlopCountAnalysis(model, torch.rand(input_sz).cuda())
    total = flops.total()  # ('149.508M', '146.713K') using another tool
    by_operator = flops.by_operator()
    by_module_and_operator = flops.by_module_and_operator()
    by_module = flops.by_module()
    return {"total": total,
            "by_operator": by_operator,
            "by_module_and_operator": by_module_and_operator,
            "by_module": by_module, }
