from IMDLBenCo.registry import POSTFUNCS
import torch

_NORM_CACHE = {} # global cache for normalization tensors per device

@POSTFUNCS.register_module()
def mymodel_post_func(data_dict):
    tensor_img = data_dict['image']
    device = tensor_img.device
    
    if device not in _NORM_CACHE:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=tensor_img.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=tensor_img.dtype).view(3, 1, 1)
        _NORM_CACHE[device] = (mean, std)
    
    mean, std = _NORM_CACHE[device]

    try:
        with torch.no_grad():
            denorm = tensor_img.detach().mul(std).add(mean)
            denorm = denorm.mul(255.0).clamp_(0, 255).to(torch.uint8)
            raw_img = denorm.permute(1, 2, 0).contiguous()

        data_dict['raw_img'] = raw_img
    except Exception as e:
        print(f"Error in post-processing function: {e}")
        raise e
