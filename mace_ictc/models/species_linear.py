# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Species GEMMs without an [N,C_out,C_in] weight expansion.

Eager CUDA FP32 uses indexed native kernels for any positive bank size.
The ordinary PyTorch formula remains available outside that coverage;
learned banks and mixed/higher derivatives stay live in both paths.
"""
import torch
import torch.nn.functional as F


def shared_species_linear(block, weight, index, bias=None, *, native=True):
    """Apply weight[S,O,I] to block[N,I,M] selected by fixed index[N]."""
    if native and supported(block, weight) and not torch.compiler.is_compiling():
        from mace_ictc.hp._species import species_linear
        return species_linear(block, weight, index, bias)
    torch._assert_async(((index>=0)&(index<weight.shape[0])).all(),"Species index outside the weight bank")
    angular=block.movedim(-2,-1).contiguous()
    flat=angular.reshape(-1,angular.shape[-1])
    result=None
    for species in range(weight.shape[0]):
        value=F.linear(flat,weight[species],None if bias is None else bias[species])
        value=value.reshape(*angular.shape[:-1],weight.shape[1])
        selected=torch.where(index[:,None,None]==species,value,0.)
        result=selected if result is None else result+selected
    return result.movedim(-1,-2)


def supported(block, weight):
    return (block.is_cuda and block.dtype==torch.float32 and weight.dtype==block.dtype
            and weight.device==block.device and weight.shape[0]>=1
            and not torch.is_autocast_enabled(block.device.type))
