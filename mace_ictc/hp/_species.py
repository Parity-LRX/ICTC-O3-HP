# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Checked arbitrary-size species banks with stable live device grouping."""
import torch


def species_linear(features, weight, species, bias=None, *, native_training=True):
    from mace_ictc.models.message_topology import stable_topology
    if (features.ndim != 3 or weight.ndim != 3 or weight.shape[0] < 1
            or features.shape[1] != weight.shape[2] or not features.is_cuda
            or features.dtype != torch.float32 or weight.dtype != features.dtype
            or weight.device != features.device or species.device != features.device
            or species.dtype != torch.int64 or species.shape != (features.shape[0],)):
        raise ValueError('Expected CUDA FP32 features[N,I,M], bank[S,O,I], S>=1 and int64 species[N]')
    if torch.compiler.is_compiling():
        raise ValueError('Native indexed species GEMMs require eager execution')
    if bias is not None and (bias.shape != weight.shape[:2] or bias.device != weight.device
            or bias.dtype != weight.dtype or features.shape[-1] != 1):
        raise ValueError('Bias requires scalar features and a matching [S,O] bank')
    if not native_training and (weight.requires_grad or (bias is not None and bias.requires_grad)):
        from mace_ictc.models.species_linear import shared_species_linear
        return shared_species_linear(features, weight, species, bias, native=False)
    torch._assert_async(((species >= 0) & (species < weight.shape[0])).all(),
                        'Species index outside the complete weight bank')
    ptr, order = stable_topology(species, weight.shape[0])
    if weight.requires_grad or (bias is not None and bias.requires_grad):
        from ._species_training import _Linear
        value = _Linear.apply(features, weight.contiguous(), False, ptr, order)
        return value if bias is None else value+bias.index_select(0, species)[:, :, None]
    from ._species_gemm import _SpeciesGemm
    return _SpeciesGemm.apply(features, weight.contiguous(),
        None if bias is None else bias.contiguous(), False, ptr, order)
