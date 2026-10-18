# ICTC-O3-HP

High-performance Cartesian tensor operators and whole-network runtimes for
MACE-compatible atomistic models. The library provides tensor products,
graph-message aggregation, direction harmonics, symmetric contractions,
channel operations, force training, and AOTInductor deployment.

This repository contains the current library, API documentation, and usage
examples. Internal tests, benchmarks, task scripts, and historical source
snapshots are maintained separately. It has no dependency on `chorus_o2u1`.

The distribution name is **`ictc-o3-hp`**. The Python namespace **`mace_ictc`**
is retained for compatibility with existing imports, model classes, and
checkpoints. The main entry point is `mace_ictc.hp`.

## Installation

Use an isolated environment with a compatible CUDA-enabled PyTorch build:

```bash
git clone https://github.com/Parity-LRX/ICTC-O3-HP.git
cd ICTC-O3-HP
python -m pip install -e '.[hp,train]'
```

The validated runtime includes PyTorch 2.7.1 with CUDA, Triton 3.3.1, and e3nn
0.5.9. CUDA tensor-product JIT compilation also requires a compatible CUDA
toolkit, C++ compiler, and Ninja. Avoid installing different `mace_ictc`
checkouts into the same environment.

GPU operations, compilation, and examples should run on an allocated compute
device. On Slurm clusters, submit them through `sbatch` rather than running
them on a login node.

## Quick start

```python
from mace_ictc.hp import (
    EdgeTensorProduct, GraphMessage, direction_harmonics,
    create_symmetric_contraction, configure_symmetric_contraction,
    channel_linear, normalized_silu,
    create_model, check_model, configure_model, energy_and_forces,
    prepare_for_export,
)

model = create_model(
    channels=96, lmax=2, num_interactions=4, correlation=2,
)
configure_model(model, training=True, policy="generated")
```

Runnable examples demonstrate operator calls and whole-network force training:

```bash
python examples/operators.py --lmax 3 --edges 257
python examples/model.py --lmax 3 --layers 3 --layout packed
```

- [Full operator and whole-network API](docs/HP_API.md)
- [Operator example](examples/operators.py)
- [Whole-network example](examples/model.py)
- [LAMMPS interface and deployment](lammps_user_mfftorch/README.md)

## Layout compatibility and derivatives

Standalone tensor products expose both `ir_mul` and `mul_ir` buffer layouts.
Use `execution="legacy"` to select the compatible execution route that
performs `mul_ir` layout conversions inside the tensor-product autograd boundary:

```python
op = EdgeTensorProduct.create(
    channels=64, input_lmax=3, edge_lmax=3,
    layout="mul_ir", execution="legacy",
)
```

Both routes use the current Cartesian coefficients and derivative
implementations. The default `execution="direct"` retains the specialized
adjoint schedule. This option applies to standalone tensor products;
whole-network execution is configured through `configure_model`.

Standalone tensor products support ordinary and mixed autograd in eager CUDA
execution. Geometry and higher derivatives retain differentiable reference
fallbacks. `GraphMessage` accepts feature blocks, while `PackedGraphMessage`
provides a native two-dimensional buffer interface. See the API guide for
path ordering, gate shapes, fixed base coefficients, and derivative coverage.

## Runtime policies

The default whole-network policy is qualified `auto` HP with `packed` layout;
`fused_pack` is an explicit alternative. Configuration preserves model
parameters, checkpoint keys, and existing trainability. It does not freeze
trainable tensor-product coefficients to enable a faster route.

GPU HP paths use strict FP32. Configuration sets PyTorch matrix precision to
`highest` and disables matmul and cuDNN TF32 for the process; these settings
are not automatically restored. Supported execution requires compatible
tensor dtypes and devices.

Architecture settings include `channels`, `lmax`, `edge_lmax`,
`num_interactions`, and `correlation`. Positive channel counts need not be
multiples of 32; interaction depth is not fixed to two layers. Whole-network
configuration requires `0 <= lmax <= edge_lmax`. Qualified coverage includes
correlation orders 2 and 3, with order 1 supported for compatibility.
Structural support does not imply that every architecture has been validated
or that every schedule improves both speed and memory.

Current policies include shared element/channel GEMMs, destination-node
aggregation, native geometry schedules, bounded high-angular-order
contractions, and optional gate or radial rematerialization. High-L schedules
use device- and workload-dependent channel tiling. Optional CSR, indexed
species, message-chunk, and gradient-role schedules are documented in the API
guide. Performance should be measured for the actual workload.

For qualified single-device L4/L5 correlation-order-three plans, dense
reference U constants may reside on CPU while factorized GPU constants,
features, learning weights, and derivatives remain on the FP32 GPU path.
Use `reference_buffer_policy="device"` when conventional device residency is
required. Distributed wrappers retain device buffers. `high_l_policy="off"`
restores the preceding high-L schedule, and `policy="reference"` selects the
reference implementation.

## Deployment

Whole-model AOTInductor preparation uses bounded contractions and partitioned
tensor products where supported. Prepared packages contain the constants and
frozen weights needed for energy-and-force inference while preserving the
checkpoint schema. Keep each `.pt2` package with its metadata sidecar.

Updating Python source does not update an existing compiled package: re-export
the model to adopt a new execution plan. Eager qualification does not establish
AOTI or LAMMPS performance for every architecture. Deployment currently
supports energy and force; follow the LAMMPS guide for build requirements,
element mapping, ownership masks, and distributed halos.

## License

New HP implementations are licensed under
[PolyForm Noncommercial 1.0.0](LICENSE). Separate
[commercial licensing](COMMERCIAL_LICENSE.md) is available. Previously
MIT-licensed code and third-party components retain their original rights;
see [LICENSE_SCOPE.md](LICENSE_SCOPE.md) and [NOTICE](NOTICE).

The complete PolyForm permitted-use terms apply, including its educational
and public-research-organization provisions. Commercial uses outside those
permissions require separate written authorization. This is a
source-available, noncommercially licensed distribution, not an OSI-approved
open-source license.

The software license does not grant licenses to third-party model weights,
patents, or trademarks. Distribution of LAMMPS/GPL interfaces and combined
binaries must respect their applicable licenses.
