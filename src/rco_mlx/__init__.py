"""RCO budget-constrained mixed-precision allocation, ported to NumPy + MLX.

Ports the optimizer half of ISTA-DASLab/RCO (Apache-2.0) so it runs anywhere
NumPy does, then wires the result into MLX's quantized-checkpoint format.

    from rco_mlx.objective import LossTable
    from rco_mlx.mlx_alloc import allocation_to_config
    from rco_mlx.validate import validate_allocation

    table = LossTable(losses=losses, fracs=fracs, bitwidths=[2, 3, 4, 6, 8])
    bits_per_group = table.assignment(target_avg_bits=4.0)
    cfg = allocation_to_config(dict(zip(group_names, bits_per_group)))
    report = validate_allocation(cfg, on_disk_keys=keys)
    assert report.ok, report.summary()

The two entry points:

    LossTable.solve / .assignment   exact knapsack for a separable objective
    ProjectedGumbelOptimizer        Riemannian Adam for a live, differentiable
                                    objective (supply a loss_and_grad callable)

See ``README.md`` for the honest scope of the port.
"""

from .grouping import (
    LayerGroup,
    attach_param_counts,
    build_layer_groups,
    build_moe_per_expert_groups,
    get_group_param_count,
    get_layer_to_group_mapping,
    parse_group_spec,
)
from .manifold import (
    budget_normal,
    project_gradient,
    retraction,
    vector_transport,
)
from .mlx_alloc import (
    MlxQuantConfig,
    allocation_to_config,
    merge_with_defaults,
    read_config,
    recover_bits_from_packed_shape,
    write_config,
)
from .adapters import (
    ModelAdapter,
    OLMoEAdapter,
    Qwen3NextMoEAdapter,
    get,
    register,
)
from .objective import (
    MLX_SUPPORTED_BITS,
    MLX_SUPPORTED_GROUP_SIZES,
    LossTable,
    effective_bits,
)
from .search import ProjectedGumbelOptimizer, budget_dp, values_from_losses
from .validate import ValidationReport, assert_coverage, validate_allocation

__version__ = "0.2.0"

__all__ = [
    # grouping
    "LayerGroup",
    "parse_group_spec",
    "build_layer_groups",
    "build_moe_per_expert_groups",
    "get_layer_to_group_mapping",
    "get_group_param_count",
    "attach_param_counts",
    # manifold
    "budget_normal",
    "project_gradient",
    "retraction",
    "vector_transport",
    # search
    "budget_dp",
    "values_from_losses",
    "ProjectedGumbelOptimizer",
    # objective
    "LossTable",
    "effective_bits",
    "MLX_SUPPORTED_BITS",
    "MLX_SUPPORTED_GROUP_SIZES",
    # mlx
    "MlxQuantConfig",
    "allocation_to_config",
    "merge_with_defaults",
    "write_config",
    "read_config",
    "recover_bits_from_packed_shape",
    # adapters
    "ModelAdapter",
    "OLMoEAdapter",
    "Qwen3NextMoEAdapter",
    "get",
    "register",
    # validation
    "ValidationReport",
    "validate_allocation",
    "assert_coverage",
]
