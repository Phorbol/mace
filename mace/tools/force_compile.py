from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Sequence
from typing import Any

import torch

from mace.tools.scatter import scatter_sum



def _noop_restore() -> None:
    return


def patch_inductor_force_int64_indexing() -> Callable[[], None]:
    try:
        from torch._inductor.codegen.simd import SIMDScheduling
    except Exception:
        return _noop_restore

    if getattr(SIMDScheduling, "_mace_force_int64_patched", False):
        return _noop_restore

    original_can_use_32bit_indexing = SIMDScheduling.can_use_32bit_indexing
    marker_was_present = hasattr(SIMDScheduling, "_mace_force_int64_patched")
    original_marker = getattr(SIMDScheduling, "_mace_force_int64_patched", None)
    SIMDScheduling.can_use_32bit_indexing = staticmethod(lambda numel, buffers: False)
    SIMDScheduling._mace_force_int64_patched = True

    def restore() -> None:
        SIMDScheduling.can_use_32bit_indexing = original_can_use_32bit_indexing
        if marker_was_present:
            SIMDScheduling._mace_force_int64_patched = original_marker
        elif hasattr(SIMDScheduling, "_mace_force_int64_patched"):
            delattr(SIMDScheduling, "_mace_force_int64_patched")

    return restore


def apply_force_compile_global_patches() -> Callable[[], None]:
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_REPORT_CHOICES_STATS", "0")
    os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
    dynamo_config = None
    previous_optimize_ddp = None
    try:
        import torch._dynamo.config as dynamo_config

        previous_optimize_ddp = dynamo_config.optimize_ddp
        dynamo_config.optimize_ddp = False
    except Exception:
        dynamo_config = None
    maybe_restore_inductor_patches = patch_inductor_force_int64_indexing()
    restore_inductor_patches = (
        maybe_restore_inductor_patches
        if callable(maybe_restore_inductor_patches)
        else _noop_restore
    )

    def restore() -> None:
        try:
            restore_inductor_patches()
        finally:
            if dynamo_config is not None and previous_optimize_ddp is not None:
                dynamo_config.optimize_ddp = previous_optimize_ddp

    return restore


def build_force_compile_inductor_options(
    *, shape_padding: bool = True, max_fusion_size: int = 8
) -> dict[str, Any]:
    compile_options: dict[str, Any] = {
        "max_autotune": False,
        "shape_padding": bool(shape_padding),
        "epilogue_fusion": False,
        "triton.cudagraphs": False,
        "max_fusion_size": int(max_fusion_size),
        "triton.persistent_reductions": False,
        "triton.mix_order_reduction": False,
        "triton.max_tiles": 1,
    }
    try:
        from torch._inductor import config as inductor_config

        valid_options = inductor_config.get_config_copy()
        return {
            key: value
            for key, value in compile_options.items()
            if key.replace("-", "_") in valid_options
        }
    except Exception:
        return compile_options


def _force_compile_decomposition_table() -> dict[Any, Any] | None:
    try:
        from torch._decomp import get_decompositions

        return get_decompositions([torch.ops.aten.silu_backward.default])
    except Exception:
        return None


@dataclasses.dataclass(frozen=True)
class ForceClosureTraceResult:
    graph_module: torch.fx.GraphModule
    detach_nodes_before: int
    detach_nodes_after: int


def edge_gradient_to_atomic_forces(
    edge_grad: torch.Tensor,
    *,
    edge_index: torch.Tensor,
    num_atoms: int,
) -> torch.Tensor:
    sender = edge_index[0]
    receiver = edge_index[1]
    sender_force = scatter_sum(edge_grad, sender, dim=0, dim_size=num_atoms)
    receiver_force = scatter_sum(edge_grad, receiver, dim=0, dim_size=num_atoms)
    return sender_force - receiver_force


def _is_detach_node(node: torch.fx.Node) -> bool:
    return node.op == "call_function" and node.target == torch.ops.aten.detach.default


def count_fx_detach_nodes(gm: torch.fx.GraphModule) -> int:
    return sum(1 for node in gm.graph.nodes if _is_detach_node(node))


def strip_fx_saved_tensor_detach(
    gm: torch.fx.GraphModule, *, remove_all: bool = False
) -> None:
    to_remove: list[torch.fx.Node] = []
    for node in gm.graph.nodes:
        if not _is_detach_node(node):
            continue
        if remove_all:
            to_remove.append(node)
            continue
        input_node = node.args[0]
        users = list(node.users.keys())
        is_chain_inner = _is_detach_node(input_node)
        is_dead = len(users) == 0
        is_chain_head = len(users) > 0 and all(_is_detach_node(user) for user in users)
        if is_chain_inner or is_dead or is_chain_head:
            to_remove.append(node)
    for node in to_remove:
        node.replace_all_uses_with(node.args[0])
        gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()


def rebuild_fx_graph_module(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    new_graph = torch.fx.Graph()
    value_map: dict[torch.fx.Node, torch.fx.Node] = {}
    for node in gm.graph.nodes:
        value_map[node] = new_graph.node_copy(node, lambda old: value_map[old])
    new_graph.lint()
    return torch.fx.GraphModule(gm, new_graph)


def trace_force_closure(
    fn: Callable[..., Any],
    example_inputs: Sequence[torch.Tensor],
    *,
    tracing_mode: str,
    strip_detach: bool,
    strip_all_detach: bool = False,
) -> ForceClosureTraceResult:
    from torch.fx.experimental.proxy_tensor import make_fx

    decomp_table = _force_compile_decomposition_table()
    make_fx_kwargs: dict[str, Any] = {"tracing_mode": tracing_mode}
    if tracing_mode == "symbolic":
        make_fx_kwargs["_allow_non_fake_inputs"] = True
    if decomp_table is not None:
        make_fx_kwargs["decomposition_table"] = decomp_table
    traced = make_fx(fn, **make_fx_kwargs)(*example_inputs)
    detach_nodes_before = count_fx_detach_nodes(traced)
    if strip_detach:
        strip_fx_saved_tensor_detach(traced, remove_all=strip_all_detach)
        traced = rebuild_fx_graph_module(traced)
    detach_nodes_after = count_fx_detach_nodes(traced)
    return ForceClosureTraceResult(
        graph_module=traced,
        detach_nodes_before=detach_nodes_before,
        detach_nodes_after=detach_nodes_after,
    )


def compile_fx_graph_module(
    graph_module: torch.fx.GraphModule,
    *,
    compile_graph: bool,
    compile_mode: str,
    compile_dynamic: bool,
    shape_padding: bool = True,
    max_fusion_size: int = 8,
) -> tuple[Callable[..., Any], dict[str, Any] | None]:
    if not compile_graph:
        return graph_module, None

    dynamo_config = None
    previous_optimize_ddp = None
    try:
        import torch._dynamo.config as dynamo_config

        previous_optimize_ddp = dynamo_config.optimize_ddp
    except Exception:
        dynamo_config = None

    restore_global_patches: Callable[[], None] = _noop_restore
    try:
        maybe_restore = apply_force_compile_global_patches()
        if callable(maybe_restore):
            restore_global_patches = maybe_restore
        compile_kwargs: dict[str, Any] = {
            "backend": "inductor",
            "dynamic": compile_dynamic,
            "options": build_force_compile_inductor_options(
                shape_padding=shape_padding, max_fusion_size=max_fusion_size
            ),
        }
        if compile_mode != "default":
            compile_kwargs["mode"] = compile_mode

        try:
            import torch._functorch.config as functorch_config
        except (ImportError, AttributeError):
            return torch.compile(graph_module, **compile_kwargs), compile_kwargs

        previous_donated_buffer = functorch_config.donated_buffer
        functorch_config.donated_buffer = False
        try:
            executable = torch.compile(graph_module, **compile_kwargs)
        finally:
            functorch_config.donated_buffer = previous_donated_buffer
        return executable, compile_kwargs
    finally:
        restore_global_patches()
        if dynamo_config is not None and previous_optimize_ddp is not None:
            dynamo_config.optimize_ddp = previous_optimize_ddp
