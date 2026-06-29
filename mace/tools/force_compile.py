from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import torch

from mace.tools.scatter import scatter_sum


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


def strip_fx_saved_tensor_detach(gm: torch.fx.GraphModule) -> None:
    to_remove: list[torch.fx.Node] = []
    for node in gm.graph.nodes:
        if not _is_detach_node(node):
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
) -> ForceClosureTraceResult:
    from torch.fx.experimental.proxy_tensor import make_fx

    traced = make_fx(fn, tracing_mode=tracing_mode)(*example_inputs)
    detach_nodes_before = count_fx_detach_nodes(traced)
    if strip_detach:
        strip_fx_saved_tensor_detach(traced)
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
) -> tuple[Callable[..., Any], dict[str, Any] | None]:
    if not compile_graph:
        return graph_module, None
    compile_kwargs: dict[str, Any] = {
        "backend": "inductor",
        "dynamic": compile_dynamic,
    }
    if compile_mode != "default":
        compile_kwargs["mode"] = compile_mode
    return torch.compile(graph_module, **compile_kwargs), compile_kwargs
