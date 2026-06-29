from __future__ import annotations

import torch

from mace.tools.force_compile import (
    compile_fx_graph_module,
    count_fx_detach_nodes,
    edge_gradient_to_atomic_forces,
    rebuild_fx_graph_module,
    strip_fx_saved_tensor_detach,
    trace_force_closure,
)


def test_edge_gradient_to_atomic_forces_matches_position_gradient():
    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, -1.0, 0.5]], requires_grad=True)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    vector = positions[edge_index[1]] - positions[edge_index[0]]
    energy = 0.5 * vector.square().sum()
    reference_force = -torch.autograd.grad(energy, positions)[0]

    edge_force = edge_gradient_to_atomic_forces(
        vector.detach(),
        edge_index=edge_index,
        num_atoms=positions.shape[0],
    )

    torch.testing.assert_close(edge_force, reference_force)


def test_strip_fx_saved_tensor_detach_removes_make_fx_detach_chain():
    from torch.fx.experimental.proxy_tensor import make_fx

    def fn(x):
        return x + x.detach().detach() * 2.0

    x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    traced = make_fx(fn, tracing_mode="real")(x)
    expected = traced(x)

    assert count_fx_detach_nodes(traced) >= 2

    strip_fx_saved_tensor_detach(traced)
    rebuilt = rebuild_fx_graph_module(traced)

    assert count_fx_detach_nodes(rebuilt) == 0
    torch.testing.assert_close(rebuilt(x), expected)


def test_trace_force_closure_repairs_detach_chain_and_preserves_outputs():
    def fn(x):
        y = x + x.detach().detach() * 2.0
        return y, y.square().sum()

    x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)

    result = trace_force_closure(
        fn,
        (x,),
        tracing_mode="real",
        strip_detach=True,
    )
    output, loss = result.graph_module(x)
    expected_output, expected_loss = fn(x)

    assert result.detach_nodes_before >= 2
    assert result.detach_nodes_after == 0
    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(loss, expected_loss)


def test_compile_fx_graph_module_returns_eager_graph_when_disabled():
    def fn(x):
        return x + 1.0

    x = torch.tensor([1.0])
    result = trace_force_closure(fn, (x,), tracing_mode="real", strip_detach=False)

    executable, compile_kwargs = compile_fx_graph_module(
        result.graph_module,
        compile_graph=False,
        compile_mode="default",
        compile_dynamic=True,
    )

    assert executable is result.graph_module
    assert compile_kwargs is None
    torch.testing.assert_close(executable(x), torch.tensor([2.0]))
