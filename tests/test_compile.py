import os
from functools import wraps
from typing import Callable

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from e3nn import o3
from torch.testing import assert_close

from mace import data, modules, tools
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.tools import compile as mace_compile
from mace.tools import torch_geometric

table = tools.AtomicNumberTable([6])
atomic_energies = np.array([1.0], dtype=float)
cutoff = 5.0


def setup_cueq(enable: bool, device: str):
    if not enable:
        return None

    return CuEquivarianceConfig(
        enabled=True,
        layout="ir_mul",
        group="O3_e3nn",
        optimize_all=True,
        conv_fusion=(device == "cuda"),
    )


def create_mace(device: str, seed: int = 1702, enable_cueq: bool = False):
    torch_geometric.seed_everything(seed)

    model_config = {
        "r_max": cutoff,
        "num_bessel": 8,
        "num_polynomial_cutoff": 6,
        "max_ell": 3,
        "interaction_cls": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "interaction_cls_first": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "num_interactions": 2,
        "num_elements": 1,
        "hidden_irreps": o3.Irreps("128x0e + 128x1o"),
        "MLP_irreps": o3.Irreps("16x0e"),
        "gate": F.silu,
        "atomic_energies": atomic_energies,
        "avg_num_neighbors": 8,
        "atomic_numbers": table.zs,
        "correlation": 3,
        "radial_type": "bessel",
        "atomic_inter_scale": 1.0,
        "atomic_inter_shift": 0.0,
        "cueq_config": setup_cueq(enable_cueq, device),
    }
    model = modules.ScaleShiftMACE(**model_config)
    return model.to(device)


def create_batch(device: str):
    from ase import build

    size = 2
    atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
    atoms_list = [atoms.repeat((size, size, size))]
    print("Number of atoms", len(atoms_list[0]))

    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=table, cutoff=cutoff)
            for config in configs
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    batch = batch.to(device)
    batch = batch.to_dict()
    return batch


def time_func(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        torch._inductor.cudagraph_mark_step_begin()  # pylint: disable=W0212
        outputs = func(*args, **kwargs)
        torch.cuda.synchronize()
        return outputs

    return wrapper


@pytest.fixture(params=[torch.float32, torch.float64], ids=["fp32", "fp64"])
def default_dtype(request):
    with tools.torch_tools.default_dtype(request.param):
        yield torch.get_default_dtype()


# skip if on windows
@pytest.mark.skipif(os.name == "nt", reason="Not supported on Windows")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mace(device, default_dtype):  # pylint: disable=W0621
    print(f"using default dtype = {default_dtype}")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="cuda is not available")

    model_defaults = create_mace(device)
    tmp_model = mace_compile.prepare(create_mace)(device)
    model_compiled = torch.compile(tmp_model, mode="default", fullgraph=True)

    batch1 = create_batch(device)
    output1 = model_defaults(batch1, training=True)

    batch2 = create_batch(device)
    batch2["positions"].requires_grad_(True)
    output2 = model_compiled(batch2, training=True)
    assert_close(output1["energy"], output2["energy"])
    assert_close(output1["forces"], output2["forces"])


@pytest.mark.skipif(os.name == "nt", reason="Not supported on Windows")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mace_compile_stress(device, default_dtype):  # pylint: disable=W0621
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="cuda is not available")

    model_eager = create_mace(device)
    batch_eager = create_batch(device)
    output_eager = model_eager(batch_eager, compute_stress=True)

    tmp_model = mace_compile.prepare(create_mace)(device)
    model_compiled = torch.compile(tmp_model, mode="default", fullgraph=True)
    batch_compiled = create_batch(device)
    batch_compiled["positions"].requires_grad_(True)
    output_compiled = model_compiled(batch_compiled, training=True, compute_stress=True)

    assert_close(output_eager["energy"], output_compiled["energy"])
    assert_close(output_eager["forces"], output_compiled["forces"])
    assert output_eager["stress"] is not None
    assert output_compiled["stress"] is not None
    assert_close(output_eager["stress"], output_compiled["stress"])


@pytest.mark.skipif(os.name == "nt", reason="Not supported on Windows")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
@pytest.mark.parametrize("enable_cueq", [True, False])
def test_eager_benchmark(
    benchmark, default_dtype, enable_cueq
):  # pylint: disable=W0621
    print(f"using default dtype = {default_dtype}")
    batch = create_batch("cuda")
    model = create_mace("cuda", enable_cueq=enable_cueq)
    model = time_func(model)
    benchmark(model, batch, training=True)


@pytest.mark.skipif(os.name == "nt", reason="Not supported on Windows")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
@pytest.mark.parametrize("compile_mode", ["default", "reduce-overhead", "max-autotune"])
@pytest.mark.parametrize("enable_amp", [False, True], ids=["fp32", "mixed"])
@pytest.mark.parametrize("enable_cueq", [False, True])
def test_compile_benchmark(benchmark, compile_mode, enable_amp, enable_cueq):
    _torch_version = tuple(
        int(x) for x in torch.__version__.split("+")[0].split(".")[:2]
    )
    if (
        enable_cueq
        and compile_mode in ("reduce-overhead", "max-autotune")
        and _torch_version < (2, 10)
    ):
        pytest.skip("cueq + CUDA graphs requires PyTorch >= 2.10")
    with tools.torch_tools.default_dtype(torch.float32):
        torch.compiler.reset()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        batch = create_batch("cuda")
        batch["positions"].requires_grad_(True)
        model = mace_compile.prepare(create_mace)("cuda", enable_cueq=enable_cueq)
        model = torch.compile(model, mode=compile_mode)
        model = time_func(model)

        with torch.autocast("cuda", enabled=enable_amp):
            benchmark(model, batch, training=True)


@pytest.mark.skipif(os.name == "nt", reason="Not supported on Windows")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_graph_breaks(device):
    import torch._dynamo as dynamo

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="cuda is not available")

    batch = create_batch(device)
    batch["positions"].requires_grad_(True)
    model = mace_compile.prepare(create_mace)(device)
    explanation = dynamo.explain(model)(batch, training=False)

    # these clutter the output but might be useful for investigating graph breaks
    explanation.ops_per_graph = None
    explanation.out_guards = None
    print(explanation)
    assert explanation.graph_break_count == 0


def test_training_compile_helper_noop_cpu():
    from mace.tools.training_compile import prepare_model_for_training_compile

    model = create_mace("cpu")
    compiled = prepare_model_for_training_compile(
        model,
        enabled=False,
        mode="default",
        fullgraph=False,
        allow_fallback=True,
    )

    assert compiled is model


def test_edge_force_compile_config_defaults_to_disabled():
    from mace.tools.training_compile import EdgeForceCompileConfig

    config = EdgeForceCompileConfig()

    assert config.enabled is False
    assert config.tracing_mode == "real"
    assert config.strip_detach is True
    assert config.compile_graph is True
    assert config.compile_mode == "default"
    assert config.compile_dynamic is True
    assert config.allow_fallback is True
    assert config.atol == 1.0e-5
    assert config.rtol == 1.0e-4


def test_edge_force_compile_gate_rejects_disabled_config():
    from mace.tools.training_compile import (
        EdgeForceCompileConfig,
        edge_force_compile_gate,
    )

    result = edge_force_compile_gate(
        model=torch.nn.Linear(1, 1),
        batch=object(),
        config=EdgeForceCompileConfig(enabled=False),
    )

    assert result.enabled is False
    assert result.accepted is False
    assert result.fallback_reason == "disabled"
    assert result.detach_nodes_before is None
    assert result.detach_nodes_after is None
    assert result.node_count is None


def test_edge_force_compile_gate_rejects_unsupported_outputs():
    from mace.tools.training_compile import (
        EdgeForceCompileConfig,
        edge_force_compile_gate,
    )

    result = edge_force_compile_gate(
        model=torch.nn.Linear(1, 1),
        batch=object(),
        config=EdgeForceCompileConfig(enabled=True),
        compute_virials=True,
    )

    assert result.enabled is True
    assert result.accepted is False
    assert result.fallback_reason == "unsupported_outputs"


def test_training_compile_allow_fallback_suppresses_dynamo_errors():
    import torch._dynamo.config as dynamo_config

    from mace.tools.training_compile import prepare_model_for_training_compile

    previous = dynamo_config.suppress_errors
    dynamo_config.suppress_errors = False
    try:
        model = torch.nn.Linear(1, 1)
        compiled = prepare_model_for_training_compile(
            model,
            enabled=True,
            mode="default",
            fullgraph=False,
            allow_fallback=True,
        )

        assert compiled is not model
        assert dynamo_config.suppress_errors is True
    finally:
        dynamo_config.suppress_errors = previous


class _MiniBatch:
    def __init__(self):
        self.x = torch.ones(1)

    def to(self, device, **kwargs):
        self.x = self.x.to(device, **kwargs)
        return self

    def to_dict(self):
        return {"x": self.x}


class _NoCallModel(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("base model should not be used for compiled training step")


class _TrainingOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.calls = 0

    def forward(self, batch, **kwargs):
        self.calls += 1
        return {"value": batch["x"].sum() * self.weight}


class _MiniLoss(torch.nn.Module):
    def forward(self, pred, ref):
        return pred["value"]


class _MiniLogger:
    def __init__(self):
        self.records = []

    def log(self, metrics):
        self.records.append(metrics)


class _MiniEma:
    def __init__(self):
        self.updates = 0

    def update(self):
        self.updates += 1


def test_train_one_epoch_uses_optional_training_model():
    from mace.tools.train import train_one_epoch

    base_model = _NoCallModel()
    training_model = _TrainingOnlyModel()
    optimizer = torch.optim.SGD(training_model.parameters(), lr=0.1)
    logger = _MiniLogger()

    train_one_epoch(
        model=base_model,
        training_model=training_model,
        loss_fn=_MiniLoss(),
        data_loader=[_MiniBatch()],
        optimizer=optimizer,
        epoch=0,
        output_args={"forces": False, "virials": False, "stress": False},
        max_grad_norm=None,
        ema=None,
        logger=logger,
        device=torch.device("cpu"),
        distributed=False,
    )

    assert training_model.calls == 1
    assert logger.records[0]["mode"] == "opt"


class _RaisesOnForward(torch.nn.Module):
    def forward(self, x):
        raise RuntimeError("compiled path failed")


def test_runtime_compile_fallback_retries_eager_model():
    from mace.tools.training_compile import RuntimeFallbackCompiledModule

    eager = torch.nn.Linear(1, 1)
    wrapper = RuntimeFallbackCompiledModule(
        eager_model=eager,
        compiled_model=_RaisesOnForward(),
        allow_fallback=True,
    )
    x = torch.ones(1, 1)

    output = wrapper(x)

    assert torch.allclose(output, eager(x))
    assert wrapper.disabled is True


def test_energy_only_compile_wrapper_preserves_force_outputs_cpu():
    from mace.tools.training_compile import EnergyOnlyForceCompiledModule

    model = create_mace("cpu")
    wrapper = EnergyOnlyForceCompiledModule(
        eager_model=model,
        compiled_model=model,
        allow_fallback=False,
    )

    eager_output = model(
        create_batch("cpu"),
        training=True,
        compute_force=True,
        compute_virials=False,
        compute_stress=False,
    )
    wrapped_output = wrapper(
        create_batch("cpu"),
        training=True,
        compute_force=True,
        compute_virials=False,
        compute_stress=False,
    )

    assert_close(wrapped_output["energy"], eager_output["energy"])
    assert_close(wrapped_output["forces"], eager_output["forces"])


class _EnergyOnlyCompiledModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.calls = 0

    def forward(self, batch, **kwargs):
        self.calls += 1
        return {"energy": batch["x"].sum() * self.weight}


def test_energy_only_compile_wrapper_uses_compiled_model_without_forces():
    from mace.tools.training_compile import EnergyOnlyForceCompiledModule

    eager = _NoCallModel()
    compiled = _EnergyOnlyCompiledModel()
    wrapper = EnergyOnlyForceCompiledModule(
        eager_model=eager,
        compiled_model=compiled,
        allow_fallback=False,
    )

    output = wrapper(
        {"x": torch.ones(2)},
        training=True,
        compute_force=False,
        compute_virials=False,
        compute_stress=False,
    )
    output["energy"].backward()

    assert compiled.calls == 1
    assert compiled.weight.grad is not None


class _BackwardFails(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError("compiled backward failed")


class _BackwardFallbackModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.disabled = False
        self.disable_calls = 0
        self.forward_calls = 0

    def forward(self, batch, **kwargs):
        self.forward_calls += 1
        value = batch["x"].sum() * self.weight
        if not self.disabled:
            value = _BackwardFails.apply(value)
        return {"value": value}

    def disable_compile_fallback(self, exc):
        self.disable_calls += 1
        self.disabled = True
        return True


def test_take_step_retries_eager_after_compile_backward_failure():
    from mace.tools.train import take_step

    model = _BackwardFallbackModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    loss, metrics = take_step(
        model=model,
        loss_fn=_MiniLoss(),
        batch=_MiniBatch(),
        optimizer=optimizer,
        ema=None,
        output_args={"forces": False, "virials": False, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
    )

    assert model.disabled is True
    assert model.disable_calls == 1
    assert model.forward_calls == 2
    assert metrics["loss"] == 1.0
    assert loss.requires_grad is True


class _TransferRecordingBatch:
    def __init__(self):
        self.to_calls = []

    def to(self, device, **kwargs):
        self.to_calls.append((device, kwargs))
        return self

    def to_dict(self):
        return {"value": torch.tensor([1.0])}


class _TransferModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, batch_dict, **kwargs):
        return {"value": self.weight * batch_dict["value"]}


def test_take_step_uses_non_blocking_batch_transfer_when_requested():
    from mace.tools.train import take_step

    batch = _TransferRecordingBatch()
    model = _TransferModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    take_step(
        model=model,
        loss_fn=_MiniLoss(),
        batch=batch,
        optimizer=optimizer,
        ema=None,
        output_args={"forces": False, "virials": False, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
        non_blocking_transfer=True,
    )

    assert batch.to_calls == [(torch.device("cpu"), {"non_blocking": True})]


def test_take_step_loss_skip_prevents_optimizer_and_ema_update():
    from mace.tools.train import take_step
    from mace.tools.training_guards import LossSkipController, TrainingGuardConfig

    model = _TrainingOnlyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ema = _MiniEma()
    controller = LossSkipController(
        manual_threshold=0.5,
        start_step=0,
        ema_window=2,
        multiplier=2.0,
        skip_nan=True,
        skip_large=True,
    )

    loss, metrics = take_step(
        model=model,
        loss_fn=_MiniLoss(),
        batch=_MiniBatch(),
        optimizer=optimizer,
        ema=ema,
        output_args={"forces": False, "virials": False, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
        guard_config=TrainingGuardConfig(loss_skip=True, loss_skip_threshold=0.5),
        loss_skip_controller=controller,
        global_step=0,
    )

    assert loss.item() == pytest.approx(1.0)
    assert model.weight.item() == pytest.approx(1.0)
    assert model.weight.grad is None
    assert ema.updates == 0
    assert metrics["loss_skipped"] is True
    assert metrics["loss_skip_reason"] == "large"


def test_take_step_default_guards_preserve_optimizer_and_ema_update():
    from mace.tools.train import take_step

    model = _TrainingOnlyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ema = _MiniEma()

    loss, metrics = take_step(
        model=model,
        loss_fn=_MiniLoss(),
        batch=_MiniBatch(),
        optimizer=optimizer,
        ema=ema,
        output_args={"forces": False, "virials": False, "stress": False},
        max_grad_norm=None,
        device=torch.device("cpu"),
    )

    assert loss.item() == pytest.approx(1.0)
    assert model.weight.item() == pytest.approx(0.9)
    assert model.weight.grad is not None
    assert ema.updates == 1
    assert metrics["loss_skipped"] is False


class _RecordingCheckpointHandler:
    def __init__(self):
        self.calls = []

    def save(self, **kwargs):
        self.calls.append(kwargs)


def test_save_checkpoint_after_guard_blocks_nonfinite_gradients():
    from mace.tools.train import _save_checkpoint_after_guard
    from mace.tools.training_guards import NonFiniteGradGuard

    param = torch.nn.Parameter(torch.tensor([1.0]))
    param.grad = torch.tensor([float("inf")])
    guard = NonFiniteGradGuard()
    guard.update(torch.tensor(float("inf")))
    handler = _RecordingCheckpointHandler()

    with pytest.raises(RuntimeError, match="Non-finite gradient norm"):
        _save_checkpoint_after_guard(
            checkpoint_handler=handler,
            state=object(),
            epochs=3,
            keep_last=True,
            grad_guard=guard,
            named_parameters=lambda: [("weight", param)],
        )

    assert handler.calls == []


def test_save_checkpoint_after_guard_saves_when_guard_is_clear():
    from mace.tools.train import _save_checkpoint_after_guard

    handler = _RecordingCheckpointHandler()
    state = object()

    _save_checkpoint_after_guard(
        checkpoint_handler=handler,
        state=state,
        epochs=4,
        keep_last=False,
        grad_guard=None,
        named_parameters=lambda: [],
    )

    assert handler.calls == [
        {"state": state, "epochs": 4, "keep_last": False}
    ]


def test_train_one_epoch_passes_non_blocking_transfer_to_take_step(monkeypatch):
    import importlib

    train_module = importlib.import_module("mace.tools.train")

    seen = {}

    def fake_take_step(**kwargs):
        seen["non_blocking_transfer"] = kwargs["non_blocking_transfer"]
        return torch.tensor(0.0), {"loss": 0.0}

    class FakeLogger:
        def log(self, metrics):
            seen["logged"] = metrics

    monkeypatch.setattr(train_module, "take_step", fake_take_step)

    train_module.train_one_epoch(
        model=torch.nn.Linear(1, 1),
        loss_fn=_MiniLoss(),
        data_loader=[object()],
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.tensor([1.0]))], lr=0.1),
        epoch=3,
        output_args={"forces": False, "virials": False, "stress": False},
        max_grad_norm=None,
        ema=None,
        logger=FakeLogger(),
        device=torch.device("cpu"),
        distributed=False,
        non_blocking_transfer=True,
    )

    assert seen["non_blocking_transfer"] is True
    assert seen["logged"]["epoch"] == 3
