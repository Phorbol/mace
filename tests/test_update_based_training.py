from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from mace.tools import build_default_arg_parser


class _MiniLoss(torch.nn.Module):
    def forward(self, *_args, **_kwargs):
        return torch.tensor(0.0, requires_grad=True)


class _FakeLogger:
    def __init__(self):
        self.records = []

    def log(self, metrics):
        self.records.append(metrics)


class _FakeScheduler:
    step_on_batch = False

    def __init__(self):
        self.epoch_steps = []

    def step(self, metrics=None, epoch=None):
        self.epoch_steps.append((metrics, epoch))


class _FakeCheckpointHandler:
    def __init__(self):
        self.saved = []

    def save(self, state, epochs, keep_last, **kwargs):
        self.saved.append((epochs, keep_last, kwargs))


def _eval_metrics():
    return {
        "rmse_e_per_atom": 0.0,
        "rmse_f": 0.0,
        "mae_e_per_atom": 0.0,
        "mae_f": 0.0,
    }


def test_arg_parser_accepts_update_based_training_flags():
    parser = build_default_arg_parser()

    args = parser.parse_args([
        "--name",
        "update_based_parser_test",
        "--max_num_updates",
        "20000",
        "--start_stage_two_update",
        "15000",
        "--eval_interval_updates",
        "1000",
        "--checkpoint_interval_updates",
        "5000",
    ])

    assert args.max_num_updates == 20000
    assert args.start_swa_update == 15000
    assert args.eval_interval_updates == 1000
    assert args.checkpoint_interval_updates == 5000


def test_train_one_epoch_stops_after_max_steps(monkeypatch):
    train_module = importlib.import_module("mace.tools.train")
    calls = []

    def fake_take_step(**kwargs):
        calls.append(kwargs["global_step"])
        return torch.tensor(0.0), {"loss": 0.0}

    monkeypatch.setattr(train_module, "take_step", fake_take_step)

    steps = train_module.train_one_epoch(
        model=torch.nn.Linear(1, 1),
        loss_fn=_MiniLoss(),
        data_loader=[object(), object(), object(), object()],
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.tensor([1.0]))], lr=0.1),
        epoch=2,
        output_args={"forces": False, "virials": False, "stress": False},
        max_grad_norm=None,
        ema=None,
        logger=_FakeLogger(),
        device=torch.device("cpu"),
        distributed=False,
        global_step_start=10,
        max_steps=2,
    )

    assert steps == 2
    assert calls == [10, 11]


def test_train_stops_at_max_num_updates_without_full_extra_epoch(monkeypatch):
    train_module = importlib.import_module("mace.tools.train")
    train_calls = []

    def fake_evaluate(**_kwargs):
        return 0.0, _eval_metrics()

    def fake_train_one_epoch(**kwargs):
        max_steps = kwargs.get("max_steps")
        loader_steps = len(kwargs["data_loader"])
        steps = loader_steps if max_steps is None else min(max_steps, loader_steps)
        train_calls.append({
            "epoch": kwargs["epoch"],
            "global_step_start": kwargs["global_step_start"],
            "max_steps": max_steps,
        })
        return steps

    monkeypatch.setattr(train_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(train_module, "train_one_epoch", fake_train_one_epoch)

    train_module.train(
        model=torch.nn.Linear(1, 1),
        loss_fn=_MiniLoss(),
        train_loader=[object(), object(), object()],
        valid_loaders={"valid": [object()]},
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.tensor([1.0]))], lr=0.1),
        lr_scheduler=_FakeScheduler(),
        start_epoch=0,
        max_num_epochs=10,
        patience=999,
        checkpoint_handler=_FakeCheckpointHandler(),
        logger=_FakeLogger(),
        eval_interval=1,
        output_args={"forces": False, "virials": False, "stress": False},
        device=torch.device("cpu"),
        log_errors="PerAtomMAE",
        max_num_updates=5,
    )

    assert train_calls == [
        {"epoch": 0, "global_step_start": 0, "max_steps": 3},
        {"epoch": 1, "global_step_start": 3, "max_steps": 2},
    ]


def test_train_activates_stage_two_from_global_update(monkeypatch):
    train_module = importlib.import_module("mace.tools.train")
    stage_one_loss = _MiniLoss()
    stage_two_loss = _MiniLoss()
    seen_losses = []

    def fake_evaluate(**_kwargs):
        return 0.0, _eval_metrics()

    def fake_train_one_epoch(**kwargs):
        seen_losses.append(kwargs["loss_fn"])
        return len(kwargs["data_loader"])

    class FakeAveragedModel:
        def __init__(self):
            self.updated = 0

        def update_parameters(self, _model):
            self.updated += 1

    class FakeSwaScheduler:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

    monkeypatch.setattr(train_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(train_module, "train_one_epoch", fake_train_one_epoch)

    averaged_model = FakeAveragedModel()
    train_module.train(
        model=torch.nn.Linear(1, 1),
        loss_fn=stage_one_loss,
        train_loader=[object(), object(), object()],
        valid_loaders={"valid": [object()]},
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.tensor([1.0]))], lr=0.1),
        lr_scheduler=_FakeScheduler(),
        start_epoch=0,
        max_num_epochs=3,
        patience=999,
        checkpoint_handler=_FakeCheckpointHandler(),
        logger=_FakeLogger(),
        eval_interval=1,
        output_args={"forces": False, "virials": False, "stress": False},
        device=torch.device("cpu"),
        log_errors="PerAtomMAE",
        max_num_updates=6,
        swa=SimpleNamespace(
            start=999,
            start_update=3,
            loss_fn=stage_two_loss,
            model=averaged_model,
            scheduler=FakeSwaScheduler(),
        ),
    )

    assert seen_losses == [stage_one_loss, stage_two_loss]
    assert averaged_model.updated == 1


def test_train_saves_update_interval_checkpoints_without_waiting_for_validation(monkeypatch):
    train_module = importlib.import_module("mace.tools.train")
    checkpoint_handler = _FakeCheckpointHandler()

    def fake_evaluate(**_kwargs):
        return 1.0, _eval_metrics()

    def fake_train_one_epoch(**kwargs):
        max_steps = kwargs.get("max_steps")
        loader_steps = len(kwargs["data_loader"])
        return loader_steps if max_steps is None else min(max_steps, loader_steps)

    monkeypatch.setattr(train_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(train_module, "train_one_epoch", fake_train_one_epoch)

    train_module.train(
        model=torch.nn.Linear(1, 1),
        loss_fn=_MiniLoss(),
        train_loader=[object(), object(), object()],
        valid_loaders={"valid": [object()]},
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.tensor([1.0]))], lr=0.1),
        lr_scheduler=_FakeScheduler(),
        start_epoch=0,
        max_num_epochs=3,
        patience=999,
        checkpoint_handler=checkpoint_handler,
        logger=_FakeLogger(),
        eval_interval=999,
        output_args={"forces": False, "virials": False, "stress": False},
        device=torch.device("cpu"),
        log_errors="PerAtomMAE",
        max_num_updates=6,
        checkpoint_interval_updates=3,
    )

    assert (
        0,
        True,
        {"updates": 3, "checkpoint_kind": "update"},
    ) in checkpoint_handler.saved
    assert (
        1,
        True,
        {"updates": 6, "checkpoint_kind": "update"},
    ) in checkpoint_handler.saved


def test_train_resume_uses_explicit_start_update(monkeypatch):
    train_module = importlib.import_module("mace.tools.train")
    train_calls = []

    def fake_evaluate(**_kwargs):
        return 0.0, _eval_metrics()

    def fake_train_one_epoch(**kwargs):
        train_calls.append({
            "epoch": kwargs["epoch"],
            "global_step_start": kwargs["global_step_start"],
            "max_steps": kwargs.get("max_steps"),
        })
        return kwargs.get("max_steps")

    monkeypatch.setattr(train_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(train_module, "train_one_epoch", fake_train_one_epoch)

    train_module.train(
        model=torch.nn.Linear(1, 1),
        loss_fn=_MiniLoss(),
        train_loader=[object(), object(), object()],
        valid_loaders={"valid": [object()]},
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.tensor([1.0]))], lr=0.1),
        lr_scheduler=_FakeScheduler(),
        start_epoch=4,
        start_update=4,
        max_num_epochs=10,
        patience=999,
        checkpoint_handler=_FakeCheckpointHandler(),
        logger=_FakeLogger(),
        eval_interval=999,
        output_args={"forces": False, "virials": False, "stress": False},
        device=torch.device("cpu"),
        log_errors="PerAtomMAE",
        max_num_updates=5,
    )

    assert train_calls == [
        {"epoch": 4, "global_step_start": 4, "max_steps": 1},
    ]

