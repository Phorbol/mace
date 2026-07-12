from types import SimpleNamespace

import pytest
import torch

from mace.tools import build_default_arg_parser
from mace.tools.scripts_utils import LRScheduler


def _args(**overrides):
    base = {
        "optimizer": "adam",
        "scheduler": "WSD",
        "max_num_epochs": 10,
        "lr_wsd_warmup_steps": 2,
        "lr_wsd_warmup_ratio": 0.03,
        "lr_wsd_warmup_start_factor": 0.1,
        "lr_wsd_stop_lr_ratio": 0.01,
        "lr_wsd_decay_phase_ratio": 0.2,
        "lr_wsd_decay_type": "inverse_linear",
        "lr_scheduler_interval": "epoch",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_wsd_scheduler_follows_warmup_stable_decay_curve():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scheduler = LRScheduler(optimizer, _args())

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)

    scheduler.step(epoch=1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.55)

    scheduler.step(epoch=2)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)

    scheduler.step(epoch=8)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)

    scheduler.step(epoch=9)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0 / (0.5 / 0.01 + 0.5))

    scheduler.step(epoch=10)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


def test_arg_parser_accepts_wsd_scheduler_flags():
    parser = build_default_arg_parser()
    args = parser.parse_args([
        "--name",
        "wsd_parser_test",
        "--scheduler",
        "WSD",
        "--lr_wsd_warmup_steps",
        "7",
        "--lr_wsd_warmup_ratio",
        "0.02",
        "--lr_wsd_warmup_start_factor",
        "0.2",
        "--lr_wsd_stop_lr_ratio",
        "0.0005",
        "--lr_wsd_decay_phase_ratio",
        "0.15",
        "--lr_wsd_decay_type",
        "cosine",
        "--lr_scheduler_interval",
        "step",
    ])

    assert args.scheduler == "WSD"
    assert args.lr_wsd_warmup_steps == 7
    assert args.lr_wsd_warmup_ratio == pytest.approx(0.02)
    assert args.lr_wsd_warmup_start_factor == pytest.approx(0.2)
    assert args.lr_wsd_stop_lr_ratio == pytest.approx(0.0005)
    assert args.lr_wsd_decay_phase_ratio == pytest.approx(0.15)
    assert args.lr_wsd_decay_type == "cosine"
    assert args.lr_scheduler_interval == "step"


def test_wsd_scheduler_defaults_to_per_step_with_train_loader_length():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scheduler = LRScheduler(
        optimizer,
        _args(
            max_num_epochs=4,
            lr_wsd_warmup_steps=4,
            lr_scheduler_interval="auto",
        ),
        steps_per_epoch=5,
    )

    assert scheduler.step_on_batch is True
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)

    scheduler.step(metrics=123.0, epoch=2)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)

    scheduler.step_batch(global_step=1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.325)

    scheduler.step_batch(global_step=4)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)


def test_wsd_scheduler_step_schedule_prefers_max_num_updates():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scheduler = LRScheduler(
        optimizer,
        _args(
            max_num_epochs=100,
            max_num_updates=20,
            lr_wsd_warmup_steps=4,
            lr_scheduler_interval="step",
        ),
        steps_per_epoch=5,
    )

    assert scheduler.summary()["num_steps"] == 20


def test_wsd_scheduler_describes_resolved_per_step_schedule():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scheduler = LRScheduler(
        optimizer,
        _args(
            max_num_epochs=4,
            lr_wsd_warmup_steps=0,
            lr_wsd_warmup_ratio=0.25,
            lr_wsd_warmup_start_factor=0.2,
            lr_wsd_stop_lr_ratio=0.001,
            lr_wsd_decay_phase_ratio=0.1,
            lr_wsd_decay_type="cosine",
            lr_scheduler_interval="auto",
        ),
        steps_per_epoch=5,
    )

    assert scheduler.summary() == {
        "scheduler": "WSD",
        "interval": "step",
        "step_on_batch": True,
        "steps_per_epoch": 5,
        "num_steps": 20,
        "warmup_steps": 5,
        "warmup_ratio": 0.25,
        "warmup_start_factor": 0.2,
        "stop_lr_ratio": 0.001,
        "decay_phase_ratio": 0.1,
        "decay_type": "cosine",
    }


def test_wsd_scheduler_state_dict_reloads_current_lr():
    source_param = torch.nn.Parameter(torch.tensor([1.0]))
    source_optimizer = torch.optim.SGD([source_param], lr=1.0)
    source_scheduler = LRScheduler(source_optimizer, _args())
    source_scheduler.step(epoch=9)

    target_param = torch.nn.Parameter(torch.tensor([1.0]))
    target_optimizer = torch.optim.SGD([target_param], lr=1.0)
    target_scheduler = LRScheduler(target_optimizer, _args())
    target_scheduler.load_state_dict(source_scheduler.state_dict())

    assert target_optimizer.param_groups[0]["lr"] == pytest.approx(
        source_optimizer.param_groups[0]["lr"]
    )
    assert target_scheduler.get_last_lr() == pytest.approx(source_scheduler.get_last_lr())
