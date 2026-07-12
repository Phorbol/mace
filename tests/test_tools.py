import tempfile

import numpy as np
import torch
import torch.nn.functional
from torch import nn, optim

from mace.tools import (
    AtomicNumberTable,
    CheckpointHandler,
    CheckpointState,
    atomic_numbers_to_indices,
)


def test_atomic_number_table():
    table = AtomicNumberTable(zs=[1, 8])
    array = np.array([8, 8, 1])
    indices = atomic_numbers_to_indices(array, z_table=table)
    expected = np.array([1, 1, 0], dtype=int)
    assert np.allclose(expected, indices)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)

    def forward(self, x):
        return torch.nn.functional.relu(self.linear(x))


def test_save_load():
    model = MyModel()
    initial_lr = 0.001
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

    with tempfile.TemporaryDirectory() as directory:
        handler = CheckpointHandler(directory=directory, tag="test", keep=True)
        handler.save(state=CheckpointState(model, optimizer, scheduler), epochs=50)

        optimizer.step()
        scheduler.step()
        assert not np.isclose(optimizer.param_groups[0]["lr"], initial_lr)

        handler.load_latest(state=CheckpointState(model, optimizer, scheduler))
        assert np.isclose(optimizer.param_groups[0]["lr"], initial_lr)


def test_update_checkpoint_load_latest_preserves_epoch_and_update():
    model = MyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

    with tempfile.TemporaryDirectory() as directory:
        handler = CheckpointHandler(directory=directory, tag="test", keep=True)
        handler.save(
            state=CheckpointState(model, optimizer, scheduler),
            epochs=2,
            updates=7,
            checkpoint_kind="update",
        )

        result = handler.load_latest_with_metadata(
            state=CheckpointState(model, optimizer, scheduler),
            prefer_update=True,
        )

    assert result is not None
    assert result.epoch == 2
    assert result.update == 7
    assert result.kind == "update"



def test_update_checkpoint_uses_update_stage_two_boundary_for_swa():
    model = MyModel()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

    with tempfile.TemporaryDirectory() as directory:
        handler = CheckpointHandler(
            directory=directory,
            tag="test",
            keep=True,
            swa_start=999,
            swa_start_update=10,
        )
        handler.save(
            state=CheckpointState(model, optimizer, scheduler),
            epochs=2,
            updates=12,
            checkpoint_kind="update",
        )

        result = handler.load_latest_with_metadata(
            state=CheckpointState(model, optimizer, scheduler),
            swa=True,
            prefer_update=True,
        )

    assert result is not None
    assert result.epoch == 2
    assert result.update == 12
    assert result.kind == "update"
