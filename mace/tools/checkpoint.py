###########################################################################################
# Checkpointing
# Authors: Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import dataclasses
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

Checkpoint = Dict[str, Any]


@dataclasses.dataclass
class CheckpointState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR


class CheckpointBuilder:
    @staticmethod
    def create_checkpoint(
        state: CheckpointState, metadata: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        checkpoint = {
            "model": state.model.state_dict(),
            "optimizer": state.optimizer.state_dict(),
            "lr_scheduler": state.lr_scheduler.state_dict(),
        }
        if metadata is not None:
            checkpoint["metadata"] = metadata
        return checkpoint

    @staticmethod
    def load_checkpoint(
        state: CheckpointState, checkpoint: Checkpoint, strict: bool
    ) -> None:
        state.model.load_state_dict(checkpoint["model"], strict=strict)  # type: ignore
        state.optimizer.load_state_dict(checkpoint["optimizer"])
        state.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])


@dataclasses.dataclass
class CheckpointPathInfo:
    path: str
    tag: str
    epochs: int
    swa: bool
    updates: Optional[int] = None
    kind: str = "epoch"


@dataclasses.dataclass
class CheckpointLoadResult:
    epoch: int
    update: Optional[int] = None
    kind: str = "epoch"


class CheckpointIO:
    def __init__(
        self,
        directory: str,
        tag: str,
        keep: bool = False,
        swa_start: int = None,
        swa_start_update: int = None,
    ) -> None:
        self.directory = directory
        self.tag = tag
        self.keep = keep
        self.old_path: Optional[str] = None
        self.swa_start = swa_start
        self.swa_start_update = swa_start_update

        self._epochs_string = "_epoch-"
        self._updates_string = "_update-"
        self._filename_extension = "pt"

    def _is_swa_checkpoint(
        self,
        *,
        epochs: int,
        updates: Optional[int],
        checkpoint_kind: str,
        swa_start: Optional[int],
    ) -> bool:
        if (
            checkpoint_kind == "update"
            and self.swa_start_update is not None
            and updates is not None
        ):
            return updates >= self.swa_start_update
        return swa_start is not None and epochs >= swa_start

    def _get_checkpoint_filename(
        self,
        epochs: int,
        swa_start=None,
        updates: Optional[int] = None,
        checkpoint_kind: str = "epoch",
    ) -> str:
        if checkpoint_kind == "update":
            if updates is None:
                raise ValueError("updates must be provided for update checkpoints")
            suffix = f"{self._updates_string}{updates}"
            if self._is_swa_checkpoint(
                epochs=epochs,
                updates=updates,
                checkpoint_kind=checkpoint_kind,
                swa_start=swa_start,
            ):
                suffix += "_swa"
            return self.tag + suffix + "." + self._filename_extension
        if checkpoint_kind != "epoch":
            raise ValueError(f"Unsupported checkpoint kind: {checkpoint_kind!r}")
        if self._is_swa_checkpoint(
            epochs=epochs,
            updates=updates,
            checkpoint_kind=checkpoint_kind,
            swa_start=swa_start,
        ):
            return (
                self.tag
                + self._epochs_string
                + str(epochs)
                + "_swa"
                + "."
                + self._filename_extension
            )
        return (
            self.tag
            + self._epochs_string
            + str(epochs)
            + "."
            + self._filename_extension
        )

    def _list_file_paths(self) -> List[str]:
        if not os.path.isdir(self.directory):
            return []
        all_paths = [
            os.path.join(self.directory, f) for f in os.listdir(self.directory)
        ]
        return [path for path in all_paths if os.path.isfile(path)]

    def _parse_checkpoint_path(self, path: str) -> Optional[CheckpointPathInfo]:
        filename = os.path.basename(path)
        regex = re.compile(
            rf"^(?P<tag>.+){self._epochs_string}(?P<epochs>\d+)\.{self._filename_extension}$"
        )
        regex2 = re.compile(
            rf"^(?P<tag>.+){self._epochs_string}(?P<epochs>\d+)_swa\.{self._filename_extension}$"
        )
        update_regex = re.compile(
            rf"^(?P<tag>.+){self._updates_string}(?P<updates>\d+)\.{self._filename_extension}$"
        )
        update_regex2 = re.compile(
            rf"^(?P<tag>.+){self._updates_string}(?P<updates>\d+)_swa\.{self._filename_extension}$"
        )
        match = regex.match(filename)
        match2 = regex2.match(filename)
        update_match = update_regex.match(filename)
        update_match2 = update_regex2.match(filename)
        swa = False
        if match or match2:
            if not match:
                match = match2
                swa = True
            return CheckpointPathInfo(
                path=path,
                tag=match.group("tag"),
                epochs=int(match.group("epochs")),
                swa=swa,
            )
        if update_match or update_match2:
            if not update_match:
                update_match = update_match2
                swa = True
            return CheckpointPathInfo(
                path=path,
                tag=update_match.group("tag"),
                epochs=0,
                updates=int(update_match.group("updates")),
                swa=swa,
                kind="update",
            )
        return None

    def _get_latest_checkpoint_path(
        self, swa, prefer_update: bool = False
    ) -> Optional[str]:
        all_file_paths = self._list_file_paths()
        checkpoint_info_list = [
            self._parse_checkpoint_path(path) for path in all_file_paths
        ]
        selected_checkpoint_info_list = [
            info for info in checkpoint_info_list if info and info.tag == self.tag
        ]

        if len(selected_checkpoint_info_list) == 0:
            logging.warning(
                f"Cannot find checkpoint with tag '{self.tag}' in '{self.directory}'"
            )
            return None

        selected_checkpoint_info_list_swa = []
        selected_checkpoint_info_list_no_swa = []

        for ckp in selected_checkpoint_info_list:
            if ckp.swa:
                selected_checkpoint_info_list_swa.append(ckp)
            else:
                selected_checkpoint_info_list_no_swa.append(ckp)
        candidates = (
            selected_checkpoint_info_list_swa
            if swa
            else selected_checkpoint_info_list_no_swa
        )
        if not candidates:
            if swa:
                logging.warning(
                    "No SWA checkpoint found, while SWA is enabled. Compare the swa_start parameter and the latest checkpoint."
                )
            return None
        if prefer_update:
            update_candidates = [info for info in candidates if info.kind == "update"]
            if update_candidates:
                latest_checkpoint_info = max(
                    update_candidates,
                    key=lambda info: info.updates if info.updates is not None else -1,
                )
                return latest_checkpoint_info.path
        epoch_candidates = [info for info in candidates if info.kind == "epoch"]
        if epoch_candidates:
            latest_checkpoint_info = max(
                epoch_candidates, key=lambda info: info.epochs
            )
        else:
            latest_checkpoint_info = max(
                candidates,
                key=lambda info: info.updates if info.updates is not None else -1,
            )
        return latest_checkpoint_info.path

    def save(
        self,
        checkpoint: Checkpoint,
        epochs: int,
        keep_last: bool = False,
        updates: Optional[int] = None,
        checkpoint_kind: str = "epoch",
    ) -> None:
        if not self.keep and self.old_path and not keep_last:
            logging.debug(f"Deleting old checkpoint file: {self.old_path}")
            os.remove(self.old_path)

        filename = self._get_checkpoint_filename(
            epochs,
            self.swa_start,
            updates=updates,
            checkpoint_kind=checkpoint_kind,
        )
        path = os.path.join(self.directory, filename)
        logging.debug(f"Saving checkpoint: {path}")
        os.makedirs(self.directory, exist_ok=True)
        torch.save(obj=checkpoint, f=path)
        self.old_path = path

    def load_latest(
        self, swa: Optional[bool] = False, device: Optional[torch.device] = None
    ) -> Optional[Tuple[Checkpoint, int]]:
        path = self._get_latest_checkpoint_path(swa=swa)
        if path is None:
            return None

        return self.load(path, device=device)

    def load_latest_with_metadata(
        self,
        swa: Optional[bool] = False,
        device: Optional[torch.device] = None,
        prefer_update: bool = False,
    ) -> Optional[Tuple[Checkpoint, CheckpointLoadResult]]:
        path = self._get_latest_checkpoint_path(
            swa=swa, prefer_update=prefer_update
        )
        if path is None:
            return None

        return self.load_with_metadata(path, device=device)

    def load(
        self, path: str, device: Optional[torch.device] = None
    ) -> Tuple[Checkpoint, int]:
        checkpoint_info = self._parse_checkpoint_path(path)

        if checkpoint_info is None:
            raise RuntimeError(f"Cannot find path '{path}'")

        logging.info(f"Loading checkpoint: {checkpoint_info.path}")
        return (
            torch.load(f=checkpoint_info.path, map_location=device),
            checkpoint_info.epochs,
        )

    def load_with_metadata(
        self, path: str, device: Optional[torch.device] = None
    ) -> Tuple[Checkpoint, CheckpointLoadResult]:
        checkpoint_info = self._parse_checkpoint_path(path)

        if checkpoint_info is None:
            raise RuntimeError(f"Cannot find path '{path}'")

        logging.info(f"Loading checkpoint: {checkpoint_info.path}")
        checkpoint = torch.load(f=checkpoint_info.path, map_location=device)
        metadata = checkpoint.get("metadata", {})
        kind = str(metadata.get("kind", checkpoint_info.kind))
        epoch = int(metadata.get("epoch", checkpoint_info.epochs))
        update = metadata.get("update", checkpoint_info.updates)
        update = None if update is None else int(update)
        return checkpoint, CheckpointLoadResult(
            epoch=epoch,
            update=update,
            kind=kind,
        )


class CheckpointHandler:
    def __init__(self, *args, **kwargs) -> None:
        self.io = CheckpointIO(*args, **kwargs)
        self.builder = CheckpointBuilder()

    def save(
        self,
        state: CheckpointState,
        epochs: int,
        keep_last: bool = False,
        updates: Optional[int] = None,
        checkpoint_kind: str = "epoch",
    ) -> None:
        metadata = {"epoch": epochs, "kind": checkpoint_kind}
        if updates is not None:
            metadata["update"] = updates
        checkpoint = self.builder.create_checkpoint(state, metadata=metadata)
        self.io.save(
            checkpoint,
            epochs,
            keep_last,
            updates=updates,
            checkpoint_kind=checkpoint_kind,
        )

    def load_latest(
        self,
        state: CheckpointState,
        swa: Optional[bool] = False,
        device: Optional[torch.device] = None,
        strict=False,
    ) -> Optional[int]:
        result = self.io.load_latest(swa=swa, device=device)
        if result is None:
            return None

        checkpoint, epochs = result
        self.builder.load_checkpoint(state=state, checkpoint=checkpoint, strict=strict)
        return epochs

    def load_latest_with_metadata(
        self,
        state: CheckpointState,
        swa: Optional[bool] = False,
        device: Optional[torch.device] = None,
        strict=False,
        prefer_update: bool = False,
    ) -> Optional[CheckpointLoadResult]:
        result = self.io.load_latest_with_metadata(
            swa=swa, device=device, prefer_update=prefer_update
        )
        if result is None:
            return None

        checkpoint, checkpoint_result = result
        self.builder.load_checkpoint(state=state, checkpoint=checkpoint, strict=strict)
        return checkpoint_result

    def load(
        self,
        state: CheckpointState,
        path: str,
        strict=False,
        device: Optional[torch.device] = None,
    ) -> int:
        checkpoint, epochs = self.io.load(path, device=device)
        self.builder.load_checkpoint(state=state, checkpoint=checkpoint, strict=strict)
        return epochs
