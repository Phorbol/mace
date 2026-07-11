###########################################################################################
# Training script
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import dataclasses
import logging
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from torch.optim import LBFGS
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_ema import ExponentialMovingAverage
from torchmetrics import Metric

from mace.cli.visualise_train import TrainingPlotter

from . import torch_geometric
from .checkpoint import CheckpointHandler, CheckpointState
from .force_compile import disable_functorch_donated_buffer
from .precision import (
    TrainingPrecisionConfig,
    get_autocast_context,
    get_float32_matmul_precision_context,
)
from .torch_tools import to_numpy
from .training_guards import (
    LossSkipController,
    NonFiniteGradGuard,
    TrainingGuardConfig,
    stable_clip_grad_norm_,
)
from .utils import (
    MetricsLogger,
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
    filter_nonzero_weight,
)


_EDGE_FORCE_COMPILE_SETUP_PHASES = (
    "input_prep",
    "trace",
    "gate_compile",
    "gate_reference",
    "gate_candidate",
    "training_compile",
)


@dataclasses.dataclass
class SWAContainer:
    model: AveragedModel
    scheduler: SWALR
    start: int
    loss_fn: torch.nn.Module


def _make_loss_skip_controller(
    guard_config: TrainingGuardConfig,
) -> LossSkipController | None:
    if not guard_config.loss_skip:
        return None
    return LossSkipController(
        manual_threshold=guard_config.loss_skip_threshold,
        start_step=guard_config.loss_skip_start_step,
        ema_window=guard_config.loss_skip_ema_window,
        multiplier=guard_config.loss_skip_multiplier,
        skip_nan=guard_config.loss_skip_nan,
        skip_large=guard_config.loss_skip_large,
    )


def _save_checkpoint_after_guard(
    *,
    checkpoint_handler: CheckpointHandler,
    state: CheckpointState,
    epochs: int,
    keep_last: bool,
    grad_guard: NonFiniteGradGuard | None,
    named_parameters,
) -> None:
    if grad_guard is not None:
        grad_guard.raise_if_nonfinite(named_parameters)
    checkpoint_handler.save(state=state, epochs=epochs, keep_last=keep_last)


def valid_err_log(
    valid_loss,
    eval_metrics,
    logger,
    log_errors,
    epoch=None,
    valid_loader_name="Default",
):
    eval_metrics["mode"] = "eval"
    eval_metrics["epoch"] = epoch
    eval_metrics["head"] = valid_loader_name
    logger.log(eval_metrics)
    if epoch is None:
        inintial_phrase = "Initial"
    else:
        inintial_phrase = f"Epoch {epoch}"
    if log_errors == "PerAtomRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A"
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_stress"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_stress = eval_metrics["rmse_stress"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_stress={error_stress:8.2f} meV / A^3",
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_virials = eval_metrics["rmse_virials_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_virials_per_atom={error_virials:8.2f} meV",
        )
    elif (
        log_errors == "PerAtomMAEstressvirials"
        and eval_metrics["mae_stress_per_atom"] is not None
    ):
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        error_stress = eval_metrics["mae_stress"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A, MAE_stress={error_stress:8.2f} meV / A^3"
        )
    elif (
        log_errors == "PerAtomMAEstressvirials"
        and eval_metrics["mae_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        error_virials = eval_metrics["mae_virials"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A, MAE_virials={error_virials:8.2f} meV"
        )
    elif log_errors == "TotalRMSE":
        error_e = eval_metrics["rmse_e"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "PerAtomMAE":
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E_per_atom={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "TotalMAE":
        error_e = eval_metrics["mae_e"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, MAE_E={error_e:8.2f} meV, MAE_F={error_f:8.2f} meV / A",
        )
    elif log_errors == "DipoleRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_MU_per_atom={error_mu:8.2f} mDebye",
        )
    elif log_errors == "DipolePolarRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        error_polarizability = eval_metrics["rmse_polarizability_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:.4f}, RMSE_MU_per_atom={error_mu:.2f} me A, RMSE_polarizability_per_atom={error_polarizability:.2f} me A^2 / V",
        )
    elif log_errors == "EnergyDipoleRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"{inintial_phrase}: head: {valid_loader_name}, loss={valid_loss:8.8f}, RMSE_E_per_atom={error_e:8.2f} meV, RMSE_F={error_f:8.2f} meV / A, RMSE_Mu_per_atom={error_mu:8.2f} mDebye",
        )


def train(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    valid_loaders: Dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
    start_epoch: int,
    max_num_epochs: int,
    patience: int,
    checkpoint_handler: CheckpointHandler,
    logger: MetricsLogger,
    eval_interval: int,
    output_args: Dict[str, bool],
    device: torch.device,
    log_errors: str,
    swa: Optional[SWAContainer] = None,
    ema: Optional[ExponentialMovingAverage] = None,
    max_grad_norm: Optional[float] = 10.0,
    log_wandb: bool = False,
    distributed: bool = False,
    save_all_checkpoints: bool = False,
    plotter: TrainingPlotter = None,
    distributed_model: Optional[DistributedDataParallel] = None,
    train_sampler: Optional[DistributedSampler] = None,
    rank: Optional[int] = 0,
    precision_config: Optional[TrainingPrecisionConfig] = None,
    training_model: Optional[torch.nn.Module] = None,
    non_blocking_transfer: bool = False,
    guard_config: Optional[TrainingGuardConfig] = None,
):
    lowest_loss = np.inf
    valid_loss = np.inf
    patience_counter = 0
    swa_start = True
    keep_last = False
    if log_wandb:
        import wandb

    if guard_config is None:
        guard_config = TrainingGuardConfig()
    loss_skip_controller = _make_loss_skip_controller(guard_config)
    nonfinite_grad_guard = (
        NonFiniteGradGuard() if guard_config.nonfinite_grad_guard else None
    )

    if max_grad_norm is not None:
        logging.info(f"Using gradient clipping with tolerance={max_grad_norm:.3f}")

    logging.info("")
    logging.info("===========TRAINING===========")
    logging.info("Started training, reporting errors on validation set")
    logging.info("Loss metrics on validation set")
    epoch = start_epoch

    # log validation loss before _any_ training
    for valid_loader_name, valid_loader in valid_loaders.items():
        valid_loss_head, eval_metrics = evaluate(
            model=model,
            loss_fn=loss_fn,
            data_loader=valid_loader,
            output_args=output_args,
            device=device,
            non_blocking_transfer=non_blocking_transfer,
        )
        valid_err_log(
            valid_loss_head, eval_metrics, logger, log_errors, None, valid_loader_name
        )
    valid_loss = valid_loss_head  # consider only the last head for the checkpoint

    # variable used for broadcast by rank == 0 if epoch loop is exited early, e.g. patience
    exit_now = torch.zeros(1, device=device) if distributed else None
    while epoch < max_num_epochs:
        # LR scheduler and SWA update
        if swa is None or epoch < swa.start:
            if epoch > start_epoch:
                lr_scheduler.step(
                    metrics=valid_loss
                )  # Can break if exponential LR, TODO fix that!
        else:
            if swa_start:
                logging.info("Changing loss based on Stage Two Weights")
                lowest_loss = np.inf
                swa_start = False
                keep_last = True
            loss_fn = swa.loss_fn
            swa.model.update_parameters(model)
            if epoch > start_epoch and not getattr(lr_scheduler, "step_on_batch", False):
                swa.scheduler.step()

        # Train
        if distributed:
            train_sampler.set_epoch(epoch)
        if "ScheduleFree" in type(optimizer).__name__:
            optimizer.train()
        try:
            global_step_start = epoch * len(train_loader)
        except TypeError:
            global_step_start = epoch
        train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            ema=ema,
            logger=logger,
            device=device,
            distributed=distributed,
            distributed_model=distributed_model,
            rank=rank,
            precision_config=precision_config,
            training_model=training_model,
            non_blocking_transfer=non_blocking_transfer,
            guard_config=guard_config,
            loss_skip_controller=loss_skip_controller,
            nonfinite_grad_guard=nonfinite_grad_guard,
            global_step_start=global_step_start,
            lr_scheduler=lr_scheduler,
        )
        if distributed:
            torch.distributed.barrier()

        # Validate
        if epoch % eval_interval == 0:
            model_to_evaluate = (
                model if distributed_model is None else distributed_model
            )
            param_context = (
                ema.average_parameters() if ema is not None else nullcontext()
            )
            if "ScheduleFree" in type(optimizer).__name__:
                optimizer.eval()
            with param_context:
                wandb_log_dict = {}
                for valid_loader_name, valid_loader in valid_loaders.items():
                    valid_loss_head, eval_metrics = evaluate(
                        model=model_to_evaluate,
                        loss_fn=loss_fn,
                        data_loader=valid_loader,
                        output_args=output_args,
                        device=device,
                        non_blocking_transfer=non_blocking_transfer,
                    )
                    if rank == 0:
                        valid_err_log(
                            valid_loss_head,
                            eval_metrics,
                            logger,
                            log_errors,
                            epoch,
                            valid_loader_name,
                        )
                        if log_wandb:
                            wandb_log_dict[valid_loader_name] = {
                                "epoch": epoch,
                                "valid_loss": valid_loss_head,
                                "valid_rmse_e_per_atom": eval_metrics[
                                    "rmse_e_per_atom"
                                ],
                                "valid_rmse_f": eval_metrics["rmse_f"],
                            }
                if plotter and epoch % plotter.plot_frequency == 0:
                    try:
                        plotter.plot(epoch, model_to_evaluate, rank)
                    except Exception as e:  # pylint: disable=broad-except
                        logging.debug(f"Plotting failed: {e}")
                valid_loss = (
                    valid_loss_head  # consider only the last head for the checkpoint
                )
            if log_wandb:
                wandb.log(wandb_log_dict)
            if rank == 0:
                if valid_loss >= lowest_loss:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if swa is not None and epoch < swa.start:
                            logging.info(
                                f"Stopping optimization after {patience_counter} epochs without improvement and starting Stage Two"
                            )
                            epoch = swa.start
                        else:
                            logging.info(
                                f"Stopping optimization after {patience_counter} epochs without improvement"
                            )
                            if exit_now is not None:
                                exit_now.fill_(1)
                    if save_all_checkpoints:
                        param_context = (
                            ema.average_parameters()
                            if ema is not None
                            else nullcontext()
                        )
                        with param_context:
                            _save_checkpoint_after_guard(
                                checkpoint_handler=checkpoint_handler,
                                state=CheckpointState(model, optimizer, lr_scheduler),
                                epochs=epoch,
                                keep_last=True,
                                grad_guard=nonfinite_grad_guard,
                                named_parameters=model.named_parameters,
                            )
                else:
                    lowest_loss = valid_loss
                    patience_counter = 0
                    param_context = (
                        ema.average_parameters() if ema is not None else nullcontext()
                    )
                    with param_context:
                        _save_checkpoint_after_guard(
                            checkpoint_handler=checkpoint_handler,
                            state=CheckpointState(model, optimizer, lr_scheduler),
                            epochs=epoch,
                            keep_last=keep_last,
                            grad_guard=nonfinite_grad_guard,
                            named_parameters=model.named_parameters,
                        )
                        keep_last = False or save_all_checkpoints
        if distributed:
            torch.distributed.barrier()
        if exit_now is not None:
            torch.distributed.broadcast(exit_now, src=0)
            if exit_now == 1:
                break

        epoch += 1

    logging.info("Training complete")


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    ema: Optional[ExponentialMovingAverage],
    logger: MetricsLogger,
    device: torch.device,
    distributed: bool,
    distributed_model: Optional[DistributedDataParallel] = None,
    rank: Optional[int] = 0,
    precision_config: Optional[TrainingPrecisionConfig] = None,
    training_model: Optional[torch.nn.Module] = None,
    non_blocking_transfer: bool = False,
    guard_config: Optional[TrainingGuardConfig] = None,
    loss_skip_controller: Optional[LossSkipController] = None,
    nonfinite_grad_guard: Optional[NonFiniteGradGuard] = None,
    global_step_start: int = 0,
    lr_scheduler: Optional[Any] = None,
) -> None:
    if distributed_model is not None:
        model_to_train = distributed_model
    else:
        model_to_train = training_model if training_model is not None else model

    if guard_config is None:
        guard_config = TrainingGuardConfig()

    edge_force_summary = defaultdict(int)
    edge_force_reasons = defaultdict(int)
    edge_force_buckets = defaultdict(int)
    edge_force_compile_setup_seconds = 0.0
    edge_force_compile_phase_seconds = defaultdict(float)
    edge_force_step_seconds = 0.0
    edge_force_parity_max = defaultdict(float)
    edge_force_parity_worst_grad = "none"
    edge_force_parity_failed_names = set()
    edge_force_fixed_probe_max = defaultdict(float)
    edge_force_fixed_probe_worst_grad = "none"
    edge_force_fixed_probe_failed_names = set()
    edge_force_fixed_probe_shape = "none"

    def update_edge_force_summary(opt_metrics: Dict[str, Any]) -> None:
        nonlocal edge_force_compile_setup_seconds, edge_force_step_seconds
        nonlocal edge_force_parity_worst_grad, edge_force_fixed_probe_worst_grad
        nonlocal edge_force_fixed_probe_shape
        if "edge_force_compile" not in opt_metrics:
            return
        edge_force_summary["steps"] += 1
        edge_force_step_seconds += float(opt_metrics.get("time", 0.0) or 0.0)
        bucket_atoms = opt_metrics.get("edge_force_bucket_atoms")
        bucket_edges = opt_metrics.get("edge_force_bucket_edges")
        if bucket_atoms is not None and bucket_edges is not None:
            edge_force_buckets[f"{int(bucket_atoms)}x{int(bucket_edges)}"] += 1
        if bool(opt_metrics.get("edge_force_compile")):
            edge_force_summary["compiled_steps"] += 1
            if bool(opt_metrics.get("edge_force_cache_hit")):
                edge_force_summary["cache_hits"] += 1
            else:
                edge_force_summary["new_compiles"] += 1
                edge_force_compile_setup_seconds += float(
                    opt_metrics.get("edge_force_compile_setup_seconds", 0.0) or 0.0
                )
                for phase_name in _EDGE_FORCE_COMPILE_SETUP_PHASES:
                    metric_name = f"edge_force_compile_{phase_name}_seconds"
                    edge_force_compile_phase_seconds[phase_name] += float(
                        opt_metrics.get(metric_name, 0.0) or 0.0
                    )
            if bool(opt_metrics.get("edge_force_runtime_recompile")):
                edge_force_summary["runtime_recompiles"] += 1
        else:
            edge_force_summary["fallback_steps"] += 1
            reason = opt_metrics.get("edge_force_compile_disabled_reason")
            if reason:
                edge_force_reasons[str(reason)] += 1

        if bool(opt_metrics.get("edge_force_parity_check")):
            edge_force_summary["parity_checks"] += 1
            if bool(opt_metrics.get("edge_force_parity_accepted")):
                edge_force_summary["parity_accepted"] += 1
            else:
                edge_force_summary["parity_failed"] += 1
            for key, metric_name in (
                ("energy", "edge_force_parity_energy_max_abs_diff"),
                ("forces", "edge_force_parity_forces_max_abs_diff"),
                ("loss", "edge_force_parity_loss_abs_diff"),
                ("grad", "edge_force_parity_param_grad_max_abs_diff"),
            ):
                edge_force_parity_max[key] = max(
                    edge_force_parity_max[key],
                    float(opt_metrics.get(metric_name, 0.0) or 0.0),
                )
            worst_grad = str(opt_metrics.get("edge_force_parity_param_grad_worst") or "none")
            if worst_grad != "none" and edge_force_parity_max["grad"] == float(
                opt_metrics.get("edge_force_parity_param_grad_max_abs_diff", 0.0) or 0.0
            ):
                edge_force_parity_worst_grad = worst_grad
            failed_names = str(opt_metrics.get("edge_force_parity_failed_check_names") or "")
            for failed_name in failed_names.split(","):
                failed_name = failed_name.strip()
                if failed_name:
                    edge_force_parity_failed_names.add(failed_name)

        if bool(opt_metrics.get("edge_force_fixed_probe_check")):
            edge_force_summary["fixed_probe_checks"] += 1
            if bool(opt_metrics.get("edge_force_fixed_probe_accepted")):
                edge_force_summary["fixed_probe_accepted"] += 1
            else:
                edge_force_summary["fixed_probe_failed"] += 1
            probe_atoms = opt_metrics.get("edge_force_fixed_probe_num_atoms")
            probe_edges = opt_metrics.get("edge_force_fixed_probe_num_edges")
            if probe_atoms is not None and probe_edges is not None:
                edge_force_fixed_probe_shape = f"{int(probe_atoms)}x{int(probe_edges)}"
            for key, metric_name in (
                ("energy", "edge_force_fixed_probe_energy_max_abs_diff"),
                ("forces", "edge_force_fixed_probe_forces_max_abs_diff"),
                ("loss", "edge_force_fixed_probe_loss_abs_diff"),
                ("grad", "edge_force_fixed_probe_param_grad_max_abs_diff"),
            ):
                edge_force_fixed_probe_max[key] = max(
                    edge_force_fixed_probe_max[key],
                    float(opt_metrics.get(metric_name, 0.0) or 0.0),
                )
            worst_grad = str(
                opt_metrics.get("edge_force_fixed_probe_param_grad_worst") or "none"
            )
            if worst_grad != "none" and edge_force_fixed_probe_max["grad"] == float(
                opt_metrics.get("edge_force_fixed_probe_param_grad_max_abs_diff", 0.0)
                or 0.0
            ):
                edge_force_fixed_probe_worst_grad = worst_grad
            failed_names = str(
                opt_metrics.get("edge_force_fixed_probe_failed_check_names") or ""
            )
            for failed_name in failed_names.split(","):
                failed_name = failed_name.strip()
                if failed_name:
                    edge_force_fixed_probe_failed_names.add(failed_name)

    def log_edge_force_summary() -> None:
        if rank != 0 or edge_force_summary["steps"] == 0:
            return
        reason_text = ", ".join(
            f"{reason}:{count}" for reason, count in sorted(edge_force_reasons.items())
        ) or "none"
        bucket_text = ", ".join(
            f"{bucket}:{count}" for bucket, count in sorted(edge_force_buckets.items())
        ) or "none"
        phase_text = ", ".join(
            f"{phase}:{edge_force_compile_phase_seconds[phase]:.3f}"
            for phase in _EDGE_FORCE_COMPILE_SETUP_PHASES
            if edge_force_compile_phase_seconds[phase] > 0.0
        ) or "none"
        parity_text = "none"
        if edge_force_summary["parity_checks"] > 0:
            failed_names = ",".join(sorted(edge_force_parity_failed_names)[:8]) or "none"
            parity_text = (
                f"checks={edge_force_summary['parity_checks']} "
                f"accepted={edge_force_summary['parity_accepted']} "
                f"failed={edge_force_summary['parity_failed']} "
                f"max_energy={edge_force_parity_max['energy']:.3e} "
                f"max_forces={edge_force_parity_max['forces']:.3e} "
                f"max_loss={edge_force_parity_max['loss']:.3e} "
                f"max_grad={edge_force_parity_max['grad']:.3e} "
                f"worst_grad={edge_force_parity_worst_grad} "
                f"failed_names={failed_names}"
            )
        fixed_probe_text = "none"
        if edge_force_summary["fixed_probe_checks"] > 0:
            failed_names = (
                ",".join(sorted(edge_force_fixed_probe_failed_names)[:8]) or "none"
            )
            fixed_probe_text = (
                f"checks={edge_force_summary['fixed_probe_checks']} "
                f"accepted={edge_force_summary['fixed_probe_accepted']} "
                f"failed={edge_force_summary['fixed_probe_failed']} "
                f"shape={edge_force_fixed_probe_shape} "
                f"max_energy={edge_force_fixed_probe_max['energy']:.3e} "
                f"max_forces={edge_force_fixed_probe_max['forces']:.3e} "
                f"max_loss={edge_force_fixed_probe_max['loss']:.3e} "
                f"max_grad={edge_force_fixed_probe_max['grad']:.3e} "
                f"worst_grad={edge_force_fixed_probe_worst_grad} "
                f"failed_names={failed_names}"
            )
        logging.info(
            "Edge-force compile epoch %s summary: steps=%d, compiled=%d, "
            "cache_hits=%d, new_compiles=%d, fallbacks=%d, runtime_recompiles=%d, "
            "compile_setup_seconds=%.3f, opt_step_seconds=%.3f, fallback_reasons=%s, "
            "buckets=%s, setup_phases=%s, parity=%s, fixed_probe=%s",
            epoch,
            edge_force_summary["steps"],
            edge_force_summary["compiled_steps"],
            edge_force_summary["cache_hits"],
            edge_force_summary["new_compiles"],
            edge_force_summary["fallback_steps"],
            edge_force_summary["runtime_recompiles"],
            edge_force_compile_setup_seconds,
            edge_force_step_seconds,
            reason_text,
            bucket_text,
            phase_text,
            parity_text,
            fixed_probe_text,
        )

    if isinstance(optimizer, LBFGS):
        _, opt_metrics = take_step_lbfgs(
            model=model_to_train,
            loss_fn=loss_fn,
            data_loader=data_loader,
            optimizer=optimizer,
            ema=ema,
            output_args=output_args,
            max_grad_norm=max_grad_norm,
            device=device,
            distributed=distributed,
            rank=rank,
            non_blocking_transfer=non_blocking_transfer,
        )
        opt_metrics["mode"] = "opt"
        opt_metrics["epoch"] = epoch
        update_edge_force_summary(opt_metrics)
        if rank == 0:
            logger.log(opt_metrics)
        log_edge_force_summary()
    else:
        for step_index, batch in enumerate(data_loader):
            _, opt_metrics = take_step(
                model=model_to_train,
                loss_fn=loss_fn,
                batch=batch,
                optimizer=optimizer,
                ema=ema,
                output_args=output_args,
                max_grad_norm=max_grad_norm,
                device=device,
                precision_config=precision_config,
                non_blocking_transfer=non_blocking_transfer,
                guard_config=guard_config,
                loss_skip_controller=loss_skip_controller,
                nonfinite_grad_guard=nonfinite_grad_guard,
                global_step=global_step_start + step_index,
            )
            opt_metrics["mode"] = "opt"
            opt_metrics["epoch"] = epoch
            if (
                lr_scheduler is not None
                and getattr(lr_scheduler, "step_on_batch", False)
                and not bool(opt_metrics.get("loss_skipped", False))
            ):
                lr_scheduler.step_batch(global_step=global_step_start + step_index + 1)
                if hasattr(lr_scheduler, "get_last_lr"):
                    opt_metrics["lr"] = lr_scheduler.get_last_lr()[0]
            update_edge_force_summary(opt_metrics)
            if rank == 0:
                logger.log(opt_metrics)
        log_edge_force_summary()


def take_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: torch_geometric.batch.Batch,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
    precision_config: Optional[TrainingPrecisionConfig] = None,
    non_blocking_transfer: bool = False,
    guard_config: Optional[TrainingGuardConfig] = None,
    loss_skip_controller: Optional[LossSkipController] = None,
    nonfinite_grad_guard: Optional[NonFiniteGradGuard] = None,
    global_step: int = 0,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    batch = batch.to(device, non_blocking=non_blocking_transfer)
    batch_dict = batch.to_dict()
    if precision_config is None:
        precision_config = TrainingPrecisionConfig(enabled=False, dtype=None)
    if guard_config is None:
        guard_config = TrainingGuardConfig()
    if guard_config.loss_skip and loss_skip_controller is None:
        loss_skip_controller = _make_loss_skip_controller(guard_config)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        compile_metrics = None
        compiled_force_loss = getattr(model, "compiled_force_training_loss", None)
        has_compiled_force_loss = callable(compiled_force_loss)
        can_use_compiled_force_loss = bool(
            has_compiled_force_loss and output_args["forces"]
        )
        with get_float32_matmul_precision_context(precision_config):
            if can_use_compiled_force_loss:
                loss, compile_metrics = compiled_force_loss(
                    batch=batch,
                    loss_fn=loss_fn,
                    output_args=output_args,
                )
            else:
                with get_autocast_context(precision_config):
                    output = model(
                        batch_dict,
                        training=True,
                        compute_force=output_args["forces"],
                        compute_virials=output_args["virials"],
                        compute_stress=output_args["stress"],
                    )
                    loss = loss_fn(pred=output, ref=batch)
        retain_graph_for_backward = False
        compiled_param_grad_tensors = ()
        if compile_metrics is not None:
            retain_graph_for_backward = bool(
                compile_metrics.pop("_retain_graph_for_backward", False)
            )
            compiled_param_grad_tensors = compile_metrics.pop(
                "_compiled_param_grad_tensors", ()
            )

        skip_result = None
        grad_norm = None
        if guard_config.loss_skip and loss_skip_controller is not None:
            skip_result = loss_skip_controller.check(loss, global_step=global_step)
            if skip_result.skip:
                return loss, skip_result, grad_norm, compile_metrics

        backward_context = (
            disable_functorch_donated_buffer()
            if retain_graph_for_backward
            else nullcontext()
        )
        with backward_context:
            loss.backward(retain_graph=retain_graph_for_backward)
        if compiled_param_grad_tensors:
            named_parameters = dict(model.named_parameters())
            for name, grad_source in compiled_param_grad_tensors:
                parameter = named_parameters.get(name)
                if parameter is None:
                    parameter = named_parameters.get(f"model.{name}")
                if parameter is None:
                    raise KeyError(name)
                if grad_source.grad is None:
                    parameter.grad = None
                    continue
                grad = grad_source.grad.detach().to(
                    device=parameter.device, dtype=parameter.dtype
                )
                if parameter.grad is None:
                    parameter.grad = torch.empty_like(parameter)
                parameter.grad.copy_(grad)
        if max_grad_norm is not None:
            grad_norm = stable_clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
                stable=guard_config.stable_grad_clip,
            )
        elif nonfinite_grad_guard is not None:
            grad_norm = stable_clip_grad_norm_(
                model.parameters(),
                max_norm=float("inf"),
                stable=True,
            )
        if nonfinite_grad_guard is not None and grad_norm is not None:
            nonfinite_grad_guard.update(grad_norm)

        return loss, skip_result, grad_norm, compile_metrics

    try:
        loss, skip_result, grad_norm, compile_metrics = closure()
    except RuntimeError as exc:
        disable_compile_fallback = getattr(model, "disable_compile_fallback", None)
        if disable_compile_fallback is None or not disable_compile_fallback(exc):
            raise
        loss, skip_result, grad_norm, compile_metrics = closure()

    loss_skipped = skip_result.skip if skip_result is not None else False
    if not loss_skipped:
        optimizer.step()

        if ema is not None:
            ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
        "loss_skipped": loss_skipped,
        "loss_skip_reason": (
            skip_result.reason if skip_result is not None else "none"
        ),
    }
    if compile_metrics is not None:
        loss_dict.update(compile_metrics)
    if skip_result is not None:
        loss_dict["loss_skip_threshold"] = skip_result.threshold
        loss_dict["loss_ema"] = skip_result.loss_ema
    if grad_norm is not None:
        loss_dict["grad_norm"] = to_numpy(grad_norm)

    return loss, loss_dict


def take_step_lbfgs(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    output_args: Dict[str, bool],
    max_grad_norm: Optional[float],
    device: torch.device,
    distributed: bool,
    rank: int,
    non_blocking_transfer: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    logging.debug(
        f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )

    total_sample_count = 0
    for batch in data_loader:
        total_sample_count += batch.num_graphs

    if distributed:
        global_sample_count = torch.tensor(total_sample_count, device=device)
        torch.distributed.all_reduce(
            global_sample_count, op=torch.distributed.ReduceOp.SUM
        )
        total_sample_count = global_sample_count.item()

    signal = torch.zeros(1, device=device) if distributed else None

    def closure():
        if distributed:
            if rank == 0:
                signal.fill_(1)
                torch.distributed.broadcast(signal, src=0)

            for param in model.parameters():
                torch.distributed.broadcast(param.data, src=0)

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)

        # Process each batch and then collect the results we pass to the optimizer
        for batch in data_loader:
            batch = batch.to(device, non_blocking=non_blocking_transfer)
            batch_dict = batch.to_dict()
            output = model(
                batch_dict,
                training=True,
                compute_force=output_args["forces"],
                compute_virials=output_args["virials"],
                compute_stress=output_args["stress"],
            )
            batch_loss = loss_fn(pred=output, ref=batch)
            batch_loss = batch_loss * (batch.num_graphs / total_sample_count)

            batch_loss.backward()
            total_loss += batch_loss

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        if distributed:
            torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
        return total_loss

    if distributed:
        if rank == 0:
            loss = optimizer.step(closure)
            signal.fill_(0)
            torch.distributed.broadcast(signal, src=0)
        else:
            while True:
                # Other ranks wait for signals from rank 0
                torch.distributed.broadcast(signal, src=0)
                if signal.item() == 0:
                    break
                if signal.item() == 1:
                    loss = closure()

        for param in model.parameters():
            torch.distributed.broadcast(param.data, src=0)
    else:
        loss = optimizer.step(closure)

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    return loss, loss_dict


# Keep parameters frozen/active after evaluation
@contextmanager
def preserve_grad_state(model):
    # save the original requires_grad state for all parameters
    requires_grad_backup = {param: param.requires_grad for param in model.parameters()}
    try:
        # temporarily disable gradients for all parameters
        for param in model.parameters():
            param.requires_grad = False
        yield  # perform evaluation here
    finally:
        # restore the original requires_grad states
        for param, requires_grad in requires_grad_backup.items():
            param.requires_grad = requires_grad


def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    output_args: Dict[str, bool],
    device: torch.device,
    non_blocking_transfer: bool = False,
) -> Tuple[float, Dict[str, Any]]:

    metrics = MACELoss(loss_fn=loss_fn).to(device)

    start_time = time.time()

    with preserve_grad_state(model):
        for batch in data_loader:
            batch = batch.to(device, non_blocking=non_blocking_transfer)
            batch_dict = batch.to_dict()
            output = model(
                batch_dict,
                training=False,
                compute_force=output_args["forces"],
                compute_virials=output_args["virials"],
                compute_stress=output_args["stress"],
            )
            avg_loss, aux = metrics(batch, output)
    avg_loss, aux = metrics.compute()
    aux["time"] = time.time() - start_time
    metrics.reset()

    return avg_loss, aux


class MACELoss(Metric):
    def __init__(self, loss_fn: torch.nn.Module):
        super().__init__()
        self.loss_fn = loss_fn
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_data", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("E_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("delta_es", default=[], dist_reduce_fx="cat")
        self.add_state("delta_es_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Fs_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fs", default=[], dist_reduce_fx="cat")
        self.add_state("delta_fs", default=[], dist_reduce_fx="cat")
        self.add_state(
            "stress_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_stress", default=[], dist_reduce_fx="cat")
        self.add_state(
            "virials_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_virials", default=[], dist_reduce_fx="cat")
        self.add_state("delta_virials_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state("Mus_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus", default=[], dist_reduce_fx="cat")
        self.add_state("delta_mus_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state(
            "polarizability_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_polarizability", default=[], dist_reduce_fx="cat")
        self.add_state(
            "delta_polarizability_per_atom", default=[], dist_reduce_fx="cat"
        )

    def update(self, batch, output):  # pylint: disable=arguments-differ
        loss = self.loss_fn(pred=output, ref=batch)
        self.total_loss += loss
        self.num_data += batch.num_graphs

        if output.get("energy") is not None and batch.energy is not None:
            self.delta_es.append(batch.energy - output["energy"])
            self.delta_es_per_atom.append(
                (batch.energy - output["energy"]) / (batch.ptr[1:] - batch.ptr[:-1])
            )
            self.E_computed += filter_nonzero_weight(
                batch, self.delta_es, batch.weight, batch.energy_weight
            )
        if output.get("forces") is not None and batch.forces is not None:
            self.fs.append(batch.forces)
            self.delta_fs.append(batch.forces - output["forces"])
            self.Fs_computed += filter_nonzero_weight(
                batch,
                self.delta_fs,
                batch.weight,
                batch.forces_weight,
                spread_atoms=True,
            )
        if output.get("stress") is not None and batch.stress is not None:
            self.delta_stress.append(batch.stress - output["stress"])
            self.stress_computed += filter_nonzero_weight(
                batch, self.delta_stress, batch.weight, batch.stress_weight
            )
        if output.get("virials") is not None and batch.virials is not None:
            self.delta_virials.append(batch.virials - output["virials"])
            self.delta_virials_per_atom.append(
                (batch.virials - output["virials"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
            self.virials_computed += filter_nonzero_weight(
                batch, self.delta_virials, batch.weight, batch.virials_weight
            )
        if output.get("dipole") is not None and batch.dipole is not None:
            self.mus.append(batch.dipole)
            self.delta_mus.append(batch.dipole - output["dipole"])
            self.delta_mus_per_atom.append(
                (batch.dipole - output["dipole"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1)
            )
            self.Mus_computed += filter_nonzero_weight(
                batch,
                self.delta_mus,
                batch.weight,
                batch.dipole_weight,
                spread_quantity_vector=False,
            )
        if (
            output.get("polarizability") is not None
            and batch.polarizability is not None
        ):
            self.delta_polarizability.append(
                batch.polarizability - output["polarizability"]
            )
            self.delta_polarizability_per_atom.append(
                (batch.polarizability - output["polarizability"])
                / (batch.ptr[1:] - batch.ptr[:-1]).unsqueeze(-1).unsqueeze(-1)
            )
            self.polarizability_computed += filter_nonzero_weight(
                batch,
                self.delta_polarizability,
                batch.weight,
                batch.polarizability_weight,
                spread_quantity_vector=False,
            )

    def convert(self, delta: Union[torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        if isinstance(delta, list):
            delta = torch.cat(delta)
        return to_numpy(delta)

    def compute(self):

        class NoneMultiply:
            def __mul__(self, other):
                return NoneMultiply()

            def __rmul__(self, other):
                return NoneMultiply()

            def __imul__(self, other):
                return NoneMultiply()

            def __format__(self, format_spec):
                return str(None)

        aux = defaultdict(NoneMultiply)
        aux["loss"] = to_numpy(self.total_loss / self.num_data).item()
        if self.E_computed:
            delta_es = self.convert(self.delta_es)
            delta_es_per_atom = self.convert(self.delta_es_per_atom)
            aux["mae_e"] = compute_mae(delta_es)
            aux["mae_e_per_atom"] = compute_mae(delta_es_per_atom)
            aux["rmse_e"] = compute_rmse(delta_es)
            aux["rmse_e_per_atom"] = compute_rmse(delta_es_per_atom)
            aux["q95_e"] = compute_q95(delta_es)
        if self.Fs_computed:
            fs = self.convert(self.fs)
            delta_fs = self.convert(self.delta_fs)
            aux["mae_f"] = compute_mae(delta_fs)
            aux["rel_mae_f"] = compute_rel_mae(delta_fs, fs)
            aux["rmse_f"] = compute_rmse(delta_fs)
            aux["rel_rmse_f"] = compute_rel_rmse(delta_fs, fs)
            aux["q95_f"] = compute_q95(delta_fs)
        if self.stress_computed:
            delta_stress = self.convert(self.delta_stress)
            aux["mae_stress"] = compute_mae(delta_stress)
            aux["rmse_stress"] = compute_rmse(delta_stress)
            aux["q95_stress"] = compute_q95(delta_stress)
        if self.virials_computed:
            delta_virials = self.convert(self.delta_virials)
            delta_virials_per_atom = self.convert(self.delta_virials_per_atom)
            aux["mae_virials"] = compute_mae(delta_virials)
            aux["rmse_virials"] = compute_rmse(delta_virials)
            aux["rmse_virials_per_atom"] = compute_rmse(delta_virials_per_atom)
            aux["q95_virials"] = compute_q95(delta_virials)
        if self.Mus_computed:
            mus = self.convert(self.mus)
            delta_mus = self.convert(self.delta_mus)
            delta_mus_per_atom = self.convert(self.delta_mus_per_atom)
            aux["mae_mu"] = compute_mae(delta_mus)
            aux["mae_mu_per_atom"] = compute_mae(delta_mus_per_atom)
            aux["rel_mae_mu"] = compute_rel_mae(delta_mus, mus)
            aux["rmse_mu"] = compute_rmse(delta_mus)
            aux["rmse_mu_per_atom"] = compute_rmse(delta_mus_per_atom)
            aux["rel_rmse_mu"] = compute_rel_rmse(delta_mus, mus)
            aux["q95_mu"] = compute_q95(delta_mus)
        if self.polarizability_computed:
            delta_polarizability = self.convert(self.delta_polarizability)
            delta_polarizability_per_atom = self.convert(
                self.delta_polarizability_per_atom
            )
            aux["mae_polarizability"] = compute_mae(delta_polarizability)
            aux["mae_polarizability_per_atom"] = compute_mae(
                delta_polarizability_per_atom
            )
            aux["rmse_polarizability"] = compute_rmse(delta_polarizability)
            aux["rmse_polarizability_per_atom"] = compute_rmse(
                delta_polarizability_per_atom
            )
            aux["q95_polarizability"] = compute_q95(delta_polarizability)

        return aux["loss"], aux
