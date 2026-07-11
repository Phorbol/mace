from importlib import import_module

_LAZY_EXPORTS = {
    "TensorDict": ".torch_tools",
    "AtomicNumberTable": ".utils",
    "atomic_numbers_to_indices": ".utils",
    "to_numpy": ".torch_tools",
    "to_one_hot": ".torch_tools",
    "build_default_arg_parser": ".arg_parser",
    "check_args": ".arg_parser_tools",
    "DefaultKeys": ".default_keys",
    "set_seeds": ".torch_tools",
    "init_device": ".torch_tools",
    "setup_logger": ".utils",
    "get_tag": ".utils",
    "count_parameters": ".torch_tools",
    "MetricsLogger": ".utils",
    "get_atomic_number_table_from_zs": ".utils",
    "train": ".train",
    "evaluate": ".train",
    "SWAContainer": ".train",
    "CheckpointHandler": ".checkpoint",
    "CheckpointIO": ".checkpoint",
    "CheckpointState": ".checkpoint",
    "set_default_dtype": ".torch_tools",
    "compute_mae": ".utils",
    "compute_rel_mae": ".utils",
    "compute_rmse": ".utils",
    "compute_rel_rmse": ".utils",
    "compute_q95": ".utils",
    "compute_c": ".utils",
    "U_matrix_real": ".cg",
    "spherical_to_cartesian": ".torch_tools",
    "cartesian_to_spherical": ".torch_tools",
    "voigt_to_matrix": ".torch_tools",
    "init_wandb": ".torch_tools",
    "load_foundations": ".finetuning_utils",
    "load_foundations_elements": ".finetuning_utils",
    "build_preprocess_arg_parser": ".arg_parser",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted((*globals(), *__all__))
