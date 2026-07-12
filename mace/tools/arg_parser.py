###########################################################################################
# Parsing functionalities
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse
import os
from typing import Dict, Optional

from .default_keys import DefaultKeys


def build_default_arg_parser() -> argparse.ArgumentParser:
    try:
        import configargparse

        parser = configargparse.ArgumentParser(
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add(
            "--config",
            type=str,
            is_config_file=True,
            help="config file to aggregate options",
        )
    except ImportError:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    # Name and seed
    parser.add_argument("--name", help="experiment name", required=True)
    parser.add_argument("--seed", help="random seed", type=int, default=123)

    # Directories
    parser.add_argument(
        "--work_dir",
        help="set directory for all files and folders",
        type=str,
        default=".",
    )
    parser.add_argument(
        "--log_dir", help="directory for log files", type=str, default=None
    )
    parser.add_argument(
        "--model_dir", help="directory for final model", type=str, default=None
    )
    parser.add_argument(
        "--checkpoints_dir",
        help="directory for checkpoint files",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--results_dir", help="directory for results", type=str, default=None
    )
    parser.add_argument(
        "--downloads_dir", help="directory for downloads", type=str, default=None
    )

    # Device and logging
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda", "mps", "xpu"],
        default="cpu",
    )
    parser.add_argument(
        "--default_dtype",
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument(
        "--train_amp_dtype",
        help="Autocast dtype for training forward pass: none, bf16, or fp16",
        type=str,
        choices=["none", "bf16", "fp16"],
        default="none",
    )
    parser.add_argument(
        "--train_tf32",
        help="Use torch.set_float32_matmul_precision('high') during CUDA training",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--train_compile",
        help="Enable torch.compile for the training model after cueq/OEQ conversion",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--train_compile_mode",
        help="torch.compile mode for training",
        type=str,
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
    )
    parser.add_argument(
        "--train_compile_fullgraph",
        help="Request fullgraph=True for training torch.compile",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--train_compile_allow_fallback",
        help="Continue eager training when training torch.compile setup fails",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--edge_force_compile",
        help="Enable DPA4-style edge-vector force-loss compile for training",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--edge_force_compile_mode",
        help="torch.compile mode for the edge-force FX graph",
        type=str,
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
    )
    parser.add_argument(
        "--edge_force_compile_tracing_mode",
        help="make_fx tracing mode for the edge-force closure",
        type=str,
        choices=["real", "symbolic"],
        default="real",
    )
    parser.add_argument(
        "--edge_force_compile_graph",
        help="Run torch.compile on the repaired edge-force FX graph",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--edge_force_compile_dynamic",
        help="Use dynamic=True for the edge-force FX graph compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--edge_force_compile_reuse_executable",
        help=(
            "Reuse graph-compiled edge-force executables across optimizer steps. "
            "Experimental: default is disabled because retained executables have "
            "failed long-run parity checks on some PyTorch/e3nn stacks."
        ),
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--edge_force_compile_max_executable_reuse_steps",
        help=(
            "Maximum number of graph-compiled edge-force calls to run before "
            "refreshing a reused executable. Use 0 for no age limit."
        ),
        type=int,
        default=0,
    )
    parser.add_argument(
        "--edge_force_compile_shape_padding",
        help="Allow Inductor shape_padding for the edge-force FX graph compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--edge_force_compile_max_fusion_size",
        help="Inductor max_fusion_size for the edge-force FX graph compile",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--edge_force_compile_spherical_harmonics",
        help="Spherical harmonics implementation inside edge-force compile",
        type=str,
        choices=["polynomial", "e3nn"],
        default="polynomial",
    )
    parser.add_argument(
        "--edge_force_compile_force_gradient_mode",
        help=(
            "Force-gradient construction used by edge-force compile: edge keeps "
            "the stable edge-vector leaf path; positions compiles the fuller "
            "positions-to-force derivative path"
        ),
        type=str,
        choices=["edge", "positions"],
        default="edge",
    )
    parser.add_argument(
        "--edge_force_compile_strip_detach",
        help=(
            "Strip aten.detach nodes from the traced edge-force FX graph. "
            "This is disabled by default because broad detach removal can "
            "change higher-order force-training autograd semantics."
        ),
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--edge_force_compile_setup_gate",
        help=(
            "Initial setup equivalence gate for edge-force compile. "
            "Use 'strict' for the safe default, or 'none' only in guarded "
            "benchmarks that rely on cache-hit or periodic parity checks."
        ),
        type=str,
        choices=["strict", "none"],
        default="strict",
    )
    parser.add_argument(
        "--edge_force_compile_cache_hit_gate",
        help=(
            "Gate cache hits against the position-gradient baseline. This is a "
            "diagnostic option and is disabled by default because the full "
            "gradient gate is expensive and can perturb higher-order autograd "
            "state when repeated in the same live training process."
        ),
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--edge_force_compile_direct_closure_check",
        help=(
            "During the strict setup gate, compare the raw force-training "
            "closure against eager before comparing the traced/compiled FX "
            "executable. Diagnostic only; does not change gate semantics."
        ),
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--edge_force_compile_atol",
        help="Absolute tolerance for edge-force compile equivalence gates",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument(
        "--edge_force_compile_rtol",
        help="Relative tolerance for edge-force compile equivalence gates",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument(
        "--edge_force_compile_cache_policy",
        help="Cache policy for edge-force compile",
        type=str,
        choices=["shape", "repeat_only", "bucket", "dynamic", "break_even"],
        default="repeat_only",
    )
    parser.add_argument(
        "--edge_force_compile_min_repeats",
        help="Minimum exact-shape repeats before repeat_only policy compiles",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--edge_force_compile_break_even_expected_remaining_hits",
        help=(
            "Expected remaining hits for a bucket when using the break_even "
            "edge-force compile cache policy"
        ),
        type=int,
        default=0,
    )
    parser.add_argument(
        "--edge_force_compile_max_cache_entries",
        help=(
            "Maximum number of compiled edge-force executables to retain. "
            "Set to 0 to disable executable caching."
        ),
        type=int,
        default=32,
    )
    parser.add_argument(
        "--edge_force_compile_disable_negative_speedup",
        help="Disable compile for shapes or buckets that show negative speedup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--edge_force_compile_bucket_atoms",
        help="Comma-separated atom-count buckets for edge-force compile bucket mode",
        type=str,
        default="",
    )
    parser.add_argument(
        "--edge_force_compile_bucket_edges",
        help="Comma-separated edge-count buckets for edge-force compile bucket mode",
        type=str,
        default="",
    )
    parser.add_argument(
        "--edge_force_compile_bucket_margin",
        help=(
            "Maximum bucket/input size ratio allowed for bucket mode; set to 0 "
            "to disable this lower-fill guard and always choose the nearest "
            "bucket that can hold the input"
        ),
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--edge_force_compile_parity_check_interval",
        help=(
            "Run an eager-vs-compiled edge-force parity diagnostic every N "
            "compiled training steps; 0 disables the diagnostic"
        ),
        type=int,
        default=0,
    )
    parser.add_argument(
        "--edge_force_compile_parity_check_gradients",
        help="Include parameter-gradient comparisons in periodic edge-force parity diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--edge_force_compile_parity_check_strict",
        help="Raise on failed periodic edge-force parity diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--edge_force_compile_fixed_probe_interval",
        help=(
            "Run eager-vs-compiled diagnostics on the first captured compiled "
            "batch every N compiled training steps; 0 disables the diagnostic"
        ),
        type=int,
        default=0,
    )
    parser.add_argument(
        "--edge_force_compile_fixed_probe_gradients",
        help="Include parameter-gradient comparisons in fixed edge-force probe diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--edge_force_compile_fixed_probe_strict",
        help="Raise on failed fixed edge-force probe diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--edge_force_compile_allow_fallback",
        help="Continue eager training if edge-force compile setup or runtime fails",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--distributed",
        help="train in multi-GPU data parallel mode",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--launcher",
        default="slurm",
        choices=["slurm", "torchrun", "mpi", "none"],
        help="How the job was launched",
    )
    parser.add_argument("--log_level", help="log level", type=str, default="INFO")

    parser.add_argument(
        "--plot",
        help="Plot results of training",
        type=str2bool,
        default=True,
    )

    parser.add_argument(
        "--plot_frequency",
        help="Set plotting frequency: '0' for only at the end or an integer N to plot every N epochs.",
        type=int,
        default="0",
    )

    parser.add_argument(
        "--plot_interaction_e",
        help="Whether to plot energy without E0s",
        type=str2bool,
        default=False,
    )

    parser.add_argument(
        "--error_table",
        help="Type of error table produced at the end of the training",
        type=str,
        choices=[
            "PerAtomRMSE",
            "TotalRMSE",
            "PerAtomRMSEstressvirials",
            "PerAtomMAEstressvirials",
            "PerAtomMAE",
            "TotalMAE",
            "DipoleRMSE",
            "DipoleMAE",
            "DipolePolarRMSE",
            "EnergyDipoleRMSE",
        ],
        default="PerAtomRMSE",
    )

    # Model
    parser.add_argument(
        "--model",
        help="model type",
        default="MACE",
        choices=[
            "BOTNet",
            "MACE",
            "ScaleShiftMACE",
            "PolarMACE",
            "MACELES",
            "ScaleShiftBOTNet",
            "AtomicDipolesMACE",
            "AtomicDielectricMACE",
            "EnergyDipolesMACE",
        ],
    )
    parser.add_argument(
        "--r_max", help="distance cutoff (in Ang)", type=float, default=5.0
    )
    parser.add_argument(
        "--radial_type",
        help="type of radial basis functions",
        type=str,
        default="bessel",
        choices=["bessel", "gaussian", "chebyshev"],
    )
    parser.add_argument(
        "--num_radial_basis",
        help="number of radial basis functions",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_cutoff_basis",
        help="number of basis functions for smooth cutoff",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--pair_repulsion",
        help="use pair repulsion term with ZBL potential",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--distance_transform",
        help="use distance transform for radial basis functions",
        default="None",
        choices=["None", "Agnesi", "Soft"],
    )
    parser.add_argument(
        "--apply_cutoff",
        help="apply cutoff to the radial basis functions before MLP",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--use_last_readout_only",
        help="use only the last readout for the final output",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--use_embedding_readout",
        help="use embedding readout for the final output",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--interaction",
        help="name of interaction block",
        type=str,
        default="RealAgnosticResidualInteractionBlock",
        choices=[
            "RealAgnosticResidualInteractionBlock",
            "RealAgnosticAttResidualInteractionBlock",
            "RealAgnosticInteractionBlock",
            "RealAgnosticDensityInteractionBlock",
            "RealAgnosticDensityResidualInteractionBlock",
            "RealAgnosticResidualNonLinearInteractionBlock",
        ],
    )
    parser.add_argument(
        "--interaction_first",
        help="name of interaction block",
        type=str,
        default="RealAgnosticResidualInteractionBlock",
        choices=[
            "RealAgnosticResidualInteractionBlock",
            "RealAgnosticInteractionBlock",
            "RealAgnosticDensityInteractionBlock",
            "RealAgnosticDensityResidualInteractionBlock",
            "RealAgnosticResidualNonLinearInteractionBlock",
        ],
    )
    parser.add_argument(
        "--max_ell", help=r"highest \ell of spherical harmonics", type=int, default=3
    )
    parser.add_argument(
        "--correlation", help="correlation order at each layer", type=int, default=3
    )
    parser.add_argument(
        "--use_reduced_cg",
        help="use reduced generalized Clebsch-Gordan coefficients",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--use_so3",
        help="use SO(3) irreps instead of O(3) irreps",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--use_agnostic_product",
        help="use element agnostic product",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--num_interactions", help="number of interactions", type=int, default=2
    )
    parser.add_argument(
        "--MLP_irreps",
        help="hidden irreps of the MLP in last readout",
        type=str,
        default="16x0e",
    )
    parser.add_argument(
        "--radial_MLP",
        help="width of the radial MLP",
        type=str,
        default="[64, 64, 64]",
    )
    parser.add_argument(
        "--hidden_irreps",
        help="irreps for hidden node states",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--edge_irreps",
        help="irreps for edge states",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_edge_irreps_first",
        help="use edge irreps in the first interaction block",
        type=str2bool,
        default=False,
    )
    # add option to specify irreps by channel number and max L
    parser.add_argument(
        "--num_channels",
        help="number of embedding channels",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_L",
        help="max L equivariance of the message",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--gate",
        help="non linearity for last readout",
        type=str,
        default="silu",
        choices=["silu", "tanh", "abs", "None"],
    )
    parser.add_argument(
        "--kspace_cutoff_factor",
        help="k-space cutoff factor used by PolarMACE",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--atomic_multipoles_max_l",
        help="maximum l for atomic multipoles in PolarMACE",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--atomic_multipoles_smearing_width",
        help="Gaussian smearing width for atomic multipoles in PolarMACE",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--field_feature_max_l",
        help="maximum l for projected field features in PolarMACE",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--field_feature_widths",
        help="list of field feature widths for PolarMACE",
        type=str,
        default="[1.0]",
    )
    parser.add_argument(
        "--field_feature_norms",
        help="optional list of field feature norms for PolarMACE",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num_recursion_steps",
        help="number of fixed-point recursion steps in PolarMACE",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--field_si",
        help="include self-interaction when projecting local fields in PolarMACE",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--include_electrostatic_self_interaction",
        help="include electrostatic self interaction in PolarMACE",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--add_local_electron_energy",
        help="add local electron energy correction in PolarMACE",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--quadrupole_feature_corrections",
        help="enable quadrupole feature corrections in PolarMACE",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--return_electrostatic_potentials",
        help="return electrostatic potentials from PolarMACE forward pass",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--field_norm_factor",
        help="global normalization factor for field features in PolarMACE",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--fixedpoint_update_config",
        help="dict-like config for PolarMACE fixed-point update block",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--field_readout_config",
        help="dict-like config for PolarMACE field readout block",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--scaling",
        help="type of scaling to the output",
        type=str,
        default="rms_forces_scaling",
        choices=["std_scaling", "rms_forces_scaling", "no_scaling"],
    )
    parser.add_argument(
        "--avg_num_neighbors",
        help="normalization factor for the message",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--compute_avg_num_neighbors",
        help="normalization factor for the message",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--compute_stress",
        help="Select True to compute stress",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--compute_forces",
        help="Select True to compute forces",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--compute_polarizability",
        help="Select True to compute polarizability",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--compute_atomic_dipole",
        help="Select True to compute dipoles",
        type=str2bool,
        default=False,
    )

    # Dataset
    parser.add_argument(
        "--train_file",
        help="Training set file, format is .xyz or .h5",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--valid_file",
        help="Validation set .xyz or .h5 file",
        default=None,
        type=str,
        required=False,
    )
    parser.add_argument(
        "--valid_fraction",
        help="Fraction of training set used for validation",
        type=float,
        default=0.1,
        required=False,
    )
    parser.add_argument(
        "--test_file",
        help="Test set .xyz pt .h5 file",
        type=str,
    )
    parser.add_argument(
        "--test_dir",
        help="Path to directory with test files named as test_*.h5",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--multi_processed_test",
        help="Boolean value for whether the test data was multiprocessed",
        type=str2bool,
        default=False,
        required=False,
    )
    parser.add_argument(
        "--num_workers",
        help="Number of workers for data loading",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--pin_memory",
        help="Pin memory for data loading",
        default=True,
        type=str2bool,
    )
    parser.add_argument(
        "--non_blocking_transfer",
        help="Use non-blocking host-to-device batch transfers when supported",
        default=False,
        type=str2bool,
    )
    parser.add_argument(
        "--shuffle",
        help="Shuffle the training dataset",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--atomic_numbers",
        help="List of atomic numbers",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--mean",
        help="Mean energy per atom of training set",
        type=float,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--std",
        help="Standard deviation of force components in the training set",
        type=float,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--statistics_file",
        help="json file containing statistics of training set",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--les_arguments",
        help="Path to the LES arguments file",
        type=read_yaml,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--E0s",
        help="Dictionary of isolated atom energies",
        type=str,
        default=None,
        required=False,
    )

    # Fine-tuning
    parser.add_argument(
        "--pseudolabel_replay",
        help="Use pseudolabels from foundation model for replay data in multihead finetuning",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--pseudolabel_replay_compute_stress",
        help="When replay pseudolabels are generated, always generate stress labels even if the original replay data lacked stress",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--foundation_filter_elements",
        help="Filter element during fine-tuning",
        type=str2bool,
        default=True,
        required=False,
    )
    parser.add_argument(
        "--heads",
        help="Dict of heads: containing individual files and E0s",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--multiheads_finetuning",
        help="Boolean value for whether the model is multiheaded",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--foundation_head",
        help="Name of the head to use for fine-tuning",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--weight_pt_head",
        help="Weight of the pretrained head in the loss function",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--real_pt_data_ratio_threshold",
        help="threshold of real data to replay data below which real data (sum over all real heads) is duplicated",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--num_samples_pt",
        help="Number of samples in the pretrained head",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--force_mh_ft_lr",
        help="Force the multiheaded fine-tuning to use arg_parser lr",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--subselect_pt",
        help="Method to subselect the configurations of the pretraining set",
        choices=["fps", "random"],
        default="random",
    )
    parser.add_argument(
        "--filter_type_pt",
        help="Filtering method for collecting the pretraining set",
        choices=["none", "combinations", "inclusive", "exclusive"],
        default="none",
    )
    parser.add_argument(
        "--disallow_random_padding_pt",
        help="do not allow random padding of the configurations to match the number of samples",
        action="store_false",
        dest="allow_random_padding_pt",
    )
    parser.add_argument(
        "--pt_train_file",
        help="Training set file for the pretrained head",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pt_valid_file",
        help="Validation set file for the pretrained head",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--foundation_model_elements",
        help="Keep all elements of the foundation model during fine-tuning",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--keep_isolated_atoms",
        help="Keep isolated atoms in the dataset, useful for transfer learning",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--lora",
        help="Use Low-Rank Adaptation for the fine-tuning",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--lora_rank",
        help="Rank of the LoRA matrices",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--lora_alpha",
        help="Scaling factor for LoRA",
        type=float,
        default=1.0,
    )

    # Keys
    parser.add_argument(
        "--energy_key",
        help="Key of reference energies in training xyz",
        type=str,
        default=DefaultKeys.ENERGY.value,
    )
    parser.add_argument(
        "--forces_key",
        help="Key of reference forces in training xyz",
        type=str,
        default=DefaultKeys.FORCES.value,
    )
    parser.add_argument(
        "--virials_key",
        help="Key of reference virials in training xyz",
        type=str,
        default=DefaultKeys.VIRIALS.value,
    )
    parser.add_argument(
        "--stress_key",
        help="Key of reference stress in training xyz",
        type=str,
        default=DefaultKeys.STRESS.value,
    )
    parser.add_argument(
        "--dipole_key",
        help="Key of reference dipoles in training xyz",
        type=str,
        default=DefaultKeys.DIPOLE.value,
    )
    parser.add_argument(
        "--polarizability_key",
        help="Key of polarizability in training xyz",
        type=str,
        default=DefaultKeys.POLARIZABILITY.value,
    )
    parser.add_argument(
        "--head_key",
        help="Key of head in training xyz",
        type=str,
        default=DefaultKeys.HEAD.value,
    )
    parser.add_argument(
        "--charges_key",
        help="Key of atomic charges in training xyz",
        type=str,
        default=DefaultKeys.CHARGES.value,
    )
    parser.add_argument(
        "--elec_temp_key",
        help="Key of electronic temperature in training xyz",
        type=str,
        default=DefaultKeys.ELEC_TEMP.value,
    )
    parser.add_argument(
        "--total_spin_key",
        help="Key of total spin in training xyz",
        type=str,
        default=DefaultKeys.TOTAL_SPIN.value,
    )
    parser.add_argument(
        "--total_charge_key",
        help="Key of total charge in training xyz",
        type=str,
        default=DefaultKeys.TOTAL_CHARGE.value,
    )
    parser.add_argument(
        "--embedding_specs",
        help=(
            "Dict of feature‐spec dictionaries. "
            "embedding_specs:\n"
            "  total_spin:\n"
            "    type: categorical\n"
            "    per: graph\n"
            "    num_classes: 101\n"
            "    emb_dim: 64\n"
            "  total_charge:\n"
            "    type: categorical\n"
            "    per: graph\n"
            "    num_classes: 201\n"
            "    emb_dim: 64\n"
            "  temperature:\n"
            "    type: continuous\n"
            "    per: graph\n"
            "    in_dim: 1\n"
            "    emb_dim: 32\n"
        ),
        default=None,
    )
    parser.add_argument(
        "--skip_evaluate_heads",
        help="Comma-separated list of heads to skip during final evaluation",
        type=str,
        default="pt_head",
    )

    # Loss and optimization
    parser.add_argument(
        "--loss",
        help="type of loss",
        default="weighted",
        choices=[
            "ef",
            "weighted",
            "forces_only",
            "virials",
            "stress",
            "dipole",
            "dipole_polar",
            "huber",
            "universal",
            "energy_forces_dipole",
            "l1l2energyforces",
        ],
    )
    parser.add_argument(
        "--forces_weight", help="weight of forces loss", type=float, default=100.0
    )
    parser.add_argument(
        "--swa_forces_weight",
        "--stage_two_forces_weight",
        help="weight of forces loss after starting Stage Two (previously called swa)",
        type=float,
        default=100.0,
        dest="swa_forces_weight",
    )
    parser.add_argument(
        "--energy_weight", help="weight of energy loss", type=float, default=1.0
    )
    parser.add_argument(
        "--swa_energy_weight",
        "--stage_two_energy_weight",
        help="weight of energy loss after starting Stage Two (previously called swa)",
        type=float,
        default=1000.0,
        dest="swa_energy_weight",
    )
    parser.add_argument(
        "--virials_weight", help="weight of virials loss", type=float, default=1.0
    )
    parser.add_argument(
        "--swa_virials_weight",
        "--stage_two_virials_weight",
        help="weight of virials loss after starting Stage Two (previously called swa)",
        type=float,
        default=10.0,
        dest="swa_virials_weight",
    )
    parser.add_argument(
        "--stress_weight", help="weight of stress loss", type=float, default=1.0
    )
    parser.add_argument(
        "--swa_stress_weight",
        "--stage_two_stress_weight",
        help="weight of stress loss after starting Stage Two (previously called swa)",
        type=float,
        default=10.0,
        dest="swa_stress_weight",
    )
    parser.add_argument(
        "--dipole_weight", help="weight of dipoles loss", type=float, default=1.0
    )
    parser.add_argument(
        "--swa_dipole_weight",
        "--stage_two_dipole_weight",
        help="weight of dipoles after starting Stage Two (previously called swa)",
        type=float,
        default=1.0,
        dest="swa_dipole_weight",
    )
    parser.add_argument(
        "--swa_polarizability_weight",
        "--stage_two_polarizability_weight",
        help="weight of polarizability after starting Stage Two (previously called swa)",
        type=float,
        default=1.0,
        dest="swa_polarizability_weight",
    )
    parser.add_argument(
        "--polarizability_weight",
        help="weight of polarizability loss",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--config_type_weights",
        help="String of dictionary containing the weights for each config type",
        type=str,
        default='{"Default":1.0}',
    )
    parser.add_argument(
        "--huber_delta",
        help="delta parameter for huber loss",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--optimizer",
        help="Optimizer for parameter optimization",
        type=str,
        default="adam",
        choices=["adam", "adamw", "schedulefree", "hybrid_muon"],
    )
    parser.add_argument(
        "--beta",
        help="Beta parameter for the optimizer",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--beta1_schedulefree",
        help="Beta1 parameter for the ScheduleFree optimizer",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--beta2_schedulefree",
        help="Beta2 parameter for the ScheduleFree optimizer",
        type=float,
        default=0.98,
    )
    parser.add_argument(
        "--warmup_steps_schedulefree",
        help="Number of linear LR warmup steps for the ScheduleFree optimizer",
        type=int,
        default=0,
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=10)
    parser.add_argument(
        "--valid_batch_size", help="Validation batch size", type=int, default=10
    )
    parser.add_argument(
        "--lr", help="Learning rate of optimizer", type=float, default=0.01
    )
    parser.add_argument(
        "--swa_lr",
        "--stage_two_lr",
        help="Learning rate of optimizer in Stage Two (previously called swa)",
        type=float,
        default=1e-3,
        dest="swa_lr",
    )
    parser.add_argument(
        "--weight_decay", help="weight decay (L2 penalty)", type=float, default=5e-7
    )
    parser.add_argument(
        "--hybrid_muon_weight_decay",
        help="Decoupled weight decay for HybridMuon-routed matrix parameters",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--hybrid_muon_lr_factor",
        help="Learning-rate factor applied only to HybridMuon-routed matrix parameters",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--hybrid_muon_stage_two_lr_factor",
        help=(
            "Extra multiplier applied once to HybridMuon-routed matrix parameter "
            "groups when Stage Two starts. Use 1.0 to keep the Stage One "
            "Muon LR; use 0.0 to freeze Muon-routed matrices in Stage Two."
        ),
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--hybrid_muon_stage_two_route",
        help=(
            "Optimizer route for HybridMuon-routed matrix parameter groups when "
            "Stage Two starts. 'keep' preserves Muon; 'adamw' or 'adam' switch "
            "those groups to the corresponding Adam variant at the base LR."
        ),
        type=str,
        default="keep",
        choices=["keep", "adam", "adamw"],
    )
    parser.add_argument(
        "--hybrid_muon_mode",
        help=(
            "HybridMuon matrix routing mode. '2d' preserves the conservative "
            "MACE default; 'slice' applies Muon independently to trailing "
            "matrices of selected equivariant/path tensors."
        ),
        type=str,
        default="2d",
        choices=["2d", "slice"],
    )
    parser.add_argument(
        "--hybrid_muon_routing",
        help=(
            "HybridMuon parameter coverage policy. 'mace' keeps conservative "
            "MACE-safe routing; 'tace' follows TACE/DPA4-style broad matrix "
            "routing with MACE hard exclusions for embeddings, atomic heads, "
            "biases, norms, and scales; 'module' uses OptimSpec declarations "
            "on owning modules."
        ),
        type=str,
        default="mace",
        choices=["mace", "tace", "module"],
    )
    parser.add_argument(
        "--hybrid_muon_magma_lite",
        help=(
            "Enable DPA4/TACE-style Magma-lite damping for Muon-routed matrix "
            "updates based on momentum-gradient alignment."
        ),
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--hybrid_muon_adam_variant",
        help=(
            "Optimizer variant for HybridMuon fallback parameters. 'adamw' uses "
            "decoupled weight decay; 'adam' uses coupled L2 weight decay."
        ),
        type=str,
        default="adamw",
        choices=["adam", "adamw"],
    )
    parser.add_argument(
        "--hybrid_muon_lr_scale_mode",
        help=(
            "Muon update scaling mode. 'original' preserves the existing "
            "sqrt(max(1, rows / cols)) scale; 'match_rms' uses "
            "hybrid_muon_match_rms_coeff * sqrt(max(rows, cols)); 'none' "
            "disables shape scaling."
        ),
        type=str,
        default="original",
        choices=["original", "match_rms", "none"],
    )
    parser.add_argument(
        "--hybrid_muon_match_rms_coeff",
        help="Coefficient used by --hybrid_muon_lr_scale_mode=match_rms",
        type=float,
        default=0.18,
    )
    parser.add_argument(
        "--hybrid_muon_magma_initial_score",
        help="Initial Magma-lite EMA score for new Muon matrix blocks",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--hybrid_muon_magma_warmup_steps",
        help="Number of Muon steps to update Magma-lite EMA without damping",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--hybrid_muon_magma_bypass_first_step",
        help="Bypass Magma-lite damping on the first update for each matrix block",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--lr_params_factors",
        help="Learning rate factors to multiply on the original lr",
        type=str,
        default='{"embedding_lr_factor": 1.0, "interactions_lr_factor": 1.0, "products_lr_factor": 1.0, "readouts_lr_factor": 1.0}',
    )
    parser.add_argument(
        "--freeze",
        help="Freeze layers from 1 to N. Can be positive or negative, e.g. -1 means the last layer is frozen. 0 or None means all layers are active and is a default setting",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--amsgrad",
        help="use amsgrad variant of optimizer",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--scheduler",
        help="Type of scheduler",
        type=str,
        default="ReduceLROnPlateau",
        choices=["ReduceLROnPlateau", "ExponentialLR", "WSD"],
    )
    parser.add_argument(
        "--lr_factor", help="Learning rate factor", type=float, default=0.8
    )
    parser.add_argument(
        "--scheduler_patience", help="Learning rate factor", type=int, default=50
    )
    parser.add_argument(
        "--lr_scheduler_gamma",
        help="Gamma of learning rate scheduler",
        type=float,
        default=0.9993,
    )
    parser.add_argument(
        "--lr_scheduler_interval",
        help=(
            "When to step the selected LR scheduler. 'auto' uses per-step "
            "updates for WSD and per-epoch updates for legacy schedulers."
        ),
        type=str,
        default="auto",
        choices=["auto", "epoch", "step"],
    )
    parser.add_argument(
        "--lr_wsd_warmup_steps",
        help="Warmup steps for WSD when per-step, or epochs when per-epoch; if >0, overrides --lr_wsd_warmup_ratio",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--lr_wsd_warmup_ratio",
        help="Warmup fraction for WSD scheduler when --lr_wsd_warmup_steps is 0",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--lr_wsd_warmup_start_factor",
        help="Initial WSD warmup LR as a factor of the base LR",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--lr_wsd_stop_lr_ratio",
        help="Final WSD LR as a factor of the base LR",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--lr_wsd_decay_phase_ratio",
        help="Final training fraction used for WSD decay",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--lr_wsd_decay_type",
        help="Decay rule for WSD scheduler",
        type=str,
        default="inverse_linear",
        choices=["inverse_linear", "cosine", "linear"],
    )
    parser.add_argument(
        "--swa",
        "--stage_two",
        help="use Stage Two loss weight, which decreases the learning rate and increases the energy weight at the end of the training to help converge them",
        action="store_true",
        default=False,
        dest="swa",
    )
    parser.add_argument(
        "--start_swa",
        "--start_stage_two",
        help="Number of epochs before changing to Stage Two loss weights",
        type=int,
        default=None,
        dest="start_swa",
    )
    parser.add_argument(
        "--start_swa_update",
        "--start_stage_two_update",
        help="Number of optimizer updates before changing to Stage Two loss weights",
        type=int,
        default=None,
        dest="start_swa_update",
    )
    parser.add_argument(
        "--lbfgs",
        help="Switch to L-BFGS optimizer",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ema",
        help="use Exponential Moving Average",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ema_decay",
        help="Exponential Moving Average decay",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--max_num_epochs", help="Maximum number of epochs", type=int, default=2048
    )
    parser.add_argument(
        "--max_num_updates",
        help="Maximum number of optimizer updates; if set, training may stop mid-epoch",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--patience",
        help="Maximum number of consecutive epochs of increasing loss",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--foundation_model",
        help="Path to the foundation model for transfer learning",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--foundation_model_kwargs",
        help="Additional kwargs for the foundation model for transfer learning",
        type=str,
        default="{}",
    )
    parser.add_argument(
        "--foundation_model_readout",
        help="Use readout of foundation model for transfer learning",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--finetune_dipoles_polarizabilities",
        help="Fine-tune an existing AtomicDielectricMACE (MACE-MDP) model on dipoles and polarizabilities only. Requires --foundation_model pointing to the pretrained MDP checkpoint.",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--eval_interval", help="evaluate model every <n> epochs", type=int, default=1
    )
    parser.add_argument(
        "--eval_interval_updates",
        help="evaluate model every <n> optimizer updates; overrides epoch interval when set",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--checkpoint_interval_updates",
        help="save a keep-last checkpoint every <n> optimizer updates; useful for max_num_updates training",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--keep_checkpoints",
        help="keep all checkpoints",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save_all_checkpoints",
        help="save all checkpoints",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--restart_latest",
        help="restart optimizer from latest checkpoint",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save_cpu",
        help="Save a model to be loaded on cpu",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--clip_grad",
        help="Gradient Clipping Value",
        type=check_float_or_none,
        default=10.0,
    )
    parser.add_argument(
        "--loss_skip",
        help="Skip optimizer steps for non-finite or unusually large losses",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--loss_skip_nan",
        help="Skip optimizer steps when the loss is NaN or infinite",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--loss_skip_large",
        help="Skip optimizer steps when the loss exceeds the configured threshold",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--loss_skip_ema_window",
        help="EMA window used to estimate the dynamic large-loss threshold",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--loss_skip_multiplier",
        help="Multiplier applied to the EMA loss for dynamic large-loss skipping",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--loss_skip_start_step",
        help="Global step at which large-loss skipping starts",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--loss_skip_threshold",
        help="Manual large-loss skip threshold; none uses only the dynamic EMA threshold",
        type=check_float_or_none,
        default=None,
    )
    parser.add_argument(
        "--stable_grad_clip",
        help="Use overflow-stable gradient norm computation for clipping",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--nonfinite_grad_guard",
        help="Raise before checkpointing if a non-finite gradient norm was observed",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--dry_run",
        help="Run all steps upto training to test settings.",
        action="store_true",
        default=False,
    )
    # option for cuequivariance acceleration
    parser.add_argument(
        "--enable_cueq",
        help="Enable cuequivariance acceleration",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--only_cueq",
        help="Only use cuequivariance acceleration",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--cueq_layout",
        "--cueq-layout",
        help="cuequivariance irreps layout used during e3nn-to-cueq conversion",
        type=str,
        choices=["mul_ir", "ir_mul"],
        default="ir_mul",
    )
    parser.add_argument(
        "--cueq_optimize_all",
        "--cueq-optimize-all",
        help="Enable all cuequivariance optimizations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cueq_optimize_linear",
        "--cueq-optimize-linear",
        help="Enable cuequivariance Linear replacement",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--cueq_optimize_channelwise",
        "--cueq-optimize-channelwise",
        help="Enable cuequivariance channelwise tensor products",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--cueq_optimize_symmetric",
        "--cueq-optimize-symmetric",
        help="Enable cuequivariance symmetric contractions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--cueq_optimize_fctp",
        "--cueq-optimize-fctp",
        help="Enable cuequivariance fully connected tensor products",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--cueq_conv_fusion",
        "--cueq-conv-fusion",
        help="Enable cuequivariance convolution fusion; default enables it only on CUDA",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    # option for openequivariance acceleration
    parser.add_argument(
        "--enable_oeq",
        help="Enable openequivariance acceleration",
        type=str2bool,
        default=False,
    )
    # options for using Weights and Biases for experiment tracking
    # to install see https://wandb.ai
    parser.add_argument(
        "--wandb",
        help="Use Weights and Biases for experiment tracking",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--wandb_dir",
        help="An absolute path to a directory where Weights and Biases metadata will be stored",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--wandb_project",
        help="Weights and Biases project name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--wandb_entity",
        help="Weights and Biases entity name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--wandb_name",
        help="Weights and Biases experiment name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--wandb_log_hypers",
        help="The hyperparameters to log in Weights and Biases",
        nargs="+",
        default=[
            "num_channels",
            "max_L",
            "correlation",
            "lr",
            "swa_lr",
            "weight_decay",
            "batch_size",
            "max_num_epochs",
            "start_swa",
            "energy_weight",
            "forces_weight",
        ],
    )
    return parser


def build_preprocess_arg_parser() -> argparse.ArgumentParser:
    try:
        import configargparse

        parser = configargparse.ArgumentParser(
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add(
            "--config",
            type=str,
            is_config_file=True,
            help="config file to aggregate options",
        )
    except ImportError:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument(
        "--train_file",
        help="Training set h5 file",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--valid_file",
        help="Training set xyz file",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--num_process",
        help="The user defined number of processes to use, as well as the number of files created.",
        type=int,
        default=int(os.cpu_count() / 4),
    )
    parser.add_argument(
        "--valid_fraction",
        help="Fraction of training set used for validation",
        type=float,
        default=0.1,
        required=False,
    )
    parser.add_argument(
        "--test_file",
        help="Test set xyz file",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--work_dir",
        help="set directory for all files and folders",
        type=str,
        default=".",
    )
    parser.add_argument(
        "--h5_prefix",
        help="Prefix for h5 files when saving",
        type=str,
        default="",
    )
    parser.add_argument(
        "--r_max", help="distance cutoff (in Ang)", type=float, default=5.0
    )
    parser.add_argument(
        "--config_type_weights",
        help="String of dictionary containing the weights for each config type",
        type=str,
        default='{"Default":1.0}',
    )
    parser.add_argument(
        "--energy_key",
        help="Key of reference energies in training xyz",
        type=str,
        default=DefaultKeys.ENERGY.value,
    )
    parser.add_argument(
        "--forces_key",
        help="Key of reference forces in training xyz",
        type=str,
        default=DefaultKeys.FORCES.value,
    )
    parser.add_argument(
        "--virials_key",
        help="Key of reference virials in training xyz",
        type=str,
        default=DefaultKeys.VIRIALS.value,
    )
    parser.add_argument(
        "--stress_key",
        help="Key of reference stress in training xyz",
        type=str,
        default=DefaultKeys.STRESS.value,
    )
    parser.add_argument(
        "--dipole_key",
        help="Key of reference dipoles in training xyz",
        type=str,
        default=DefaultKeys.DIPOLE.value,
    )
    parser.add_argument(
        "--polarizability_key",
        help="Key of polarizability in training xyz",
        type=str,
        default=DefaultKeys.POLARIZABILITY.value,
    )
    parser.add_argument(
        "--charges_key",
        help="Key of atomic charges in training xyz",
        type=str,
        default=DefaultKeys.CHARGES.value,
    )
    parser.add_argument(
        "--atomic_numbers",
        help="List of atomic numbers",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--compute_statistics",
        help="Compute statistics for the dataset",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--batch_size",
        help="batch size to compute average number of neighbours",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--scaling",
        help="type of scaling to the output",
        type=str,
        default="rms_forces_scaling",
        choices=["std_scaling", "rms_forces_scaling", "no_scaling"],
    )
    parser.add_argument(
        "--E0s",
        help="Dictionary of isolated atom energies",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--shuffle",
        help="Shuffle the training dataset",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--seed",
        help="Random seed for splitting training and validation sets",
        type=int,
        default=123,
    )
    parser.add_argument(
        "--head_key",
        help="Key of head in training xyz",
        type=str,
        default=DefaultKeys.HEAD.value,
    )
    parser.add_argument(
        "--heads",
        help="Dict of heads: containing individual files and E0s",
        type=str,
        default=None,
        required=False,
    )
    return parser


def check_float_or_none(value: str) -> Optional[float]:
    try:
        return float(value)
    except ValueError:
        if value != "None":
            raise argparse.ArgumentTypeError(
                f"{value} is an invalid value (float or None)"
            ) from None
        return None


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def read_yaml(value: str) -> Dict:
    from pathlib import Path

    import yaml

    if not Path(value).is_file():
        raise argparse.ArgumentTypeError(f"File {value} does not exist.")
    with open(value, "r", encoding="utf-8") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise argparse.ArgumentTypeError(
                f"Error parsing YAML file {value}: {exc}"
            ) from exc
