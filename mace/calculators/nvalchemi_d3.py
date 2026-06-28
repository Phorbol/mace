from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress


_D3_BJ_PARAMETERS: dict[tuple[str, str], dict[str, float]] = {
    ("pbe", "bj"): {"a1": 0.4289, "a2": 4.4407, "s8": 0.7875},
}


def get_d3_bj_parameters(xc: str, damping: str) -> dict[str, float]:
    key = (xc.lower(), damping.lower())
    try:
        return dict(_D3_BJ_PARAMETERS[key])
    except KeyError as exc:
        raise NotImplementedError(
            "The nvalchemi D3 backend currently exposes only PBE-D3(BJ) "
            "because nvalchemi's DFTD3ModelWrapper requires explicit BJ "
            "damping parameters. Use dispersion_backend='torch_dftd' for "
            "other XC/damping combinations until their parameters are mapped "
            "and verified."
        ) from exc


class NvalchemiDFTD3Calculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        xc: str = "pbe",
        damping: str = "bj",
        cutoff: float = 40.0 * units.Bohr,
        smoothing_fraction: float = 0.2,
        auto_download: bool = True,
        param_file: Path | str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.device = device
        self.dtype = dtype
        self.xc = xc
        self.damping = damping
        self.cutoff = cutoff
        params = get_d3_bj_parameters(xc=xc, damping=damping)

        try:
            from nvalchemi.data import AtomicData, Batch
            from nvalchemi.models.dftd3 import DFTD3ModelWrapper
            from nvalchemi.neighbors import compute_neighbors
        except ImportError as exc:
            raise RuntimeError(
                "nvalchemi D3 dispersion requires nvalchemi-toolkit, "
                "nvalchemiops, and their CUDA/Warp runtime dependencies. "
                "Install the NVIDIA nvalchemi stack or use "
                "dispersion_backend='torch_dftd'."
            ) from exc

        self._atomic_data_cls = AtomicData
        self._batch_cls = Batch
        self._compute_neighbors = compute_neighbors
        self.model = DFTD3ModelWrapper(
            **params,
            cutoff=cutoff,
            smoothing_fraction=smoothing_fraction,
            auto_download=auto_download,
            param_file=param_file,
        ).to(device=device, dtype=dtype)

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        data = self._atomic_data_cls.from_atoms(self.atoms)
        batch = self._batch_cls.from_data_list([data], device=self.device)
        self._compute_neighbors(batch, config=self.model.model_config.neighbor_config)

        active_outputs = self.model.model_config.active_outputs
        if "stress" in properties:
            active_outputs.add("stress")
        else:
            active_outputs.discard("stress")

        output = self.model(batch)
        energy = output["energy"].detach().reshape(-1)[0].cpu().item()
        forces = output["forces"].detach().cpu().numpy()

        self.results["energy"] = energy
        self.results["forces"] = np.asarray(forces, dtype=float)
        if "stress" in output:
            stress = output["stress"].detach().reshape(-1, 3, 3)[0].cpu().numpy()
            self.results["stress"] = full_3x3_to_voigt_6_stress(stress)
