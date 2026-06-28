from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load_parse_metrics():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmarks"
        / "recio8k_accel"
        / "parse_metrics.py"
    )
    spec = importlib.util.spec_from_file_location("recio8k_parse_metrics", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_log_reports_epoch_timestamps_and_mean_seconds_per_epoch(tmp_path):
    parse_metrics = load_parse_metrics()
    log = tmp_path / "train.log"
    log.write_text(
        "2026-06-29 04:07:06.930 INFO: Epoch 0: head: Default, "
        "loss=16.75404358, MAE_E_per_atom=  302.25 meV, "
        "MAE_F=  600.83 meV / A\n"
        "2026-06-29 04:08:39.691 INFO: Epoch 5: head: Default, "
        "loss=6.58858395, MAE_E_per_atom=  136.44 meV, "
        "MAE_F=  400.70 meV / A\n"
        "2026-06-29 04:10:00.272 INFO: Epoch 10: head: Default, "
        "loss=4.27184916, MAE_E_per_atom=  101.65 meV, "
        "MAE_F=  331.37 meV / A\n"
    )

    summary = parse_metrics.parse_log(log)

    assert summary["last"] == {
        "epoch": 10,
        "mae_e_mev_atom": 101.65,
        "mae_f_mev_a": 331.37,
        "timestamp": "2026-06-29T04:10:00.272000",
    }
    assert summary["timing"]["report_intervals"] == [
        {"from_epoch": 0, "to_epoch": 5, "seconds": pytest.approx(92.761), "seconds_per_epoch": pytest.approx(18.5522)},
        {"from_epoch": 5, "to_epoch": 10, "seconds": pytest.approx(80.581), "seconds_per_epoch": pytest.approx(16.1162)},
    ]
    assert summary["timing"]["mean_seconds_per_epoch"] == pytest.approx(17.3342)


def test_parse_nvdmon_summarizes_gpu_memory_and_utilization(tmp_path):
    parse_metrics = load_parse_metrics()
    nvdmon = tmp_path / "nvdmon_job-123.log"
    nvdmon.write_text(
        "#Time         gpu    pwr  gtemp  mtemp     sm    mem    enc    dec    jpg    ofa   mclk   pclk  pviol  tviol     fb   bar1   ccpm  rxpci  txpci  sbecc  dbecc    pci\n"
        "#HH:MM:SS     Idx      W      C      C      %      %      %      %      %      %    MHz    MHz      %   bool     MB     MB     MB   MB/s   MB/s   errs   errs   errs\n"
        " 04:12:44       0    130     37     37     66     17      0      0      -      -    877   1530      0      0   4198      5      0    248     34      0      0      0\n"
        " 04:12:45       0    100     37     37     65     18      0      0      -      -    877   1530      0      0   4201      5      0    268     39      0      0      0\n"
    )

    summary = parse_metrics.parse_nvdmon(nvdmon)

    assert summary == {
        "path": str(nvdmon),
        "samples": 2,
        "max_fb_memory_mb": 4201,
        "mean_sm_util_percent": pytest.approx(65.5),
        "mean_mem_util_percent": pytest.approx(17.5),
        "mean_power_w": pytest.approx(115.0),
    }
