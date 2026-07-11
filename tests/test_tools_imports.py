import subprocess
import sys


def _run_import_check(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_importing_mace_tools_does_not_eagerly_import_cg_stack():
    code = """
import sys
import mace.tools
assert "cuequivariance" not in sys.modules
assert "networkx" not in sys.modules
"""

    result = _run_import_check(code)

    assert result.returncode == 0, result.stderr


def test_importing_training_compile_does_not_eagerly_import_cg_stack():
    code = """
import sys
import mace.tools.training_compile
assert "cuequivariance" not in sys.modules
assert "networkx" not in sys.modules
"""

    result = _run_import_check(code)

    assert result.returncode == 0, result.stderr
