from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import yaml

class AmberError(RuntimeError):
    """An AMBER step failed"""


def which(exe: str) -> str:
    path = shutil.which(exe)
    if path is None:
        raise AmberError(f"'{exe}' not found on PATH. Please install AMBER/AmberTools.")
    return path

def amberhome() -> Path:
    home = os.environ.get("AMBERHOME")
    if not home:
        raise AmberError("AMBERHOME is not set. Please install AMBER/AmberTools.")
    return Path(home)

def call(
        cmd: list[str],
        cwd: Path,
        log: Path,
        env: dict | None = None
) -> int:
    """Run `cmd` in `cwd`, writing stdout and stderr to `log`. Returns the exit code."""
    with open(log, "w") as fh:
        return subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode

def read_manifest(workdir: Path) -> dict:
    path = workdir / "manifest.yaml"
    return yaml.safe_load(path.read_text()) if path.exists() else {}

def write_manifest(workdir: Path, manifest: dict) -> None:
    (workdir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    