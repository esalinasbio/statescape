from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Sequence

import mdtraj as md

from ._common import AmberError, call, read_manifest, which, write_manifest
from .protocol import Stage

BOX_ERROR = "Periodic box dimensions have changed too much"


def _exe(exe: str | Sequence[str]) -> list[str]:
    return [exe] if isinstance(exe, str) else list(exe)


def _tail(path: Path, n: int = 15) -> str:
    return "\n".join(path.read_text(errors="ignore").splitlines()[-n:]) if path.exists() else ""


def _ca_rmsd(seed: Path, prmtop: Path, coords: Path) -> float:
    """C-alpha RMSD (Angstrom) of `coords` to the seed, after superposition."""
    ref = md.load(str(seed))
    frame = md.load(str(coords), top=str(prmtop))
    ref = ref.atom_slice(ref.topology.select("name CA and element C"))
    frame = frame.atom_slice(frame.topology.select("name CA and element C"))
    if ref.n_atoms != frame.n_atoms:
        raise AmberError(f"Seed has {ref.n_atoms} CA atoms but {coords} has {frame.n_atoms}.")
    return float(md.rmsd(frame, ref)[0] * 10)


def _done(out: Path) -> bool:
    if not out.exists():
        return False
    text = out.read_text(errors="ignore")
    return "Total wall time" in text and "NaN" not in text


def _segment(stage, primary, mdin, prmtop, coords, ref, files, workdir, env, k) -> str:
    """Run one segment: retry once on the GPU box error, then the stage fallback. Returns the executable used."""
    failures: list[str] = []

    def attempt(exe: list[str]) -> bool:
        for f in files.values():
            f.unlink(missing_ok=True)
        cmd = [*exe, "-O", "-i", mdin, "-o", files["out"], "-p", prmtop, "-c", coords,
               "-r", files["ncrst"], "-x", files["nc"], "-inf", files["info"]]
        if ref is not None:
            cmd += ["-ref", ref]
        call([str(c) for c in cmd], workdir, files["log"], env)
        if _done(files["out"]):
            return True
        failures.append(_tail(files["out"], 50) + "\n" + _tail(files["log"], 50))
        for ext in ("out", "log"):
            if files[ext].exists():
                files[ext].rename(f"{files[ext]}.failed{len(failures)}")
        return False

    if attempt(primary):
        return " ".join(primary)
    if BOX_ERROR in failures[-1] and attempt(primary):
        return " ".join(primary) + " (retry)"
    if stage.fallback and attempt(_exe(stage.fallback)):
        return " ".join(_exe(stage.fallback)) + " (fallback)"
    tail = "\n".join(failures[-1].splitlines()[-20:])
    raise AmberError(
        f"{workdir.name}: stage '{stage.name}' segment {k} failed after {len(failures)} attempt(s). "
        f"See {files['out']}.failed*\n{tail}"
    )


def run(
    workdir: str | Path,
    stages: Sequence[Stage],
    *,
    executable: str | Sequence[str] = "pmemd.cuda",
    device: int | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> None:
    """
    Run an MD protocol on one prepared seed.

    Completed segments (normal pmemd termination in the .out file) are skipped,
    so calling `run` again resumes an interrupted protocol.

    Parameters
    ----------
    workdir : directory written by `prepare`
    stages : ordered protocol, e.g. `default_protocol()`
    executable : pmemd executable, or a command list (e.g. ['mpirun', '-np', '8', 'pmemd.MPI'])
    device : GPU index, sets CUDA_VISIBLE_DEVICES for this run only
    overwrite : rerun from the first stage whose input changed, instead of raising
    verbose : print one line per completed stage
    """
    workdir = Path(workdir).resolve()
    prmtop, seed = workdir / "system.prmtop", workdir / "prep" / "seed.pdb"
    if not prmtop.exists():
        raise AmberError(f"{prmtop} not found. Run prepare() first.")
    names = [s.name for s in stages]
    if len(set(names)) != len(names):
        raise ValueError(f"Stage names must be unique, got {names}")
    exes = {_exe(s.executable or executable)[0] for s in stages} | {_exe(s.fallback)[0] for s in stages if s.fallback}
    for exe in exes:
        which(exe)

    env = os.environ.copy()
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device)

    stage_dir = workdir / "stages"
    stage_dir.mkdir(exist_ok=True)
    manifest = read_manifest(workdir)
    records = manifest.setdefault("stages", {})

    coords = workdir / "system.inpcrd"
    for i, stage in enumerate(stages, 1):
        prefix = stage_dir / f"{i:02d}_{stage.name}"
        mdin = Path(f"{prefix}.in")
        if mdin.exists() and mdin.read_text() != stage.mdin:
            if not overwrite:
                raise AmberError(f"{mdin} differs from stage '{stage.name}'. Use overwrite=True to rerun from here.")
            for f in stage_dir.iterdir():
                if int(f.name[:2]) >= i:
                    f.unlink()
        mdin.write_text(stage.mdin)

        ref = coords if stage.restrained else None
        previous = records.get(stage.name, {}).get("executables", [])
        used, seeds = [], []
        for k in range(1, stage.repeats + 1):
            seg = f"{prefix}.{k}" if stage.repeats > 1 else str(prefix)
            files = {ext: Path(f"{seg}.{ext}") for ext in ("out", "ncrst", "nc", "info", "log")}
            if _done(files["out"]):
                used.append(previous[k - 1] if k <= len(previous) else "previous run")
            else:
                primary = _exe(stage.executable or executable)
                used.append(_segment(stage, primary, mdin, prmtop, coords, ref, files, workdir, env, k))
            m = re.search(r"random seed to\s+(\d+)", files["out"].read_text(errors="ignore"))
            seeds.append(int(m.group(1)) if m else None)
            coords = files["ncrst"]

        rmsd = _ca_rmsd(seed, prmtop, coords)
        records[stage.name] = dict(index=i, segments=stage.repeats, executables=used, ig=seeds,
                                   ca_rmsd_to_seed=round(rmsd, 3))
        write_manifest(workdir, manifest)
        if verbose:
            print(f"{workdir.name} | {stage.name:<10} done   Cα RMSD to seed: {rmsd:.2f} Å")