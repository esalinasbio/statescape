from __future__ import annotations

import math
import re
import shutil
import warnings
from pathlib import Path
from typing import Literal, Mapping, Sequence

from statescape import __version__

from ._common import *

WATER = {
    "opc": ("leaprc.water.opc", "OPCBOX"),
    "opc3": ("leaprc.water.opc3", "OPC3BOX"),
    "tip3p": ("leaprc.water.tip3p", "TIP3PBOX"),
    "tip4pew": ("leaprc.water.tip4pew", "TIP4PEWBOX"),
    "spce": ("leaprc.water.spce", "SPCBOX"),
}

# residue name -> family
FAMILY = {
    "ASP": "ASP", "ASH": "ASP",
    "GLU": "GLU", "GLH": "GLU",
    "HIS": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "LYS": "LYS", "LYN": "LYS",
    "CYS": "CYS", "CYM": "CYS",
}

# titrable hydrogens for each family
TITRATABLE_H = {
    "ASP": {"HD2"},
    "GLU": {"HE2"},
    "HIS": {"HD1", "HE2"},
    "LYS": {"HZ1", "HZ2", "HZ3"},
    "CYS": {"HG"},
}

ResKey = int | tuple[str, int]

def _grab(pattern: str, text: str, log: Path) -> str:
    m = re.search(pattern, text)
    if m is None:
        raise AmberError(f"Could not find '{pattern}' in {log}")
    return m.group(1)

def _resolve(key: ResKey, residues: dict[tuple[str, int], str]) -> tuple[str, int]:
    """Map a  residue key (resSeq or (chain, resSeq)) to a residue in the seed"""
    if isinstance(key, int):
        hits = [r for r in residues if r[1] == key]
        if len(hits) != 1:
            raise ValueError(f"Residue {key} matches {len(hits)} residues in the seed. Use (chain, resSeq).")
        return hits[0]
    res = (str(key[0]), int(key[1]))
    if res not in residues:
        raise ValueError(f"Residue {res} not found in the seed.")
    return res

def _edit_seed(lines: list[str], protonation: Mapping[ResKey, str]) -> list[str]:
    """
    Protonation changes and Amber naming fixes to seeds.

    Changed residues are renamed and their titratable hydrogens removed, so tleap
    rebuilds them from the new template. The N-terminal 'H' of each chain is renamed
    'H1'. CONECT records are removed also.
    """
    atoms = [l for l in lines if l.startswith(("ATOM", "HETATM"))]
    residues: dict[tuple[str, int], str] = {}
    first: dict[str, int] = {}
    for l in atoms:
        res = (l[21], int(l[22:26]))
        residues.setdefault(res, l[17:20].strip())
        first.setdefault(res[0], res[1])

    targets = {}
    for key, name in protonation.items():
        res, name = _resolve(key, residues), name.upper()
        if name not in FAMILY or FAMILY.get(residues[res]) != FAMILY[name]:
            raise ValueError(
                f"Cannot set {residues[res]} {res[0].strip()}{res[1]} to {name!r}. "
                f"Allowed: {sorted(FAMILY)} within the same resideu type (disulfides are detected by pdb4amber)."
            )
        targets[res] = name

    out = []
    for l in lines:
        if l.startswith("CONECT"):
            continue
        if l.startswith(("ATOM", "HETATM")):
            res, atom = (l[21], int(l[22:26])), l[12:16].strip()
            if res in targets:
                if atom in TITRATABLE_H[FAMILY[targets[res]]]:
                    continue
                l = l[:17] + f"{targets[res]:>3}" + l[20:]
            if res[1] == first[res[0]] and atom == "H":
                l = l[:12] + " H1 " + l[16:]
        out.append(l)
    return out

def _disulfides(sslink: Path) -> list[tuple[int, int]]:
    """S-S bond forming residue pairs from the pdb4amber `_sslink` file"""
    if not sslink.exists():
        return []
    pairs = []
    for line in sslink.read_text().splitlines():
        nums = [int(x) for x in line.split() if x.isdigit()]
        if len(nums) >= 2:
            pairs.append((nums[0], nums[1]))
    return pairs

def _ion_counts(charge: float, n_water: int, salt: float) -> tuple[int, int]:
    """(n_cation, n_anion) with SLTCAP method (Schmit et al. 2018). salt=0 only neutralizes charge"""
    q = round(charge)
    if salt <= 0:
        return max(-q, 0), max(q, 0)
    n0 = salt * n_water / 55.5
    n_cat = round(n0 * (math.sqrt(1 + (q / (2 * n0)) ** 2) - q / (2 * n0)))
    return n_cat, n_cat + q

def _tleap(prep: Path, name: str, script: str) -> str:
    """Runs tleap script in `prep` and check its log. Returns the log text."""
    if "quit" not in script.lower():
        script += "\nquit\n"
    (prep / f"{name}.in").write_text(script)
    log = prep / f"{name}.log"
    call([which("tleap"), "-f", f"{name}.in"], prep, log)
    text = log.read_text()
    if int(_grab(r"Errors = (\d+)", text, log)) > 0:
        raise AmberError(f"tleap reported errors, see {log}")
    heavy = re.search(r"(\d+) Heavy", text)
    if heavy and int(heavy.group(1)) > 0:
        raise AmberError(f"tleap added {heavy.group(1)} missing heavy atoms, the seed is incomplete. See {log}")
    return text


def prepare(
    pdb: str | Path,
    workdir: str | Path,
    *,
    forcefields: Sequence[str] = ("protein.ff19SB",),
    water: str = "opc",
    box: Literal["octahedron", "cubic"] = "octahedron",
    buffer: float = 10.0,
    salt: float = 0.15,
    cation: str = "Na+",
    anion: str = "Cl-",
    protonation: Mapping[ResKey, str] | None = None,
    leap_template: str | Path | None = None,
    hmr: bool = False,
    overwrite: bool = False,
) -> Path:
    """
    Build an AMBER system (system.prmtop / system.inpcrd) from one seed PDB with tleap.

    Parameters
    ----------
    pdb : seed structure, protonated (e.g. by `ConformerSet.protonate`)
    workdir : output directory, one per seed
    forcefields : leaprc names without the 'leaprc.' prefix (default protein.ff19SB)
    water : water model, one of 'opc', 'opc3', 'tip3p', 'tip4pew', 'spce'
    box : 'octahedron' or 'cubic'
    buffer : solute-box distance in Angstrom
    salt : salt concentration in M, ion counts by SLTCAP. 0 only neutralizes
    cation, anion : ion residue names
    protonation : {resSeq or (chain, resSeq): variant}, e.g. {181: 'ASH'}. Seed numbering
    leap_template : user tleap script with {pdb}, {disulfides}, {prmtop}, {inpcrd} placeholders.
        Solvation and ions are then the user's responsibility
    hmr : hydrogen mass repartitioning (allows dt = 0.004 ps)
    overwrite : rebuild an already prepared workdir

    Returns
    -------
    Path : `workdir`
    """
    pdb, workdir = Path(pdb).resolve(), Path(workdir).resolve()
    if not pdb.is_file():
        raise FileNotFoundError(f"Seed {pdb} not found.")
    if water not in WATER:
        raise ValueError(f"Unknown water model {water!r}. Available: {list(WATER)}")
    if box not in ("octahedron", "cubic"):
        raise ValueError(f"box must be 'octahedron' or 'cubic', got {box!r}")

    params = dict(
        seed=str(pdb), forcefields=list(forcefields), water=water, box=box, buffer=buffer, salt=salt,
        cation=cation, anion=anion, protonation={str(k): v for k, v in (protonation or {}).items()},
        leap_template=str(leap_template) if leap_template else None, hmr=hmr,
    )
    if (workdir / "system.prmtop").exists() and not overwrite:
        if read_manifest(workdir).get("prepare", {}).get("parameters") != params:
            raise AmberError(f"{workdir} was prepared with different parameters. Use overwrite=True.")
        return workdir

    leaprcs = [f"leaprc.{ff}" for ff in forcefields] + [WATER[water][0]]
    cmd_dir = amberhome() / "dat" / "leap" / "cmd"
    missing = [l for l in leaprcs if not (cmd_dir / l).exists()]
    if missing:
        raise AmberError(f"Not found in {cmd_dir}: {missing}")
    if any("ff19SB" in ff for ff in forcefields) and water != "opc":
        warnings.warn("ff19SB was parametrized with OPC water.", stacklevel=2)

    shutil.rmtree(workdir / "stages", ignore_errors=True)
    prep = workdir / "prep"
    prep.mkdir(parents=True, exist_ok=True)
    shutil.copy(pdb, prep / "seed.pdb")
    lines = _edit_seed(pdb.read_text().splitlines(), protonation or {})
    (prep / "amber_in.pdb").write_text("\n".join(lines) + "\n")

    if call([which("pdb4amber"), "-i", "amber_in.pdb", "-o", "amber.pdb"], prep, prep / "pdb4amber.log") != 0:
        raise AmberError(f"pdb4amber failed, see {prep / 'pdb4amber.log'}")
    amber_pdb = prep / "amber.pdb"
    amber_pdb.write_text("\n".join(l for l in amber_pdb.read_text().splitlines() if not l.startswith("CONECT")) + "\n")
    bonds = _disulfides(prep / "amber_sslink")

    info = {}
    if leap_template is None:
        head = [f"source {l}" for l in leaprcs] + ["mol = loadpdb amber.pdb"]
        head += [f"bond mol.{i}.SG mol.{j}.SG" for i, j in bonds]
        solvate = f"{'solvateOct' if box == 'octahedron' else 'solvateBox'} mol {WATER[water][1]} {buffer}"

        log1 = prep / "tleap_pass1.log"
        text = _tleap(prep, "tleap_pass1", "\n".join(head + ["charge mol", solvate, "quit"]) + "\n")
        charge = float(_grab(r"Total unperturbed charge:\s*(\S+)", text, log1))
        n_water = int(_grab(r"Added (\d+) residues", text, log1))
        n_cat, n_an = _ion_counts(charge, n_water, salt)
        if n_an < 0:
            raise AmberError(f"Negative anion count for charge {charge} and salt {salt} M.")
        ions = " ".join(f"{ion} {n}" for ion, n in ((cation, n_cat), (anion, n_an)) if n > 0)

        body = [solvate] + ([f"addIonsRand mol {ions}"] if ions else [])
        body += ["saveamberparm mol ../system.prmtop ../system.inpcrd", "savepdb mol ../system.pdb", "quit"]
        _tleap(prep, "tleap_pass2", "\n".join(head + body) + "\n")
        info = dict(net_charge=charge, n_waters=n_water, n_cation=n_cat, n_anion=n_an)
    else:
        script = Path(leap_template).read_text()
        m = re.search(r"(\w+)\s*=\s*loadpdb\s+\{pdb\}", script, re.IGNORECASE)
        if m is None:
            raise ValueError("leap_template needs a line like 'mol = loadpdb {pdb}'.")
        unit = m.group(1)
        fill = {
            "{pdb}": "amber.pdb",
            "{disulfides}": "\n".join(f"bond {unit}.{i}.SG {unit}.{j}.SG" for i, j in bonds),
            "{prmtop}": "../system.prmtop",
            "{inpcrd}": "../system.inpcrd",
        }
        for key, value in fill.items():
            script = script.replace(key, value)
        _tleap(prep, "tleap_template", script)

    if not (workdir / "system.prmtop").exists():
        raise AmberError(f"tleap did not write {workdir / 'system.prmtop'}")

    if hmr:
        import parmed
        parm = parmed.load_file(str(workdir / "system.prmtop"))
        parmed.tools.HMassRepartition(parm).execute()
        parm.save(str(workdir / "system.prmtop"), overwrite=True)

    write_manifest(workdir, {
        "statescape_version": __version__,
        "prepare": {"parameters": params, "disulfides": [list(p) for p in bonds], **info},
    })
    return workdir