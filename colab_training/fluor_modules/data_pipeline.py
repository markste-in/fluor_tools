"""Data pipeline for fluorescent molecule property prediction.

Pure Python + rdkit + numpy + pandas. No torch/dgl/dgllife imports.
Handles CSV parsing, fingerprints, molecular descriptors, scaffold
detection, solvent mapping, and substructure flag computation.

Reference: design.md section "data_pipeline.py"
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Solvent mapping: German + English solvent names -> canonical SMILES
# Based on 00_solvent_mapping.csv which maps SMILES -> numeric IDs.
# ---------------------------------------------------------------------------

SOLVENT_NAME_TO_SMILES: dict[str, str] = {
    # German names
    "Toluol": "Cc1ccccc1",
    "EtOH": "CCO",
    "MeOH": "CO",
    "DCM": "ClCCl",
    "Dichlormethan": "ClCCl",
    "CHCl3": "ClC(Cl)Cl",
    "Benzol": "c1ccccc1",
    "Wasser": "O",
    "Aceton": "CC(C)=O",
    "Cyclohexan": "C1CCCCC1",
    "Hexan": "CCCCCC",
    "Acetonitril": "CC#N",
    "Diethylether": "CCOCC",
    "THF": "C1CCOC1",
    "DMSO": "CS(C)=O",
    "DMF": "CN(C)C=O",
    # English names
    "Toluene": "Cc1ccccc1",
    "Ethanol": "CCO",
    "Methanol": "CO",
    "Dichloromethane": "ClCCl",
    "Chloroform": "ClC(Cl)Cl",
    "Benzene": "c1ccccc1",
    "Water": "O",
    "Acetone": "CC(C)=O",
    "Cyclohexane": "C1CCCCC1",
    "Hexane": "CCCCCC",
    "Acetonitrile": "CC#N",
    "Diethyl ether": "CCOCC",
    "Tetrahydrofuran": "C1CCOC1",
}


def load_solvent_mapping(csv_path: str) -> dict[str, int]:
    """Load 00_solvent_mapping.csv into {smiles: numeric_id} dict.

    Args:
        csv_path: Path to the solvent mapping CSV file.

    Returns:
        Dictionary mapping solvent SMILES strings to integer IDs.
    """
    df = pd.read_csv(csv_path)
    return dict(zip(df["solvent"], df["solvent_num"]))


def map_solvent_name_to_smiles(name: str) -> str | None:
    """Map a solvent name (including German) to canonical SMILES.

    Also accepts raw SMILES strings as pass-through: if the name is not
    found in the lookup table, we try to parse it as SMILES via RDKit.

    Args:
        name: Solvent name or SMILES string.

    Returns:
        Canonical SMILES string, or None if the name is unknown and
        cannot be parsed as SMILES.
    """
    # Direct lookup first
    if name in SOLVENT_NAME_TO_SMILES:
        return SOLVENT_NAME_TO_SMILES[name]

    # Try interpreting the name as a SMILES string (pass-through)
    try:
        mol = Chem.MolFromSmiles(name)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass

    return None


def map_solvent_smiles_to_id(
    smiles: str, mapping: dict[str, int]
) -> int | None:
    """Map solvent SMILES to numeric ID via the reference mapping.

    Args:
        smiles: Canonical solvent SMILES string.
        mapping: Dictionary from load_solvent_mapping().

    Returns:
        Integer solvent ID, or None if not found in the mapping.
    """
    return mapping.get(smiles)


# ---------------------------------------------------------------------------
# Fingerprint and descriptor computation
# ---------------------------------------------------------------------------


def compute_morgan_fingerprint(
    smiles: str, radius: int = 2, n_bits: int = 1024
) -> np.ndarray | None:
    """Generate Morgan fingerprint as int32 numpy array.

    Uses the modern rdFingerprintGenerator API (replaces deprecated
    GetMorganFingerprintAsBitVect).
    Reference: https://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.html

    Args:
        smiles: SMILES string of the molecule.
        radius: Morgan fingerprint radius (default 2).
        n_bits: Number of bits in the fingerprint (default 1024).

    Returns:
        Numpy array of shape (n_bits,) with dtype int32, elements 0 or 1.
        Returns None if the SMILES cannot be parsed.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
        # GetFingerprintAsNumPy returns uint8; cast to int32 for consistency
        fp = gen.GetFingerprintAsNumPy(mol).astype(np.int32)
        return fp
    except Exception:
        logger.warning("Failed to compute Morgan fingerprint for: %s", smiles)
        return None


def compute_molecular_descriptors(smiles: str) -> dict[str, float] | None:
    """Compute molecular descriptors for a SMILES string.

    Computes: MW, LogP, TPSA, Double_Bond_Count, Ring_Count.
    Double_Bond_Count includes aromatic bonds (matching the original
    notebook behavior in 01_data_preprocessing.py).

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        Dict with keys MW, LogP, TPSA, Double_Bond_Count, Ring_Count.
        Returns None if the SMILES cannot be parsed.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)

        # Count double bonds including aromatic bonds
        # (matches original notebook: bond.GetBondType() == DOUBLE or bond.GetIsAromatic())
        double_bond_count = sum(
            1
            for bond in mol.GetBonds()
            if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
            or bond.GetIsAromatic()
        )

        ring_count = mol.GetRingInfo().NumRings()

        return {
            "MW": mw,
            "LogP": logp,
            "TPSA": tpsa,
            "Double_Bond_Count": float(double_bond_count),
            "Ring_Count": float(ring_count),
        }
    except Exception:
        logger.warning("Failed to compute descriptors for: %s", smiles)
        return None


# ---------------------------------------------------------------------------
# Scaffold detection
#
# The scaffold dictionary below is ported from Fluor-RLAT/01_data_preprocessing.py.
# It defines 16 scaffold families, each with a list of SMILES patterns.
# Order matters: first match wins.
# Tag numbers: SquaricAcid=0, Naphthalimide=1, Coumarin=2, Carbazole=3,
#   Cyanine=4, BODIPY=5, Triphenylamine=6, Porphyrin=7, PAHs=8,
#   Acridines=9, 5p6=10, 6p6=11, 5n6=12, 6n6=13, Azo=14, Benz=15
# ---------------------------------------------------------------------------

_SCAFFOLD_SMILES: dict[str, list[str]] = {
    "SquaricAcid": [
        "O=c1ccc1=O",
        "O=C1CC([O-])C1",
        "O=C1C=C(O)C1",
        "OC1=CCC1",
        "C=c1c(=O)c(=C)c1=O",
        "c1ccc(N2CCC2)cc1",
        "C=C1C(C=C1)=O",
    ],
    "Naphthalimide": [
        "O=C1NC(=O)c2cccc3cccc1c23",
        "O=C(C1=C2C(C=CC=C23)=CC=C1)NC3=O",
    ],
    "Coumarin": [
        "C1=Cc2ccccc2OC1",
        "O=c1ccc2ccccc2o1",
        "S=c1ccc2ccccc2o1",
        "O=C1C=Cc2ccccc2C1(F)F",
        "O=c1ccc2ccccc2[nH]1",
        "C[Si]1(C)C(=O)C=Cc2ccccc21",
        "N=c1ccc2ccccc2o1",
        "O=c1cnc2ccccc2o1",
        "O=c1cnc2ccccc2[nH]1",
    ],
    "Carbazole": [
        "[nH]1c2ccccc2c3ccccc13",
    ],
    "Cyanine": [
        "NC=CC=O",
        "NC=CC=[OH+]",
        "NC=CC=[NH2+]",
        "NC=CC=CC=O",
        "NC=CC=CC=[OH+]",
        "NC=CC=CC=[NH2+]",
        "NC=CC=CC=CC=O",
        "NC=CC=CC=CC=[OH+]",
        "NC=CC=CC=CC=[NH2+]",
        "NC=CC=CC=CC=CC=O",
        "NC=CC=CC=CC=CC=[OH+]",
        "NC=CC=CC=CC=CC=[NH2+]",
        "NC=CC=CC=CC=CC=CC=O",
        "NC=CC=CC=CC=CC=CC=[OH+]",
        "NC=CC=CC=CC=CC=CC=[NH2+]",
        "NC=CC=CC=CC=CC=CC=CC=O",
        "NC=CC=CC=CC=CC=CC=CC=[OH+]",
        "NC=CC=CC=CC=CC=CC=CC=[NH2+]",
        "NC=CC=CC=CC=CC=CC=CC=CC=O",
        "NC=CC=CC=CC=CC=CC=CC=CC=[OH+]",
        "NC=CC=CC=CC=CC=CC=CC=CC=[NH2+]",
    ],
    "BODIPY": [
        "B(n1cccc1)n1cccc1",
        "N1([BH2-]n2cccc2)C=CCC1",
        "[BH2-](N1CC=CC1)n1cccc1",
        "n1([BH2-][N+]2=CC=CC2)cccc1",
        "[BH2-](n1cccc1)[N+]1=CC=CC1",
        "[N+][BH2-][N+]",
        "N[BH2-][N+]",
        "N[BH2-]N",
        "[N+]B[N+]",
        "NB[N+]",
        "NBN",
    ],
    "Triphenylamine": [
        "c1ccc(cc1)N(c2ccccc2)c3ccccc3",
        "C1=CC(=[N+](c2ccccc2)c2ccccc2)C=CC1",
        "N=C1C=C/C(C=C1)=C(C2=CN=CS2)/C3=CN=CS3",
    ],
    "Porphyrin": [
        "C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[nH]5)C=C4)[nH]3",
        "C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[nH]5)CC4)[nH]3",
        "C1=CC2=NC1=Cc1ccc([nH]1)CC1=NC(=CC3=NC(=C2)C=C3)C=C1",
        "C1=CC2=NC1=CC1=NC(=CC3=NC(=CC4=NC(=C2)C=C4)C=C3)C=C1",
        "C1=C/C2=C/c3ccc([nH]3)CC3CCC(=N3)/C=C3/CC/C(=C/C1=N2)N3",
        "C1=Cc2nc1ccc1ccc([nH]1)c1nc(ccc3ccc2[nH]3)C=C1",
        "C1=CC2=NC1=Cc1ccc([n-]1)C=C1C=CC(=CC3=NC(=C2)C=C3)[NH2+]1",
        "C1=Cc2cc3ccc(cc4nc(cc5[nH]c(cc1n2)CC5)C=C4)[nH]3",
        "c1cc2cc3nc(cc4ccc(cc5nc(cc1[nH]2)CC5)[nH]4)CC3",
        "C1=C2C=c3ccc([nH]3)=Cc3ccc([n-]3)CC3=CC=C(CC(=C1)[NH2+]2)[NH2+]3",
        "C1=C2C=c3ccc([nH]3)=Cc3ccc([n-]3)Cc3ccc([nH]3)CC(=C1)[NH2+]2",
        "C1=Cc2cc3ccc(cc4nc(cc5[nH]c(cc1n2)CC5)CC4)[nH]3",
        "C1=CC2=NC1=CC1=NC(=CC3=NC(=CC3)C=c3ccc([n-]3)=C2)CC1",
        "C1=Cc2nc1ccc1ccc([nH]1)c1nc(ccc3ccc2[nH]3)CC1",
        "C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[nH]5)C=C4)s3",
        "C1=CC2=NC1=Cc1ccc([nH]1)CC1C=CC(=N1)C=c1ccc([nH]1)=C2",
        "C1=Cc2cc3ccc(cc4nc(cn5ccc(cc1n2)c5)C=C4)[nH]3",
        "C1=CC2=[NH+]C1=CC1=NC(C=C1)CC1C=CC(=N1)C=c1ccc([nH]1)=C2",
        "C1=Cc2nc1cc1ccc([nH]1)c1ccc(cc3ccc(cc4ccc2[nH]4)o3)[nH]1",
        "C=C1C=C2C=c3ccc([n-]3)=CC3=NC(=CC4=NC(=CC1=N2)C=C4)C=C3",
        "C1=Cc2cc3ccc(cc4nc(c5ccc(ccc1n2)[nH]5)C=C4)[nH]3",
        "C1=Cc2nc1ccc1nc(c3ccc(ccc4ccc2[n-]4)[n-]3)CC1",
        "C=C1C=C2C=C3C=C4C(=O)CC(=C5CCC(=N5)C=c5ccc([n-]5)=CC1=N2)C4=N3",
        "C=C1C=C2C=C3C=CC(=N3)C=C3C=CC(=N3)C=c3ccc([n-]3)=CC1=N2",
        "C1=Cc2cc3[nH]c(cc4ccc(cc5nc(cc1n2)C=C5)[nH]4)CC3",
        "C1=Cc2cc3ccc(cc4ccc(cc5cc(cc1n2)[NH+]=C5)[nH]4)[nH]3",
        "C1=Cc2nc1ccc1nc(c3ccc(ccc4ccc2[n-]4)[nH]3)C=C1",
        "C1=Cc2cc3cnc(cc4nc(cc5ccc(cc1n2)[n-]5)C=C4)[n-]3",
        "C1=Cc2cc3ccc([nH]3)c3nc(ccc4ccc(cc1n2)[nH]4)C=C3",
        "C=C1C=C2C=C3CCC(=CC4=NC(=CC5=NC(=CC1=N2)CC5)C=C4)N3",
        "C1=CC2=NC1=Cc1ccc([n-]1)Cc1ccc([n-]1)C=C1C=CC(=N1)C2",
        "C1=CC2=NC1=Cc1ccn(c1)CC1C=CC(=N1)C=c1ccc([nH]1)=C2",
        "C1CC2CC3CCC(N3)C3CCC(CC4CCC(CC1N2)N4)N3",
        "C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)o5)C=C4)o3",
        "c1c2nc(cc3ccc(cc4nc(cc5[nH]c1CC5)CC4)[nH]3)CC2",
        "C=C1C=C2C=c3ccc([nH]3)=CC3=NC(=CC4=NC(=CC1=N2)CC4)C=C3",
        "C1=CC2=NC1=Cc1ccc([nH]1)C=C1C=CC(=N1)CC1=NC(=C2)C=C1",
        "C1=CC2=NC1=Cc1ccc([nH]1)C=C1C=CC(=N1)Cc1ccc([nH]1)C2",
        "C1=CC2C=C3CCC(=N3)C=c3ccc([nH]3)=CC3=NC(=CC3)C=C1N2",
        "C1=CC2=NC1=Cc1ccc([nH]1)C=C1C=CC(=N1)CC1C=CC(=C2)N1",
        "C1=C2CCC(=N2)C=C2CCC(=N2)C=C2CCC(C=C3CCC1=N3)N2",
        "c1c2nc(cc3ccc(cc4ccc(cc5nc1CC5)[n-]4)[n-]3)CC2",
        "C1=Cc2nc1ccc1ccc([nH]1)c1ccc(ccc3nc2C=C3)[nH]1",
        "C1=Cc2cc3ccc([n-]3)c3ccc(cc4nc(ccc1n2)C=C4)[n-]3",
        "C1=CC2=NC1=CC1CCC(C=c3ccc([n-]3)=CC3=NC(=C2)CC3)[N-]1",
        "C1=Cc2cc3ccc(cc4ccc(cc5nc(cc1n2)C=C5)[nH]4)[nH]3",
        "O=C1CC2=CC3N=C(C=c4ccc([nH]4)=CC4=CCC(=N4)C=C1[N-]2)CC3=O",
        "C1=CC2=NC1=Cc1ccc([nH]1)CC1=NC(=Cc3ccc([nH]3)C2)C=C1",
        "O=C1C2=NC(=Cc3ccc([nH]3)C=C3C=CC(=Cc4ccc1[nH]4)[N]3)C=C2",
        "C1=CC2=NC1=CC1=NC(=CC3=NC(=CC4=NC(=C2)C=C4)CC3)C=C1",
        "C1=C2CCC(=Cc3ccc([nH]3)C=c3ccc([nH]3)=Cc3ccc1[nH]3)N2",
        "C1=C2[CH]NC=1C=C1C=CC(=N1)C=C1C=CC(=N1)C=c1ccc([n-]1)=C2",
        "C1=Cc2cc3ccc(cc4ccc(cc5nc(cc1n2)CC5)[nH]4)[nH]3",
        "O=C1NC2=C=C1C=c1ccc([nH]1)=CC1=N[C](C=C1)C=c1ccc([n-]1)=C2",
        "C1=Cc2cc3nc(cc4ccc([n-]4)c4ccc(ccc1n2)[n-]4)C=C3",
        "C1=CC2=NC1=Cc1ccc([nH]1)C=C1C=CC(=N1)C=C1C=CC(=C2)[NH2+]1",
        "C1=CC2=NC1=CC1=CCC(=N1)C=C1C=CC(=N1)C=c1ccc([n-]1)=C2",
        "C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[nH]5)C=C4)o3",
        "n([BH2-][N+]1=CCCC1=C2)(cc3c4cc(cc5)[nH]c5cc6nc(C=C6)c7)c2c3c(n4)cc8[nH]c7cc8",
        "C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)s5)C=C4)s3",
        "C1=Cc2cc3ccc(cc4nc(cc5ccc(cc1n2)[n-]5)C=C4)[n-]3",
        "C1=CC2=CC3=NC(=CC4=NC(=CC5=NC(=CC(=C1)[N-]2)C=C5)C=C4)C=C3",
    ],
    "PAHs": [
        "c1ccc2ccccc2c1",
        "c1ccc2cc3ccccc3cc2c1",
        "c1ccc2c(c1)ccc3ccccc23",
        "c1cc2ccc3cccc4ccc(c1)c2c34",
        "c1cc2cccc3c4cccc5cccc(c(c1)c23)c45",
        "c1ccc2c(c1)ccc3ccc4ccc5ccccc5c4c23",
        "C1=CCC2C=c3ccccc3=CC2=C1",
        "C12=CC=CC=C1C=C3C(C=C(C=CC=C4)C4=C3)=C2",
        "C1(C(C=CC=C2)=C2C=C3)=C3C=CC=C1",
        "C1(C(C=CC=C2)=C2C3=C4C=CC=C3)=C4C=CC=C1",
        "C12=CC=C3B4N1C(C=CC4=CC=C3)=CC=C2",
        "N12C=CC=CC1=CC=C3B2C=CC=C3",
        "C=C1C=CC=C2NC=CC=C21",
        "O=C1C2=C(C(C(C=C2)=O)=O)C=C3OC=CC=C31",
        "O=C1C=C2NC=CC=C2C=C1",
    ],
    "Acridines": [
        "B1c2ccccc2Cc2ccccc21",
        "B1c2ccccc2Nc2ccccc21",
        "C1=C2CCCC=C2Sc2ccccc21",
        "C1=CC2=Cc3ccccc3CC2=CC1",
        "C1=CC2=Cc3ccccc3[GeH2]C2=CC1",
        "C1=CC2=Nc3ccccc3[SiH2]C2=CC1",
        "C1=CC2=[O+]c3ccccc3CC2C=C1",
        "C1=CC2C=c3cc4ccccc4[o+]c3=CC2NC1",
        "C1=CC2C=c3ccccc3=[O+]C2C=C1",
        "C1=CC2Oc3ccccc3CC2CC1",
        "C1=CC=C[C-]2CC3=CC=C[C+]=C3OC=12",
        "C1=CCC2=Cc3ccccc3[N]C2=C1",
        "C1=CCC2Nc3ccccc3CC2C1",
        "C1=CCC2Oc3ccccc3CC2C1",
        "C1=C[C]2NC3C=CC=CC3C=C2C=C1",
        "C1=Cc2[o+]c3ccccc3cc2CC1",
        "C1=c2ccccc2=[SiH2+]c2ccccc21",
        "C=c1ccc2c(c1)Oc1ccccc1C=2",
        "C=c1ccc2c(c1)Sc1ccccc1C=2",
        "N=C1C=CC2=Cc3ccccc3S(=O)(=O)C2=C1",
        "N=C1C=CC2=Cc3ccccc3[SiH2]C2=C1",
        "N=C1C=CC2Cc3ccccc3OC2=C1",
        "N=C1c2ccccc2[GeH2]c2ccccc21",
        "N=c1c2ccccc2oc2ccccc12",
        "N=c1ccc2cc3ccccc3[nH]c-2c1",
        "N=c1ccc2nc3ccccc3oc-2c1",
        "O=C1C2=C(CC=CC2)C(=O)C2=C1CC=CC2",
        "O=C1C2=C(CC=CC2)C(=O)c2ccccc21",
        "O=C1C2=C(CCC=C2)C(=O)c2ccccc21",
        "O=C1C2=C(CCCC2)C(=O)c2ccccc21",
        "O=C1C2=C(CCCC2)S(=O)(=O)c2ccccc21",
        "O=C1C2=C(COC=C2)C(=O)c2ccccc21",
        "O=C1C2=C(COCC2)C(=O)c2ccccc21",
        "O=C1C2=C(OC=CC2)C(=O)c2ccccc21",
        "O=C1C2=C(OCC=C2)C(=O)c2ccccc21",
        "O=C1C2=C(OCCC2)C(=O)c2ccccc21",
        "O=C1C2=CC=CCC2C(=O)c2ccccc21",
        "O=C1C2=CC=CCC2Oc2ccccc21",
        "O=C1C2=CCCCC2Oc2ccccc21",
        "O=C1C2=CCCOC2C(=O)c2ccccc21",
        "O=C1C2=CCOC=C2C(=O)c2ccccc21",
        "O=C1C=C2C(=O)c3ccccc3CC2CC1",
        "O=C1C=CC(=O)C2=C1CC1=C(O2)C(=O)C=CC1=O",
        "O=C1C=CC(=O)c2c1[nH]c1ccccc1c2=O",
        "O=C1C=CC(=O)c2c1oc1ccccc1c2=O",
        "O=C1C=CC(=O)c2c1sc1ccccc1c2=O",
        "O=C1C=CC2=Nc3ccccc3CC2=C1",
        "O=C1C=CC2Cc3ccccc3OC2=C1",
        "O=C1C=CC2Cc3ccccc3OC2C1",
        "O=C1C=CC=C2C(=O)c3ccccc3C=C12",
        "O=C1C=Cc2c([nH]c3ccccc3c2=O)C1",
        "O=C1c2ccccc2C(=O)C2C=CC=CC12",
        "O=C1c2ccccc2OC2C=CC=CC12",
        "O=[Te+]1=c2ccccc2=Cc2ccccc21",
        "O=c1c2c(oc3ccccc13)C=CCC2",
        "O=c1c2c(sc3ccccc13)CCC=C2",
        "O=c1c2ccccc2[se]c2ccccc12",
        "O=c1c2ccccc2oc2ccccc12",
        "O=c1c2ccccc2sc2ccccc12",
        "O=c1cc2oc3ccccc3nc-2c2ccccc12",
        "O=c1cc2oc3ccccc3nc-2c2cccnc12",
        "O=c1ccc(=O)c2c(=O)c3ccccc3c(=O)c1=2",
        "O=c1ccc2cc3ccccc3[nH]c-2c1",
        "O=c1ccc2cc3ccccc3oc-2c1",
        "O=c1cccc2[nH]c3ccccc3cc1-2",
        "S=c1c2ccccc2[se]c2ccccc12",
        "S=c1c2ccccc2oc2ccccc12",
        "S=c1c2ccccc2sc2ccccc12",
        "[CH]1C2=CC=CCC2=[NH+]c2ccccc21",
        "[CH]1C=C2Nc3ccccc3CC2=C[CH+]1",
        "[N+]=C1C=CC2=Cc3ccccc3[BH2-]C2=C1",
        "[O+]=c1cccc2sc3ccccc3cc1-2",
        "[OH+]=c1cccc2oc3ccccc3cc1-2",
        "c1ccc2[o+]c3ccccc3cc2c1",
        "c1ccc2[o+]c3ccccc3nc2c1",
        "c1ccc2[s+]c3ccccc3cc2c1",
        "c1ccc2[s+]c3ccccc3nc2c1",
        "c1ccc2[se+]c3ccccc3cc2c1",
        "c1ccc2[te+]c3ccccc3cc2c1",
        "c1ccc2c(c1)Cc1ccccc1C2",
        "c1ccc2c(c1)Cc1ccccc1N2",
        "c1ccc2c(c1)Cc1ccccc1O2",
        "c1ccc2c(c1)Cc1ccccc1S2",
        "c1ccc2c(c1)Cc1ccccc1[Se]2",
        "c1ccc2c(c1)Cc1ccccc1[SiH2]2",
        "c1ccc2c(c1)Nc1ccccc1N2",
        "c1ccc2c(c1)Nc1ccccc1O2",
        "c1ccc2c(c1)Nc1ccccc1S2",
        "c1ccc2c(c1)Oc1ccccc1O2",
        "c1ccc2c(c1)Oc1ccccc1S2",
        "c1ccc2c(c1)Sc1ccccc1S2",
        "c1ccc2c(c1)[SiH2]c1ccccc1[SiH2]2",
        "c1ccc2nc3ccccc3cc2c1",
        "c1ccc2nc3ccccc3nc2c1",
        "c1ccc2pc3ccccc3cc2c1",
        "O=C1C=Cc2c(cc3occcc-3c2=O)C1=O",
    ],
    "5p6": [
        "c1ccc2[nH]ccc2c1",
        "c1ccc2occc2c1",
        "c1ccc2sccc2c1",
        "c1ccc2[nH]cnc2c1",
        "c1ccc2scnc2c1",
        "c1ccc2ocnc2c1",
        "c1ccc2nonc2c1",
        "c1ccc2nsnc2c1",
        "c1ccc2scpc2c1",
        "C1=Nc2ccccc2C1",
        "c1ccn2cccc2c1",
        "c1ccn2ccnc2c1",
        "c1ccc2[nH]ncc2c1",
        "c1ccn2cnnc2c1",
        "c1ccc2cscc2c1",
        "c1ncc2nc[nH]c2n1",
        "C1=COc2n[nH]cc2C1",
        "c1cnc2ncnn2c1",
        "c1ccc2c(c1)CCO2",
        "c1cn2ccnc2cn1",
        "c1cnc2sccc2c1",
        "O=C1OCc2ccccc21",
        "c1cc[n+]2c(c1)[N-][NH2+]C2",
        "O=C1NCc2nc[nH]c2N1",
        "c1ccc2c(c1)OCO2",
        "O=C1Cc2ccccc2C1=O",
        "c1ccc2[nH]nnc2c1",
        "O=C1N=Cc2ccccc21",
        "c1ccc2c(c1)=NCN=2",
        "c1cnn2cccc2c1",
        "c1ccc2n[se]nc2c1",
        "C1=CC2=CCCN2N=C1",
        "c1nc2cnc[nH]c-2n1",
        "C=[N+]1[BH2-]N2C=CC=CC2=N1",
        "N=C1N=CC2N=CNC2N1",
        "O=c1ccc2sccc2[nH]1",
        "C1=CC2CCCC2CC1",
        "O=C1NC(=O)C2CC=CCC12",
        "O=C1CCCc2occc21",
        "c1scc2c1OCCO2",
        "O=c1ccn2nccc2o1",
        "c1ccn2cncc2c1",
        "O=C1NC(=O)C2CCCCC12",
        "c1cnc2[nH]cnc2c1",
        "c1cc2nc[nH]c2cn1",
        "c1ccc2cocc2c1",
        "c1ncc2ccsc2n1",
        "c1cc2sccc2cn1",
        "c1ncc2cc[nH]c2n1",
        "N=c1ncc2[nH]ccc2[nH]1",
        "c1cnc2[nH]cnc2n1",
        "O=c1ccsc2ncnn12",
        "c1cnc2nccn2c1",
        "c1cc2cnccn2c1",
        "C1=Cc2ccccc2C1",
        "O=S1(=O)NC=Cc2sccc21",
        "c1ccc2[pH]ccc2c1",
        "C=C1Nc2ccccc2O1",
        "c1cnc2[nH]ncc2c1",
        "c1ncc2c[nH]nc2n1",
        "c1ncn2c1CCCC2",
        "c1cc2[o+]ccc-2c[nH]1",
        "O=C1NCc2ccccc21",
        "O=[PH]1C=Cc2ccccc21",
        "c1ncc2sccc2n1",
        "c1cc2ccsc2cn1",
        "O=c1nc[nH]n2cncc12",
        "c1cnc2ncsc2c1",
        "c1nncc2oncc12",
        "O=c1ccc2c[nH]ccn1-2",
        "N=c1ncnc2[nH][nH]cc1-2",
        "C=C1Nc2ccccc2S1",
        "C1=Nn2cnnc2SC1",
        "c1ncn2cncc2n1",
        "c1cc2c[nH]nc2cn1",
        "O=c1[nH]ccn2nccc12",
        "O=c1cnn2cnnc2[nH]1",
        "O=c1[nH]ncn2cnnc12",
        "c1ccc2oncc2c1",
        "c1cc2cc[nH]c2cn1",
        "c1ncc2cscc2n1",
        "c1cnn2cnnc2c1",
        "O=c1ccn2cnnc2s1",
        "c1ccn2nccc2c1",
        "N=C1CSc2nncn2N1",
        "O=c1ccnc2sccn12",
        "O=c1cnc2c[nH]ccn1-2",
        "C=c1sc2n(c1=O)CC=CN=2",
        "c1ccc2c(c1)N=S=N2",
        "O=C1C=Nc2cncc(=O)n21",
        "c1cnc2cscc2n1",
        "O=c1[nH]ncc2nn[nH]c12",
        "O=C1CN=C2C=NC=CN12",
        "O=C1C=NN2CNN=C2N1",
        "[CH]1C=CC=C2C=CC=C12",
        "c1cc2[s+]ccc-2c[nH]1",
        "O=c1ccnc2n1CCS2",
        "C=C1N=Cc2ccccc21",
        "C=C1Nc2ccccc2N1",
        "C=C1C(=O)c2ccccc2C1=O",
        "N=c1[nH]ncc2nn[nH]c12",
        "c1cnc2ncoc2c1",
        "C=C1Sc2ccccc2C1=O",
        "C1=CCC2CCCC2=C1",
        "c1ccc2c(c1)CCN2",
        "O=S1(=O)C=Cc2ccccc21",
        "B1Oc2ccccc2O1",
        "c1cc2nonc2cn1",
        "O=c1[nH]nnc2ccnn12",
        "[BH2-]1[O+]=CN=C2SC=NN12",
        "C=C1N=C2SC=NN2[BH2-]O1",
        "C1=CC2=CCCC2CC1",
        "C1=CC2=NNCC2CC1",
        "C=C1C(=C)c2ccccc2C1=C",
        "c1ccc2[se]c[nH+]c2c1",
        "C=C1Nc2ccccc2[Se]1",
        "C=[N+]1[BH2-][n+]2ccccc2[N-]1",
        "c1cnc2[nH]ccc2c1",
        "c1ccc2c[nH]cc2c1",
        "N=C1N=Cc2ccccc21",
        "O=c1ccnc2[nH][nH]c[n+]1-2",
        "C1=Cc2ccccc2[SiH2]1",
        "C=C1N=CC2=C1CCCC2",
        "C=C1CC2=CC(=S)C=CC2=[NH+]1",
        "N=C1N=C2C=CC=CN2C1=N",
        "C=C1N=C2SC=CN2[BH2-]O1",
        "C=C1N=c2ccccc2=[O+]1",
        "N=c1[nH][nH]c2nccc(=O)n12",
        "O=C1C=NC2=CN=CCN12",
        "c1[nH]cc2c1CCCC2",
        "C=C1CCCC2CCCC12",
        "C=C1C=CCC2COCC12",
        "C1=CC2CCCN2N=C1",
    ],
    "6p6": [
        "c1ccc2c(c1)CCCN2",
        "C1=CB2C(=CC=C3C=CC=CN23)C=C1",
        "c1ccc2ncccc2c1",
        "C1=CNC2=NCN=CC2=N1",
        "c1ccc2c(c1)CCCO2",
        "N=c1n[nH+]c2ccccc2[nH]1",
        "O=C1C=CC(=O)c2ccccc21",
        "c1cnc2c(c1)CCCC2",
        "C1=COC2=C(C1)CCCC2",
        "O=c1ccoc2ccccc12",
        "c1ccc2cnccc2c1",
        "c1ccc2ncncc2c1",
        "O=C1CCCC2=C1CC=CN2",
        "C1=NCNc2ccccc21",
        "c1cnc2ncccc2c1",
        "O=c1occc2ccccc12",
        "C1=Cc2ccccc2SC1",
        "C=C1C=c2ccccc2=[O+]C1=O",
        "O=c1cnc2cncnc2[nH]1",
        "c1cc2c(cn1)CCCC2",
        "O=c1ccnc2ccccn12",
        "C1=CC2CCCCC2CC1",
        "C=C1C=C2CCCNC2=CC1=[OH+]",
        "c1cc[n+]2ccccc2c1",
        "c1cc2nncnc2cn1",
        "C=C1NCCc2ccccc21",
        "O=C1NCNc2ccccc21",
        "O=c1ccnc2cnccn12",
        "O=C1C=CC2=CNCCC2=C1",
        "O=C1C=C2CCCCC2CC1",
        "O=c1ccc2cccoc-2c1",
        "O=C1C=Cc2ncccc2C1",
        "O=S1(=O)NC=Cc2ccccc21",
        "C=c1ccc2c(c1)C=CC(=[NH2+])C=2",
        "O=c1nc2ccccn2c(=O)[nH]1",
        "C=c1ccc2c(c1)C=CC(=O)C=2",
        "N=c1ncc2nccnc2[nH]1",
        "[BH2-]1OC=Cc2cccc[n+]21",
        "O=C1CCC2CCCCC2C1",
        "C=C1C=CC(=O)c2ncccc21",
        "c1ccc2nccnc2c1",
        "S=c1cc[n+]2ccccc2[nH]1",
        "C1=NNc2ccccc2S1",
        "C1=CSC2=CCCCC2=C1",
        "c1ccc2[o+]cccc2c1",
        "O=C1C=NNC2=NN=CNN12",
        "O=C1CCC2COC=CC2C1",
        "O=C1C=C2C=COCC2CC1",
        "N=c1nc2ncccc2c[nH]1",
        "C=C1C=Cc2cccnc2C1=O",
        "C1=Cc2ccccc2CC1",
        "O=c1cnc2cnccc2[nH]1",
        "c1cc2c(nn1)CCCC2",
        "O=c1cnc2cccnc2[nH]1",
        "N=c1ccc2c[nH]ccc-2c1",
        "C=C1C=CNc2ccccc21",
        "O=c1ccc2c(=O)occc2o1",
        "[BH2-]1NC=Cc2cccc[n+]21",
        "N=C1C=CC2=CNCCC2=C1",
        "O=c1[nH]c(=O)c2nccnc2[nH]1",
        "O=c1[nH]ncc2ccccc12",
        "N=c1[nH]ccc2ncccc12",
        "C1OCC2OCOCC2O1",
        "O=C1C=C2C=CCCC2CO1",
        "C1=COc2ccccc2C1",
        "c1ncc2c(n1)CCCC2",
        "[BH2-]1OCC=C2C=CN=CN12",
        "C=C1NS(=O)(=O)c2ccccc2C1=O",
        "O=c1[nH]c(=O)c2ncnnc2[nH]1",
        "C=C1C(=O)C=Cc2ccc(=O)oc21",
        "N=c1ncc2c([nH]1)NNC=N2",
        "O=c1nc[nH]c2ncccc12",
        "C1=COC2=CCCCC2=C1",
        "c1ccc2c(c1)OCCO2",
        "[BH2-]1[O+]=CC=C2C=CN=CN12",
    ],
    "5n6": [
        "c1ccc(CC2=NCCC2)cc1",
        "C=C1CCCC1=CC1CCCCC1",
        "c1ccc(Cc2cnsc2)cc1",
        "O=C1CCCC1=CC1CCCCC1",
        "c1ccc(CC2=NCNC2)cc1",
        "c1ccc(-c2ccno2)cc1",
        "c1ccc(Cc2nnco2)cc1",
        "c1ccc(-c2ccco2)cc1",
        "O=C1N=CC=CC1C1CCCO1",
        "c1ccc(Cc2ncco2)cc1",
        "O=C1CCCC1=Cc1ccccc1",
        "c1ccc(Cc2cccs2)cc1",
        "c1ccc(-c2cnco2)cc1",
        "c1cc(-c2nccs2)ccn1",
        "c1ccc(CC2=NCCN2)cc1",
        "c1cc(N2CCCC2)ccn1",
        "c1ccc(Cc2ccon2)cc1",
        "c1ccc(Cc2cncs2)cc1",
        "c1ccc(C2=NCCO2)cc1",
        "c1ccc(-c2ncco2)cc1",
        "C1=C(Cc2ccoc2)CCCC1",
        "O=C1N=CC=C1c1ccccc1",
        "O=C1NCC=C1Cc1ccccc1",
        "c1ccc(Cc2nccs2)cc1",
        "c1ccc(-n2cccn2)nc1",
        "c1ccc(Cc2ccco2)cc1",
        "O=c1ncccn1C1=CCCO1",
        "S=c1sscc1Cc1ccccc1",
        "c1ccc(-c2cccs2)cc1",
        "C(=Nc1cccs1)c1cccs1",
        "c1ccc(-n2cccc2)cc1",
        "C(=Nc1ccccc1)c1cncs1",
        "C=C1N=C(Cc2ccccc2)OC1=O",
        "O=C1C=CC(CC2=CCCC2)=CC1",
        "C(=NN=Cc1ccco1)c1ccccc1",
        "C1=CC(=Cc2nccs2)C=CO1",
        "S=C1C=CC(C2CCCO2)C=N1",
        "c1ccc(Cc2ccc[nH]2)cc1",
        "c1ccc(Cc2c[nH]cn2)cc1",
        "C(=Cc1ccco1)c1ccccc1",
        "C1=CC(=Cc2ccc[nH]2)C=CO1",
        "C(=Cc1cccs1)c1ccccc1",
        "O=C1N=CC(=Cc2ccccc2)S1",
        "O=C1C=CC(=Cc2ccccn2)N1",
        "c1ccc(-c2ncc[nH]2)cc1",
        "C=C1C=C(c2ccccc2)C(=O)O1",
        "O=C1CCCC1=CC=Cc1ccccc1",
        "C1=CC(=Cc2ccccn2)N=C1",
        "c1ccc(Cc2cc[nH]c2)cc1",
        "C(=Cc1ccccc1)C1=CCOC1",
        "C=C1C=NC(Cc2ccccc2)=C1",
        "O=c1nc(-n2cncn2)cc[nH]1",
        "O=C1NC=NC1=Cc1ccncc1",
        "C1=CC(=CC2C=CCCC2)N=C1",
        "O=c1scc(Cc2ccccc2)s1",
        "C1=CC(=Cc2cncs2)C=CO1",
        "O=C1NC(=S)SC1=Cc1ccccc1",
        "C=C1N=CN(c2ccccc2)C1=O",
        "c1ccc(Cc2ccn[nH]2)cc1",
        "C(#Cc1cccs1)c1ccccc1",
        "c1cc[n+](C2CCCO2)cc1",
        "c1ccc(-c2ccc[nH]2)cc1",
        "O=C1C=CC=C1Cc1ccccc1",
        "O=C1NC=NC1=Cc1ccccc1",
        "O=C1C=CC=CC1=C1C=S=CS1",
        "C=C1C=C(c2ccccc2)C=N1",
        "c1ccc(Cc2nnc[nH]2)cc1",
        "C1=CC(=Cc2ccco2)C=CO1",
        "C1=CC(=Cc2cccs2)C=CO1",
        "O=C(Nc1ccccc1)c1cncs1",
        "c1ccc(-c2cccs2)nc1",
        "C=C1N=CN(CCN=Cc2ccccc2)C1=O",
        "C=C1CCN(C(=O)CN=Cc2ccccc2)C1",
        "c1cc(CNC2CCCCC2)cs1",
        "c1csc(C2CCCCC2)c1",
        "C1=C(c2cccs2)CCCC1",
        "C1=CC(C=NN=Cc2ccccc2)=S=C1",
        "c1ccc(-n2cccn2)cc1",
        "c1ccc(-n2cnnn2)cc1",
        "c1ccc(-c2nncs2)cc1",
        "C1=NC(c2ccccc2)=NC1",
        "O=C1C=C(CC2=CC(=O)OC2)CCC1",
        "c1csc(CNC2CCCCC2)c1",
        "O=C1CC=C(c2ccccc2)N1",
        "c1ccc(CCc2cccs2)cc1",
        "c1csc(N2CCCCC2)c1",
        "c1ccc(-c2ccsc2)cc1",
        "c1csc(-c2cnnnn2)c1",
        "C=C1OC(=O)C(=Cc2ccccc2)C1=O",
        "O=C(c1ccccc1)c1cnoc1",
        "c1ccc(-c2cn[nH]n2)cc1",
        "C1=C(Cc2ccccc2)COC1",
        "c1cc(-n2nccn2)ccn1",
        "c1ccc(-c2cncs2)cc1",
        "c1ccc(-c2cscn2)cc1",
        "c1ccc(-n2nccn2)cc1",
        "C1=CC(Cc2ccccc2)=S=C1",
        "O=C(C=Cc1ccccc1)c1cc[cH-]c1",
        "C1=C(Cc2ccccc2)CCC1",
        "C(=Cc1ccc[nH]1)c1ccccc1",
        "O=C1CCCC=C1Cc1cccs1",
        "C1=CC(=Cc2ccccc2)N=C1",
        "C=C1CCCC1=Cc1ccccc1",
        "c1ccc(-c2nnco2)cc1",
        "C1=NC(c2ccccc2)=CC1",
        "C(CSc1nnc[nH]1)=NN=Cc1ccccc1",
        "O=C1CCCC=C1Cc1ccco1",
        "C1=NC(c2ccccc2)CO1",
        "O=c1c[nH+]n(Cc2ccccc2)o1",
        "C(=Cc1ccccc1)SC=C1CCCC1",
        "c1ccc(-c2ncc[se]2)cc1",
        "C1=CNC(=CC=C2C=CCS2)C=C1",
        "O=C1NCCC1=Cc1ccccc1",
        "C=C(C=C1C=CC=C1)c1ccccc1",
        "O=C(OC1=NCC=C1)c1ccccc1",
        "N=c1ncn(C2CCCO2)cn1",
        "c1csc(-c2ncncn2)c1",
        "C(=Cc1cnco1)c1ccccc1",
        "c1ccc(Cc2ncon2)cc1",
        "c1ccc(C2=NCCN2)cc1",
        "C(=Cc1ncco1)c1ccccc1",
        "c1ccc(CCc2ccc[nH]2)cc1",
        "O=C1CNC=C1Cc1ccccc1",
        "C1=NCC(c2ccccc2)O1",
        "c1cc(-c2ncco2)ccn1",
        "c1ccc(N=c2nc[nH]s2)cc1",
        "c1ccc(-c2c[nH]cn2)cc1",
        "c1ccc(-c2ccn[nH]2)cc1",
        "C=C1C=C(OC(=O)c2ccccc2)C(=O)O1",
        "c1ccc(Cc2cn[nH]c2)cc1",
        "C1=CNB(c2ccccc2)N1",
        "C1=CSC(=C2C=CNC=C2)[N-]1",
    ],
    "6n6": [
        "c1ccc(Cc2ccccc2)cc1",
        "[N+]=C1C=CC(=NN=C2C=CC(=[N+])C=C2)C=C1",
        "c1ccc(CCc2ccccc2)cc1",
        "c1ccc(CNc2ccccc2)cc1",
        "C(=Cc1ccccc1)Cc1ccccc1",
        "c1ccc(Nc2ccccc2)cc1",
        "c1ccc(Oc2ccccc2)cc1",
        "B(c1ccccc1)c1ccccc1",
        "c1ccc(Pc2ccccc2)cc1",
        "C(=Cc1ccccc1)c1ccccc1",
        "C(#Cc1ccccc1)c1ccccc1",
        "C(C#Cc1ccccc1)#Cc1ccccc1",
        "C(C=Cc1ccccc1)=Cc1ccccc1",
        "C(=Cc1ccccc1)CCCC=Cc1ccccc1",
        "C(=CC=Cc1ccccc1)C=Cc1ccccc1",
        "C(=Cc1ccccc1)CCCc1ccccc1",
        "C(#Cc1ccccc1)C=Cc1ccccc1",
        "c1ccc(-c2ccccc2)cc1",
        "c1ccc(COc2ccccc2)cc1",
        "c1ccc(-c2ccncn2)cc1",
        "c1ccc(Cc2ncccn2)cc1",
        "C(=NNCc1ccccc1)c1ccccc1",
        "C1=CNCC(Cc2ccccc2)=C1",
        "C1=COC(c2ccccc2)=CC1",
        "C1=COC(OC2CCCCO2)CC1",
        "C1=CCN(Cc2ccccc2)C=C1",
        "C(C=Cc1ccccc1)=CCC=Cc1ccccc1",
        "C1=CCCC(Cc2ccccc2)=C1",
        "c1ccc(Cc2ccccn2)cc1",
        "C1=CCC(OCc2ccccc2)=CC1",
        "C1=CNCC(Cc2ccccc2)=N1",
        "C1=CCOC(c2ccccc2)=C1",
        "C1=COC(C=Cc2ccccc2)=CC1",
        "C(=Cc1cccnc1)c1ccccc1",
        "c1ccc(-c2cnccn2)cc1",
        "c1ccc(Cc2cnccn2)cc1",
        "c1ccc(-c2ccccn2)nc1",
        "C1=C(Cc2ccccc2)COCC1",
        "C1=C(Cc2ccccc2)CNCN1",
        "c1ccc(Cc2cccnc2)cc1",
        "C(=Cc1ccncn1)c1ccccc1",
        "c1ccc(-c2cccnc2)cc1",
        "c1ccc(Cc2ccncc2)cc1",
        "c1ccc(NC2OCCCO2)cc1",
        "C1=CN(Cc2ccccc2)C=CC1",
        "C1=C(Cc2ccccc2)CCCC1",
        "C1=C(Cc2ccccc2)OCCC1",
        "C(=CCCc1ccccc1)C=NCc1ccccc1",
        "c1cc(-c2ccncc2)ccn1",
        "C1=C(Cc2ccccc2)CCOC1",
        "c1ccc(COC2CCCCC2)cc1",
        "c1ccc(-c2ncncn2)cc1",
        "O=c1ccoc(Cc2ccccc2)c1",
        "c1ccc(Cc2ccncn2)cc1",
        "c1ccc(-c2ncccn2)cc1",
        "C1=C(C2CCCCC2)CCCC1",
        "c1ccc(Cc2cccnn2)cc1",
        "c1ccc(Cc2nccnn2)cc1",
        "c1ccc(Cc2cncnc2)cc1",
        "c1ccc(-c2ccccn2)cc1",
        "c1ccc(Cc2cncnn2)cc1",
        "O=c1cccnn1-c1ccccc1",
        "O=c1occcc1Cc1ccccc1",
        "C=c1ccc(=Cc2ccccc2)cc1",
        "C(=Nc1ccccc1)c1ccccc1",
        "O=c1ccoc(CC2=CCCCC2)c1",
        "C=C1NC(Cc2ccccc2)=CCS1",
        "C1=CC(c2ccccc2)C=CN1",
        "O=C1C=CCC=C1Cc1ccccc1",
        "c1ccc(CCNCc2ccccc2)cc1",
        "C1=CCCC(c2ccccc2)=C1",
        "C(=Cc1ccncc1)c1ccccc1",
        "C(=NNc1ccccc1)c1ccccc1",
        "O=c1cccc(Cc2ccccc2)o1",
        "C(=Cc1ncccn1)c1ccccc1",
        "c1ccc(CC2=NCCCN2)cc1",
        "c1ccc(OCOc2ccccc2)cc1",
        "C1=C(Cc2cncnc2)CCCC1",
        "c1ccc(-c2cc[o+]cc2)cc1",
        "C(=Cc1cccnn1)c1ccccc1",
        "C(=NNc1ccccc1)c1ccccn1",
        "C1=CN(c2ccccc2)C=CC1",
        "O=C1C=C(Nc2ccccc2)CCC1",
        "N=c1ccccn1Cn1ccccc1=O",
        "C(=Cc1ccccn1)c1ccccc1",
        "C(=NNc1ccncn1)c1ccccc1",
        "C(=Cc1cnccn1)c1ccccc1",
        "C1=CC(NC2CCCOC2)CCC1",
        "C=C1C=C(c2ccccc2)OC=N1",
        "O=C1C=CC(=C2C=CNC=C2)C=C1",
        "O=C(C=Cc1ccccc1)c1cnccn1",
        "C=C1C=C(C=Cc2ccccc2)CCC1",
        "[N+]=C1C=CC=CC1=Cc1ccccc1",
        "c1ccc(CN2CCCCC2)cc1",
        "O=C1CCCC(=O)C1=NNc1ccccc1",
        "O=C1C=CCC=C1CC1=CCC=CC1=O",
        "C(=Cc1cccnc1)Cc1ccccc1",
        "C(=NN=Cc1ccccc1)c1ccccc1",
        "C(=Cc1cccnn1)C=C1C=COC=C1",
        "[O+]=C1C=CC=CC1=Cc1ccccc1",
        "C(=Cc1ccncn1)C=C1C=COC=C1",
        "C(=Cc1cnccn1)C=C1C=COC=C1",
        "C(C=Cc1ccncc1)=Cc1ccccc1",
        "c1ccc([N+]#[N+]c2ccccc2)cc1",
        "O=C1C=CC(=CC=C2C=CNC=C2)C=C1",
        "C(=NCCN=Cc1ccccc1)c1ccccc1",
        "C(=Cc1cc[nH+]cc1)c1cc[nH+]cc1",
        "C(=Cc1ccccc1)COCCc1ccccc1",
        "O=C1NC(=S)NC(=O)C1=Cc1ccccc1",
        "O=C1C=CC(=CC2CCC=CC2=O)C=C1",
        "C(=CN=Cc1ccccc1)N=Cc1ccccc1",
        "O=C(C=Cc1ccccc1)c1cccoc1=O",
        "C(=Cc1ccccc1)C=NN=Cc1ccccc1",
        "C1=C[CH+]C(=Cc2ccccc2)C=C1",
        "O=C1C=CC(=O)C(NCc2ccccc2)=C1",
        "N=c1[nH]cncc1CNCCSC(=O)c1ccccc1",
        "C(#Cc1ccncc1)c1ccccc1",
        "C1=NCSC(Cc2ccccc2)=N1",
        "O=C1C=CC(=O)C(c2ccccc2)=C1",
        "O=C1C=CC=CC1=CC=C1C=C[NH2+]C=C1",
        "C(#Cc1cnccn1)c1ccncc1",
        "C(#Cc1cnccn1)c1ccccc1",
        "C(=C[NH2+]c1ccccc1)C=Nc1ccccc1",
        "C1CCC(OC2CCOCC2)OC1",
        "c1ccc(-c2cccnn2)cc1",
        "C(#Cc1ncccn1)c1ccccc1",
        "C(#Cc1ccncn1)c1ccccc1",
        "O=c1cnnc(Cc2ccccc2)o1",
        "[N+]=C1C=C=C([CH-]C2=C=CC=C[CH+]2)C=C1",
        "O=S(=O)(c1ccccc1)N1CCSCC1",
        "c1ncnc(N2CCOCC2)n1",
        "[N+]=C1C=CC(=NN=C2C=CC(=[N+])C=C2)C=C1",
        "O=C(CCc1ccccc1)Nc1ccccc1",
        "O=C1C=NC(=Cc2ccccc2)C(=O)N1",
        "C(=NN=Cc1ccccc1)Nc1ccccc1",
        "c1ccc(CCC[n+]2ccccc2)cc1",
        "C(=Cc1ccccc1)CC1CCCCC1",
        "[BH2-]1OC(C=CC=Cc2ccccc2)=CC=[O+]1",
        "c1ccc(Cc2ccpcc2)cc1",
        "c1ccc(-c2ccncc2)cc1",
        "C1=C[N-]C(=C2C=CC=C[N-]2)C=C1",
        "C1=CC(=Cc2ccccc2)C=CC1",
        "N=C1C=CC(C=C1)=NN=C2C=CC(C=C2)=N",
    ],
    "Azo": [
        "N=N",
        "N=[N+]",
    ],
    "Benz": [
        "C1=CC=[NH+]CC=1",
        "C1=CCCC=C1",
        "C=C1C=CNCC1",
        "O=C1C=CC(=O)C=C1",
        "O=C1C=CN=CC1",
        "O=C1N=CCC=N1",
        "S=C1N=CCC=N1",
        "c1cc[o+]cc1",
        "c1cc[s+]cc1",
        "c1ccccc1",
        "c1ccncc1",
        "c1cnccn1",
        "c1cncnc1",
        "c1ncncn1",
        "c1nncnn1",
    ],
}


# Build scaffold name -> tag number mapping (order matters, matches original code)
_SCAFFOLD_TAG: dict[str, int] = {
    name: idx for idx, name in enumerate(_SCAFFOLD_SMILES.keys())
}

# Pre-compile scaffold SMILES into (tag_number, family_name, rdkit_mol) tuples
# at module load time for efficiency. Invalid patterns are silently skipped.
_SCAFFOLD_PATTERNS: list[tuple[int, str, Any]] = []
for _family_name, _smiles_list in _SCAFFOLD_SMILES.items():
    _tag = _SCAFFOLD_TAG[_family_name]
    for _smi in _smiles_list:
        try:
            _mol = Chem.MolFromSmiles(_smi)
            if _mol is not None:
                _SCAFFOLD_PATTERNS.append((_tag, _family_name, _mol))
        except Exception:
            logger.debug("Skipping invalid scaffold SMILES: %s", _smi)


def detect_scaffold(smiles: str) -> tuple[int, str]:
    """Classify molecule into exactly one scaffold family.

    Iterates through scaffold patterns in order; first match wins.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        Tuple of (tag_number, family_name).
        Returns (-1, "Other") if no scaffold matches.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (-1, "Other")
        for tag, family_name, pattern_mol in _SCAFFOLD_PATTERNS:
            if mol.HasSubstructMatch(pattern_mol):
                return (tag, family_name)
    except Exception:
        logger.warning("Scaffold detection failed for: %s", smiles)
    return (-1, "Other")


# ---------------------------------------------------------------------------
# Substructure flag computation (136 binary flags from 00_mmp_substructure.csv)
# ---------------------------------------------------------------------------


def load_substructure_patterns(csv_path: str) -> list[tuple[int, Any]]:
    """Load 00_mmp_substructure.csv and compile SMARTS patterns.

    Args:
        csv_path: Path to the substructure CSV file.

    Returns:
        List of (index, compiled_mol) tuples. Patterns that fail to
        compile are silently skipped.
    """
    df = pd.read_csv(csv_path)
    patterns: list[tuple[int, Any]] = []
    for _, row in df.iterrows():
        idx = int(row["index"])
        fragment = str(row["fragment"])
        try:
            mol = Chem.MolFromSmarts(fragment)
            if mol is not None:
                patterns.append((idx, mol))
            else:
                logger.debug(
                    "Failed to compile SMARTS at index %d: %s", idx, fragment
                )
        except Exception:
            logger.debug(
                "Exception compiling SMARTS at index %d: %s", idx, fragment
            )
    return patterns


def compute_scaffold_flags(
    smiles: str, patterns: list[tuple[int, Any]]
) -> np.ndarray:
    """Compute 136 binary substructure flags from compiled patterns.

    Args:
        smiles: SMILES string of the molecule.
        patterns: List of (index, compiled_mol) from load_substructure_patterns().

    Returns:
        Numpy array of shape (136,) with int32 dtype, elements 0 or 1.
        All zeros if the SMILES is invalid.
    """
    flags = np.zeros(136, dtype=np.int32)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return flags
        for idx, pattern_mol in patterns:
            if 0 <= idx < 136:
                if mol.HasSubstructMatch(pattern_mol):
                    flags[idx] = 1
    except Exception:
        logger.warning("Scaffold flag computation failed for: %s", smiles)
    return flags


# ---------------------------------------------------------------------------
# CSV processing, property splitting, and data merging
# ---------------------------------------------------------------------------

# Required columns in the user input CSV.
_REQUIRED_CSV_COLUMNS: list[str] = [
    "name", "solvent", "abs", "em", "epsilon", "mw", "plqy", "smiles",
]

# Column order for the 152-column main DataFrame.
# Matches the layout in Fluor-RLAT/data/train_{property}.csv.
_MAIN_COLUMNS: list[str] = (
    ["split", "smiles", "solvent", "abs", "em", "plqy", "k",
     "tag_name", "solvent_num", "tag",
     "MW", "LogP", "TPSA", "Double_Bond_Count", "Ring_Count", "unimol_plus"]
    + [f"fragment_{i}" for i in range(1, 137)]
)


def process_input_csv(
    csv_path: str,
    solvent_mapping: dict[str, int],
    substructure_patterns: list[tuple[int, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[tuple[str, str]]]:
    """Parse a user CSV and produce training-format DataFrames.

    Reads the CSV, validates each row, computes fingerprints, descriptors,
    scaffold info, and epsilon-to-k conversion. Invalid rows are skipped
    and recorded.

    Args:
        csv_path: Path to the user input CSV file.
        solvent_mapping: Dict from load_solvent_mapping() mapping solvent
            SMILES to numeric IDs.
        substructure_patterns: List from load_substructure_patterns().

    Returns:
        Tuple of (main_df, smiles_fp_df, solvent_fp_df, skipped_rows).
        - main_df: 152 columns matching the training data layout.
        - smiles_fp_df: 1024 columns named "0".."1023" (molecule FPs).
        - solvent_fp_df: 1024 columns named "0".."1023" (solvent FPs).
        - skipped_rows: List of (name, reason) tuples for invalid rows.

    Raises:
        ValueError: If required columns are missing from the CSV.
    """
    df = pd.read_csv(csv_path)

    # Validate required columns
    missing = [c for c in _REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}"
        )

    skipped_rows: list[tuple[str, str]] = []
    main_rows: list[dict[str, Any]] = []
    smiles_fps: list[list[int]] = []
    solvent_fps: list[list[int]] = []

    fp_col_names = [str(i) for i in range(1024)]

    for _, row in df.iterrows():
        name = str(row.get("name", ""))
        smiles_val = row.get("smiles")

        # Skip rows with missing/empty SMILES
        if pd.isna(smiles_val) or str(smiles_val).strip() == "":
            skipped_rows.append((name, "Missing SMILES"))
            continue

        smiles_str = str(smiles_val).strip()

        # Map solvent name to SMILES
        solvent_name = str(row.get("solvent", ""))
        solvent_smiles = map_solvent_name_to_smiles(solvent_name)
        if solvent_smiles is None:
            skipped_rows.append(
                (name, f"Unknown solvent: {solvent_name}")
            )
            continue

        # Map solvent SMILES to numeric ID
        solvent_num = map_solvent_smiles_to_id(solvent_smiles, solvent_mapping)
        # Assumption: if solvent SMILES not in mapping, we still keep the row
        # but solvent_num will be None. This matches the original notebook
        # behavior where solvent_num can be NaN for unmapped solvents.

        # Compute Morgan fingerprints
        mol_fp = compute_morgan_fingerprint(smiles_str)
        if mol_fp is None:
            skipped_rows.append((name, "Missing SMILES"))
            continue

        sol_fp = compute_morgan_fingerprint(solvent_smiles)
        # Assumption: solvent SMILES from our mapping are always valid,
        # so sol_fp should never be None. But handle gracefully.
        if sol_fp is None:
            logger.warning(
                "Solvent FP failed for %s (solvent: %s)", name, solvent_name
            )
            skipped_rows.append(
                (name, f"Unknown solvent: {solvent_name}")
            )
            continue

        # Compute molecular descriptors
        descriptors = compute_molecular_descriptors(smiles_str)
        if descriptors is None:
            skipped_rows.append((name, "Missing SMILES"))
            continue

        # Scaffold detection
        tag, tag_name = detect_scaffold(smiles_str)

        # Scaffold flags (136 binary)
        flags = compute_scaffold_flags(smiles_str, substructure_patterns)

        # Epsilon to k conversion: k = log10(epsilon) for epsilon > 0
        epsilon_val = row.get("epsilon")
        if pd.notna(epsilon_val):
            eps = float(epsilon_val)
            k_val = np.log10(eps) if eps > 0 else np.nan
        else:
            k_val = np.nan

        # Build main row dict
        main_row: dict[str, Any] = {
            "split": "train",
            "smiles": smiles_str,
            "solvent": solvent_smiles,
            "abs": float(row["abs"]) if pd.notna(row.get("abs")) else np.nan,
            "em": float(row["em"]) if pd.notna(row.get("em")) else np.nan,
            "plqy": float(row["plqy"]) if pd.notna(row.get("plqy")) else np.nan,
            "k": k_val,
            "tag_name": tag_name,
            "solvent_num": solvent_num,
            "tag": tag,
            "MW": descriptors["MW"],
            "LogP": descriptors["LogP"],
            "TPSA": descriptors["TPSA"],
            "Double_Bond_Count": descriptors["Double_Bond_Count"],
            "Ring_Count": descriptors["Ring_Count"],
            "unimol_plus": np.nan,  # placeholder
        }
        for i in range(136):
            main_row[f"fragment_{i + 1}"] = int(flags[i])

        main_rows.append(main_row)
        smiles_fps.append(mol_fp.tolist())
        solvent_fps.append(sol_fp.tolist())

    # Build output DataFrames
    if main_rows:
        main_df = pd.DataFrame(main_rows, columns=_MAIN_COLUMNS)
        smiles_fp_df = pd.DataFrame(smiles_fps, columns=fp_col_names)
        solvent_fp_df = pd.DataFrame(solvent_fps, columns=fp_col_names)
    else:
        main_df = pd.DataFrame(columns=_MAIN_COLUMNS)
        smiles_fp_df = pd.DataFrame(
            columns=fp_col_names, dtype=np.int32
        )
        solvent_fp_df = pd.DataFrame(
            columns=fp_col_names, dtype=np.int32
        )

    return main_df, smiles_fp_df, solvent_fp_df, skipped_rows


def split_by_property(
    main_df: pd.DataFrame,
    smiles_fp_df: pd.DataFrame,
    solvent_fp_df: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Split DataFrames into per-property subsets, excluding NaN rows.

    For each property in {abs, em, plqy, k}, filters out rows where
    that property column is NaN.

    Args:
        main_df: The 152-column main DataFrame.
        smiles_fp_df: The 1024-column molecule fingerprint DataFrame.
        solvent_fp_df: The 1024-column solvent fingerprint DataFrame.

    Returns:
        Dict mapping property name to (main, smiles_fp, solvent_fp) tuple,
        each filtered to rows where the property is not NaN.
    """
    from .config import PROPERTIES

    result: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for prop in PROPERTIES:
        if prop not in main_df.columns:
            continue
        mask = main_df[prop].notna()
        result[prop] = (
            main_df.loc[mask].reset_index(drop=True),
            smiles_fp_df.loc[mask].reset_index(drop=True),
            solvent_fp_df.loc[mask].reset_index(drop=True),
        )
    return result


def merge_training_data(
    new_main: pd.DataFrame,
    new_smiles_fp: pd.DataFrame,
    new_sol_fp: pd.DataFrame,
    existing_main: pd.DataFrame,
    existing_smiles_fp: pd.DataFrame,
    existing_sol_fp: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Concatenate new and existing training DataFrames row-wise.

    Aligns fingerprint column names of the new DataFrames to match
    the existing DataFrames before concatenation.

    Args:
        new_main: New main DataFrame (152 columns).
        new_smiles_fp: New molecule fingerprint DataFrame (1024 columns).
        new_sol_fp: New solvent fingerprint DataFrame (1024 columns).
        existing_main: Existing main DataFrame.
        existing_smiles_fp: Existing molecule fingerprint DataFrame.
        existing_sol_fp: Existing solvent fingerprint DataFrame.

    Returns:
        Tuple of (merged_main, merged_smiles_fp, merged_sol_fp).
    """
    # Align FP column names: rename new columns to match existing
    if len(existing_smiles_fp.columns) > 0:
        new_smiles_fp = new_smiles_fp.copy()
        new_smiles_fp.columns = existing_smiles_fp.columns
    if len(existing_sol_fp.columns) > 0:
        new_sol_fp = new_sol_fp.copy()
        new_sol_fp.columns = existing_sol_fp.columns

    merged_main = pd.concat(
        [existing_main, new_main], ignore_index=True
    )
    merged_smiles_fp = pd.concat(
        [existing_smiles_fp, new_smiles_fp], ignore_index=True
    )
    merged_sol_fp = pd.concat(
        [existing_sol_fp, new_sol_fp], ignore_index=True
    )

    return merged_main, merged_smiles_fp, merged_sol_fp
