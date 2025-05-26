"""
In this script, we create a dataset from a file of SMILES strings to pretrain our model. We create a `MoleculeDesign`
from each SMILES to obtain a sequence of actions.
"""
import time
import pickle
from rdkit import Chem, RDLogger
from config import MoleculeConfig
from molecule_design import MoleculeDesign
from typing import Optional, Tuple, List
from tqdm import tqdm

datatypes = ["valid", "train"]
limit_num_atoms = 100

for datatype in datatypes:
    start_time = time.perf_counter()
    molecules: List[Tuple[Chem.RWMol, str]] = []
    molecule_designs: List[dict] = []

    path_to_smiles = f"./data/chembl/chembl_{datatype}_filtered.smiles"
    destination_path = f"./data/chembl/pretrain_sequences/chembl_{datatype}.pickle"
    limit_num_smiles_to = None  # Set to `None` to process all

    num_differing_smiles = 0
    print("Converting SMILES to molecules")
    with open(path_to_smiles) as f:
        for line in tqdm(f):
            smiles = line.rstrip()
            if len(smiles) > 0:
                mol = Chem.MolFromSmiles(smiles)
                Chem.SanitizeMol(mol)
                #Chem.Kekulize(mol)
                s = Chem.CanonSmiles(Chem.MolToSmiles(mol))

                if s != smiles:
                    num_differing_smiles += 1
                if mol.GetNumAtoms() <= limit_num_atoms:
                    molecules.append((mol, s))
            if len(molecules) == limit_num_smiles_to:
                break

    print(f"Created {len(molecules)} RDkit molecules from SMILES. SMILES differing in {num_differing_smiles} occasions.")
    max_num_atoms = max([x.GetNumAtoms() for x, _ in molecules])
    print(f"Maximum number of atoms: {max_num_atoms}")

    config = MoleculeConfig()
    """
    Allow everything in the config
    """
    config.max_num_atoms = max_num_atoms
    config.max_allowed_oxygen = None
    config.max_allowed_nitrogen = None
    config.min_ratio_c = None  # minimum ratio of C atoms to all atoms
    config.disallow_oxygen_bonding = False
    config.disallow_nitrogen_nitrogen_single_bond = False
    config.disallow_rings = False
    config.disallow_rings_larger_than = -1

    dont_match = []
    full_len = len(molecules)
    i = 0
    while len(molecules):
        mol, smiles = molecules.pop(0)
        i += 1
        # Create an instance of MoleculeDesign from it
        print(f"Converting {i}/{full_len} {Chem.MolToSmiles(mol)} ...")
        #molecule_design = MoleculeDesign.from_rdkit_mol(config, mol, smiles)
        smiles_to_process = [smiles]

        for s in smiles_to_process:
            molecule_design = MoleculeDesign.from_smiles(config, s, do_finish=True, compare_smiles=False)
            if Chem.CanonSmiles(molecule_design.smiles_string) != Chem.CanonSmiles(smiles):
                dont_match.append(Chem.CanonSmiles(smiles))
            instance = dict(
                start_atom=molecule_design.initial_atom,
                action_seq=molecule_design.history,
                smiles=molecule_design.smiles_string,
                obj=0.0,
                sa_score=0.0
            )
            molecule_designs.append(instance)

    print("Generated molecules didnt match with source SMILES in cases: ", len(dont_match))
    print(f"Generation took {time.perf_counter() - start_time} seconds.")

    with open(destination_path, "wb") as f:
        pickle.dump(molecule_designs, f)