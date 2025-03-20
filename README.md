# GraphXForm: Graph transformer for computer-aided molecular design

This is the repository for the paper **[GraphXForm: Graph transformer for computer-aided molecular design](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d4dd00339j)**,
published in Digital Discovery.

## Preliminary note on structure

The learning algorithm used here is a hybrid between the deep cross-entropy method and self-improvement learning (SIL) 
(from natural language processing and neural combinatorial optimization). The general structure of our code thus follows
the SIL repository [https://github.com/grimmlab/gumbeldore](https://github.com/grimmlab/gumbeldore). Please check out
this repo for general information on how the code is structured and what is happening in the folder `core`.
So note that regarding the `gumbeldore_config` dict in `config.py`: In accordance with our paper, where sequences are 
sampled without replacement for multiple rounds, the setting `gumbeldore_config.search_type` is set to `wor` 
('without replacement').

## Install

Install all requirements from `requirements.txt`. Additionally, you must install **[torch_scatter](https://github.com/rusty1s/pytorch_scatter)** with the options 
corresponding to your hardware.

Create subdirectories `results` (where model weights and experimental results will be saved), `data`, `data/chembl` and 
`data/chembl/pretrain_sequences`. 

### GuacaMol

For the drug design tasks, [guacamol](https://github.com/BenevolentAI/guacamol/tree/master) is required. A run will fail, since in guacamol's `utils.chemistry`, it needs `histogram` from scipy, which is no longer supported. You can exchange it with `from numpy import histogram`, see this [issue](https://github.com/BenevolentAI/guacamol/issues/33). 

## Pretraining

To pretrain with the settings in the paper, do the following: 

1. Download the file `chembl_35_chemreps.txt.gz` from this this link: [Download File](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35_chemreps.txt.gz)
and put the extracted `chembl_35_chemreps.txt` under the `./data/chembl` directory.
2. Run `$ python filter_molecules.py`. This will perform a train/val split with 100k validation molecules, which will be
put under `data/chembl/chembl_{train/valid}.smiles`. Then, depending on the allowed vocabulary (see `filter_molecules.py`),
it filters the smiles for all strings containing the allowed sets of characters and save them under `data/chembl/chembl_{train/valid}_filtered.smiles`.
3. Run `$ python create_pretrain_dataset.py`. This will convert the filtered SMILES to instances of `MoleculeDesign` (the class in `molecule_design.py`, which takes the role of the molecular graph environment).
The pickled molecules are saved in `./data/chembl/pretrain_sequences/chembl_{train/valid}.pickle`.
4. Run `$ python pretrain.py` to perform pretraining of the model. The general config to use (e.g., architecture) is under `config.py`.
In `pretrain.py`, you can specify pretrain-specific options directly at the entrypoint to adjust to your hardware, i.e.:
```python
if __name__ == '__main__':
    pretrain_train_dataset = "./data/pretrain_data.pickle"  # Path to the pretraining dataset
    pretrain_num_epochs = 1000   # For how many epochs to train
    batch_size = 512  # Minibatch size
    num_batches_per_epoch = 3000   # Number of minibatches per epoch.
    batch_size_validation = 512  # Batch size during validation
    training_device = "cuda:0"  # Device on which to train. Set to "cpu" if no CUDA available.
    num_dataloader_workers = 30  # Number of dataloader workers for creating batches for training
    load_checkpoint_from_path = None   # Path to checkpoint if training should be continued from existing weights.
```
4. The terminal output will show under which subdirectory (named after timestamp) in `./results` the script will save the model checkpoints.

## Finetuning

Finetuning the model is performed by running `$ python main.py`. We note that during finetuning, all layers excpet the last
are frozen. The first thing that should be done is modify `config.py`
to configure everything to your needs. The default `config.py` here in the repo reflects in general the setup in the paper, 
and we list all options and their meaning below in detail. The config as it is, is defined to perform `celecoxib_rediscovery` of the GuacaMol benchmark.

In each epoch during finetuning, the agent generates a high number of potential molecules, and stores the best ones 
under `./data/generated_molecules.pickle` (can be changed via `gumbeldore_config.destination_path` in the config, see below).
**Important:** This file is loaded as the supervised training dataset in each epoch, and it is always _merged_ with freshly
generated molecules. In particular, if the file is not deleted before an experiment run, __the experiment will continue
from this file__. 

Results are saved to a subfolder with the current timestamp under `results` (see config below to change it).
During a run, after an epoch, the current network weights will be saved: `best_model.pt` for the epoch where the overall best 
molecule so far has been found. `last_model.pt` for the most recent epoch. We also save a text file with the overall best
20 molecules found so far. We'd recommend logging everything that's being logged to `log.txt` also to Mlflow or similar.
See the method `Logger.log_metrics` in `logger.py`. 

### Config

We'll go through each config parameter in `config.py`. Also see the SIL repository [https://github.com/grimmlab/gumbeldore](https://github.com/grimmlab/gumbeldore)
for more detailed info on the parallelization (which is not really used here).

- `seed`: [int] Random seed. Defaults to 42.

#### Network architecture options

- `latent_dimension`: [int] Latent dimension $`d`$. Defaults to 128.
- `num_transformer_blocks`: [int] Number of layers in the stack of transformer blocks for the architecture. Defaults to 8.
- `num_heads`: [int] Number of heads in the multihead attention. Defaults to 8.
- `dropout`: [float] Dropout for feedforward layer in a transformer block. Defaults to 0.
- `use_rezero_transformer`: [bool] Whether ReZero normalization should be used or regular layer normalization. Defaults to `True`.

#### Molecule environment options

- `wall_clock_limit`: Optional[int]. Defaults to `None`, but is set to `3600*8` in the paper. Number of seconds how long an experiment may run.
    If `None`, no limit is set.
- `max_num_atoms`: [int] Maximum number of atoms that may exist in a designed molecule.

- `atom_vocabulary`: [dict] Specifies the vocabulary that is allowed in the design process. Important: Order matters, 
   because the indexing is derived from it. To explain how it works, it's best to look at the default:
   ```python
   self.atom_vocabulary = {  # Attention! Order matters!
    "C":    {"allowed": True, "atomic_number": 6, "valence": 4},
    "C-":   {"allowed": True, "atomic_number": 6, "valence": 3, "formal_charge": -1},
    "C+":   {"allowed": True, "atomic_number": 6, "valence": 5, "formal_charge": 1},
    "C@":   {"allowed": True, "atomic_number": 6, "valence": 4, "chiral_tag": 1},
    "C@@":  {"allowed": True, "atomic_number": 6, "valence": 4, "chiral_tag": 2},
    "N":    {"allowed": True, "atomic_number": 7, "valence": 3},
    "N-":   {"allowed": True, "atomic_number": 7, "valence": 2, "formal_charge": -1},
    "N+":   {"allowed": True, "atomic_number": 7, "valence": 4, "formal_charge": 1},
    "O":    {"allowed": True, "atomic_number": 8, "valence": 2},
    "O-":   {"allowed": True, "atomic_number": 8, "valence": 1, "formal_charge": -1},
    "O+":   {"allowed": True, "atomic_number": 8, "valence": 3, "formal_charge": 1},
    "F":    {"allowed": True, "atomic_number": 9, "valence": 1},
    "P":    {"allowed": True, "atomic_number": 15, "valence": 7},
    "P-":   {"allowed": True, "atomic_number": 15, "valence": 6, "formal_charge": -1},
    "P+":   {"allowed": True, "atomic_number": 15, "valence": 8, "formal_charge": 1},
    "S":    {"allowed": True, "atomic_number": 16, "valence": 6},
    "S-":   {"allowed": True, "atomic_number": 16, "valence": 5, "formal_charge": -1},
    "S+":   {"allowed": True, "atomic_number": 16, "valence": 7, "formal_charge": 1},
    "S@":   {"allowed": True, "atomic_number": 16, "valence": 6, "chiral_tag": 1},
    "S@@":  {"allowed": True, "atomic_number": 16, "valence": 6, "chiral_tag": 2},
    "Cl": {"allowed": True, "atomic_number": 17, "valence": 1},
    "Br": {"allowed": True, "atomic_number": 35, "valence": 1},
    "I": {"allowed": True, "atomic_number": 53, "valence": 1}
    }
    ```
    Each key is an atom name (e.g., a single letter, but naming is arbitrary) corresponding to a node type that can be placed
    on the graph. The value is again a dictionary, with `allowed` indicating if the atom may be placed (otherwise, it will be masked,
    useful for seeing how turning atom types on and off affect the performance). `atomic_number` is used to identify the atom type
    in `rdkit`. `valence` specifies to how many other non-hydrogen atoms we can bond it with. For ionization, we also set `formal_charge`, and for chirality, set `chiral_tag` (1 for rdkit's `Chem.CHI_TETRAHEDRAL_CW` and 2 for `Chem.CHI_TETRAHEDRAL_CCW`.
    **Note that for the solvent design tasks, we set `allowed` to `False` for all atoms except C, N, O.** 

- `start_from_c_chains`: [bool] If `True` (Default), the agent starts each design from a carbon chain. See `start_c_chain_max_len` below in that case.
- `start_c_chain_max_len`: [int] Defaults to 1. Only relevant if `start_from_c_chains` is `True`. Then, if integer `n` is given here, 
    the agent starts its design from multiple start points with SMILES 'C', 'CC', 'CCC', ... 'CCC...C' (`n` times). So when set to 1 (default),
    the agent always starts from a single C-atom.
- `start_from_smiles`: Optional[str] Only relevant if `start_from_c_chains` is `False`. If set, then the agent will start all its designs
   from a graph corresponding to the given SMILES string. Defaults to `None`.
- `repeat_start_instances`: [int] Defaults to 1. If larger than 1, the agent uses multiple, independent search trees (for sampling) for each
    design from which it starts. 
            # Positive value x, where the actual objective with our molecule score will be set to obj = score - x * SA_score
- `synthetic_accessibility_in_objective_scale`: [float] Defaults to 0. This is actually not used in the paper. If a positive
    value `x` is given, the objective value will be augmented with synthetic accessibility score. I.e., let `score` be the original
    objective value, then the updated one is set to `score - x * SA_score`.
- `include_structural_constraints`: [bool] If `True`, design is performed under the constraints mentioned in the paper. Used for the constrained results of the solvent design tasks. 

#### Objective function options

- `GHGNN_model_path`: [str] Path to activity coefficient prediction model. Defaults to `objective_predictor/GH_GNN_IDAC/models/GHGNN.pth`.
- `GHGNN_hidden_dim`: [int] Latent dim of activity coefficient prediction model. Defaults to 113 and should not be altered.
- `objective_type`: [str] Objective function to use for finetuning as described in paper. For **solvent design**, this is either **"IBA"** or **"DMBA_TMB"** (for TMB/DMBA). For the GuacaMol goal-directed tasks, this is one of the following: `celecoxib_rediscovery`,`troglitazone_rediscovery`,`thiothixene_rediscovery`,`aripiprazole_similarity`,`albuterol_similarity`,`mestranol_similarity`,`isomers_c11h24`,`isomers_c9h10n2o2pf2cl`,`median_camphor_menthol`,`median_tadalafil_sildenafil`,`osimertinib_mpo`,`fexofenadine_mpo`,`ranolazine_mpo`,`perindopril_rings`,`amlodipine_rings`,`sitagliptin_replacement`,`zaleplon_mpo`,`valsartan_smarts`,`deco_hop`,`scaffold_hop`.
- `num_predictor_workers`: [int] Number of parallel workers that distribute objective function evaluations between each other. Defaults to 10.
- `objective_predictor_batch_size`: [int] Batch size for inference of the activity coefficient model.
- `objective_gnn_device`: [str] Device on which the activity model lives. Defaults to "cpu".

#### Training flow options

- `load_checkpoint_from_path`: Optional[str]. If given, model checkpoint is loaded from this path. This should normally be set to the
    checkpoint after pretraining. Defaults to `None`.
- `load_optimizer_state`: [bool] If `True`, the optimizer state is also loaded from the checkpoint path above. Defaults to `False`.
- `num_dataloader_workers`: [int] Number of workers for preparing batches for supervised training. Defaults to 30.
- `CUDA_VISIBLE_DEVICES`: [str] ray.io sometimes needs help to recognize multiple GPUs. 
This variable will be set as the env var of the same name in each worker. Defaults to "0,1" (for two GPUs). Modify as needed.
- `training_device`: [str] Device on which network training is performed. Defaults to "cuda:0".
- `num_epochs`: [int] Number of epochs to train in total. Is overridden by `wall_clock_limit` above. Defaults to 1000.
- `scale_factor_level_one`: [float] Scale factor for loss at level 1 (pick atom to bond new atom to / pick first atom for bond increase). Defaults to 1. 
- `scale_factor_level_two`: [float] Scale factor for loss at level 2 (pick second atom for bond increase). Defaults to 1.
- `batch_size_training`: [int] Batch size to use for the supervised training during finetuning. Defaults to 64.
- `num_batches_per_epoch`: [int] Number of batches to use for supervised training during finetuning in each epoch. Defaults to 20. Can be `None`, then 
    one pass through the current generated dataset is done.
- `optimizer`: [dict] Optimizer configuration. Options are commented on in `config.py`.

#### Learning algorithm for finetuning options

- `gumbeldore_config`: [dict] This is the config for the self-improvement part. We go through the options below but only explain how they are used
    in our setup. For a more general discussion, please see the TASAR repository [https://github.com/grimmlab/step-and-reconsider/](https://github.com/grimmlab/step-and-reconsider/).
    - `num_trajectories_to_keep:` [int] Number of best designed molecules to keep, which are used for supervised training during finetuning. Default is 100.
    - `keep_intermediate_trajectories`: [bool] If this is `True`, all designed molecules encountered in the trie are considered,
        not only the leaves.
    - `devices_for_workers`: List[str] Number of parallel workers and on which devices their models live. Defaults to `["cuda:0"] * 1`.
    - `destination_path`: [str] Path where the generated molecules after each epoch are stored (and then loaded from to use as training dataset).
        Defaults to `"./data/generated_molecules.pickle"`. **Note**: You need to manually delete this file if you want to start fresh in a new run.
    Otherwise, it will always only be updated.
    - `batch_size_per_worker`: [int] If you start from a single atom, keep at 1.
    - `batch_size_per_cpu_worker`: [int] Same as above. This value is used for workers whose models live on the CPU.
    - `search_type`: [str] Keep at `'tasar'` ("Take a step and reconsider": sampling without replacement using stochastic beam search, following the best solution for a number of steps, before considering alternatives).
    - `beam_width`: [int] Beam width for stochastic beam search. Defaults to 16 for demonstration purposes. Value in paper is 512.
    - `replan_steps`: [int] Needed for TASAR. Defines the 'step size', i.e. for how many actions the best solution should be followed before sampling unseen alternatives. Defaults to 12 (value used in paper), indicating that we resample solutions after every 3 atoms placed.
    - `num_rounds`: Union[int, Tuple[int, int]]. (No longer used in the paper, only relevant for `search_type='wor'`). If it's a single integer, we sample for this many rounds exactly. If it's an (int, int)-tuple
      (as used in the paper), then we sample as long as it takes to obtain a new best molecule, but for a minimum of first entry rounds and a maximum of second entry rounds.
    - `deterministic`: [bool] Set to `True` to switch to deterministic beam seach. Not relevant for paper.
    - `nucleus_top_p`: [float] Top-p sampling nucleus size. Defaults to 1.0 (no nucleus sampling)
    - `pin_workers_to_core`: [bool] Default to `False`. If `True`, workers are pinned to single CPU threads, which can help with many workers on the CPU to prevent them from jamming each other with their numpy and pytorch operations.

#### Results and logging options

- `results_path`: [str] Path where to save results (checkpoint and top 20 molecules with objective values and SMILES). Defaults to 
    ```python
    os.path.join("./results", datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
    ```
- `log_to_file`: [bool] If logging output should also be saved to text file. Defaults to `True`.

## Acknowledgments

Thanks to the following repositories:

- [rezero](https://github.com/majumderb/rezero)
- [step-and-reconsider](https://github.com/grimmlab/step-and-reconsider), using
    - [gumbeldore](https://github.com/grimmlab/gumbeldore)
    - [unique-randomizer](https://github.com/google-research/unique-randomizer)
    - [stochastic-beam-search](https://github.com/wouterkool/stochastic-beam-search/tree/stochastic-beam-search)
