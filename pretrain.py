import argparse
import copy
import importlib
import os

from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from logger import Logger
from molecule_dataset import RandomMoleculeDataset

import torch
import numpy as np
from config import MoleculeConfig
from model.molecule_transformer import MoleculeTransformer, dict_to_cpu


def save_checkpoint(checkpoint: dict, filename: str, config: MoleculeConfig):
    os.makedirs(config.results_path, exist_ok=True)
    path = os.path.join(config.results_path, filename)
    torch.save(checkpoint, path)


def train_for_one_epoch(epoch: int, config: MoleculeConfig, network: MoleculeTransformer,
                        optimizer: torch.optim.Optimizer, dataset: RandomMoleculeDataset, is_validation=False):

    dataloader = DataLoader(dataset, batch_size=1, shuffle=not is_validation, num_workers=config.num_dataloader_workers,
                            pin_memory=True, persistent_workers=True)
    metrics = dict()
    # Train for one epoch
    network.train() if not is_validation else network.eval()
    accumulated_loss = 0
    accumulated_loss_lvl_zero = 0
    accumulated_loss_lvl_one = 0
    accumulated_loss_lvl_two = 0
    num_batches = len(dataloader)
    progress_bar = tqdm(range(num_batches))
    data_iter = iter(dataloader)
    for _ in progress_bar:
        data = next(data_iter)
        input_data = {k: v[0].to(network.device) for k, v in data["input"].items()}
        # targets for the logit levels
        target_zero = data["target_zero"][0].to(network.device)
        target_one = data["target_one"][0].to(network.device)
        target_two = data["target_two"][0].to(network.device)

        logits_zero, logits_one, logits_two = network(input_data)

        # We mask the output according to feasibility
        logits_zero[input_data["feasibility_mask_level_zero"]] = float("-inf")
        logits_one[input_data["feasibility_mask_level_one"]] = float("-inf")
        logits_two[input_data["feasibility_mask_level_two"]] = float("-inf")

        criterion = CrossEntropyLoss(reduction="mean", ignore_index=-1)
        loss_zero = criterion(logits_zero, target_zero)
        loss_zero = torch.tensor(0.) if torch.isnan(loss_zero) else loss_zero
        loss_one = criterion(logits_one, target_one)
        loss_one = torch.tensor(0.) if torch.isnan(loss_one) else loss_one
        loss_two = criterion(logits_two, target_two)
        loss_two = torch.tensor(0.) if torch.isnan(loss_two) else loss_two
        loss = loss_zero + config.scale_factor_level_one * loss_one + config.scale_factor_level_two * loss_two

        if not is_validation:
            # Optimization step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if config.optimizer["gradient_clipping"] > 0:
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=config.optimizer["gradient_clipping"])

            optimizer.step()

        batch_loss = loss.item()
        accumulated_loss += loss.item()
        accumulated_loss_lvl_zero += loss_zero.item()
        accumulated_loss_lvl_one += loss_one.item()
        accumulated_loss_lvl_two += loss_two.item()

        progress_bar.set_postfix({"batch_loss": batch_loss})

        del data

    metric_prefix = "" if not is_validation else "val_"
    metrics[f"{metric_prefix}full_loss"] = accumulated_loss / num_batches
    metrics[f"{metric_prefix}loss_level_zero"] = accumulated_loss_lvl_zero / num_batches
    metrics[f"{metric_prefix}loss_level_one"] = accumulated_loss_lvl_one / num_batches
    metrics[f"{metric_prefix}loss_level_two"] = accumulated_loss_lvl_two / num_batches

    return metrics


if __name__ == '__main__':
    pretrain_train_dataset = "./data/chembl/pretrain_sequences/chembl_train.pickle"
    pretrain_val_dataset = "./data/chembl/pretrain_sequences/chembl_valid.pickle"
    pretrain_num_epochs = 1000
    batch_size = 512
    num_batches_per_epoch = 3000
    batch_size_validation = 512
    training_device = "cuda:0"  # Device on which to train.
    num_dataloader_workers = 10  # Number of dataloader workers for creating batches for training
    load_checkpoint_from_path = None

    print(">> Pretraining Molecule Design")

    parser = argparse.ArgumentParser(description='Experiment')
    parser.add_argument('--config', help="Path to optional config relative to main.py")
    args = parser.parse_args()

    if args.config is not None:
        # Load config from given path
        MoleculeConfig = importlib.import_module(args.config).MoleculeConfig

    config = MoleculeConfig()
    print(f"Results path: {config.results_path}")
    config.max_num_atoms = None
    config.training_device = training_device
    config.num_dataloader_workers = num_dataloader_workers

    logger = Logger(args, config.results_path, config.log_to_file)
    logger.log_hyperparams(config)
    # Fix random number generator seed for better reproducibility
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Setup the neural network for training
    network = MoleculeTransformer(config, config.training_device)

    # Load checkpoint if needed
    if load_checkpoint_from_path is not None:
        print(f"Loading checkpoint from path {load_checkpoint_from_path}")
        checkpoint = torch.load(load_checkpoint_from_path)
        print(f"{checkpoint['pretrain_epochs_trained']} episodes have been trained in the loaded checkpoint.")
    else:
        checkpoint = {
            "model_weights": None,
            "best_model_weights": None,
            "optimizer_state": None,
            "pretrain_epochs_trained": 0,
            "pretrain_best_validation_loss": float("inf"),
            "epochs_trained": 0,
            "validation_metric": float("-inf"),   # objective of the best molecule designed during validation.
            "best_validation_metric": float("-inf")  # corresponding to best model weights
        }
    if checkpoint["model_weights"] is not None:
        network.load_state_dict(checkpoint["model_weights"])

    print(f"Policy network is on device {config.training_device}")
    network.to(network.device)
    network.eval()

    if pretrain_num_epochs > 0:
        # Training loop
        print(f"Starting pre-training for {pretrain_num_epochs} epochs.")

        best_validation_metric = checkpoint["pretrain_best_validation_loss"]

        print("Setting up optimizer.")
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config.optimizer["lr"],
            weight_decay=config.optimizer["weight_decay"]
        )
        if checkpoint["optimizer_state"] is not None and config.load_optimizer_state:
            print("Loading optimizer state from checkpoint.")
            optimizer.load_state_dict(
                checkpoint["optimizer_state"]
            )
        print("Setting up LR scheduler")
        _lambda = lambda epoch: config.optimizer["schedule"]["decay_factor"] ** (
                    checkpoint["pretrain_epochs_trained"] // config.optimizer["schedule"]["decay_lr_every_epochs"])
        scheduler = LambdaLR(optimizer, lr_lambda=_lambda)

        train_dataset = RandomMoleculeDataset(config, pretrain_train_dataset,
                                              batch_size=batch_size,
                                              custom_num_batches=num_batches_per_epoch)
        val_dataset = RandomMoleculeDataset(config, pretrain_val_dataset,
                                            batch_size=batch_size_validation,
                                            custom_num_batches=None,
                                            no_random=True)

        for epoch in range(pretrain_num_epochs):
            print("Training.")
            generated_loggable_dict = train_for_one_epoch(
                epoch, config, network, optimizer, train_dataset
            )
            checkpoint["pretrain_epochs_trained"] += 1
            scheduler.step()
            print(f">> Epoch {checkpoint['pretrain_epochs_trained']}. Avg loss level 0: {generated_loggable_dict['loss_level_zero']},"
                  f" Avg loss level 1: {generated_loggable_dict['loss_level_one']},"
                  f" Avg loss level 2: {generated_loggable_dict['loss_level_two']}")
            logger.log_metrics(generated_loggable_dict, step=epoch)

            print("Validating...")
            torch.cuda.empty_cache()
            with torch.no_grad():
                validation_metrics = train_for_one_epoch(
                    None, config, network, None, val_dataset, is_validation=True
                )
            logger.log_metrics(validation_metrics, step=epoch)
            print(f">> Validation. Avg loss level 0: {validation_metrics['val_loss_level_zero']},"
                  f" Avg loss level 1: {validation_metrics['val_loss_level_one']},"
                  f" Avg loss level 2: {validation_metrics['val_loss_level_two']}")

            # Save model
            checkpoint["model_weights"] = copy.deepcopy(network.get_weights())
            checkpoint["optimizer_state"] = copy.deepcopy(
                dict_to_cpu(optimizer.state_dict())
            )

            save_checkpoint(checkpoint, "last_model.pt", config)

            if validation_metrics["val_full_loss"] < checkpoint["pretrain_best_validation_loss"]:
                print(">> Got new best model.")
                checkpoint["pretrain_best_validation_loss"] = generated_loggable_dict["full_loss"]
                save_checkpoint(checkpoint, "best_model.pt", config)
