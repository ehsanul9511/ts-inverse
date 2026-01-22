import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset

import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment

from ts_inverse.datahandler import get_mean_std_dataloader, ConcatSliceDataset

from ts_inverse.attack_time_series_utils import interpolate
from ts_inverse.models.grad_to_input import ImprovedGradToInputNN, ImprovedGradToInputNN_2
from ts_inverse.utils import seed_worker
from ts_inverse.models import GradToInputNN

from ts_inverse.utils import set_seed, seed_worker
# from .attack_worker import AttackWorker, plot_original_and_dummy_data
from .worker import Worker
from ts_inverse.datahandler import ConcatSliceDataset
from .forecasting_worker import evaluate_model
from ts_inverse.attack_time_series_utils import SMAPELoss

# from .attack_dlg_invg_dia_worker import AttackBaselineWorker
from .attack_worker import apply_sign_transformation, apply_pruning, add_gaussian_noise


class AttackLearningToInvertWorker(Worker):
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.worker_name = "AttackLearningToInvertWorker"

    def _init_attack_worker_process(self, c, d_c, m_c, fam_c, def_c=None):
        final_model_settings = {**m_c, **fam_c}
        self.train_datasets, self.val_datasets, self.test_datasets = self.get_datasets(**d_c)
        self.g = set_seed(c["seed"])
        train_dataloader = DataLoader(
            ConcatSliceDataset(self.train_datasets),
            batch_size=c["batch_size"],
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=self.g,
        )

        mean_std_dataloader = DataLoader(
            ConcatSliceDataset(self.train_datasets),
            batch_size=c["batch_size"],
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=self.g,
        )
        self.inputs_mean, self.inputs_std, self.targets_mean, self.targets_std = get_mean_std_dataloader(
            mean_std_dataloader, c["device"]
        )

        freq_in_day = self.train_datasets[0].freq_in_day
        model_settings = {key: value for key, value in final_model_settings.items() if not key.startswith("_")}
        for key, value in model_settings.items():
            if key.startswith("input_") or key.startswith("output_"):
                model_settings[key] = value * freq_in_day
                final_model_settings[key] = value * freq_in_day

        model = final_model_settings["_model"](**model_settings)  # Create model

        final_model_settings.update(model.extra_info)
        final_config = {**c, **d_c, **final_model_settings}

        print("def_c", def_c)
        if def_c:
            final_config.update({**def_c})

        del final_config["_model"]
        final_config["model"] = model.name
        if "base_num_attack_steps" in final_config:
            final_config["num_attack_steps"] = (
                final_config["base_num_attack_steps"] * final_config["batch_size"] * final_config["_attack_step_multiplier"]
            )
        final_config["train_dataset_size"] = len(train_dataloader.dataset)
        print("Starting attack", final_config["run_number"], "with config:", final_config)
        return model, train_dataloader, final_config

    def init_logger_object(self, config):
        tags = [config["attack_method"]]
        project_names = {
            "wandb": "ts-inverse_preparation_baselines",
        }
        self._init_logger_object(project_names, tags, config)

    def worker_process(self, c, d_c, m_c, fam_c, def_c):
        model, train_dataloader, final_config = self._init_attack_worker_process(c, d_c, m_c, fam_c, def_c)
        self.init_logger_object(final_config)
        self.init_attack(model, train_dataloader, final_config)
        self.start_attack(model, config=final_config)
        self._end_logger_object()

    def init_attack(self, model, tr_dataloader, config):
        # super().init_attack(model, tr_dataloader, config)
        self.all_batch_inputs, self.all_batch_targets, self.all_model_state_dicts, self.all_model_gradients, _ = (
            self.train_model_and_record(model, tr_dataloader, config)
        )

        self.all_dummy_inputs, self.all_dummy_targets = self.generate_dummy_data(
            self.all_batch_inputs[0], self.all_batch_targets[0], config
        )

        self.inputs_mean, self.inputs_std = self.inputs_mean[model.features], self.inputs_std[model.features]
        self.targets_mean, self.targets_std = self.targets_mean[0], self.targets_std[0]
        config["model_size"] = sum(p.numel() for p in model.parameters())

        if "aux_dataset" in config and config["aux_dataset"] is not None:
            aux_dataset_config = config["aux_dataset"]
            aux_trainset_path = f"../data/_aux_datasets/train_{aux_dataset_config['dataset']}_{len(aux_dataset_config['columns'])}_{aux_dataset_config['train_stride']}_{aux_dataset_config['observation_days']}_{aux_dataset_config['future_days']}_{aux_dataset_config['normalize']}.pt"
            aux_valset_path = f"../data/_aux_datasets/val_{aux_dataset_config['dataset']}_{len(aux_dataset_config['columns'])}_{aux_dataset_config['train_stride']}_{aux_dataset_config['observation_days']}_{aux_dataset_config['future_days']}_{aux_dataset_config['normalize']}.pt"
            if os.path.exists(aux_trainset_path) and os.path.exists(aux_valset_path):
                aux_trainset = torch.load(aux_trainset_path)
                aux_valset = torch.load(aux_valset_path)
            else:
                train_sets, val_sets, test_sets = self.get_datasets(**aux_dataset_config, split_ratio=0.05)
                aux_trainset = ConcatDataset(train_sets)
                aux_valset = ConcatDataset(val_sets)
                torch.save(aux_trainset, aux_trainset_path)
                torch.save(aux_valset, aux_valset_path)
        else:
            aux_trainset = ConcatDataset(self.test_datasets)
            aux_valset = ConcatDataset(self.val_datasets)

        # Prior knowledge datasets
        self.auxiliary_train_dataloader = DataLoader(
            aux_trainset, batch_size=1, shuffle=False, worker_init_fn=seed_worker, generator=self.g
        )
        self.auxiliary_val_dataloader = DataLoader(
            aux_valset, batch_size=1, shuffle=False, worker_init_fn=seed_worker, generator=self.g
        )
        config["auxiliary_train_dataset_size"] = len(self.auxiliary_train_dataloader.dataset)
        config["auxiliary_val_dataset_size"] = len(self.auxiliary_val_dataloader.dataset)

        assert (
            config["batch_size"] % config["inversion_batch_size"] == 0
        ), "Batch size should be divisible by inversion batch size"

        self._update_config(config)

        if config["verbose"]:
            print("Loaded auxiliary dataset with", len(self.aux_gi_t_dataset), "samples")
            print("Sample size:", self.aux_gi_t_dataset[0][0].shape, self.aux_gi_t_dataset[0][1].shape)
            print("Model / gradient size:", config["model_size"])
            print("Length of dataloader:", len(self.aux_gi_t_dataloader))

    def start_attack(self, model, config):
        for batch_number in range(config["attack_number_of_batches"]):
            model.load_state_dict(self.all_model_state_dicts[batch_number])
            original_dy_dx = self.all_model_gradients[batch_number]
            dummy_inputs = self.all_dummy_inputs[batch_number]
            dummy_targets = self.all_dummy_targets[batch_number]
            batch_inputs = self.all_batch_inputs[batch_number]
            batch_targets = self.all_batch_targets[batch_number]

            self.attack_batch(
                model, config, batch_number, original_dy_dx, dummy_inputs, dummy_targets, batch_inputs, batch_targets
            )

        if config["device"] != "cpu":
            torch.cuda.empty_cache()

        model.eval()

    def attack_batch(self, model, config, batch_number, original_dy_dx, dummy_inputs, dummy_targets, batch_inputs, batch_targets):
        if "num_learn_epochs" in config and config["num_learn_epochs"] <= 0:
            return None

        _, aux_gi_t_dataloader = create_gradient_inversion_dataloader(
            self.auxiliary_train_dataloader, model, config, batch_number, dummy_inputs, dummy_targets, seed_generator=self.g
        )
        _, aux_gi_v_dataloader = create_gradient_inversion_dataloader(
            self.auxiliary_val_dataloader, model, config, batch_number, dummy_inputs, dummy_targets
        )

        inversion_model, _, _ = self.initialize_inversion_model(config, dummy_inputs, dummy_targets)

        grad_to_input_optimizer, lr_schedular = self.set_attack_optimizer_and_schedular(
            inversion_model.parameters(),
            config["learn_optimizer"],
            config["learn_learning_rate"],
            config["learn_lr_decay"],
            config["num_learn_epochs"],
        )

        def generate_model_path(config, folder_path, batch_number, model, aux_gi_t_dataloader):
            keys = [
                "inversion_model",
                "defense_name",
                "attack_batch_size",
                "attack_hidden_size",
                "model_size",
                "learn_optimizer",
                "learn_learning_rate",
                "learn_lr_decay",
                "num_learn_epochs",
                "attack_loss",
                "dataset",
                "input_size",
                "output_size",
                "attack_targets",
                "seed",
                "batch_size",
                "inversion_batch_size",
                "quantiles",
            ]

            path_components = [
                folder_path,
                "grad_inputs_targets_model",
                str(batch_number),
                *(
                    str("-".join(map(str, config[key])) if isinstance(config[key], list) else config[key])
                    for key in keys
                    if key in config and config[key]
                ),
                model.name,
                "-".join(map(str, model.features)),
                str(len(aux_gi_t_dataloader.dataset)),
            ]

            return "_".join(path_components) + ".pt"

        folder_path = "../data/_model_dataset_gradients/"
        model_path = generate_model_path(config, folder_path, batch_number, model, aux_gi_t_dataloader)
        if os.path.exists(model_path) and config["load_lti_model"]:
            inversion_model.load_state_dict(torch.load(model_path))
            print("Loaded inversion model to file.")
        else:
            for epoch in range(0, config["num_learn_epochs"] + 1):
                epoch_t_loss = self.inversion_model_epoch(
                    config, epoch, aux_gi_t_dataloader, inversion_model, grad_to_input_optimizer, lr_schedular
                )
                epoch_v_loss = self.inversion_model_epoch(config, epoch, aux_gi_v_dataloader, inversion_model)

                attack_metrics = {"epoch": epoch, "aux_t_loss": np.mean(epoch_t_loss), "aux_v_loss": np.mean(epoch_v_loss)}

                self.schedular_step(config["learn_lr_decay"], lr_schedular, attack_metrics, np.mean(epoch_v_loss))

                self.evaluate_dummy_prediction(
                    config,
                    batch_number,
                    original_dy_dx,
                    batch_inputs,
                    batch_targets,
                    dummy_inputs,
                    dummy_targets,
                    inversion_model,
                    epoch,
                    attack_metrics,
                )
            torch.save(inversion_model.state_dict(), model_path)
            print("Saved inversion model to file.")

        return inversion_model

    def evaluate_dummy_prediction(
        self,
        config,
        batch_number,
        original_dy_dx,
        batch_inputs,
        batch_targets,
        dummy_inputs,
        dummy_targets,
        inversion_model,
        epoch,
        attack_metrics,
    ):
        inversion_model.eval()
        with torch.no_grad():
            flattened_original_dy_dx = torch.cat([g.view(-1) for g in original_dy_dx]).to(config["device"])
            dummy_inputs, predicted_dummy_targets = inversion_model.inference(flattened_original_dy_dx.unsqueeze(0))
            # repeat dummy input along first dim until it matches batch_inputs
            dummy_inputs = dummy_inputs.repeat(batch_inputs.size(0) // dummy_inputs.size(1), 1, 1, 1)
            dummy_inputs = dummy_inputs.view(batch_inputs.size())
            if predicted_dummy_targets is not None:
                predicted_dummy_targets = predicted_dummy_targets.repeat(
                    batch_targets.size(0) // predicted_dummy_targets.size(1), 1, 1
                )
                dummy_targets = predicted_dummy_targets.view(batch_targets.size())
            self.evaluate_and_log_reconstruction(
                config,
                batch_inputs,
                batch_targets,
                dummy_inputs,
                dummy_targets,
                batch_number,
                epoch,
                config["num_learn_epochs"],
                attack_metrics,
                log_plots_n_times=10,
            )

    def initialize_inversion_model(self, config, batch_inputs, batch_targets):
        data_observations_shape = torch.Size([config["inversion_batch_size"]] + list(batch_inputs.shape[1:]))
        data_targets_shape = None
        if config["attack_targets"]:
            data_targets_shape = torch.Size([config["inversion_batch_size"]] + list(batch_targets.shape[1:]))

        inversion_model = None
        if config["inversion_model"] == "GradToInputNN":
            inversion_model = GradToInputNN(
                config["attack_hidden_size"], config["model_size"], data_observations_shape, data_targets_shape
            ).to(config["device"])
        if config["inversion_model"] == "ImprovedGradToInputNN":
            inversion_model = ImprovedGradToInputNN(
                config["attack_hidden_size"], config["model_size"], data_observations_shape, data_targets_shape
            ).to(config["device"])
        if config["inversion_model"] == "ImprovedGradToInputNN_2":
            inversion_model = ImprovedGradToInputNN_2(
                config["attack_hidden_size"], config["model_size"], data_observations_shape, data_targets_shape
            ).to(config["device"])
        return inversion_model, data_observations_shape, data_targets_shape

    def calculate_inversion_model_loss(
        self, inversion_model, config, a_batch_size, predicted_inputs, predicted_targets, aux_inputs, aux_targets
    ):
        # Flatten the inputs and targets to calculate the loss
        loss = torch.tensor(0.0).to(config["device"])
        if config["attack_loss"] == "mse":
            # View the predicted inputs and auxiliary inputs as flat vectors
            predicted_inputs = predicted_inputs.view(a_batch_size, config["inversion_batch_size"], -1)
            predicted_inputs = predicted_inputs.repeat(1, config["batch_size"] // config["inversion_batch_size"], 1)
            aux_inputs = aux_inputs.view(a_batch_size, config["batch_size"], -1)

            # Check if targets are provided and concatenate them with inputs if they are
            if predicted_targets is not None:
                predicted_targets = predicted_targets.view(a_batch_size, config["inversion_batch_size"], -1)
                predicted_targets = predicted_targets.repeat(1, config["batch_size"] // config["inversion_batch_size"], 1)

                aux_targets = aux_targets.view(a_batch_size, config["batch_size"], -1)

                # Concatenate inputs with targets along the last dimension
                predicted_combined = torch.cat((predicted_inputs, predicted_targets), dim=-1)
                aux_combined = torch.cat((aux_inputs, aux_targets), dim=-1)
            else:
                # Use only inputs if no targets are provided
                predicted_combined = predicted_inputs
                aux_combined = aux_inputs

            # Calculate pairwise squared Euclidean distances between combined vectors
            batch_wise_combined_loss = (torch.cdist(predicted_combined, aux_combined) ** 2) / predicted_combined.size(-1)

            # Solve the optimal assignment problem for the combined loss matrix
            for combined_loss_matrix in batch_wise_combined_loss:
                row_ind, col_ind = linear_sum_assignment(combined_loss_matrix.detach().cpu().numpy())
                loss += combined_loss_matrix[row_ind, col_ind].mean()

            # Normalize the loss by the batch size
            loss /= a_batch_size

        return loss

    def inversion_model_epoch(
        self, config, epoch, aux_gi_dataloader, inversion_model, grad_to_input_optimizer=None, lr_schedular=None
    ):
        epoch_loss = []
        for i, (aux_grads, aux_inputs, aux_targets) in enumerate(aux_gi_dataloader):
            if grad_to_input_optimizer is not None:
                grad_to_input_optimizer.zero_grad()
                inversion_model.train()
            else:
                inversion_model.eval()

            aux_grads, aux_inputs, aux_targets = (
                aux_grads.to(config["device"]),
                aux_inputs.to(config["device"]),
                aux_targets.to(config["device"]),
            )
            batch_size = int(aux_grads.size(0) / config["batch_size"])
            batch_num = batch_size * config["batch_size"]
            if batch_num != aux_grads.size(0):
                continue

            # print(batch_size, batch_num, aux_grads.shape, aux_inputs.shape, aux_targets.shape)
            aux_grads, aux_inputs, aux_targets = aux_grads[:batch_num], aux_inputs[:batch_num], aux_targets[:batch_num]
            aux_grads = aux_grads.view(batch_size, config["batch_size"], aux_grads.shape[-1]).mean(
                1
            )  # gradients are always averaged over batch size
            aux_inputs = aux_inputs.view(batch_size, config["batch_size"], *aux_inputs.shape[-2:])
            aux_targets = aux_targets.view(batch_size, config["batch_size"], *aux_targets.shape[-1:])  # Only 1 feature

            if aux_inputs.min() < 0 or aux_inputs.max() > 1:
                print("Aux inputs out of range:", aux_inputs.min(), aux_inputs.max())
            if aux_targets.min() < 0 or aux_targets.max() > 1:
                print("Aux targets out of range:", aux_targets.min(), aux_targets.max())

            predicted_inputs, predicted_targets = inversion_model(aux_grads)

            loss = self.calculate_inversion_model_loss(
                inversion_model, config, batch_size, predicted_inputs, predicted_targets, aux_inputs, aux_targets
            )

            if grad_to_input_optimizer is not None:
                loss.backward()
                grad_to_input_optimizer.step()

            epoch_loss.append(loss.detach().item())
            if config["verbose"]:
                if grad_to_input_optimizer is None:
                    print(f'\rEpoch {epoch}/{config["num_attack_steps"]}: Val Loss: {round(np.mean(epoch_loss), 5)}', end="")
                else:
                    print(
                        f'\rEpoch {epoch}/{config["num_attack_steps"]}: Train Loss: {round(np.mean(epoch_loss), 5)} lr: {lr_schedular.get_last_lr()}',
                        end="",
                    )
        if config["verbose"]:
            print()

        return epoch_loss

    def schedular_step(self, config_lr_decay, dummy_schedular, attack_metrics, dy_dx_loss):
        if dummy_schedular is not None:
            if "on_plateau" in config_lr_decay:
                dummy_schedular.step(dy_dx_loss)
            else:
                dummy_schedular.step()
            for i, lr in enumerate(dummy_schedular.get_last_lr()):
                attack_metrics[f"learning_rates/lr_{i}"] = lr

    def set_attack_optimizer_and_schedular(self, values, optimizer, lr, lr_decay, num_attack_steps):
        dummy_optimizer = None
        if optimizer == "adamW":
            dummy_optimizer = torch.optim.AdamW(values, lr=lr)
        elif optimizer == "adam":
            dummy_optimizer = torch.optim.Adam(values, lr=lr)
        elif optimizer == "lbfgs":
            dummy_optimizer = torch.optim.LBFGS(values, lr=lr)

        dummy_schedular = None
        if lr_decay == "multi_step":
            # https://github.com/JonasGeiping/invertinggradients/blob/master/inversefed/reconstruction_algorithms.py
            dummy_schedular = lr_scheduler.MultiStepLR(
                dummy_optimizer,
                milestones=[num_attack_steps // 2.667, num_attack_steps // 1.6, num_attack_steps // 1.142],
                gamma=0.1,
            )  # 3/8 5/8 7/8
        elif lr_decay == "75%":
            dummy_schedular = torch.optim.lr_scheduler.MultiStepLR(
                dummy_optimizer, milestones=[int(0.75 * num_attack_steps)], gamma=0.1
            )

        elif "on_plateau" in lr_decay:
            splitted_lr_decay = lr_decay.split("_")
            if len(splitted_lr_decay) == 3:
                dummy_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    dummy_optimizer, mode="min", factor=0.1, patience=num_attack_steps // int(splitted_lr_decay[2])
                )
            else:
                dummy_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    dummy_optimizer, mode="min", factor=0.1, patience=num_attack_steps // 10
                )

        return dummy_optimizer, dummy_schedular

    def generate_dummy_data(self, batch_input_example, batch_target_example, config):
        all_dummy_inputs, all_dummy_targets = [], []
        attack_number_of_batches = (
            config["attack_number_of_batches"] if "attack_number_of_batches" in config else config["number_of_batches"]
        )

        if "dummy_init_method" not in config or config["dummy_init_method"] == "rand":
            all_dummy_inputs = [
                torch.rand_like(batch_input_example, device=config["device"], requires_grad=True)
                for _ in range(attack_number_of_batches)
            ]
            all_dummy_targets = [
                torch.rand_like(batch_target_example, device=config["device"], requires_grad=True)
                for _ in range(attack_number_of_batches)
            ]
        elif config["dummy_init_method"] == "halves":
            all_dummy_inputs = [
                (torch.tensor(0.5) * torch.ones_like(batch_input_example, device=config["device"])).requires_grad_(True)
                for _ in range(attack_number_of_batches)
            ]
            all_dummy_targets = [
                (torch.tensor(0.5) * torch.ones_like(batch_target_example, device=config["device"])).requires_grad_(True)
                for _ in range(attack_number_of_batches)
            ]
        elif config["dummy_init_method"] == "small_randn":
            mean = torch.tensor(0.5)
            std_dev = torch.tensor(0.1)  # Adjust the standard deviation as needed to control the noise level
            all_dummy_inputs = [
                (mean + std_dev * torch.randn_like(batch_input_example, device=config["device"])).requires_grad_(True)
                for _ in range(attack_number_of_batches)
            ]
            all_dummy_targets = [
                (mean + std_dev * torch.randn_like(batch_target_example, device=config["device"])).requires_grad_(True)
                for _ in range(attack_number_of_batches)
            ]
        elif config["dummy_init_method"] == "rand_flat":
            batch_size = batch_input_example.size(0)
            input_shape = batch_input_example.shape[1:]
            target_shape = batch_target_example.shape[1:]
            all_dummy_inputs = [
                (torch.rand(batch_size, *([1] * len(input_shape)), device=config["device"]).expand(batch_size, *input_shape))
                .clone()
                .requires_grad_(True)
                for _ in range(attack_number_of_batches)
            ]

            all_dummy_targets = [
                (torch.rand(batch_size, *([1] * len(target_shape)), device=config["device"]).expand(batch_size, *target_shape))
                .clone()
                .requires_grad_(True)
                for _ in range(attack_number_of_batches)
            ]
        else:
            raise NotImplementedError("Dummy init method not found.")

        return all_dummy_inputs, all_dummy_targets

    def train_model_and_record(self, model, tr_dataloader, config):
        model.to(config["device"])
        model_optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        all_batch_inputs, all_batch_targets, all_model_state_dicts, all_model_gradients, all_model_updates = [], [], [], [], []

        total_batches_processed = 0  # Initialize total batch counter
        total_batches_needed = config["warmup_number_of_batches"] + config["number_of_batches"]
        val_dataset = ConcatSliceDataset(self.val_datasets)
        while total_batches_processed < total_batches_needed:
            for batch_inputs, batch_targets in tr_dataloader:
                if total_batches_processed >= total_batches_needed:
                    break  # Exit if we've processed the total required batches
                if not config["update_model"] and total_batches_processed > 0:
                    model.load_state_dict(all_model_state_dicts[0])

                batch_inputs, batch_targets = (
                    batch_inputs[:, :, model.features].to(config["device"]),
                    batch_targets[:, :, 0].to(config["device"]),
                )
                if total_batches_processed >= config["warmup_number_of_batches"]:
                    all_batch_inputs.append(batch_inputs.clone())
                    all_batch_targets.append(batch_targets.clone())
                model_optimizer.zero_grad()
                if total_batches_processed >= config["warmup_number_of_batches"]:
                    all_model_state_dicts.append({k: v.clone() for k, v in model.state_dict().items()})
                out = model(batch_inputs)
                y = F.mse_loss(out, batch_targets)
                y.backward()

                if "defense_name" in config:
                    # Apply gradient defenses
                    gradients = [param.grad for param in model.parameters()]
                    if "sign" in config:
                        gradients = apply_sign_transformation(gradients)
                    if "prune_rate" in config:
                        gradients = apply_pruning(gradients, config["prune_rate"])
                    if "dp_epsilon" in config:
                        gradients = add_gaussian_noise(gradients, config["dp_epsilon"])

                    # Update model parameters with modified gradients
                    for param, grad in zip(model.parameters(), gradients):
                        param.grad = grad

                if total_batches_processed >= config["warmup_number_of_batches"]:
                    all_model_gradients.append([param.grad.clone() for param in model.parameters()])

                model_optimizer.step()

                if total_batches_processed >= config["warmup_number_of_batches"]:
                    model_update = [
                        (current - prev).clone()
                        for current, prev in zip(model.state_dict().values(), all_model_state_dicts[-1].values())
                    ]
                    all_model_updates.append(model_update)

                    if "evaluate_trained_model" in config and config["evaluate_trained_model"]:
                        model_predictive_stats = evaluate_model(
                            model=deepcopy(model),
                            dataset=val_dataset,
                            device=config["device"],
                            name="model_predictive_stats",
                        )
                        self._log_metrics(model_predictive_stats, step=total_batches_processed + 1)

                total_batches_processed += 1  # Update the total number of batches processed

        return all_batch_inputs, all_batch_targets, all_model_state_dicts, all_model_gradients, all_model_updates

    def gradient_loss_function(self, dummy_dy_dx, original_dy_dx, gradient_loss):
        dy_dx_loss = torch.zeros(1, device=dummy_dy_dx[0].device)
        if gradient_loss == "log_cosh":

            def _log_cosh(x: torch.Tensor) -> torch.Tensor:
                return x + torch.nn.functional.softplus(-2.0 * x) - torch.log(torch.tensor(2.0))

            for d_g, o_g in zip(dummy_dy_dx, original_dy_dx):
                dy_dx_loss += _log_cosh(d_g - o_g).sum()

        if gradient_loss == "l1_skip_1D":
            for d_g, o_g in zip(dummy_dy_dx, original_dy_dx):
                if len(d_g.shape) == 1:
                    continue
                dy_dx_loss += F.l1_loss(d_g, o_g, reduction="sum")

        if gradient_loss == "top20percent_l1":
            for d_g, o_g in zip(dummy_dy_dx, original_dy_dx):
                d_g_flat = d_g.view(-1)
                o_g_flat = o_g.view(-1)
                k = int(len(o_g_flat) * 0.2)
                _, top_k_indices = torch.topk(o_g_flat.abs(), k, largest=True)
                dy_dx_loss += F.l1_loss(d_g_flat[top_k_indices], o_g_flat[top_k_indices], reduction="sum")

        if gradient_loss == "last_2_layers_l1":
            for d_g, o_g in zip(dummy_dy_dx[-2:], original_dy_dx[-2:]):
                dy_dx_loss += F.l1_loss(d_g, o_g, reduction="sum")
        if gradient_loss == "double_outer_l1":
            for i, (d_g, o_g) in enumerate(zip(dummy_dy_dx, original_dy_dx)):
                bounds = len(dummy_dy_dx) // 4
                weight = 1
                if i < bounds or i > len(dummy_dy_dx) - bounds:
                    weight = 2
                dy_dx_loss += F.l1_loss(d_g, o_g, reduction="sum") * weight

        if gradient_loss == "l1":
            for d_g, o_g in zip(dummy_dy_dx, original_dy_dx):
                dy_dx_loss += F.l1_loss(d_g, o_g, reduction="sum")

        if gradient_loss == "euclidean":
            dy_dx_loss += sum(F.mse_loss(d_g, o_g, reduction="sum") for d_g, o_g in zip(dummy_dy_dx, original_dy_dx))

        def cosine_invg():
            pnorm = [0, 0]
            costs = 0
            for d_g, o_g in zip(dummy_dy_dx, original_dy_dx):
                costs -= (d_g * o_g).sum()
                pnorm[0] += d_g.pow(2).sum()
                pnorm[1] += o_g.pow(2).sum()
            return 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        if gradient_loss == "cosine_invg":
            # # https://github.com/JonasGeiping/invertinggradients/blob/master/inversefed/reconstruction_algorithms.py#L325
            dy_dx_loss += cosine_invg()

        def cosine_dia():
            total_loss, dummy_norm, orig_norm = 0, 0, 0
            for d_g, o_g in zip(dummy_dy_dx, original_dy_dx):
                partial_loss = (d_g * o_g).sum()
                partial_d_norm = d_g.pow(2).sum()
                partial_o_norm = o_g.pow(2).sum()

                total_loss += partial_loss
                dummy_norm += partial_d_norm
                orig_norm += partial_o_norm
            return 1 - total_loss / (dummy_norm.sqrt() * orig_norm.sqrt() + 1e-16)

        if gradient_loss == "cosine_dia":
            # https://github.com/dAI-SY-Group/DropoutInversionAttack/blob/main/src/loss.py#L14
            dy_dx_loss += cosine_dia()

        if "l1norm" in gradient_loss and "cosine" in gradient_loss:
            # 1_l1_0.005_cosine
            splitted = gradient_loss.split("_")
            l1_weight = float(splitted[0])
            cosine_weight = float(splitted[2])
            dy_dx_loss += l1_weight * sum((d_g - o_g).norm(p=1) for d_g, o_g in zip(dummy_dy_dx, original_dy_dx))
            dy_dx_loss += cosine_weight * cosine_dia()

        if "inorm" in gradient_loss and "icosine" in gradient_loss:
            # 1_norm_0.005_cosine
            splitted = gradient_loss.split("_")
            norm_weight = float(splitted[0])
            cosine_weight = float(splitted[2])
            for i, (d_g, o_g) in enumerate(zip(dummy_dy_dx, original_dy_dx)):
                if len(d_g.shape) == 1:
                    continue

                layer_loss = 0
                layer_loss += cosine_weight * (1 - (d_g * o_g).sum() / (d_g.norm() * o_g.norm() + 1e-16))
                layer_loss += norm_weight * (d_g - o_g).norm()
                dy_dx_loss += layer_loss

        elif "l2norm" in gradient_loss and "cosine" in gradient_loss:
            splitted = gradient_loss.split("_")
            norm_weight = float(splitted[0])
            cosine_weight = float(splitted[2])
            dy_dx_loss += norm_weight * sum((d_g - o_g).norm(p=2) for d_g, o_g in zip(dummy_dy_dx, original_dy_dx))
            dy_dx_loss += cosine_weight * cosine_dia()

        return dy_dx_loss

    def evaluate_and_log_reconstruction(
        self,
        config,
        batch_inputs,
        batch_targets,
        dummy_inputs,
        dummy_targets,
        batch_number,
        attack_step,
        num_attack_steps,
        attack_metrics,
        attack_step_offset=0,
        log_plots_n_times=10,
    ):
        with torch.no_grad():
            # sample_mapping = self.get_batch_sample_mapping(batch_inputs, dummy_inputs)
            standard_mapping = np.arange(0, batch_inputs.shape[0])
            input_sample_mapping = get_batch_sample_mapping(batch_inputs, dummy_inputs)
            target_sample_mapping = get_batch_sample_mapping(batch_targets, dummy_targets)

            sample_mapping = np.arange(0, batch_inputs.shape[0])
            if not (standard_mapping == input_sample_mapping).all():
                sample_mapping = input_sample_mapping
            if (standard_mapping == input_sample_mapping).all() and not (standard_mapping == target_sample_mapping).all():
                sample_mapping = target_sample_mapping
            # if not (standard_mapping == input_sample_mapping).all() and not (standard_mapping == target_sample_mapping).all():
            #     if not (input_sample_mapping == target_sample_mapping).all():
            #         raise ValueError('Input and target sample mappings are not equal while being different from the standard mapping.')

            mean_evaluation = {
                "inputs/mse/mean": F.mse_loss(dummy_inputs[sample_mapping], batch_inputs).item(),
                "targets/mse/mean": F.mse_loss(dummy_targets[sample_mapping], batch_targets).item(),
                "inputs/rmse/mean": torch.sqrt(F.mse_loss(dummy_inputs[sample_mapping], batch_inputs)).item(),
                "targets/rmse/mean": torch.sqrt(F.mse_loss(dummy_targets[sample_mapping], batch_targets)).item(),
                "inputs/mae/mean": F.l1_loss(dummy_inputs[sample_mapping], batch_inputs).item(),
                "targets/mae/mean": F.l1_loss(dummy_targets[sample_mapping], batch_targets).item(),
                "inputs/smape/mean": SMAPELoss(dummy_inputs[sample_mapping], batch_inputs).item(),
                "targets/smape/mean": SMAPELoss(dummy_targets[sample_mapping], batch_targets).item(),
            }
            attack_metrics.update(mean_evaluation)

            # individual batch sample metrics
            for i, j in enumerate(sample_mapping):
                individual_evaluation = {
                    f"inputs/mse/{i}": F.mse_loss(dummy_inputs[j], batch_inputs[i]).item(),
                    f"targets/mse/{i}": F.mse_loss(dummy_targets[j], batch_targets[i]).item(),
                    f"inputs/rmse/{i}": torch.sqrt(F.mse_loss(dummy_inputs[j], batch_inputs[i])).item(),
                    f"targets/rmse/{i}": torch.sqrt(F.mse_loss(dummy_targets[j], batch_targets[i])).item(),
                    f"inputs/mae/{i}": F.l1_loss(dummy_inputs[j], batch_inputs[i]).item(),
                    f"targets/mae/{i}": F.l1_loss(dummy_targets[j], batch_targets[i]).item(),
                    f"inputs/smape/{i}": SMAPELoss(dummy_inputs[j], batch_inputs[i]).item(),
                    f"targets/smape/{i}": SMAPELoss(dummy_targets[j], batch_targets[i]).item(),
                }
                attack_metrics.update(individual_evaluation)

            if attack_step % (num_attack_steps // log_plots_n_times) == 0:
                df, fig = plot_original_and_dummy_data(
                    config, sample_mapping, dummy_inputs, dummy_targets, batch_inputs, batch_targets
                )
                self._log_dataframe(df, attack_step + attack_step_offset, log_name=f"_batch_{batch_number}")
                self._log_matplotlib_figure(fig, step=attack_step + attack_step_offset, log_name=f"_batch_{batch_number}")
            self._log_metrics(attack_metrics, step=attack_step + attack_step_offset)

    def plot_gradients(self, dummy_dy_dx, original_dy_dx, config):
        """Plot all gradients along the x-axis as bars. The goal is to show the values"""
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        dummy_grad_list = []
        original_grad_list = []
        split_indexes = []
        for i, grad in enumerate(dummy_dy_dx):
            dummy_grad_list.extend(grad.cpu().detach().flatten().numpy().tolist())
            split_indexes += [len(dummy_grad_list)]
        for i, grad in enumerate(original_dy_dx):
            original_grad_list.extend(grad.cpu().detach().flatten().numpy().tolist())

        grad_diff = np.abs(np.array(dummy_grad_list) - np.array(original_grad_list))
        axes[0].plot(dummy_grad_list)
        axes[0].set_title("Dummy Gradients")
        axes[0].set_xlabel("Parameter Index")
        axes[0].set_ylabel("Gradient Value")

        axes[1].plot(original_grad_list)
        axes[1].set_title("Dummy Gradients")
        axes[1].set_xlabel("Parameter Index")
        axes[1].set_ylabel("Gradient Value")

        axes[2].plot(grad_diff, label="Gradient Difference")
        axes[2].set_title(config["gradient_loss"])
        axes[2].set_xlabel("Parameter Index")
        axes[2].set_ylabel("Gradient Difference Value")

        for split_index in split_indexes:
            axes[0].axvline(x=split_index, color="r", linestyle="--", linewidth=0.8)
            axes[1].axvline(x=split_index, color="r", linestyle="--", linewidth=0.8)
            axes[2].axvline(x=split_index, color="r", linestyle="--", linewidth=0.8)

        grad_dict = {"dummy_grad_list": dummy_grad_list, "original_grad_list": original_grad_list, "split_indexes": split_indexes}

        if config["verbose"]:
            fig.tight_layout()
            plt.show()
        else:
            plt.close(fig)

        return grad_dict, fig


    def after_effect(self, config, model, dummy_inputs, dummy_targets, attack_step):
        with torch.no_grad():
            if "TCN" in model.name and config["optimize_dropout"] and config["clamp_dropout"] > 0:
                if attack_step % config["clamp_dropout"] == 0:
                    model.clamp_dropout_layers(*config["clamp_dropout_min_max"])

            if "boxed" in config["after_effect"]:
                dummy_inputs.data = torch.max(
                    torch.min(dummy_inputs, (1 - self.inputs_mean) / self.inputs_std), -self.inputs_mean / self.inputs_std
                )
                dummy_targets.data = torch.max(
                    torch.min(dummy_targets, (1 - self.targets_mean) / self.targets_std), -self.targets_mean / self.targets_std
                )
            if "clamp_1" in config["after_effect"]:
                dummy_inputs.data = torch.clamp(dummy_inputs, 0, 1)
                dummy_targets.data = torch.clamp(dummy_targets, 0, 1)
            if "clamp_2" in config["after_effect"]:
                if attack_step % 2 == 0:
                    dummy_inputs.data = torch.clamp(dummy_inputs, 0, 1)
                    dummy_targets.data = torch.clamp(dummy_targets, 0, 1)
            if "clamp_min0_2" in config["after_effect"]:
                if attack_step % 2 == 0:
                    dummy_inputs.data = torch.clamp(dummy_inputs, 0)
                    dummy_targets.data = torch.clamp(dummy_targets, 0)


def plot_original_and_dummy_data(config, sample_mapping, dummy_inputs, dummy_targets, batch_inputs, batch_targets):
    batch_size = batch_inputs.shape[0]
    original_x_axis = np.arange(0, batch_inputs.shape[1] + batch_targets.shape[1])
    # warped_dummy_x_axis = np.linspace(0, original_x_axis[-1], dummy_inputs.shape[1]+dummy_targets.shape[1])
    # x_axis = np.arange(0, dummy_inputs.shape[1]+dummy_targets.shape[1])
    fig, axes = plt.subplots(nrows=batch_size, figsize=(20, 3.5 * batch_size), sharex=True)
    for i, j in enumerate(sample_mapping):
        ax = axes[i] if batch_size > 1 else axes
        ax.plot(original_x_axis[: batch_inputs.shape[1]], batch_inputs[i, :].detach().cpu().numpy(), label=f"Dataset Input ({i})")
        ax.plot(
            original_x_axis[: batch_inputs.shape[1]],
            dummy_inputs[j, :].detach().cpu().numpy(),
            label=f"Gradient Recovered Input ({j})",
        )

        ax.plot(
            original_x_axis[batch_inputs.shape[1] :], batch_targets[i, :].detach().cpu().numpy(), label=f"Dataset Target ({i})"
        )
        ax.plot(
            original_x_axis[batch_inputs.shape[1] :],
            dummy_targets[j, :].detach().cpu().numpy(),
            label=f"Gradient Recovered Target ({j})",
        )

        if i == batch_size - 1:
            ax.set_xlabel("Time")
        ax.set_ylabel("Output")
        ax.set_title(
            f'Dataset {config["dataset"]}: Comparison Inputs & Targets & of Dataset & Recovery - Batch Sample {i} & Dummy Sample {j}'
        )
        ax.legend()

    # Create weights and biases plot & dataframe
    fill_nan = np.empty(batch_targets.shape[1])
    fill_nan.fill(np.nan)
    if len(batch_inputs.shape) == 3:
        batch_inputs_dict = {
            f"batch_inputs_{i}_{f}": np.concatenate((batch_inputs[i, :, f].detach().cpu().numpy(), fill_nan))
            for i in range(batch_inputs.shape[0])
            for f in range(batch_inputs.shape[2])
        }
        dummy_inputs_dict = {
            f"dummy_inputs_{i}_{f}": np.concatenate((dummy_inputs[i, :, f].detach().cpu().numpy(), fill_nan))
            for i in range(dummy_inputs.shape[0])
            for f in range(dummy_inputs.shape[2])
        }
    elif len(batch_inputs.shape) == 2:
        batch_inputs_dict = {
            f"batch_inputs_{i}": np.concatenate((batch_inputs[i, :].detach().cpu().numpy(), fill_nan))
            for i in range(batch_inputs.shape[0])
        }
        dummy_inputs_dict = {
            f"dummy_inputs_{i}": np.concatenate((dummy_inputs[i, :].detach().cpu().numpy(), fill_nan))
            for i in range(dummy_inputs.shape[0])
        }

    fill_nan = np.empty(batch_inputs.shape[1])
    fill_nan.fill(np.nan)
    batch_targets_dict = {
        f"batch_targets_{i}": np.concatenate((fill_nan, batch_targets[i, :].detach().cpu().numpy()))
        for i in range(batch_targets.shape[0])
    }
    dummy_targets_dict = {
        f"dummy_targets_{i}": np.concatenate((fill_nan, dummy_targets[i, :].detach().cpu().numpy()))
        for i in range(dummy_targets.shape[0])
    }
    input_data = {**batch_inputs_dict, **dummy_inputs_dict, **batch_targets_dict, **dummy_targets_dict}
    df = pd.DataFrame(input_data)

    if config["verbose"]:
        plt.show()
    else:
        plt.close(fig)

    return df, fig


def get_batch_sample_mapping(original_data, dummy_data, function=F.l1_loss):
    batch_size = original_data.shape[0]
    sample_mapping = np.arange(0, batch_size)
    for i in range(batch_size):
        smallest_loss = float("inf")
        for j in range(batch_size):
            loss = (
                function(original_data[i], dummy_data[j]).detach().item()
            )  # instead of MSE because L1 is less sensitive to outliers
            if loss < smallest_loss:
                smallest_loss = loss
                sample_mapping[i] = j

    # Check if sample_mapping contains duplicates; if so, reset it
    if len(np.unique(sample_mapping)) != batch_size:
        sample_mapping = np.arange(0, batch_size)
    return sample_mapping

def total_variation_time_series(x):
    diffs = (x[:, 1:, :] - x[:, :-1, :]).abs().mean()
    return diffs



def create_gradient_inversion_dataloader(
    aux_dataloader, model, config, batch_number, dummy_inputs, dummy_targets, seed_generator=None
):
    model.to(config["device"])

    folder_path = "../data/_model_dataset_gradients/"
    # Path where the dataset will be saved or loaded from
    dataset_path = f"{folder_path}grad_inputs_targets_dataset_{config['defense_name']}_{batch_number}_{model.name}_{'-'.join(map(str, model.features))}_{config['dataset']}_{len(aux_dataloader.dataset)}_{config['input_size']}_{config['output_size']}_{config['seed']}_{config['inversion_batch_size']}.pt"

    # Check if the dataset file exists and load file
    if os.path.exists(dataset_path):
        print("Loaded gradient to inputs targets dataset:", dataset_path)
        grad_inputs_targets_dataset = torch.load(dataset_path, weights_only=False)
        config["loaded_grad_to_inputs_targets_dataset_from_file"] = True
    else:
        # If the file does not exist, proceed with creating the dataset
        config["loaded_grad_to_inputs_targets_dataset_from_file"] = False
        aux_dy_dx_inputs, aux_inputs_targets, aux_targets_targets = [], [], []
        for i, (aux_batch_inputs, aux_batch_targets) in enumerate(aux_dataloader):
            aux_batch_inputs, aux_batch_targets = (
                aux_batch_inputs[:, :, model.features].to(config["device"]),
                aux_batch_targets[:, :, 0].to(config["device"]),
            )

            if aux_batch_inputs.shape[1] != dummy_inputs.shape[1]:
                aux_batch_inputs = interpolate(aux_batch_inputs, dummy_inputs.shape[1])
            if aux_batch_targets.shape[1] != dummy_targets.shape[1]:
                aux_batch_targets = interpolate(aux_batch_targets.unsqueeze(-1), dummy_targets.shape[1]).squeeze(-1)

            model.zero_grad()
            aux_out = model(aux_batch_inputs)
            aux_y = F.mse_loss(aux_out, aux_batch_targets)
            aux_y.backward()

            if "defense_name" in config:
                # Apply gradient defenses
                gradients = [param.grad for param in model.parameters()]
                if "sign" in config:
                    gradients = apply_sign_transformation(gradients)
                if "prune_rate" in config:
                    gradients = apply_pruning(gradients, config["prune_rate"])
                if "dp_epsilon" in config:
                    gradients = add_gaussian_noise(gradients, config["dp_epsilon"])

                # Update model parameters with modified gradients
                for param, grad in zip(model.parameters(), gradients):
                    param.grad = grad

            flattened_aux_dy_dx = torch.cat([p.grad.detach().view(-1) for p in model.parameters()]).unsqueeze(
                0
            )  # add batch dimension but is always 1

            # flattend aux_dy_dx should be of shape (batch_size, model_size)
            aux_dy_dx_inputs.append(flattened_aux_dy_dx.clone().detach().cpu())
            aux_inputs_targets.append(aux_batch_inputs.clone().detach().cpu())
            aux_targets_targets.append(aux_batch_targets.clone().detach().cpu())

        aux_dy_dx_inputs, aux_inputs_targets, aux_targets_targets = (
            torch.stack(aux_dy_dx_inputs),
            torch.stack(aux_inputs_targets),
            torch.stack(aux_targets_targets),
        )
        grad_inputs_targets_dataset = TensorDataset(aux_dy_dx_inputs, aux_inputs_targets, aux_targets_targets)

        # Save the dataset to file
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        torch.save(grad_inputs_targets_dataset, dataset_path)

    if seed_generator is not None:
        return grad_inputs_targets_dataset, DataLoader(
            grad_inputs_targets_dataset,
            batch_size=config["attack_batch_size"] * config["batch_size"],
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=seed_generator,
        )
    return grad_inputs_targets_dataset, DataLoader(
        grad_inputs_targets_dataset, batch_size=config["attack_batch_size"] * config["batch_size"]
    )

def apply_sign_transformation(gradients):
    return [torch.sign(g) for g in gradients]


def apply_pruning(gradients, prune_rate):
    if prune_rate is None:
        return gradients
    pruned_gradients = []
    for grad in gradients:
        mask = torch.zeros(grad.size()).to(grad.device)
        rank = torch.argsort(grad.abs().view(-1))[-int(grad.numel() * (1 - prune_rate)) :]
        mask.view(-1)[rank] = 1
        pruned_gradients.append(grad * mask)
    return pruned_gradients


def add_gaussian_noise(gradients, noise_level):
    noisy_gradients = []
    for grad in gradients:
        noise = torch.randn_like(grad) * noise_level
        noisy_gradients.append(grad + noise)
    return noisy_gradients
