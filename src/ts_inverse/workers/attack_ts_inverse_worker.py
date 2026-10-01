import os
from copy import deepcopy
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset


from matplotlib import pyplot as plt
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ts_inverse.attack_time_series_utils import (
    interpolate,
    periodicity_regularization,
    pinball_loss,
    temporal_resolution_warping,
    trend_consistency_regularization,
    SMAPELoss
)
# from .attack_learning_to_invert_worker import AttackLearningToInvertWorker

from ts_inverse.models.grad_to_input import ImprovedGradToInputNN, ImprovedGradToInputNN_2, ImprovedGradToInputNN_Quantile
from ts_inverse.models import GradToInputNN, ImprovedGradToInputNN_Probabilistic

from ts_inverse.utils import set_seed, seed_worker
# from .attack_worker import AttackWorker, plot_original_and_dummy_data
from .worker import Worker
from ts_inverse.datahandler import (
    ConcatSliceDataset,
    get_mean_std_dataloader,
    get_har_dataset,
    get_motionsense_dataset,
    get_ett_dataset,
)
from .forecasting_worker import evaluate_model

CLASSIFICATION_DATASETS = {
    "realworld",
    "motionsense",
}

ETT_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]

FEATURE_NAMES = [
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "gravity_x", "gravity_y", "gravity_z",
    "rot_rate_x", "rot_rate_y", "rot_rate_z",
]

FEATURE_GROUPS = {
    "acc": [0, 1, 2],
    "gyro": [3, 4, 5],
    "gravity": [6, 7, 8],
    "rot_rate": [9, 10, 11],
}

class AttackTSInverseWorker(Worker):
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.worker_name = "AttackTSInverseWorker"

    def _init_attack_worker_process(self, c, d_c, m_c, fam_c, def_c=None):
        final_model_settings = {**m_c, **fam_c}
        dataset_name = d_c["dataset"]

        if dataset_name == "realworld":
            self.train_datasets, self.val_datasets, self.test_datasets = (
                get_har_dataset()
            )
            print("Loaded RealWorld HAR dataset")

        elif dataset_name == "motionsense":
            self.train_datasets, self.val_datasets, self.test_datasets = (
                get_motionsense_dataset(
                    data_path=d_c["data_path"],
                    seq_len=d_c.get("seq_len", 50),
                    stride=d_c.get("stride", 10),
                    seed=c["seed"],
                )
            )
            print("Loaded MotionSense activity dataset")
        
        elif dataset_name in ETT_DATASETS:
            self.train_datasets, self.val_datasets, self.test_datasets = (
                get_ett_dataset(
                    dataset=dataset_name,
                    root_path=d_c["root_path"],
                    data_path=d_c.get("data_path"),
                    seq_len=d_c.get("seq_len", 96),
                    label_len=d_c.get("label_len", 48),
                    pred_len=d_c.get("pred_len", 96),
                    features=d_c.get("features", "M"),
                    target=d_c.get("target", "OT"),
                    scale=d_c.get("scale", True),
                    timeenc=d_c.get("timeenc", 0),
                    freq=d_c.get("freq", "h"),
                )
            )
            print(f"Loaded ETT dataset")


        else:
            self.train_datasets, self.val_datasets, self.test_datasets = (
                self.get_datasets(**d_c)
            )
        self.g = set_seed(c["seed"])

        # if dataset == "realworld":
        #     train_dataloader = DataLoader(self.train_datasets, shuffle=True, batch_size=c["batch_size"], worker_init_fn=seed_worker, generator=self.g)
        if dataset_name in CLASSIFICATION_DATASETS:
            train_dataloader = DataLoader(
                ConcatDataset(self.train_datasets),
                batch_size=c["batch_size"],
                shuffle=(dataset_name != "motionsense"),
                worker_init_fn=seed_worker,
                generator=self.g,
            )
        else:
            train_dataloader = DataLoader(
                ConcatSliceDataset(self.train_datasets),
                batch_size=c["batch_size"],
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=self.g,
            )

        if dataset_name in CLASSIFICATION_DATASETS:
            mean_std_dataset = ConcatDataset(self.train_datasets)
        else:
            mean_std_dataset = ConcatSliceDataset(self.train_datasets)

        mean_std_dataloader = DataLoader(
            mean_std_dataset,
            batch_size=c["batch_size"],
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=self.g,
        )

        (
            self.inputs_mean,
            self.inputs_std,
            self.targets_mean,
            self.targets_std,
        ) = get_mean_std_dataloader(
            mean_std_dataloader,
            c["device"],
            target_length=self.train_datasets[0].pred_len if dataset_name in ETT_DATASETS else None
        )

        if dataset_name in {"realworld", *ETT_DATASETS}:
            model = final_model_settings["_model"]()

        elif dataset_name == "motionsense":
            model = final_model_settings["_model"](
                num_features=d_c.get("num_features", 12),
                window_size=d_c.get("seq_len", 50),
                num_classes=d_c.get("num_classes", 4),
            )

        else:
            freq_in_day = self.train_datasets[0].freq_in_day

            model_settings = {
                key: value
                for key, value in final_model_settings.items()
                if not key.startswith("_")
            }

            for key, value in model_settings.items():
                if key.startswith("input_") or key.startswith("output_"):
                    model_settings[key] = value * freq_in_day
                    final_model_settings[key] = value * freq_in_day

            model = final_model_settings["_model"](**model_settings)
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

    def worker_process(self, c, d_c, m_c, fam_c, def_c):
        model, train_dataloader, final_config = self._init_attack_worker_process(c, d_c, m_c, fam_c, def_c)
        print("Initialized attack worker process")
        self.init_logger_object(final_config)
        self.init_attack(model, train_dataloader, final_config)
        self.start_attack(model, config=final_config)
        self._end_logger_object()

    def init_logger_object(self, config):
        tags = [config["attack_method"]]
        project_names = {
            "wandb": "ts-inverse_preparation",
        }
        return self._init_logger_object(project_names, tags, config)

    def init_attack(self, model, tr_dataloader, config):
        # super().init_attack(model, tr_dataloader, config)
        self.all_batch_inputs, self.all_batch_targets, self.all_model_state_dicts, self.all_model_gradients, _ = (
            self.train_model_and_record(model, tr_dataloader, config)
        )

        if config["verbose"]:
            print(
                f"Recorded {len(self.all_batch_inputs)} gradient batches for attack."
            )
            print(
                f"Batch input shape: {self.all_batch_inputs[0].shape}, "
                f"Batch target shape: {self.all_batch_targets[0].shape}"
            )
            print(
                f"Model state dict keys: {list(self.all_model_state_dicts[0].keys())}"
            )
            print(
                f"Model gradient shapes: {[g.shape for g in self.all_model_gradients[0]]}"
            )


        number_of_attacks = config["attack_number_of_batches"]

        if number_of_attacks > len(self.all_batch_inputs):
            raise ValueError(
                f"Requested {number_of_attacks} attacks, but only "
                f"{len(self.all_batch_inputs)} gradient batches were recorded. "
                "Set number_of_batches >= attack_number_of_batches."
            )

        self.all_dummy_inputs = []
        self.all_dummy_targets = []

        for batch_number in range(number_of_attacks):
            single_batch_config = dict(config)
            single_batch_config["attack_number_of_batches"] = 1

            dummy_inputs, dummy_targets = self.generate_dummy_data(
                self.all_batch_inputs[batch_number],
                self.all_batch_targets[batch_number],
                single_batch_config,
            )

            self.all_dummy_inputs.append(dummy_inputs[0])
            self.all_dummy_targets.append(dummy_targets[0])

        dataset_name = config["dataset"]

        if dataset_name in ETT_DATASETS:
            # ETT uses all input channels.
            # For M/S, retain all target statistics already calculated.
            if config.get("features", "M") == "MS":
                self.targets_mean = self.targets_mean[-1:]
                self.targets_std = self.targets_std[-1:]

        elif dataset_name not in CLASSIFICATION_DATASETS:
            self.inputs_mean, self.inputs_std = (
                self.inputs_mean[model.features],
                self.inputs_std[model.features],
            )
            self.targets_mean, self.targets_std = (
                self.targets_mean[0],
                self.targets_std[0],
            )
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
        if dataset_name in CLASSIFICATION_DATASETS:
            # Build one auxiliary gradient per sample.  inversion_model_epoch
            # later groups these records to reproduce the attacked batch size
            # before averaging their gradients.
            self.auxiliary_train_dataloader = DataLoader(
                aux_trainset,
                batch_size=1,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=self.g,
            )
            self.auxiliary_val_dataloader = DataLoader(
                aux_valset,
                batch_size=1,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=self.g,
            )
        else:
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

        if config["verbose"] and config["aux_dataset"] is not None:
            print("Loaded auxiliary dataset with", len(self.aux_gi_t_dataset), "samples")
            print("Sample size:", self.aux_gi_t_dataset[0][0].shape, self.aux_gi_t_dataset[0][1].shape)
            print("Model / gradient size:", config["model_size"])
            print("Length of dataloader:", len(self.aux_gi_t_dataloader))

    def start_attack(self, model, config):
        number_of_attacks = config["attack_number_of_batches"]
        number_of_recorded_batches = len(self.all_model_state_dicts)

        if number_of_attacks > number_of_recorded_batches:
            raise ValueError(
                f"attack_number_of_batches={number_of_attacks}, but only "
                f"{number_of_recorded_batches} batches were recorded. "
                "Increase number_of_batches."
            )
        
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

        self.log_aggregate_reconstruction_metrics(config)

        if config["device"] != "cpu":
            torch.cuda.empty_cache()

        model.eval()

    def attack_batch(self, model, config, batch_number, original_dy_dx, dummy_inputs, dummy_targets, batch_inputs, batch_targets):
        # Get the inversion model from super class.
        # inversion_model = super().attack_batch(
        #     model, config, batch_number, original_dy_dx, dummy_inputs, dummy_targets, batch_inputs, batch_targets
        # )
        if "num_learn_epochs" in config and config["num_learn_epochs"] <= 0:
            print("Skipping inversion model training as num_learn_epochs <= 0")
            inversion_model = None
        else:
            _, aux_gi_t_dataloader = create_gradient_inversion_dataloader(
                self.auxiliary_train_dataloader, model, config, batch_number, dummy_inputs, dummy_targets, seed_generator=self.g
            )
            _, aux_gi_v_dataloader = create_gradient_inversion_dataloader(
                self.auxiliary_val_dataloader, model, config, batch_number, dummy_inputs, dummy_targets
            )

            print("Created gradient inversion dataloaders")
            inversion_model, _, _ = self.initialize_inversion_model(config, dummy_inputs, dummy_targets)

            print("Initialized Inversion Model")
            grad_to_input_optimizer, lr_schedular = self.set_attack_optimizer_and_schedular(
                inversion_model.parameters(),
                config["learn_optimizer"],
                config["learn_learning_rate"],
                config["learn_lr_decay"],
                config["num_learn_epochs"],
            )

            def generate_model_path(config, folder_path):
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
                dataset_name = config["dataset"]

                path_components = [
                    folder_path,
                    "grad_inputs_targets_model",
                    # str(batch_number),
                    dataset_name,
                ]

                return "_".join(path_components) + ".pt"

            folder_path = "../data/_model_dataset_gradients/"
            # TODO: Make the checkpoint path specific to the experiment configuration
            # and victim model weights. For now, assume one fixed setup per dataset.
            model_path = generate_model_path(config, folder_path)
            if os.path.exists(model_path) and config["load_lti_model"]:
                print(f"Loading inversion model from file: {model_path}")
                inversion_model.load_state_dict(torch.load(model_path))
                print(f"Loaded inversion model to file: {model_path}")
            else:

                for epoch in tqdm(range(config["num_learn_epochs"])):
                    epoch_t_loss = self.inversion_model_epoch(
                        config, epoch, aux_gi_t_dataloader, inversion_model, grad_to_input_optimizer, lr_schedular
                    )
                    epoch_v_loss = self.inversion_model_epoch(config, epoch, aux_gi_v_dataloader, inversion_model)

                    attack_metrics = {"epoch": epoch, "aux_t_loss": np.mean(epoch_t_loss), "aux_v_loss": np.mean(epoch_v_loss)}

                    self.schedular_step(config["learn_lr_decay"], lr_schedular, attack_metrics, np.mean(epoch_v_loss))

                    # self.evaluate_dummy_prediction(
                    #     config,
                    #     batch_number,
                    #     original_dy_dx,
                    #     batch_inputs,
                    #     batch_targets,
                    #     dummy_inputs,
                    #     dummy_targets,
                    #     inversion_model,
                    #     epoch,
                    #     attack_metrics,
                    # )
                torch.save(inversion_model.state_dict(), model_path)
                print(f"Saved inversion model to file: {model_path}")


        if config["num_attack_steps"] == 0:
            return

        if (
            config["dataset"] not in CLASSIFICATION_DATASETS
            and config["batch_size"] == 1
            and config["one_shot_targets"]
        ):
            with torch.no_grad():
                grad_w = original_dy_dx[-2]
                grad_b = original_dy_dx[-1].unsqueeze(-1)
                reconstructed_head_inputs = torch.mm(torch.pinverse(grad_b), grad_w)
                dummy_targets = model.fc(reconstructed_head_inputs) - (grad_b.T / (2 / batch_targets.shape[1]))
                dummy_targets.detach().clone()
                print(dummy_inputs.shape, dummy_targets.shape)

        if inversion_model is not None:
            inversion_model.eval()  # To be used for inference here
            flattened_original_dy_dx = torch.cat([g.view(-1) for g in original_dy_dx]).to(config["device"])
            if config["attack_loss"] == "quantile":
                dummy_quantile_inputs, predicted_dummy_quantile_targets = inversion_model.inference(
                    flattened_original_dy_dx.unsqueeze(0)
                )
                regularization_inputs = dummy_quantile_inputs.squeeze(0)  # remove batch dimension
                regularization_targets = None
                # Repeat regularization inputs and targets to match the batch size
                regularization_inputs = regularization_inputs.repeat(
                    batch_inputs.shape[0] // config["inversion_batch_size"], 1, 1, 1
                )
                if predicted_dummy_quantile_targets is not None:
                    if config["dataset"] in ETT_DATASETS:
                        regularization_targets = (
                            predicted_dummy_quantile_targets.squeeze(0).repeat(
                                batch_targets.shape[0]
                                // config["inversion_batch_size"],
                                1,
                                1,
                                1,
                            )
                        )
                    else:
                        regularization_targets = (
                            predicted_dummy_quantile_targets.squeeze(0).repeat(
                                batch_targets.shape[0]
                                // config["inversion_batch_size"],
                                1,
                                1,
                            )
                        )

                # The regularized inputs, targets last dimension are the quantiles and these should be swaped with the batch dimension
                # dummy_inputs = regularization_inputs.permute(3, 1, 2, 0).squeeze(-1).detach().requires_grad_(True)
                # dummy_targets = regularization_targets.permute(2, 1, 0).squeeze(-1).detach().requires_grad_(True)

                # dummy_inputs = regularization_inputs.clone().reshape(batch_inputs.size()).detach().requires_grad_(True)
                # dummy_targets = regularization_targets.clone().reshape(batch_targets.size()).detach().requires_grad_(True)
                # print(regularization_inputs.shape, regularization_targets.shape, batch_inputs.shape, batch_targets.shape)
            else:
                regularization_inputs, regularization_targets = inversion_model(flattened_original_dy_dx.unsqueeze(0))
                regularization_inputs = regularization_inputs.view(batch_inputs.size()).detach()
                dummy_inputs = regularization_inputs.detach().clone().requires_grad_(True)
                if regularization_targets is not None:
                    regularization_targets = (
                        regularization_targets.view(batch_targets.size()).detach()
                    )
                    dummy_targets = (
                        regularization_targets.clone().requires_grad_(True)
                    )

        optimization_space = [dummy_inputs]

        if config["attack_targets"] and (
            config["dataset"] in ETT_DATASETS
            or config["batch_size"] > 1
            or not config["one_shot_targets"]
        ):
            optimization_space.append(dummy_targets)

        dummy_optimizer, dummy_schedular = self.set_attack_optimizer_and_schedular(
            optimization_space,
            config["attack_opti_optimizer"],
            config["optimization_learning_rate"],
            config["attack_opti_lr_decay"],
            config["num_attack_steps"],
        )

        sample_mapping = np.arange(0, batch_inputs.shape[0])
        if config["dataset"] not in CLASSIFICATION_DATASETS and config["dataset"] not in ETT_DATASETS:
            plot_original_and_dummy_data(config, sample_mapping, dummy_inputs, dummy_targets, batch_inputs, batch_targets)

        for attack_step in range(0, config["num_attack_steps"] + 1):
            attack_metrics = {
                "step": attack_step + config["num_learn_epochs"],
                "sample_mapping": sample_mapping,
            }

            def closure():
                dummy_optimizer.zero_grad()
                model.zero_grad()
                zero_loss = dummy_inputs.new_zeros(())
                loss_components = {
                    "gradient_matching": zero_loss,
                    "dropout_regularization": zero_loss,
                    "inversion_regularization": zero_loss,
                    "total_variation_inputs": zero_loss,
                    "total_variation_targets": zero_loss,
                    "lower_resolution_inputs": zero_loss,
                    "lower_resolution_targets": zero_loss,
                    "trend_regularization": zero_loss,
                    "periodicity_regularization": zero_loss,
                }

                if config["dataset"] in ETT_DATASETS:
                    pred_len = self.train_datasets[0].pred_len
                    label_len = self.train_datasets[0].label_len
                    feature_mode = self.train_datasets[0].features
                    f_dim = -1 if feature_mode == "MS" else 0

                    if not 0 <= label_len <= dummy_inputs.shape[1]:
                        raise ValueError(
                            "ETT label_len must be between 0 and "
                            "the input sequence length."
                        )

                    # Decoder history overlaps the end of the input window.
                    # Build it from the candidate input, preserving gradients.
                    decoder_history = dummy_inputs[
                        :, dummy_inputs.shape[1] - label_len:, :
                    ]
                    decoder_future = dummy_inputs.new_zeros(
                        dummy_inputs.shape[0],
                        pred_len,
                        dummy_inputs.shape[2],
                    )
                    dec_inp = torch.cat(
                        [decoder_history, decoder_future],
                        dim=1,
                    )

                    dummy_out = model(
                        dummy_inputs,
                        self.all_batch_input_marks[batch_number],
                        dec_inp,
                        self.all_batch_target_marks[batch_number],
                    )
                    if isinstance(dummy_out, tuple):
                        dummy_out = dummy_out[0]

                    dummy_out = dummy_out[:, -pred_len:, f_dim:]

                    if dummy_out.shape != dummy_targets.shape:
                        raise ValueError(
                            f"ETT output shape {tuple(dummy_out.shape)} "
                            "does not match dummy target shape "
                            f"{tuple(dummy_targets.shape)}."
                        )

                    dummy_y = F.mse_loss(dummy_out, dummy_targets)

                else:
                    dummy_out = model(dummy_inputs)

                    if config["dataset"] in CLASSIFICATION_DATASETS:
                        dummy_y = nn.CrossEntropyLoss()(
                            dummy_out,
                            dummy_targets,
                        )
                    else:
                        dummy_y = F.mse_loss(
                            dummy_out,
                            dummy_targets,
                        )
                dummy_dy_dx = torch.autograd.grad(dummy_y, model.parameters(), create_graph=True)

                if attack_step >= config["num_attack_steps"]:
                    grad_dict, fig = self.plot_gradients(dummy_dy_dx, original_dy_dx, config)
                    attack_metrics.update(grad_dict)
                    self._log_matplotlib_figure(
                        fig, attack_step + config["num_learn_epochs"], log_name="gradients", matplotlib_only=True
                    )

                loss_components["gradient_matching"] = self.gradient_loss_function(
                    dummy_dy_dx,
                    original_dy_dx,
                    config["gradient_loss"],
                )

                if (
                    "TCN" in model.name
                    and config["optimize_dropout"]
                    and config["dropout"] > 0
                    and config["dropout_probability_regularizer"] > 0
                ):
                    dropout_regularization = zero_loss
                    for dropout_layer in model.get_dropout_layers():
                        dropout_regularization = dropout_regularization + (
                            config["dropout_probability_regularizer"]
                            * ((1 - dropout_layer.do_mask.mean()) - dropout_layer.p).abs()
                        )
                    loss_components["dropout_regularization"] = dropout_regularization

                if inversion_model is not None and (
                    config["inversion_regularization_term_inputs"] > 0 or config["inversion_regularization_term_targets"] > 0
                ):
                    loss_components["inversion_regularization"] = self.learned_prior_regularization(
                        dummy_inputs, dummy_targets, regularization_inputs, regularization_targets, config
                    )

                if "total_variation_alpha_inputs" in config and config["total_variation_alpha_inputs"] > 0:
                    loss_components["total_variation_inputs"] = (
                        config["total_variation_alpha_inputs"]
                        * total_variation_time_series(dummy_inputs)
                    )
                    
                if (
                    config["dataset"] not in CLASSIFICATION_DATASETS
                    and config.get("total_variation_beta_targets", 0) > 0
                ):
                    if config["dataset"] in ETT_DATASETS:
                        tv_targets = dummy_targets
                    else:
                        tv_targets = dummy_targets.unsqueeze(-1)

                    loss_components["total_variation_targets"] = (
                        config["total_variation_beta_targets"]
                        * total_variation_time_series(tv_targets)
                    )

                if config["dataset"] == "motionsense":
                    # Input-only low-resolution regularization.
                    lower_res_weight = config.get(
                        "lower_res_term_inputs",
                        config.get("lower_res_term", 0),
                    )

                    if lower_res_weight > 0:
                        with torch.no_grad():
                            warped_inputs = temporal_resolution_warping(
                                dummy_inputs,
                                2,
                            )
                            filtered_inputs = interpolate(
                                warped_inputs,
                                dummy_inputs.shape[1],
                            )

                        loss_components["lower_resolution_inputs"] = (
                            lower_res_weight
                            * F.l1_loss(filtered_inputs, dummy_inputs)
                        )

                    '''
                    if config.get("trend_term", 0) > 0:
                        trend_lr_term = 1.0

                        if (
                            attack_step > 0
                            and config.get("trend_reduce_lr", False)
                            and dummy_schedular is not None
                        ):
                            trend_lr_term = (
                                dummy_schedular.get_last_lr()[0]
                                / config["optimization_learning_rate"]
                            )

                        dy_dx_loss += (
                            trend_lr_term
                            * config["trend_term"]
                            * featurewise_trend_regularization(
                                dummy_inputs,
                                config["trend_loss"],
                            )
                        )
                    '''
                    trend_weights = config.get("trend_term", {})

                    if any(weight > 0 for weight in trend_weights.values()):
                        trend_lr_term = 1.0

                        if (
                            attack_step > 0
                            and config.get("trend_reduce_lr", False)
                            and dummy_schedular is not None
                        ):
                            trend_lr_term = (
                                dummy_schedular.get_last_lr()[0]
                                / config["optimization_learning_rate"]
                            )

                        loss_components["trend_regularization"] = (
                            trend_lr_term
                            * featurewise_trend_regularization(
                                dummy_inputs,
                                config["trend_loss"],
                                group_weights=trend_weights,
                            )
                        )
                        
                    if config.get("periodicity_term", 0) > 0:
                        period = int(config["periodicity_period"])

                        periodicity_lr_term = 1.0

                        if (
                            attack_step > 0
                            and config.get("periodicity_reduce_lr", False)
                            and dummy_schedular is not None
                        ):
                            periodicity_lr_term = (
                                dummy_schedular.get_last_lr()[0]
                                / config["optimization_learning_rate"]
                            )

                        loss_components["periodicity_regularization"] = (
                            periodicity_lr_term
                            * config["periodicity_term"]
                            * featurewise_periodicity_regularization(
                                dummy_inputs,
                                period=period,
                                loss=config["periodicity_loss"],
                            )
                        )

                elif config["dataset"] not in CLASSIFICATION_DATASETS:
                    if config["dataset"] in ETT_DATASETS:
                        if self.train_datasets[0].features == "MS":
                            history = dummy_inputs[..., -1:]
                        else:
                            history = dummy_inputs

                        combined_dummy_data = torch.cat(
                            [history, dummy_targets], dim=1
                        )
                    else:
                        combined_dummy_data_first_feature = torch.cat(
                            [
                                dummy_inputs[:, :, 0],
                                dummy_targets[:, :],
                            ],
                            dim=1,
                        )

                    if "lower_res_term_inputs" in config and config["lower_res_term_inputs"] > 0:
                        with torch.no_grad():
                            warped_inputs = temporal_resolution_warping(dummy_inputs, 2)
                            filtered_inputs = interpolate(warped_inputs, dummy_inputs.shape[1])
                        loss_components["lower_resolution_inputs"] = (
                            config["lower_res_term_inputs"]
                            * F.l1_loss(filtered_inputs, dummy_inputs)
                        )

                    if config.get("lower_res_term_targets", 0) > 0:
                        with torch.no_grad():
                            if config["dataset"] in ETT_DATASETS:
                                targets_for_filtering = dummy_targets
                            else:
                                targets_for_filtering = dummy_targets.unsqueeze(-1)

                            warped_targets = temporal_resolution_warping(
                                targets_for_filtering, 2
                            )
                            filtered_targets = interpolate(
                                warped_targets, dummy_targets.shape[1]
                            )

                            if config["dataset"] not in ETT_DATASETS:
                                filtered_targets = filtered_targets.squeeze(-1)

                        loss_components["lower_resolution_targets"] = (
                            config["lower_res_term_targets"]
                            * F.l1_loss(filtered_targets, dummy_targets)
                        )

                    trend_term = config.get("trend_term", 0)
                    is_group_weighted = isinstance(trend_term, dict)

                    trend_enabled = (
                        any(weight > 0 for weight in trend_term.values())
                        if is_group_weighted
                        else trend_term > 0
                    )

                    if trend_enabled:
                        lr_term = (
                            dummy_schedular.get_last_lr()[0]
                            / config["optimization_learning_rate"]
                            if (
                                attack_step > 0
                                and config.get("trend_reduce_lr", False)
                                and dummy_schedular is not None
                            )
                            else 1
                        )

                        if config["dataset"] in ETT_DATASETS or config["dataset"] in CLASSIFICATION_DATASETS:
                            trend_loss = featurewise_trend_regularization(
                                combined_dummy_data,
                                config["trend_loss"],
                                group_weights=(
                                    trend_term if is_group_weighted else None
                                ),
                            )
                            if not is_group_weighted:
                                trend_loss = trend_term * trend_loss
                        else:
                            if is_group_weighted:
                                raise ValueError(
                                    "This forecasting branch uses a single "
                                    "feature and requires a scalar trend_term."
                                )

                            trend_loss = (
                                trend_term
                                * trend_consistency_regularization(
                                    combined_dummy_data_first_feature,
                                    config["trend_loss"],
                                )
                            )

                        loss_components["trend_regularization"] = (
                            lr_term * trend_loss
                        )

                    if config.get("periodicity_term", 0) > 0:
                        lr_term = (
                            dummy_schedular.get_last_lr()[0]
                            / config["optimization_learning_rate"]
                            if attack_step > 0 and config["trend_reduce_lr"]
                            else 1
                        )

                        if config["dataset"] in ETT_DATASETS:
                            periodicity_loss = featurewise_periodicity_regularization(
                                combined_dummy_data,
                                period=int(config["periodicity_period"]),
                                loss=config["periodicity_loss"],
                            )
                        else:
                            periodicity_loss = periodicity_regularization(
                                combined_dummy_data_first_feature,
                                period=dummy_targets.shape[1],
                                loss=config["periodicity_loss"],
                            )

                        loss_components["periodicity_regularization"] = (
                            lr_term
                            * config["periodicity_term"]
                            * periodicity_loss
                        )

                total_loss = sum(loss_components.values(), zero_loss)
                loss_components["total"] = total_loss
                total_loss.backward()

                if config["grad_signs_for_inputs"]:
                    dummy_inputs.grad.sign_()
                if config["dataset"] in ETT_DATASETS:
                    if (
                        config["attack_targets"]
                        and config["grad_signs_for_targets"]
                    ):
                        if dummy_targets.grad is None:
                            raise RuntimeError(
                                "ETT target optimization is enabled, "
                                "but dummy_targets has no gradient."
                            )
                        dummy_targets.grad.sign_()
                elif config["grad_signs_for_targets"]:
                    dummy_targets.grad.sign_()
                if (
                    "TCN" in model.name
                    and config["optimize_dropout"]
                    and config["dropout"] > 0
                    and config["grad_signs_for_dropouts"]
                ):
                    for dropout_layer in model.get_dropout_layers():
                        dropout_layer.do_mask.grad.sign_()

                return {
                    name: value.detach()
                    for name, value in loss_components.items()
                }

            if isinstance(dummy_optimizer, torch.optim.LBFGS):
                latest_loss_components = None

                def lbfgs_closure():
                    nonlocal latest_loss_components
                    latest_loss_components = closure()
                    return latest_loss_components["total"]

                dummy_optimizer.step(lbfgs_closure)
                loss_components = latest_loss_components
            else:
                loss_components = dummy_optimizer.step(closure)

            if not isinstance(loss_components, dict):
                raise RuntimeError(
                    "The attack closure did not return its loss components."
                )

            dy_dx_loss = loss_components["total"]
            attack_metrics.update(
                {
                    f"loss/{name}": value.item()
                    for name, value in loss_components.items()
                }
            )
            # Preserve the existing dashboard key while making it represent
            # only the gradient-matching term rather than the total loss.
            attack_metrics["grad_diff_loss_mse"] = (
                loss_components["gradient_matching"].item()
            )

            self.after_effect(config, model, dummy_inputs, dummy_targets, attack_step)

            self.schedular_step(config["attack_opti_lr_decay"], dummy_schedular, attack_metrics, dy_dx_loss)

            # Should calcualte evalaution metrics and log them
            if attack_step % (config["num_attack_steps"] // min(config["num_attack_steps"], 200)) == 0:
                self.evaluate_and_log_reconstruction(
                    config,
                    batch_inputs,
                    batch_targets,
                    dummy_inputs,
                    dummy_targets,
                    batch_number,
                    attack_step,
                    config["num_attack_steps"],
                    attack_metrics,
                    attack_step_offset=config["num_learn_epochs"],
                )

        self.all_dummy_inputs[batch_number] = (
            dummy_inputs.detach().clone()
        )
        self.all_dummy_targets[batch_number] = (
            dummy_targets.detach().clone()
        )

        if config["dataset"] == "motionsense":
            # Match reconstructed samples to the order of the original batch.
            final_mapping = get_batch_sample_mapping(
                batch_inputs,
                dummy_inputs,
            )

            reconstructed_data = (
                dummy_inputs[final_mapping]
                .detach()
                .cpu()
            )

            corresponding_genders = (
                self.all_batch_genders[batch_number]
                .detach()
                .cpu()
            )

            output_directory = config.get(
                "reconstruction_output_dir",
                "../data/_reconstructions",
            )

            os.makedirs(
                output_directory,
                exist_ok=True,
            )

            output_path = os.path.join(
                output_directory,
                (
                    f"motionsense_reconstruction_"
                    f"run_{config['run_number']}_"
                    f"batch_{batch_number}.pt"
                ),
            )

            torch.save(
                {
                    "reconstructed_data": reconstructed_data,
                    "gender": corresponding_genders,
                },
                output_path,
            )

            print(
                "Saved reconstructed data and gender:",
                output_path,
            )

    def _load_timegan_attack_prior(
        self,
        config,
        batch_inputs,
        batch_targets,
    ):
        dataset_name = config["dataset"]
        if dataset_name not in {"motionsense", "mobifall"}:
            raise ValueError(
                "The TimeGAN GIFD experiment supports MotionSense and "
                "MobiFall classification experiments."
            )

        unique_labels = torch.unique(batch_targets.detach()).cpu()
        if unique_labels.numel() != 1:
            raise ValueError(
                "A TimeGAN GIFD attack batch must contain one activity. "
                "Use batch_size=1, or construct activity-homogeneous batches."
            )

        activity_label = int(unique_labels.item())
        if activity_label not in ACTIVITY_LABEL_TO_TIMEGAN_ACTIVITY:
            raise ValueError(
                f"Unsupported {dataset_name} activity label: "
                f"{activity_label}"
            )

        activity = ACTIVITY_LABEL_TO_TIMEGAN_ACTIVITY[
            activity_label
        ]
        checkpoint_path = (
            Path(config["timegan_checkpoint_root"])
            / activity
            / "timegan_checkpoint.pt"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"TimeGAN checkpoint not found: {checkpoint_path}"
            )

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
            )

        prior = _TimeGANAttackPrior(
            checkpoint["timegan_config"]
        )
        states = checkpoint["state_dict"]
        prior.generator.load_state_dict(
            states["generator"],
            strict=True,
        )
        prior.supervisor.load_state_dict(
            states["supervisor"],
            strict=True,
        )
        prior.recovery.load_state_dict(
            states["recovery"],
            strict=True,
        )
        prior = prior.to(batch_inputs.device).train()

        for parameter in prior.parameters():
            parameter.requires_grad_(False)

        if prior.sequence_length != batch_inputs.shape[1]:
            raise ValueError(
                "TimeGAN sequence length does not match the victim input: "
                f"{prior.sequence_length} != {batch_inputs.shape[1]}"
            )
        if dataset_name == "motionsense" and (
            prior.data_dim != batch_inputs.shape[2]
        ):
            raise ValueError(
                "TimeGAN feature count does not match the victim input: "
                f"{prior.data_dim} != {batch_inputs.shape[2]}"
            )
        if dataset_name == "mobifall":
            if batch_inputs.shape[2] != 9:
                raise ValueError(
                    "The MobiFall victim must use the nine native sensor "
                    f"features; received {batch_inputs.shape[2]}"
                )
            if prior.data_dim not in {9, 12}:
                raise ValueError(
                    "The MobiFall TimeGAN prior must produce either nine "
                    "native features or the 12 engineered features from "
                    f"train_mobifall_timegan.py; received {prior.data_dim}"
                )

        preprocessing = checkpoint["preprocessing"]
        if dataset_name == "mobifall" and prior.data_dim == 12:
            if (
                preprocessing.get("external_standardization_mean")
                is not None
                or preprocessing.get("external_standardization_std")
                is not None
            ):
                raise ValueError(
                    "The 12-to-9 MobiFall adapter requires a TimeGAN "
                    "checkpoint trained without external standardization."
                )
        timegan_minimum = torch.as_tensor(
            preprocessing["timegan_minimum"],
            dtype=batch_inputs.dtype,
            device=batch_inputs.device,
        ).view(1, 1, -1)
        timegan_range = torch.as_tensor(
            preprocessing["timegan_range"],
            dtype=batch_inputs.dtype,
            device=batch_inputs.device,
        ).view(1, 1, -1)

        dataset = self.train_datasets[0]
        if not hasattr(dataset, "normalization_mean") or not hasattr(
            dataset,
            "normalization_std",
        ):
            raise AttributeError(
                f"{dataset_name} dataset normalization metadata is "
                "missing."
            )

        classifier_mean = torch.as_tensor(
            dataset.normalization_mean,
            dtype=batch_inputs.dtype,
            device=batch_inputs.device,
        ).view(1, 1, -1)
        classifier_std = torch.as_tensor(
            dataset.normalization_std,
            dtype=batch_inputs.dtype,
            device=batch_inputs.device,
        ).view(1, 1, -1)
        classifier_std = classifier_std.clamp_min(1e-8)

        print(
            f"Loaded {activity} TimeGAN prior for {dataset_name} label "
            f"{activity_label}: {checkpoint_path}"
        )

        return (
            prior,
            timegan_minimum,
            timegan_range,
            classifier_mean,
            classifier_std,
        )

    @staticmethod
    def _decode_timegan_for_classifier(
        normalized_output,
        timegan_minimum,
        timegan_range,
        classifier_mean,
        classifier_std,
        dataset_name,
    ):
        raw_output = (
            normalized_output * timegan_range + timegan_minimum
        )
        if dataset_name == "mobifall" and raw_output.shape[-1] == 12:
            # train_mobifall_timegan.py generated MotionSense-compatible
            # channels. Convert them back to MobiFall's nine native sensor
            # channels for the victim classifier:
            #   acceleration = (gravity + user acceleration) * g
            #   orientation APR = attitude RPY reordered and rad -> degree.
            acceleration = (
                raw_output[..., 3:6] + raw_output[..., 9:12]
            ) * MOBIFALL_GRAVITY_M_S2
            gyroscope = raw_output[..., 6:9]
            orientation_apr = torch.rad2deg(
                raw_output[..., [2, 1, 0]]
            )
            raw_output = torch.cat(
                [acceleration, gyroscope, orientation_apr],
                dim=-1,
            )
        return (
            raw_output - classifier_mean
        ) / classifier_std

    @staticmethod
    def _project_feature_l1_ball_(
        feature,
        reference,
        radius,
    ):
        """Project each sample onto an L1 ball around its initial feature."""
        with torch.no_grad():
            if radius < 0:
                raise ValueError(
                    "timegan_gifd_feature_l1_radius must be non-negative"
                )
            if radius == 0:
                feature.copy_(reference)
                return

            delta = (feature - reference).reshape(
                feature.shape[0],
                -1,
            )
            l1_norm = delta.abs().sum(dim=1)
            outside = l1_norm > radius
            if outside.any():
                selected = delta[outside]
                absolute = selected.abs()
                sorted_absolute, _ = torch.sort(
                    absolute,
                    dim=1,
                    descending=True,
                )
                cumulative = torch.cumsum(
                    sorted_absolute,
                    dim=1,
                )
                indexes = torch.arange(
                    1,
                    selected.shape[1] + 1,
                    device=feature.device,
                    dtype=feature.dtype,
                ).view(1, -1)
                positive = (
                    sorted_absolute
                    - (cumulative - radius) / indexes
                ) > 0
                rho = positive.sum(dim=1).clamp_min(1) - 1
                theta = (
                    cumulative.gather(1, rho.unsqueeze(1)).squeeze(1)
                    - radius
                ) / (rho.to(feature.dtype) + 1)
                projected = selected.sign() * torch.clamp(
                    absolute - theta.unsqueeze(1),
                    min=0,
                )
                delta[outside] = projected

            projected_feature = (
                reference + delta.reshape_as(feature)
            ).clamp(0.0, 1.0)
            feature.copy_(projected_feature)

    def _timegan_gifd_gradient_loss(
        self,
        model,
        reconstructed_inputs,
        dummy_targets,
        original_dy_dx,
        config,
    ):
        dummy_out = model(reconstructed_inputs)
        dummy_y = nn.CrossEntropyLoss()(
            dummy_out,
            dummy_targets,
        )
        dummy_dy_dx = torch.autograd.grad(
            dummy_y,
            model.parameters(),
            create_graph=True,
        )
        loss = self.gradient_loss_function(
            dummy_dy_dx,
            original_dy_dx,
            config["gradient_loss"],
        )

        if config.get("total_variation_alpha_inputs", 0) > 0:
            loss += (
                config["total_variation_alpha_inputs"]
                * total_variation_time_series(reconstructed_inputs)
            )

        lower_res_weight = config.get(
            "lower_res_term_inputs",
            config.get("lower_res_term", 0),
        )
        if lower_res_weight > 0:
            with torch.no_grad():
                warped_inputs = temporal_resolution_warping(
                    reconstructed_inputs,
                    2,
                )
                filtered_inputs = interpolate(
                    warped_inputs,
                    reconstructed_inputs.shape[1],
                )
            loss += lower_res_weight * F.l1_loss(
                filtered_inputs,
                reconstructed_inputs,
            )

        # if config.get("trend_term", 0) > 0:
        #     loss += (
        #         config["trend_term"]
        #         * featurewise_trend_regularization(
        #             reconstructed_inputs,
        #             config["trend_loss"],
        #         )
        #     )


        trend_weights = config.get("trend_term", {})

        if any(weight > 0 for weight in trend_weights.values()):
            trend_lr_term = 1.0

            loss += (
                trend_lr_term
                * featurewise_trend_regularization(
                    reconstructed_inputs,
                    config["trend_loss"],
                    group_weights=trend_weights,
                )
            )

        if config.get("periodicity_term", 0) > 0:
            loss += (
                config["periodicity_term"]
                * featurewise_periodicity_regularization(
                    reconstructed_inputs,
                    period=int(config["periodicity_period"]),
                    loss=config["periodicity_loss"],
                )
            )

        return loss

    def _log_timegan_gifd_reconstruction(
        self,
        config,
        batch_inputs,
        batch_targets,
        reconstructed_inputs,
        dummy_targets,
        batch_number,
        attack_step,
        loss,
        extra_metrics,
    ):
        metrics = {
            "step": attack_step,
            "grad_diff_loss_mse": loss.detach().item(),
            **extra_metrics,
        }
        self.evaluate_and_log_reconstruction(
            config,
            batch_inputs,
            batch_targets,
            reconstructed_inputs,
            dummy_targets,
            batch_number,
            attack_step,
            config["num_attack_steps"],
            metrics,
            attack_step_offset=config.get(
                "num_learn_epochs",
                0,
            ),
        )

    def attack_batch_with_timegan_gifd(
        self,
        model,
        config,
        batch_number,
        original_dy_dx,
        dummy_targets,
        batch_inputs,
        batch_targets,
    ):
        """Optimize TimeGAN noise, then its generated feature sequence.

        The second stage is the recurrent TimeGAN analogue of GIFD feature
        domain optimization. Its search is constrained to an L1 ball around
        the feature sequence produced at the end of latent optimization.
        """
        if config.get("attack_targets", False):
            raise ValueError(
                "TimeGAN GIFD assumes known activity labels; "
                "set attack_targets=False."
            )

        latent_steps = int(config["timegan_gifd_latent_steps"])
        feature_steps = int(config["timegan_gifd_feature_steps"])
        total_steps = latent_steps + feature_steps
        if total_steps <= 0:
            raise ValueError(
                "At least one TimeGAN GIFD optimization step is required."
            )
        if total_steps != config["num_attack_steps"]:
            raise ValueError(
                "num_attack_steps must equal TimeGAN latent plus feature "
                "steps."
            )

        (
            prior,
            timegan_minimum,
            timegan_range,
            classifier_mean,
            classifier_std,
        ) = self._load_timegan_attack_prior(
            config,
            batch_inputs,
            batch_targets,
        )

        dummy_targets = (
            batch_targets.detach().clone().to(batch_inputs.device)
        )
        latent = torch.rand(
            batch_inputs.shape[0],
            prior.sequence_length,
            prior.z_dim,
            dtype=batch_inputs.dtype,
            device=batch_inputs.device,
            requires_grad=True,
        )
        log_interval = max(
            1,
            total_steps // min(total_steps, 200),
        )

        def decode_normalized(normalized_output):
            return self._decode_timegan_for_classifier(
                normalized_output,
                timegan_minimum,
                timegan_range,
                classifier_mean,
                classifier_std,
                config["dataset"],
            )

        latent_optimizer, latent_scheduler = (
            self.set_attack_optimizer_and_schedular(
                [latent],
                config.get(
                    "timegan_gifd_optimizer",
                    "adam",
                ),
                config["timegan_gifd_latent_learning_rate"],
                config.get(
                    "timegan_gifd_lr_decay",
                    "on_plateau_10",
                ),
                max(latent_steps, 1),
            )
        )
        if latent_optimizer is None:
            raise ValueError(
                "Unsupported TimeGAN GIFD optimizer: "
                f"{config.get('timegan_gifd_optimizer')}"
            )

        print(
            "TimeGAN GIFD stage 1/2: optimizing latent noise for "
            f"{latent_steps} steps"
        )
        for stage_step in range(latent_steps):
            attack_step = stage_step

            def latent_closure():
                latent_optimizer.zero_grad()
                model.zero_grad()
                reconstructed_inputs = decode_normalized(
                    prior(latent)
                )
                loss = self._timegan_gifd_gradient_loss(
                    model,
                    reconstructed_inputs,
                    dummy_targets,
                    original_dy_dx,
                    config,
                )
                loss.backward()
                if config.get(
                    "timegan_gifd_sign_gradients",
                    False,
                ):
                    latent.grad.sign_()
                return loss

            loss = latent_optimizer.step(latent_closure)
            with torch.no_grad():
                latent.clamp_(0.0, 1.0)

            scheduler_metrics = {}
            self.schedular_step(
                config.get(
                    "timegan_gifd_lr_decay",
                    "on_plateau_10",
                ),
                latent_scheduler,
                scheduler_metrics,
                loss,
            )

            if (
                attack_step % log_interval == 0
                or stage_step == latent_steps - 1
            ):
                with torch.no_grad():
                    reconstructed_inputs = decode_normalized(
                        prior(latent)
                    )
                self._log_timegan_gifd_reconstruction(
                    config,
                    batch_inputs,
                    batch_targets,
                    reconstructed_inputs,
                    dummy_targets,
                    batch_number,
                    attack_step,
                    loss,
                    {
                        "timegan_gifd/stage": 0,
                        "timegan_gifd/latent_l2": (
                            latent.detach().square().sum().sqrt().item()
                        ),
                        **scheduler_metrics,
                    },
                )

        with torch.no_grad():
            feature_reference = prior.generated_features(
                latent
            ).detach()
        optimized_feature = (
            feature_reference.clone().requires_grad_(True)
        )
        feature_radius = float(
            config["timegan_gifd_feature_l1_radius"]
        )

        feature_optimizer, feature_scheduler = (
            self.set_attack_optimizer_and_schedular(
                [optimized_feature],
                config.get(
                    "timegan_gifd_optimizer",
                    "adam",
                ),
                config[
                    "timegan_gifd_feature_learning_rate"
                ],
                config.get(
                    "timegan_gifd_lr_decay",
                    "on_plateau_10",
                ),
                max(feature_steps, 1),
            )
        )
        if feature_optimizer is None:
            raise ValueError(
                "Unsupported TimeGAN GIFD optimizer: "
                f"{config.get('timegan_gifd_optimizer')}"
            )

        print(
            "TimeGAN GIFD stage 2/2: optimizing generated features for "
            f"{feature_steps} steps inside an L1 ball of radius "
            f"{feature_radius}"
        )
        for stage_step in range(feature_steps):
            attack_step = latent_steps + stage_step

            def feature_closure():
                feature_optimizer.zero_grad()
                model.zero_grad()
                reconstructed_inputs = decode_normalized(
                    prior.decode_features(optimized_feature)
                )
                loss = self._timegan_gifd_gradient_loss(
                    model,
                    reconstructed_inputs,
                    dummy_targets,
                    original_dy_dx,
                    config,
                )
                loss.backward()
                if config.get(
                    "timegan_gifd_sign_gradients",
                    False,
                ):
                    optimized_feature.grad.sign_()
                return loss

            loss = feature_optimizer.step(feature_closure)
            self._project_feature_l1_ball_(
                optimized_feature,
                feature_reference,
                feature_radius,
            )

            scheduler_metrics = {}
            self.schedular_step(
                config.get(
                    "timegan_gifd_lr_decay",
                    "on_plateau_10",
                ),
                feature_scheduler,
                scheduler_metrics,
                loss,
            )

            if (
                attack_step % log_interval == 0
                or stage_step == feature_steps - 1
            ):
                with torch.no_grad():
                    reconstructed_inputs = decode_normalized(
                        prior.decode_features(optimized_feature)
                    )
                    feature_l1 = (
                        optimized_feature - feature_reference
                    ).abs().flatten(1).sum(dim=1).mean().item()
                self._log_timegan_gifd_reconstruction(
                    config,
                    batch_inputs,
                    batch_targets,
                    reconstructed_inputs,
                    dummy_targets,
                    batch_number,
                    attack_step,
                    loss,
                    {
                        "timegan_gifd/stage": 1,
                        "timegan_gifd/feature_l1": feature_l1,
                        **scheduler_metrics,
                    },
                )

        with torch.no_grad():
            if feature_steps > 0:
                reconstructed_inputs = decode_normalized(
                    prior.decode_features(optimized_feature)
                )
            else:
                reconstructed_inputs = decode_normalized(
                    prior(latent)
                )

        self.all_dummy_inputs[batch_number] = (
            reconstructed_inputs.detach().clone()
        )
        self.all_dummy_targets[batch_number] = (
            dummy_targets.detach().clone()
        )

        final_mapping = get_batch_sample_mapping(
            batch_inputs,
            reconstructed_inputs,
        )
        reconstructed_data = (
            reconstructed_inputs[final_mapping].detach().cpu()
        )
        corresponding_genders = (
            self.all_batch_genders[batch_number].detach().cpu()
        )
        output_directory = config.get(
            "reconstruction_output_dir",
            "../data/_reconstructions",
        )
        os.makedirs(output_directory, exist_ok=True)
        output_path = os.path.join(
            output_directory,
            (
                f"{config['dataset']}_reconstruction_"
                f"run_{config['run_number']}_"
                f"batch_{batch_number}.pt"
            ),
        )
        torch.save(
            {
                "reconstructed_data": reconstructed_data,
                "gender": corresponding_genders,
                "attack": "timegan_gifd",
            },
            output_path,
        )
        print(
            "Saved TimeGAN GIFD reconstructed data and gender:",
            output_path,
        )

    def learned_prior_regularization(self, dummy_inputs, dummy_targets, regularization_inputs, regularization_targets, config):
        learned_prior_regularization = torch.zeros(1, device=dummy_inputs.device)
        if config["attack_loss"] == "quantile":
            if config["inversion_regularization_loss"] == "quantile":
                # Make sure the regularization_inputs, which contains the quantile predictions, has the same batch_size as the dummy_inputs
                if config["inversion_regularization_term_inputs"] > 0:
                    reshaped_reg_inputs = regularization_inputs.view(config["batch_size"], -1, len(config["quantiles"]))
                    learned_prior_regularization += config["inversion_regularization_term_inputs"] * pinball_loss(
                        reshaped_reg_inputs, dummy_inputs.view(config["batch_size"], -1), config["quantiles"]
                    )
                if config["inversion_regularization_term_targets"] > 0:
                    # behind quantiles
                    reshaped_reg_targets = regularization_targets.unsqueeze(-2).view(
                        config["batch_size"], -1, len(config["quantiles"])
                    )
                    learned_prior_regularization += config["inversion_regularization_term_targets"] * pinball_loss(
                        reshaped_reg_targets, dummy_targets.view(config["batch_size"], -1), config["quantiles"]
                    )
            if config["inversion_regularization_loss"] == "quantile_bounds":

                def out_of_bound_loss(sequence, sequence_quantiles):
                    bound_loss = torch.zeros(1, device=sequence.device)
                    for tau_q in range(sequence_quantiles.shape[-1] // 2):
                        quantile_lower_bound = sequence_quantiles[..., tau_q].reshape(sequence.shape)
                        quantile_upper_bound = sequence_quantiles[..., -tau_q - 1].reshape(sequence.shape)
                        bound_loss += F.relu(quantile_lower_bound - sequence).mean()
                        bound_loss += F.relu(sequence - quantile_upper_bound).mean()
                    return bound_loss / 2

                if config["inversion_regularization_term_inputs"] > 0:
                    learned_prior_regularization += config["inversion_regularization_term_inputs"] * out_of_bound_loss(
                        dummy_inputs, regularization_inputs
                    )
                if config["inversion_regularization_term_targets"] > 0:
                    learned_prior_regularization += config["inversion_regularization_term_targets"] * out_of_bound_loss(
                        dummy_targets, regularization_targets
                    )

        elif config["inversion_regularization_loss"] == "l1":
            if config["inversion_regularization_term_inputs"] > 0:
                learned_prior_regularization += config["inversion_regularization_term_inputs"] * F.l1_loss(
                    dummy_inputs, regularization_inputs
                )
            if config["inversion_regularization_term_targets"] > 0:
                learned_prior_regularization += config["inversion_regularization_term_targets"] * F.l1_loss(
                    dummy_targets, regularization_targets
                )
        else:
            raise NotImplementedError(f"Inversion regularization loss not found: {config['inversion_regularization_loss']}")
        return learned_prior_regularization

    def initialize_inversion_model(self, config, batch_inputs, batch_targets):
        # inversion_model, input_shape, target_shape = super().initialize_inversion_model(config, batch_inputs, batch_targets)
        data_observations_shape = torch.Size([config["inversion_batch_size"]] + list(batch_inputs.shape[1:]))
        data_targets_shape = None
        if config["attack_targets"]:
            data_targets_shape = torch.Size([config["inversion_batch_size"]] + list(batch_targets.shape[1:]))

        inversion_model = None
        if config["inversion_model"] == "GradToInputNN":
            inversion_model = GradToInputNN(
                config["attack_hidden_size"], config["model_size"], data_observations_shape, data_targets_shape
            ).to(config["device"])
        elif config["inversion_model"] == "ImprovedGradToInputNN":
            inversion_model = ImprovedGradToInputNN(
                config["attack_hidden_size"], config["model_size"], data_observations_shape, data_targets_shape
            ).to(config["device"])
        elif config["inversion_model"] == "ImprovedGradToInputNN_2":
            inversion_model = ImprovedGradToInputNN_2(
                config["attack_hidden_size"], config["model_size"], data_observations_shape, data_targets_shape
            ).to(config["device"])
        elif config["inversion_model"] == "ImprovedGradToInputNN_Probabilistic":
            inversion_model = ImprovedGradToInputNN_Probabilistic(
                config["attack_hidden_size"],
                config["model_size"],
                data_observations_shape,
                data_targets_shape,
                distribution=config["attack_loss"],
            ).to(config["device"])
        elif config["inversion_model"] == "ImprovedGradToInputNN_Quantile":
            inversion_model = ImprovedGradToInputNN_Quantile(
                config["attack_hidden_size"],
                config["model_size"],
                data_observations_shape,
                data_targets_shape,
                quantiles=config["quantiles"],
            ).to(config["device"])
        return inversion_model, data_observations_shape, data_targets_shape

    def calculate_inversion_model_loss(
        self, inversion_model, config, attack_batch_size, predicted_inputs, predicted_targets, aux_inputs, aux_targets
    ):
        # loss = super().calculate_inversion_model_loss(
        #     inversion_model, config, attack_batch_size, predicted_inputs, predicted_targets, aux_inputs, aux_targets
        # )
        # if loss != 0:
        #     return loss

        loss = torch.tensor(0.0).to(config["device"])
        if config["attack_loss"] == "mse":
            # View the predicted inputs and auxiliary inputs as flat vectors
            a_batch_size = attack_batch_size
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
        elif config["attack_loss"] == "quantile":
            assert predicted_inputs is not None, "Predicted inputs should not be None for quantile loss"

            predicted_inputs = predicted_inputs.repeat(1, config["batch_size"] // config["inversion_batch_size"], 1, 1, 1)

            # Activity labels are assumed known for MotionSense and MobiFall.
            # Their inversion model therefore predicts quantiles for the IMU
            # input only, unlike forecasting TS-Inverse which also predicts a
            # future target sequence.
            if predicted_targets is None:
                predicted_inputs = predicted_inputs.reshape(
                    attack_batch_size * config["batch_size"],
                    -1,
                    len(config["quantiles"]),
                )
                aux_inputs = aux_inputs.reshape(
                    attack_batch_size * config["batch_size"],
                    -1,
                )
                return pinball_loss(
                    predicted_inputs,
                    aux_inputs,
                    config["quantiles"],
                )

            if config["dataset"] in ETT_DATASETS:
                # ETT targets already have a feature dimension.
                predicted_targets = predicted_targets.repeat(
                    1,
                    config["batch_size"] // config["inversion_batch_size"],
                    1,
                    1,
                    1,
                )
            else:
                # Preserve the existing univariate forecasting behavior.
                predicted_targets = predicted_targets.unsqueeze(-2)
                predicted_targets = predicted_targets.repeat(
                    1,
                    config["batch_size"] // config["inversion_batch_size"],
                    1,
                    1,
                    1,
                )
                aux_targets = aux_targets.unsqueeze(-1)

            # Calculate pairwise pinball losses between combined vectors
            # Dimensions of Predicted data = (attack_batch_size, config['batch_size'], sequence_length, n_features, n_quantiles)
            # Dimensions of Auxiliary data = (attack_batch_size, config['batch_size'], sequence_length, n_features)
            if config["inversion_batch_size"] != config["batch_size"] or (
                config["inversion_batch_size"] == config["batch_size"] and config["batch_size"] == 1
            ):
                predicted_inputs = predicted_inputs.view(attack_batch_size * config["batch_size"], -1, len(config["quantiles"]))
                aux_inputs = aux_inputs.view(attack_batch_size * config["batch_size"], -1)
                predicted_targets = predicted_targets.view(attack_batch_size * config["batch_size"], -1, len(config["quantiles"]))
                aux_targets = aux_targets.view(attack_batch_size * config["batch_size"], -1)

                combined_predicteds = torch.cat([predicted_inputs, predicted_targets], dim=1)
                combined_auxes = torch.cat([aux_inputs, aux_targets], dim=1)

                return pinball_loss(combined_predicteds, combined_auxes, config["quantiles"])
            else:
                # This would generate 4 individual quantile predictions.
                batch_wise_combined_loss = torch.empty(
                    (attack_batch_size, config["batch_size"], config["batch_size"]), device=config["device"]
                )
                for i in range(attack_batch_size):
                    for j in range(config["batch_size"]):
                        for k in range(config["batch_size"]):
                            inputs_loss = pinball_loss(predicted_inputs[i, j], aux_inputs[i, k], config["quantiles"])
                            targets_loss = pinball_loss(predicted_targets[i, j], aux_targets[i, k], config["quantiles"])
                            batch_wise_combined_loss[i, j, k] = (inputs_loss + targets_loss) / 2
        elif config["attack_loss"] == "normal" or config["attack_loss"] == "cauchy" or config["attack_loss"] == "beta":
            batch_wise_combined_loss = inversion_model.calculate_batchwise_prob_loss(
                predicted_inputs, predicted_targets, aux_inputs, aux_targets, config
            )
        else:
            raise ValueError(f"Invalid attack loss function: {config['attack_loss']}")

        # Solve the optimal assignment problem for the combined loss matrix
        for combined_loss_matrix in batch_wise_combined_loss:
            row_ind, col_ind = linear_sum_assignment(combined_loss_matrix.detach().cpu().numpy())
            loss += combined_loss_matrix[row_ind, col_ind].mean()

        # Normalize the loss by the attack batch size
        loss /= attack_batch_size

        return loss

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
        if config["attack_loss"] == "quantile":
            with torch.no_grad():
                flattened_original_dy_dx = torch.cat([g.view(-1) for g in original_dy_dx]).to(config["device"])
                dummy_quantile_inputs, predicted_dummy_quantile_targets = inversion_model.inference(
                    flattened_original_dy_dx.unsqueeze(0)
                )
                dummy_quantile_inputs = dummy_quantile_inputs.squeeze(0).repeat(
                    batch_inputs.shape[0] // config["inversion_batch_size"], 1, 1, 1
                )
                dummy_inputs = dummy_quantile_inputs

                def pinball_loss_simple(batch_inputs, dummy_inputs):
                    return pinball_loss(dummy_inputs, batch_inputs, config["quantiles"])

                if config["dataset"] in {"motionsense", "mobifall"} or ( config["dataset"] in ETT_DATASETS and predicted_dummy_quantile_targets is None):
                    input_sample_mapping = get_batch_sample_mapping(
                        batch_inputs,
                        dummy_inputs,
                        function=pinball_loss_simple,
                    )
                    attack_metrics["inputs/pinball/mean"] = (
                        pinball_loss_simple(
                            batch_inputs,
                            dummy_inputs[input_sample_mapping],
                        ).item()
                    )
                    for i, j in enumerate(input_sample_mapping):
                        attack_metrics[f"inputs/pinball/{i}"] = (
                            pinball_loss_simple(
                                batch_inputs[i],
                                dummy_inputs[j],
                            ).item()
                        )
                    self._log_metrics(attack_metrics, step=epoch)
                    return

                if predicted_dummy_quantile_targets is not None:
                    if config["dataset"] in ETT_DATASETS:
                        dummy_quantile_targets = (
                            predicted_dummy_quantile_targets.squeeze(0).repeat(
                                batch_targets.shape[0] // config["inversion_batch_size"],
                                1,
                                1,
                                1,
                            )
                        )
                    else:
                        dummy_quantile_targets = (
                            predicted_dummy_quantile_targets.squeeze(0).repeat(
                                batch_targets.shape[0] // config["inversion_batch_size"],
                                1,
                                1,
                            )
                        )

                    dummy_targets = dummy_quantile_targets

                standard_mapping = np.arange(0, batch_inputs.shape[0])
                input_sample_mapping = get_batch_sample_mapping(batch_inputs, dummy_inputs, function=pinball_loss_simple)
                target_sample_mapping = get_batch_sample_mapping(batch_targets, dummy_targets, function=pinball_loss_simple)

                sample_mapping = np.arange(0, batch_inputs.shape[0])
                if not (standard_mapping == input_sample_mapping).all():
                    sample_mapping = input_sample_mapping
                if (standard_mapping == input_sample_mapping).all() and not (standard_mapping == target_sample_mapping).all():
                    sample_mapping = target_sample_mapping

                mean_evaluation = {
                    "inputs/pinball/mean": pinball_loss_simple(batch_inputs, dummy_inputs[sample_mapping]).item(),
                    "targets/pinball/mean": pinball_loss_simple(batch_targets, dummy_targets[sample_mapping]).item(),
                }
                attack_metrics.update(mean_evaluation)

                # individual batch sample metrics
                for i, j in enumerate(sample_mapping):
                    individual_evaluation = {
                        f"inputs/pinball/{i}": pinball_loss_simple(batch_inputs[i], dummy_inputs[j]).item(),
                        f"targets/pinball/{i}": pinball_loss_simple(batch_targets[i], dummy_targets[j]).item(),
                    }
                    attack_metrics.update(individual_evaluation)

                if config["dataset"] in ETT_DATASETS:
                    plot_interval = max(1, config["num_learn_epochs"] // 10)
                else:
                    plot_interval = config["num_learn_epochs"] // 10

                if epoch % plot_interval == 0:
                    quantile_df, fig = plot_quantile_dummy_data(
                        config, sample_mapping, dummy_inputs, dummy_targets, batch_inputs, batch_targets
                    )

                    self._log_dataframe(quantile_df, step=epoch, log_name=f"_quantile_batch_{batch_number}")
                    self._log_matplotlib_figure(fig, step=epoch, log_name=f"_quantile_batch_{batch_number}", matplotlib_only=True)
                self._log_metrics(attack_metrics, step=epoch)
        else:
            with torch.no_grad():
                flattened_original_dy_dx = torch.cat([g.view(-1) for g in original_dy_dx]).to(config["device"])
                dummy_inputs, predicted_dummy_targets = inversion_model.inference(flattened_original_dy_dx.unsqueeze(0))
                # repeat dummy input along first dim until it matches batch_inputs
                dummy_inputs = dummy_inputs.repeat(batch_inputs.size(0) // dummy_inputs.size(1), 1, 1, 1)
                dummy_inputs = dummy_inputs.view(batch_inputs.size())
                if predicted_dummy_targets is not None:
                    if config["dataset"] in ETT_DATASETS:
                        predicted_dummy_targets = predicted_dummy_targets.repeat(
                            batch_targets.size(0) // predicted_dummy_targets.size(1),
                            1,
                            1,
                            1,
                        )
                    else:
                        predicted_dummy_targets = predicted_dummy_targets.repeat(
                            batch_targets.size(0) // predicted_dummy_targets.size(1),
                            1,
                            1,
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

            if config["dataset"] in ETT_DATASETS:
                aux_targets = aux_targets.reshape(
                    batch_size,
                    config["batch_size"],
                    *aux_targets.shape[-2:],
                )
            else:
                aux_targets = aux_targets.view(
                    batch_size,
                    config["batch_size"],
                    *aux_targets.shape[-1:],
                )

            if (
                config["dataset"] not in CLASSIFICATION_DATASETS
                and config["dataset"] not in ETT_DATASETS
                and (aux_inputs.min() < 0 or aux_inputs.max() > 1)
            ):
                print("Aux inputs out of range:", aux_inputs.min(), aux_inputs.max())

            if (
                config["dataset"] not in CLASSIFICATION_DATASETS
                and config["dataset"] not in ETT_DATASETS
                and (aux_targets.min() < 0 or aux_targets.max() > 1)
            ):
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
            if config["dataset"] == "motionsense":
                # MotionSense inputs are standardized, approximately mean=0, std=1.
                all_dummy_inputs = [
                    torch.randn_like(
                        batch_input_example,
                        device=config["device"],
                        requires_grad=True,
                    )
                    for _ in range(attack_number_of_batches)
                ]
            else:
                # Existing initialization for datasets normalized to [0, 1].
                all_dummy_inputs = [
                    torch.rand_like(
                        batch_input_example,
                        device=config["device"],
                        requires_grad=True,
                    )
                    for _ in range(attack_number_of_batches)
                ]
            
            if config["dataset"] in CLASSIFICATION_DATASETS:
                # Minimal classification attack: the activity label is known.
                all_dummy_targets = [
                    batch_target_example.detach().clone().to(config["device"])
                    for _ in range(attack_number_of_batches)
                ]
            else:
                all_dummy_targets = [
                    torch.rand_like(
                        batch_target_example,
                        device=config["device"],
                        requires_grad=True,
                    )
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
        self.all_batch_genders = []
        dataset = config["dataset"]
        model.to(config["device"])
        all_batch_inputs, all_batch_targets, all_model_state_dicts, all_model_gradients, all_model_updates = [], [], [], [], []
        if dataset in CLASSIFICATION_DATASETS:
            model_train_loader = DataLoader(
                tr_dataloader.dataset,
                batch_size=config.get("model_train_batch_size", 64),
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=self.g,
            )

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.get("model_train_learning_rate", 1e-3),
            )

            loss_function = nn.CrossEntropyLoss()

            # Train the activity classifier.
            for epoch in range(config.get("model_train_epochs", 20)):
                model.train()

                epoch_loss = 0.0
                epoch_correct = 0
                epoch_samples = 0

                for inputs, labels in model_train_loader:
                    inputs = inputs.to(config["device"])
                    labels = labels.to(config["device"])

                    optimizer.zero_grad()

                    logits = model(inputs)
                    loss = loss_function(logits, labels)

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item() * len(labels)
                    epoch_correct += (
                        logits.argmax(dim=1) == labels
                    ).sum().item()
                    epoch_samples += len(labels)

                print(
                    f"Activity epoch {epoch + 1:02d}/"
                    f"{config.get('model_train_epochs', 20)} | "
                    f"loss {epoch_loss / epoch_samples:.4f} | "
                    f"accuracy {epoch_correct / epoch_samples:.4f}"
                )

            # Disable dropout while recording and reconstructing the gradient.
            # Otherwise the original and dummy passes would use different masks.
            model.eval()

            sample_offset = 0

            total_batches_processed = 0
            total_batches_needed = config["number_of_batches"]

            for inputs, labels in tr_dataloader:
                if total_batches_processed >= total_batches_needed:
                    break

                inputs = inputs.to(config["device"])
                labels = labels.to(config["device"])

                batch_size = inputs.shape[0]

                if dataset == "motionsense":
                    batch_genders = self.train_datasets[0].genders[
                        sample_offset : sample_offset + batch_size
                    ]

                    batch_genders = torch.as_tensor(
                        batch_genders,
                        dtype=torch.long,
                    )

                    self.all_batch_genders.append(batch_genders)

                sample_offset += batch_size

                model.zero_grad(set_to_none=True)

                logits = model(inputs)
                loss = loss_function(logits, labels)
                loss.backward()

                all_batch_inputs.append(inputs.detach().clone())
                all_batch_targets.append(labels.detach().clone())

                all_model_state_dicts.append(
                    {
                        key: value.detach().clone()
                        for key, value in model.state_dict().items()
                    }
                )

                all_model_gradients.append(
                    [
                        parameter.grad.detach().clone()
                        if parameter.grad is not None
                        else torch.zeros_like(parameter)
                        for parameter in model.parameters()
                    ]
                )

                if config["update_model"]:
                    optimizer.step()

                total_batches_processed += 1

        elif dataset in ETT_DATASETS:
            pred_len = self.train_datasets[0].pred_len
            label_len = self.train_datasets[0].label_len
            feature_mode = self.train_datasets[0].features
            f_dim = -1 if feature_mode == "MS" else 0

            model_train_epochs = config.get("model_train_epochs", 10)
            number_of_batches = config["number_of_batches"]

            if len(tr_dataloader) == 0:
                raise ValueError("The ETT dataloader is empty.")

            if number_of_batches > len(tr_dataloader):
                raise ValueError(
                    f"Requested {number_of_batches} gradient batches, "
                    f"but the ETT dataloader contains "
                    f"{len(tr_dataloader)} batches."
                )

            model_train_loader = DataLoader(
                tr_dataloader.dataset,
                batch_size=config.get("model_train_batch_size", 32),
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=self.g,
            )

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.get("model_train_learning_rate", 1e-4),
            )

            loss_function = nn.MSELoss()

            # Train the forecasting model.
            for epoch in range(model_train_epochs):
                model.train()

                epoch_squared_error = 0.0
                epoch_target_elements = 0

                for batch_x, batch_y, batch_x_mark, batch_y_mark in model_train_loader:
                    batch_x = batch_x.float().to(config["device"])
                    batch_y = batch_y.float().to(config["device"])
                    batch_x_mark = batch_x_mark.float().to(config["device"])
                    batch_y_mark = batch_y_mark.float().to(config["device"])

                    dec_inp = torch.cat(
                        [
                            batch_y[:, :label_len, :],
                            torch.zeros_like(batch_y[:, -pred_len:, :]),
                        ],
                        dim=1,
                    )

                    optimizer.zero_grad(set_to_none=True)

                    outputs = model(
                        batch_x,
                        batch_x_mark,
                        dec_inp,
                        batch_y_mark,
                    )
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                    outputs = outputs[:, -pred_len:, f_dim:]
                    targets = batch_y[:, -pred_len:, f_dim:]

                    if outputs.shape != targets.shape:
                        raise ValueError(
                            f"ETT output shape {tuple(outputs.shape)} "
                            f"does not match target shape {tuple(targets.shape)}."
                        )

                    loss = loss_function(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    epoch_squared_error += loss.item() * targets.numel()
                    epoch_target_elements += targets.numel()

                print(
                    f"ETT epoch {epoch + 1:02d}/{model_train_epochs} | "
                    f"MSE {epoch_squared_error / epoch_target_elements:.6f}"
                )

            # Record deterministic gradients, with dropout disabled.
            # Reconstruction must also use model.eval().
            model.eval()

            self.all_batch_input_marks = []
            self.all_batch_target_marks = []

            for batch_number, batch in enumerate(tr_dataloader):
                if batch_number >= number_of_batches:
                    break

                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x = batch_x.float().to(config["device"])
                batch_y = batch_y.float().to(config["device"])
                batch_x_mark = batch_x_mark.float().to(config["device"])
                batch_y_mark = batch_y_mark.float().to(config["device"])

                dec_inp = torch.cat(
                    [
                        batch_y[:, :label_len, :],
                        torch.zeros_like(batch_y[:, -pred_len:, :]),
                    ],
                    dim=1,
                )

                state_before = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }

                model.zero_grad(set_to_none=True)

                outputs = model(
                    batch_x,
                    batch_x_mark,
                    dec_inp,
                    batch_y_mark,
                )
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                outputs = outputs[:, -pred_len:, f_dim:]
                targets = batch_y[:, -pred_len:, f_dim:]

                if outputs.shape != targets.shape:
                    raise ValueError(
                        f"ETT output shape {tuple(outputs.shape)} "
                        f"does not match target shape {tuple(targets.shape)}."
                    )

                loss = loss_function(outputs, targets)
                loss.backward()

                # Preserve model.parameters() ordering. Do not silently
                # skip parameters or replace missing gradients with zeros.
                gradients = []
                for name, parameter in model.named_parameters():
                    if parameter.grad is None:
                        raise RuntimeError(
                            f"No gradient recorded for parameter {name!r}: "
                            f"shape={tuple(parameter.shape)}, "
                            f"requires_grad={parameter.requires_grad}. "
                            "Inspect why this parameter has no gradient "
                            "before continuing the experiment."
                        )
                    if not torch.isfinite(parameter.grad).all():
                        raise RuntimeError(
                            f"Non-finite gradient recorded for parameter {name!r}."
                        )
                    gradients.append(parameter.grad.detach().clone())

                # Apply configured defenses only after checking raw gradients.
                if config.get("defense_name", "none") not in {None, "none"}:
                    if config.get("sign", False):
                        gradients = apply_sign_transformation(gradients)
                    if config.get("prune_rate") is not None:
                        gradients = apply_pruning(
                            gradients, config["prune_rate"]
                        )
                    if config.get("dp_epsilon", 0) > 0:
                        gradients = add_gaussian_noise(
                            gradients, config["dp_epsilon"]
                        )

                all_batch_inputs.append(batch_x.detach().clone())
                all_batch_targets.append(targets.detach().clone())
                all_model_state_dicts.append(state_before)
                all_model_gradients.append(
                    [gradient.detach().clone() for gradient in gradients]
                )

                self.all_batch_input_marks.append(
                    batch_x_mark.detach().clone()
                )
                self.all_batch_target_marks.append(
                    batch_y_mark.detach().clone()
                )

                if config["update_model"]:
                    for parameter, gradient in zip(
                        model.parameters(), gradients
                    ):
                        parameter.grad = gradient.detach().clone()
                    optimizer.step()

                all_model_updates.append(
                    [
                        (value - state_before[name]).detach().clone()
                        for name, value in model.state_dict().items()
                    ]
                )
        else:
            model_optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

            total_batches_processed = 0  # Initialize total batch counter
            total_batches_needed = config["warmup_number_of_batches"] + config["number_of_batches"]
            val_dataset = ConcatSliceDataset(self.val_datasets)
            while total_batches_processed < total_batches_needed:
                for batch_inputs, batch_targets in tr_dataloader:
                    if total_batches_processed >= total_batches_needed:
                        break  # Exit if we've processed the total required batches
                    if not config["update_model"] and total_batches_processed > 0:
                        model.load_state_dict(all_model_state_dicts[0])

                    print(config)
                    if config["dataset"] == "realworld":
                        batch_inputs, batch_targets = (
                            batch_inputs[:, :, model.features].to(config["device"]),
                            batch_targets[:].to(config["device"]),
                        )
                    else:
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

    def log_aggregate_reconstruction_metrics(self, config):
        """Log final reconstruction errors across every attacked batch.

        Input MAE and MSE are calculated over all reconstructed tensor
        elements. Therefore, batches of different sizes are handled correctly.

        For forecasting datasets, target metrics are also aggregated.
        Classification targets are known labels and are therefore ignored.
        """
        number_of_attacks = config["attack_number_of_batches"]

        if number_of_attacks <= 0:
            return

        input_absolute_error_sum = 0.0
        input_squared_error_sum = 0.0
        input_element_count = 0
        input_batch_maes = []

        target_absolute_error_sum = 0.0
        target_squared_error_sum = 0.0
        target_element_count = 0
        target_smape_weighted_sum = 0.0
        target_batch_maes = []

        input_smape_weighted_sum = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_number in range(number_of_attacks):
                batch_inputs = self.all_batch_inputs[batch_number]
                dummy_inputs = self.all_dummy_inputs[batch_number]

                batch_targets = self.all_batch_targets[batch_number]
                dummy_targets = self.all_dummy_targets[batch_number]

                standard_mapping = np.arange(batch_inputs.shape[0])

                input_mapping = get_batch_sample_mapping(
                    batch_inputs,
                    dummy_inputs,
                )

                sample_mapping = input_mapping

                # Preserve the existing forecasting matching behavior.
                if config["dataset"] not in CLASSIFICATION_DATASETS:
                    target_mapping = get_batch_sample_mapping(
                        batch_targets,
                        dummy_targets,
                    )

                    if (
                        np.array_equal(input_mapping, standard_mapping)
                        and not np.array_equal(
                            target_mapping,
                            standard_mapping,
                        )
                    ):
                        sample_mapping = target_mapping

                mapping_tensor = torch.as_tensor(
                    sample_mapping,
                    dtype=torch.long,
                    device=dummy_inputs.device,
                )

                reconstructed_inputs = dummy_inputs.index_select(
                    0,
                    mapping_tensor,
                )

                input_difference = (
                    reconstructed_inputs - batch_inputs
                )

                input_absolute_error_sum += (
                    input_difference.abs().sum().item()
                )
                input_squared_error_sum += (
                    input_difference.square().sum().item()
                )
                input_element_count += input_difference.numel()
                total_samples += batch_inputs.shape[0]

                input_batch_maes.append(
                    input_difference.abs().mean().item()
                )

                if config["dataset"] not in CLASSIFICATION_DATASETS:
                    target_mapping_tensor = torch.as_tensor(
                        sample_mapping,
                        dtype=torch.long,
                        device=dummy_targets.device,
                    )

                    reconstructed_targets = dummy_targets.index_select(
                        0,
                        target_mapping_tensor,
                    )

                    target_difference = (
                        reconstructed_targets - batch_targets
                    )

                    target_absolute_error_sum += (
                        target_difference.abs().sum().item()
                    )
                    target_squared_error_sum += (
                        target_difference.square().sum().item()
                    )
                    target_element_count += target_difference.numel()

                    target_batch_maes.append(
                        target_difference.abs().mean().item()
                    )

                    # Weight batch SMAPE by its number of elements.
                    input_smape_weighted_sum += (
                        SMAPELoss(
                            reconstructed_inputs,
                            batch_inputs,
                        ).item()
                        * input_difference.numel()
                    )

                    target_smape_weighted_sum += (
                        SMAPELoss(
                            reconstructed_targets,
                            batch_targets,
                        ).item()
                        * target_difference.numel()
                    )

        if input_element_count == 0:
            print("No reconstructed elements available for aggregation.")
            return

        aggregate_input_mae = (
            input_absolute_error_sum / input_element_count
        )
        aggregate_input_mse = (
            input_squared_error_sum / input_element_count
        )

        aggregate_metrics = {
            "aggregate/inputs/mae/mean": aggregate_input_mae,
            "aggregate/inputs/mse/mean": aggregate_input_mse,
            "aggregate/inputs/rmse/mean": np.sqrt(
                aggregate_input_mse
            ),
            "aggregate/inputs/mae/std_across_batches": float(
                np.std(input_batch_maes)
            ),
            "aggregate/number_of_attack_batches": number_of_attacks,
            "aggregate/number_of_samples": total_samples,
        }

        if (
            config["dataset"] not in CLASSIFICATION_DATASETS
            and target_element_count > 0
        ):
            aggregate_target_mae = (
                target_absolute_error_sum / target_element_count
            )
            aggregate_target_mse = (
                target_squared_error_sum / target_element_count
            )

            aggregate_metrics.update(
                {
                    "aggregate/inputs/smape/mean": (
                        input_smape_weighted_sum
                        / input_element_count
                    ),
                    "aggregate/targets/mae/mean": (
                        aggregate_target_mae
                    ),
                    "aggregate/targets/mse/mean": (
                        aggregate_target_mse
                    ),
                    "aggregate/targets/rmse/mean": np.sqrt(
                        aggregate_target_mse
                    ),
                    "aggregate/targets/smape/mean": (
                        target_smape_weighted_sum
                        / target_element_count
                    ),
                    "aggregate/targets/mae/std_across_batches": (
                        float(np.std(target_batch_maes))
                    ),
                }
            )

        # Use a step after the last ordinary reconstruction log.
        aggregate_step = (
            config["num_attack_steps"]
            + config.get("num_learn_epochs", 0)
            + 1
        )

        self._log_metrics(
            aggregate_metrics,
            step=aggregate_step,
        )

        print(
            "Aggregate reconstruction metrics across "
            f"{number_of_attacks} attack batches:",
            aggregate_metrics,
        )

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
            dataset_name = config["dataset"]
            if dataset_name not in CLASSIFICATION_DATASETS:
                target_sample_mapping = get_batch_sample_mapping(batch_targets, dummy_targets)

            sample_mapping = np.arange(0, batch_inputs.shape[0])
            if not (standard_mapping == input_sample_mapping).all():
                sample_mapping = input_sample_mapping
            if (
                dataset_name not in CLASSIFICATION_DATASETS
                and (standard_mapping == input_sample_mapping).all()
                and not (standard_mapping == target_sample_mapping).all()
            ):
                sample_mapping = target_sample_mapping
            # if not (standard_mapping == input_sample_mapping).all() and not (standard_mapping == target_sample_mapping).all():
            #     if not (input_sample_mapping == target_sample_mapping).all():
            #         raise ValueError('Input and target sample mappings are not equal while being different from the standard mapping.')

            if dataset_name in CLASSIFICATION_DATASETS:
                mean_evaluation = {
                    "inputs/mae/mean": F.l1_loss(
                        dummy_inputs[sample_mapping],
                        batch_inputs,
                    ).item(),
                    "inputs/mse/mean": F.mse_loss(
                        dummy_inputs[sample_mapping],
                        batch_inputs,
                    ).item(),
                    "inputs/rmse/mean": torch.sqrt(
                        F.mse_loss(
                            dummy_inputs[sample_mapping],
                            batch_inputs,
                        )
                    ).item(),
                }
            else:
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
                if dataset_name in CLASSIFICATION_DATASETS:
                    individual_evaluation = {
                        # f"inputs/accuracy/{i}": (dummy_inputs[j].argmax(dim=-1) == batch_inputs[i].argmax(dim=-1)).float().mean().item(),
                        f"inputs/mae/{i}": F.l1_loss(dummy_inputs[j], batch_inputs[i]).item(),
                        f"inputs/mse/{i}": F.mse_loss(dummy_inputs[j], batch_inputs[i]).item(),
                        f"inputs/rmse/{i}": torch.sqrt(F.mse_loss(dummy_inputs[j], batch_inputs[i])).item(),
                    }
                else:
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

            if dataset_name not in CLASSIFICATION_DATASETS and dataset_name not in ETT_DATASETS and attack_step % (num_attack_steps // log_plots_n_times) == 0:
                df, fig = plot_original_and_dummy_data(
                    config, sample_mapping, dummy_inputs, dummy_targets, batch_inputs, batch_targets
                )
                self._log_dataframe(df, attack_step + attack_step_offset, log_name=f"_batch_{batch_number}")
                self._log_matplotlib_figure(fig, step=attack_step + attack_step_offset, log_name=f"_batch_{batch_number}")

            # Changes number of steps before new plots are logged for the MotionSense data
            elif attack_step % (num_attack_steps // log_plots_n_times) == 0:
                for sample_idx in range(batch_inputs.shape[0]):
                    df, fig = plot_channel_grid_original_vs_reconstructed(
                        config, sample_mapping, dummy_inputs, batch_inputs, batch_number, sample_idx=sample_idx
                    )
                    self._log_dataframe(df, attack_step + attack_step_offset, log_name=f"_batch_{batch_number}_sample_{sample_idx}")
                    self._log_matplotlib_figure(
                        fig, step=attack_step + attack_step_offset, log_name=f"_batch_{batch_number}_sample_{sample_idx}", matplotlib_only=True
                    )
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
                    dummy_inputs.data = torch.clamp(
                        dummy_inputs,
                        0,
                        1,
                    )

                    if config["dataset"] not in CLASSIFICATION_DATASETS:
                        dummy_targets.data = torch.clamp(
                            dummy_targets,
                            0,
                            1,
                        )
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

def plot_channel_grid_original_vs_reconstructed(config, sample_mapping, dummy_inputs, batch_inputs, batch_number, sample_idx=0):
    original = batch_inputs[sample_idx].detach().cpu().numpy()  # [seq_len, n_features]
    reconstructed = dummy_inputs[sample_mapping[sample_idx]].detach().cpu().numpy()
 
    num_features = original.shape[-1]
 
    fig, axes = plt.subplots(2, num_features, figsize=(2.5 * num_features, 5), sharex=True)
 
    for feature_idx in range(num_features):
        feature_name = FEATURE_NAMES[feature_idx] if feature_idx < len(FEATURE_NAMES) else f"feature_{feature_idx}"
 
        ax_original = axes[0, feature_idx]
        ax_reconstructed = axes[1, feature_idx]
 
        ax_original.plot(original[:, feature_idx], color="tab:blue", linewidth=1.2)
        ax_reconstructed.plot(reconstructed[:, feature_idx], color="tab:orange", linewidth=1.2)
 
        ax_original.set_title(feature_name, fontsize=9)
 
        if feature_idx == 0:
            ax_original.set_ylabel("Original", fontsize=11)
            ax_reconstructed.set_ylabel("Reconstructed", fontsize=11)
 
        ax_reconstructed.set_xlabel("Time step", fontsize=8)
 
    fig.suptitle(
        f'{config["dataset"]}: Batch {batch_number}'
        f'\u2014 Original vs. Reconstructed by Channel',
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
 
    data = {}
    for feature_idx in range(num_features):
        feature_name = FEATURE_NAMES[feature_idx] if feature_idx < len(FEATURE_NAMES) else f"feature_{feature_idx}"
        data[f"original_{feature_name}"] = original[:, feature_idx]
        data[f"reconstructed_{feature_name}"] = reconstructed[:, feature_idx]
    df = pd.DataFrame(data)

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
    model.eval()

    folder_path = "../data/_model_dataset_gradients/"
    task_kind = (
        "classification_ce_v1"
        if config["dataset"] in CLASSIFICATION_DATASETS
        else "forecasting_mse_v1"
    )
    # Path where the dataset will be saved or loaded from
    if config["dataset"] in ETT_DATASETS:
        # TODO: Include the complete configuration, victim weights, and
        # auxiliary-data identity. Currently assumes one fixed setup per dataset.
        dataset_path = (
            f"{folder_path}grad_inputs_targets_dataset_"
            f"{config.get('defense_name', 'none')}_"
            f"ett_forecasting_mse_v1_{batch_number}_{model.name}_"
            f"{config['dataset']}_{len(aux_dataloader.dataset)}_"
            f"{dummy_inputs.shape[1]}_{dummy_inputs.shape[2]}_"
            f"{dummy_targets.shape[1]}_{dummy_targets.shape[2]}_"
            f"{config['seed']}_{config['inversion_batch_size']}.pt"
        )
    else:
        dataset_path = f"{folder_path}grad_inputs_targets_dataset_{config.get('defense_name', 'none')}_{task_kind}_{batch_number}_{model.name}_{'-'.join(map(str, model.features))}_{config['dataset']}_{len(aux_dataloader.dataset)}_{config['input_size']}_{config['output_size']}_{config['seed']}_{config['inversion_batch_size']}.pt"

    # Check if the dataset file exists and load file
    if os.path.exists(dataset_path):
        print("Loaded gradient to inputs targets dataset:", dataset_path)
        grad_inputs_targets_dataset = torch.load(dataset_path, weights_only=False)
        config["loaded_grad_to_inputs_targets_dataset_from_file"] = True
    else:
        # If the file does not exist, proceed with creating the dataset
        print("Creating gradient to inputs targets dataset:", dataset_path)
        config["loaded_grad_to_inputs_targets_dataset_from_file"] = False
        aux_dy_dx_inputs, aux_inputs_targets, aux_targets_targets = [], [], []
        for i, batch in enumerate(aux_dataloader):
            model.zero_grad(set_to_none=True)

            if config["dataset"] in ETT_DATASETS:
                (
                    aux_batch_inputs,
                    aux_batch_targets,
                    aux_input_marks,
                    aux_target_marks,
                ) = batch

                aux_batch_inputs = aux_batch_inputs.float().to(config["device"])
                aux_batch_targets = aux_batch_targets.float().to(config["device"])
                aux_input_marks = aux_input_marks.float().to(config["device"])
                aux_target_marks = aux_target_marks.float().to(config["device"])

                pred_len = dummy_targets.shape[1]

                # The ETT loader returns label context followed by the future.
                label_len = aux_batch_targets.shape[1] - pred_len
                if not 0 <= label_len <= aux_batch_inputs.shape[1]:
                    raise ValueError(
                        "ETT auxiliary label context is incompatible "
                        "with the input and prediction lengths."
                    )

                if aux_batch_inputs.shape[0] != 1:
                    raise ValueError(
                        "Auxiliary gradient generation expects one "
                        "ETT window per batch."
                    )

                if aux_batch_inputs.shape[1:] != dummy_inputs.shape[1:]:
                    raise ValueError(
                        "ETT auxiliary input shape does not match "
                        "the attacked input shape."
                    )

                dec_inp = torch.cat(
                    [
                        aux_batch_targets[:, :label_len, :],
                        torch.zeros_like(
                            aux_batch_targets[:, -pred_len:, :]
                        ),
                    ],
                    dim=1,
                )

                aux_out = model(
                    aux_batch_inputs,
                    aux_input_marks,
                    dec_inp,
                    aux_target_marks,
                )
                if isinstance(aux_out, tuple):
                    aux_out = aux_out[0]

                f_dim = -1 if config.get("features", "M") == "MS" else 0
                aux_out = aux_out[:, -pred_len:, f_dim:]
                aux_batch_targets = aux_batch_targets[:, -pred_len:, f_dim:]

                if aux_batch_targets.shape[1:] != dummy_targets.shape[1:]:
                    raise ValueError(
                        "ETT auxiliary target shape does not match "
                        "the attacked target shape."
                    )

                if aux_out.shape != aux_batch_targets.shape:
                    raise ValueError(
                        f"ETT output shape {tuple(aux_out.shape)} "
                        f"does not match target shape "
                        f"{tuple(aux_batch_targets.shape)}."
                    )

                aux_y = F.mse_loss(aux_out, aux_batch_targets)

            else:
                # Existing dataset behavior.
                aux_batch_inputs, aux_batch_targets = batch

                if aux_batch_inputs.ndim != 3:
                    raise ValueError(
                        "Auxiliary MotionSense/MobiFall inputs must have shape "
                        "[batch, time, features]; received "
                        f"{tuple(aux_batch_inputs.shape)}"
                    )

                aux_batch_inputs = aux_batch_inputs[
                    :, :, model.features
                ].to(config["device"])

                if aux_batch_inputs.shape[1] != dummy_inputs.shape[1]:
                    aux_batch_inputs = interpolate(
                        aux_batch_inputs, dummy_inputs.shape[1]
                    )

                if config["dataset"] in CLASSIFICATION_DATASETS:
                    aux_batch_targets = aux_batch_targets.reshape(-1).to(
                        config["device"],
                        dtype=torch.long,
                    )
                    if aux_batch_targets.numel() != aux_batch_inputs.shape[0]:
                        raise ValueError(
                            "Expected one activity label per auxiliary window; "
                            f"received inputs {tuple(aux_batch_inputs.shape)} "
                            f"and labels {tuple(aux_batch_targets.shape)}"
                        )

                    aux_logits = model(aux_batch_inputs)
                    aux_y = F.cross_entropy(
                        aux_logits,
                        aux_batch_targets,
                    )
                else:
                    aux_batch_targets = aux_batch_targets[
                        :, :, 0
                    ].to(config["device"])

                    if aux_batch_targets.shape[1] != dummy_targets.shape[1]:
                        aux_batch_targets = interpolate(
                            aux_batch_targets.unsqueeze(-1),
                            dummy_targets.shape[1],
                        ).squeeze(-1)

                    aux_out = model(aux_batch_inputs)
                    aux_y = F.mse_loss(aux_out, aux_batch_targets)

            aux_y.backward()

            if config["verbose"]:
                if config["dataset"] in ETT_DATASETS:
                    for name, parameter in model.named_parameters():
                        if parameter.grad is None:
                            raise RuntimeError(
                                f"No auxiliary gradient for parameter {name!r}: "
                                f"shape={tuple(parameter.shape)}, "
                                f"requires_grad={parameter.requires_grad}."
                            )

                        if not torch.isfinite(parameter.grad).all():
                            raise RuntimeError(
                                f"Non-finite auxiliary gradient for parameter {name!r}."
                            )

            if config.get("defense_name", "none") not in {
                None,
                "none",
            }:
                # Apply gradient defenses
                gradients = [param.grad for param in model.parameters()]
                if config.get("sign", False):
                    gradients = apply_sign_transformation(gradients)
                if config.get("prune_rate") is not None:
                    gradients = apply_pruning(gradients, config["prune_rate"])
                if config.get("dp_epsilon", 0) > 0:
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
            drop_last=True,
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


def plot_quantile_dummy_data(config, sample_mapping, dummy_quantile_inputs, dummy_quantile_targets, batch_inputs, batch_targets):
    batch_size = batch_inputs.shape[0]
    num_quantiles = dummy_quantile_inputs.shape[3]  # Number of quantiles is in the last dimension for dummy data
    original_x_axis = np.arange(0, batch_inputs.shape[1] + batch_targets.shape[1])
    fig, axes = plt.subplots(nrows=batch_size, figsize=(20, 3.5 * batch_size), sharex=True)
    axes = np.array(axes).reshape(-1)  # Ensure axes is always iterable

    for i, j in enumerate(sample_mapping):
        ax = axes[i]
        # Plotting for batch inputs (no quantiles)
        batch_input_values = batch_inputs[i, :].detach().cpu().numpy()  # Squeeze out the singleton dimension for plotting
        ax.plot(original_x_axis[: batch_inputs.shape[1]], batch_input_values, label=f"Batch Input ({i})")

        # Plotting quantiles for dummy inputs
        for q in range(num_quantiles):
            dummy_input_quantiles = dummy_quantile_inputs[j, :, 0, q].detach().cpu().numpy()
            ax.plot(original_x_axis[: batch_inputs.shape[1]], dummy_input_quantiles, label=f"Dummy Input Quantile {q} ({j})")

        # Plotting for batch targets (assuming no quantiles)
        batch_target_values = batch_targets[i, :].detach().cpu().numpy()
        ax.plot(original_x_axis[batch_inputs.shape[1] :], batch_target_values, label=f"Batch Target ({i})")

        # Plotting quantiles for dummy targets
        for q in range(num_quantiles):
            if config["dataset"] in ETT_DATASETS:
                for feature in range(dummy_quantile_targets.shape[2]):
                    dummy_target_quantiles = (
                        dummy_quantile_targets[j, :, feature, q]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    ax.plot(
                        original_x_axis[batch_inputs.shape[1]:],
                        dummy_target_quantiles,
                        label=(
                            f"Dummy Target Feature {feature} "
                            f"Quantile {q} ({j})"
                        ),
                    )
            else:
                dummy_target_quantiles = (
                    dummy_quantile_targets[j, :, q]
                    .detach()
                    .cpu()
                    .numpy()
                )
                ax.plot(
                    original_x_axis[batch_inputs.shape[1]:],
                    dummy_target_quantiles,
                    label=f"Dummy Target Quantile {q} ({j})",
                )

        ax.set_xlabel("Time" if i == batch_size - 1 else "")
        ax.set_ylabel("Output")
        ax.set_title(
            f'Dataset {config["dataset"]}: Comparison of Inputs & Targets between Dataset & Recovery - Batch Sample {i} & Dummy Sample {j}'
        )
        ax.legend()

    # Create weights and biases plot & dataframe
    fill_nan = np.empty(batch_targets.shape[1])
    fill_nan.fill(np.nan)
    if config["dataset"] in ETT_DATASETS:
        batch_targets_dict = {
            f"batch_targets_{i}_{feature}": np.concatenate(
                (
                    fill_nan,
                    batch_targets[i, :, feature].detach().cpu().numpy(),
                )
            )
            for i in range(batch_targets.shape[0])
            for feature in range(batch_targets.shape[2])
        }

        dummy_quantile_targets_dict = {
            f"dummy_quantile_targets_{i}_{feature}_{q}": np.concatenate(
                (
                    fill_nan,
                    dummy_quantile_targets[i, :, feature, q]
                    .detach()
                    .cpu()
                    .numpy(),
                )
            )
            for i in range(dummy_quantile_targets.shape[0])
            for feature in range(dummy_quantile_targets.shape[2])
            for q in range(dummy_quantile_targets.shape[3])
        }
    else:
        batch_targets_dict = {
            f"batch_targets_{i}": np.concatenate(
                (
                    fill_nan,
                    batch_targets[i, :].detach().cpu().numpy(),
                )
            )
            for i in range(batch_targets.shape[0])
        }

        dummy_quantile_targets_dict = {
            f"dummy_quantile_targets_{i}_{q}": np.concatenate(
                (
                    fill_nan,
                    dummy_quantile_targets[i, :, q].detach().cpu().numpy(),
                )
            )
            for i in range(dummy_quantile_targets.shape[0])
            for q in range(dummy_quantile_targets.shape[2])
        }
    if len(batch_inputs.shape) == 3:
        batch_inputs_dict = {
            f"batch_inputs_{i}_{f}": np.concatenate((batch_inputs[i, :, f].detach().cpu().numpy(), fill_nan))
            for i in range(batch_inputs.shape[0])
            for f in range(batch_inputs.shape[2])
        }
        dummy_quantile_inputs_dict = {
            f"dummy_quantile_inputs_{i}_{f}_{q}": np.concatenate(
                (dummy_quantile_inputs[i, :, f, q].detach().cpu().numpy(), fill_nan)
            )
            for i in range(dummy_quantile_inputs.shape[0])
            for f in range(dummy_quantile_inputs.shape[2])
            for q in range(dummy_quantile_inputs.shape[3])
        }
    elif len(batch_inputs.shape) == 2:
        batch_inputs_dict = {
            f"batch_inputs_{i}": np.concatenate((batch_inputs[i, :].detach().cpu().numpy(), fill_nan))
            for i in range(batch_inputs.shape[0])
        }
        dummy_quantile_inputs_dict = {
            f"dummy_quantile_inputs_{i}_{q}": np.concatenate((dummy_quantile_inputs[i, :, q].detach().cpu().numpy(), fill_nan))
            for i in range(dummy_quantile_inputs.shape[0])
            for q in range(dummy_quantile_inputs.shape[2])
        }

    fill_nan = np.empty(batch_inputs.shape[1])
    fill_nan.fill(np.nan)
    batch_targets_dict = {
        f"batch_targets_{i}": np.concatenate((fill_nan, batch_targets[i, :].detach().cpu().numpy()))
        for i in range(batch_targets.shape[0])
    }
    dummy_quantile_targets_dict = {
        f"dummy_quantile_targets_{i}_{q}": np.concatenate((fill_nan, dummy_quantile_targets[i, :, q].detach().cpu().numpy()))
        for i in range(dummy_quantile_targets.shape[0])
        for q in range(dummy_quantile_targets.shape[2])
    }
    input_data = {**batch_inputs_dict, **dummy_quantile_inputs_dict, **batch_targets_dict, **dummy_quantile_targets_dict}
    df = pd.DataFrame(input_data)

    if config["verbose"]:
        plt.show()
    else:
        plt.close(fig)

    return df, fig

def featurewise_trend_regularization(x, loss, group_weights=None):
    """Average the trend loss over every sample and feature.

    x has shape [batch, time, features].

    group_weights: optional dict mapping FEATURE_GROUPS keys ("acc", "gyro",
    "gravity", "rot_rate") to a weight. If provided, each feature's trend
    loss is scaled by its group's weight; groups with weight <= 0 are
    skipped. If None, every feature is weighted equally.
    """
    if group_weights is None:
        losses = [
            trend_consistency_regularization(x[b : b + 1, :, feature], loss)
            for b in range(x.shape[0])
            for feature in range(x.shape[2])
        ]
        return torch.stack(losses).mean()

    weighted_losses = []
    for group_name, feature_indices in FEATURE_GROUPS.items():
        weight = group_weights.get(group_name, 0)
        if weight <= 0:
            continue
        for b in range(x.shape[0]):
            for feature in feature_indices:
                if feature >= x.shape[2]:
                    continue
                weighted_losses.append(weight * trend_consistency_regularization(x[b : b + 1, :, feature], loss))

    if not weighted_losses:
        return torch.zeros(1, device=x.device)
    return torch.stack(weighted_losses).mean()


def featurewise_periodicity_regularization(x, period, loss):
    """Average the periodicity loss over every sample and feature.

    x has shape [batch, time, features].
    """
    if period <= 0 or period >= x.shape[1]:
        raise ValueError(
            f"period must be between 1 and {x.shape[1] - 1}; "
            f"received {period}"
        )

    losses = [
        periodicity_regularization(
            x[b : b + 1, :, feature],
            period=period,
            loss=loss,
        )
        for b in range(x.shape[0])
        for feature in range(x.shape[2])
    ]

    return torch.stack(losses).mean()
