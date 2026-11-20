import os
import json
import hashlib
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from copy import deepcopy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION
from contextlib import nullcontext
import threading
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset, Subset


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
    get_ett_dataset,
    get_ett_client_datasets,
    get_custom_dataset,
    get_custom_client_datasets,
)
from .forecasting_worker import evaluate_model

ETT_DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]

CUSTOM_DATASETS = ["weather"]


FEATURE_NAMES = [
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "gravity_x", "gravity_y", "gravity_z",
    "rot_rate_x", "rot_rate_y", "rot_rate_z",
]

WEATHER_FEATURE_NAMES = [
    "H2OC", "PAR", "SWDR", "T", "Tdew", "Tlog", "Tpot",
    "VPact", "VPdef", "VPmax", "max_PAR", "max_wv",
    "p", "rain", "raining", "rh", "rho", "sh", "wd", "wv",
    "OT",
]

FEATURE_GROUPS = {
    "acc": [0, 1, 2],
    "gyro": [3, 4, 5],
    "gravity": [6, 7, 8],
    "rot_rate": [9, 10, 11],
}

ETT_FEATURE_NAMES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

class AttackTSInverseWorker(Worker):
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.worker_name = "AttackTSInverseWorker"

    def _init_attack_worker_process(self, c, d_c, m_c, fam_c, def_c=None):
        final_model_settings = {**m_c, **fam_c}
        dataset_name = d_c["dataset"]

        if dataset_name in ETT_DATASETS:
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
        
        elif dataset_name in CUSTOM_DATASETS:
            self.train_datasets, self.val_datasets, self.test_datasets = (
                get_custom_dataset(
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
            print(f"Loaded custom dataset {dataset_name}")


        else:
            self.train_datasets, self.val_datasets, self.test_datasets = (
                self.get_datasets(**d_c)
            )
        self.g = set_seed(c["seed"])

        train_dataloader = DataLoader(
            ConcatSliceDataset(self.train_datasets),
            batch_size=c["batch_size"],
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=self.g,
        )
        self.val_dataloader = DataLoader(
            ConcatSliceDataset(self.val_datasets),
            batch_size=1024,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=self.g,
        )

        

        mean_std_dataset = ConcatSliceDataset(self.train_datasets)

        mean_std_dataloader = DataLoader(
            mean_std_dataset,
            batch_size=c["batch_size"],
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=self.g,
        )

        # Per-client datasets/dataloaders for FL simulation (ETT only).
        # train_dataloader above is kept untouched: it is still used for mean/std.
        self.client_datasets = None
        self.client_dataloaders = None
        if dataset_name in ETT_DATASETS + CUSTOM_DATASETS:
            num_clients = d_c.get("num_clients", c.get("num_clients", 10))
            client_getter = (
                get_custom_client_datasets
                if dataset_name in CUSTOM_DATASETS
                else get_ett_client_datasets
            )
            self.client_datasets = client_getter(
                dataset=dataset_name,
                root_path=d_c["root_path"],
                data_path=d_c.get("data_path"),
                num_clients=num_clients,
                seq_len=d_c.get("seq_len", 96),
                label_len=d_c.get("label_len", 48),
                pred_len=d_c.get("pred_len", 96),
                features=d_c.get("features", "M"),
                target=d_c.get("target", "OT"),
                scale=d_c.get("scale", True),
                timeenc=d_c.get("timeenc", 0),
                freq=d_c.get("freq", "h"),
            )
            self.client_dataloaders = [
                DataLoader(
                    client_dataset,
                    batch_size=c.get("model_train_batch_size", 32),
                    shuffle=True,
                    worker_init_fn=seed_worker,
                    generator=self.g,
                )
                for client_dataset in self.client_datasets
            ]

        (
            self.inputs_mean,
            self.inputs_std,
            self.targets_mean,
            self.targets_std,
        ) = get_mean_std_dataloader(
            mean_std_dataloader,
            c["device"],
            target_length=self.train_datasets[0].pred_len if dataset_name in ETT_DATASETS + CUSTOM_DATASETS else None
        )

        if dataset_name in ETT_DATASETS + CUSTOM_DATASETS:
            model = final_model_settings["_model"]()

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
        if c.get("fl_poisoning_enabled", False) and c.get("poison_feasibility_only", False):
            raise ValueError("FL poisoning does not support poison_feasibility_only=True; disable it before training.")
        model, train_dataloader, final_config = self._init_attack_worker_process(c, d_c, m_c, fam_c, def_c)
        print("Initialized attack worker process")
        self._uses_direct_property_inference(final_config)
        self.init_logger_object(final_config)
        if final_config.get("poison_feasibility_only", False):
            try:
                if not final_config.get("poison_global_model", False):
                    raise ValueError("poison_feasibility_only requires poison_global_model=True.")
                self.load_or_train_poisoned_global_model(model, final_config)
                self._update_config(final_config)
                print("[Poison] Auxiliary feasibility check complete; skipping client attack and reconstruction.")
            finally:
                self._end_logger_object()
            return
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
        direct = self._uses_direct_property_inference(config)
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


        number_of_attacks = config["attack_number_of_batches"]

        if number_of_attacks > len(self.all_batch_inputs):
            raise ValueError(
                f"Requested {number_of_attacks} attacks, but only "
                f"{len(self.all_batch_inputs)} gradient batches were recorded. "
                "Set number_of_batches >= attack_number_of_batches."
            )

        self.all_dummy_inputs = []
        self.all_dummy_targets = []

        if direct and config.get("attack_mode") != "both":
            self.all_dummy_inputs = [None] * number_of_attacks
            self.all_dummy_targets = [None] * number_of_attacks
            self._update_config(config)
            return

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

        if dataset_name in ETT_DATASETS + CUSTOM_DATASETS:
            # ETT uses all input channels.
            # For M/S, retain all target statistics already calculated.
            if config.get("features", "M") == "MS":
                self.targets_mean = self.targets_mean[-1:]
                self.targets_std = self.targets_std[-1:]

        else:
            self.inputs_mean, self.inputs_std = (
                self.inputs_mean[model.features],
                self.inputs_std[model.features],
            )
            self.targets_mean, self.targets_std = (
                self.targets_mean[0],
                self.targets_std[0],
            )
        config["model_size"] = sum(
            p.numel() for p in model.parameters()
            if config["dataset"] not in ETT_DATASETS + CUSTOM_DATASETS or p.requires_grad
        )

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

        if config["verbose"] and config["aux_dataset"] is not None:
            print("Loaded auxiliary dataset with", len(self.aux_gi_t_dataset), "samples")
            print("Sample size:", self.aux_gi_t_dataset[0][0].shape, self.aux_gi_t_dataset[0][1].shape)
            print("Model / gradient size:", config["model_size"])
            print("Length of dataloader:", len(self.aux_gi_t_dataloader))

    def start_attack(self, model, config):
        direct = self._uses_direct_property_inference(config)
        if direct:
            self._property_inference_rows = {}
        number_of_attacks = config["attack_number_of_batches"]
        number_of_recorded_batches = len(self.all_model_state_dicts)

        if number_of_attacks > number_of_recorded_batches:
            raise ValueError(
                f"attack_number_of_batches={number_of_attacks}, but only "
                f"{number_of_recorded_batches} batches were recorded. "
                "Increase number_of_batches."
            )
        
        # Normalize strings such as "cuda" and "cuda:0", as well as torch.device objects.
        is_cuda = torch.device(config["device"]).type == "cuda"
        parallel = is_cuda and number_of_attacks > 1 and config.get("parallel_attacks", True)
        self._attack_context = threading.local()
        self._attack_logging_lock = threading.RLock()
        self._attack_print_lock = threading.Lock()
        self._inversion_setup_lock = threading.Lock()
        self._attack_progress_lock = threading.Lock()
        self._attack_stop = threading.Event()
        self._attack_progress = {
            batch: [0, "pending"] for batch in range(number_of_attacks)
        }
        self._parallel_attacks = parallel
        self._attack_started_at = time.monotonic()
        if config.get("dataset") in ETT_DATASETS + CUSTOM_DATASETS and (not direct or config.get("attack_mode") == "both"):
            self._reconstruction_run_dir = self._create_reconstruction_run(config)

        def run_batch(batch_number):
            self._attack_context.batch_number = batch_number
            stream = None
            try:
                if parallel:
                    # A stream and model per batch isolate weights, gradients,
                    # buffers, dropout masks, and optimizer state.
                    stream = torch.cuda.Stream(device=config["device"])
                    stream.wait_stream(producer_stream)
                with torch.cuda.stream(stream) if parallel else nullcontext():
                    batch_model = deepcopy(model) if parallel else model
                    batch_model.load_state_dict(self.all_model_state_dicts[batch_number])
                    self._set_attack_progress(batch_number, status="preparing")
                    result = self.attack_batch(
                        batch_model, dict(config), batch_number,
                        self.all_model_gradients[batch_number],
                        self.all_dummy_inputs[batch_number],
                        self.all_dummy_targets[batch_number],
                        self.all_batch_inputs[batch_number],
                        self.all_batch_targets[batch_number],
                    )
                    if direct:
                        with self._attack_logging_lock:
                            self._property_inference_rows[batch_number] = result
                        if config.get("attack_mode") != "both":
                            self._set_attack_progress(batch_number, completed=1)
                    if stream is not None:
                        stream.synchronize()
                status = "cancelled" if self._attack_stop.is_set() else "completed"
                self._set_attack_progress(batch_number, status=status)
                if parallel and batch_number == number_of_attacks - 1:
                    return batch_model.state_dict()
            except BaseException:
                self._set_attack_progress(batch_number, status="failed")
                self._attack_stop.set()
                raise
            finally:
                # Keep tensors/models alive until their stream has finished,
                # including when an exception interrupts an attack.
                if stream is not None:
                    stream.synchronize()
                del self._attack_context.batch_number

        try:
            if parallel:
                producer_stream = torch.cuda.current_stream(config["device"])
                print(f"Running {number_of_attacks} attack batches in parallel on {config['device']}", flush=True)
                with ThreadPoolExecutor(max_workers=number_of_attacks) as executor:
                    futures = [executor.submit(run_batch, batch) for batch in range(number_of_attacks)]
                    pending = set(futures)
                    while pending:
                        done, pending = wait(pending, timeout=60, return_when=FIRST_EXCEPTION)
                        for future in done:
                            future.result()
                        self._print_attack_progress(config)
                # Preserve the sequential path's final model state.
                model.load_state_dict(futures[-1].result())
            else:
                reason = (
                    f"device {config['device']} is not CUDA"
                    if not is_cuda
                    else f"attack_number_of_batches={number_of_attacks}; parallel execution needs at least 2"
                )
                print(f"Running attack batches sequentially: {reason}", flush=True)
                for batch_number in range(number_of_attacks):
                    run_batch(batch_number)
        finally:
            self._parallel_attacks = False

        if direct:
            self._save_client_property_inference_report(config)
            if config.get("attack_mode") != "both":
                model.eval()
                return
        self.log_aggregate_reconstruction_metrics(config)

        if torch.device(config["device"]).type == "cuda":
            torch.cuda.empty_cache()

        model.eval()

        # All batch workers have finished saving; evaluate this run explicitly.
        if config.get("dataset") in ETT_DATASETS + CUSTOM_DATASETS and any(self._reconstruction_run_dir.glob("batch_*.pt")):
            from pi_eval.evaluate_reconstruction_run import main as evaluate_reconstruction_run

            wandb_run = None
            if "wandb" in getattr(self, "logger_object_dict", {}):
                import wandb

                wandb_run = wandb.run
            evaluate_reconstruction_run(
                [str(self._reconstruction_run_dir.resolve())], wandb_run=wandb_run
            )

    def _set_attack_progress(self, batch_number, completed=None, status=None):
        with self._attack_progress_lock:
            progress = self._attack_progress[batch_number]
            if completed is not None:
                progress[0] = completed
            if status is not None:
                progress[1] = status

    def _print_attack_progress(self, config):
        # Preserve the inclusive attack loop: steps 0 through num_attack_steps.
        direct = self._uses_direct_property_inference(config) and config.get("attack_mode") != "both"
        unit = "property measurements" if direct else "attack steps"
        total = (1 if direct else
                 config["num_attack_steps"] + 1 if config["num_attack_steps"] > 0 else 0)
        with self._attack_progress_lock:
            overall_completed = sum(completed for completed, _ in self._attack_progress.values())
            overall_total = total * len(self._attack_progress)
            lines = [
                f"[Batch {batch}] {completed}/{total} {unit} ({status})"
                for batch, (completed, status) in self._attack_progress.items()
            ]
        elapsed_minutes = (time.monotonic() - self._attack_started_at) / 60
        self._print_batch(
            f"Attack progress — elapsed {elapsed_minutes:.1f} min | "
            f"Overall: {overall_completed}/{overall_total} {unit}\n"
            + "\n".join(lines)
        )

    def _create_reconstruction_run(self, config):
        root = Path(config.get(
            "reconstruction_output_dir",
            Path(__file__).resolve().parents[3] / "reconstructions",
        ))
        timestamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_%f%z")
        output_directory = root / f"{config['dataset']}_{timestamp}"
        output_directory.mkdir(parents=True, exist_ok=False)

        def json_value(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, (Path, torch.device)):
                return str(value)
            raise TypeError(f"Cannot serialize config value of type {type(value).__name__}")

        with (output_directory / "config.json").open("w") as handle:
            json.dump(config, handle, indent=2, default=json_value)
            handle.write("\n")
        if config.get("attack_batch_sampling") == "property_percentiles":
            distribution = config["attack_sampling_percentiles"]
            selected = [row for row in config["attack_sampling_selected_batches"] if row["attacked"]]
            fig, ax = plt.subplots(figsize=(8, 4))
            try:
                ax.plot([row["percentile"] for row in distribution],
                        [row["property_score"] for row in distribution], label="Candidate score percentiles")
                ax.scatter([row["rank_percentile"] for row in selected],
                           [row["property_score"] for row in selected], color="tab:red", label="Selected attack batches")
                ax.set(xlabel="Percentile", ylabel="Mean elevated fraction",
                       title=f"Attack batch selection: {config['attack_sampling_target_channel']}")
                ax.legend()
                fig.tight_layout()
                fig.savefig(output_directory / "attack_sampling_percentiles.png")
            finally:
                plt.close(fig)
        self._print_batch("Saving run config and reconstructions to:", output_directory)
        return output_directory

    def _save_batch_reconstruction(self, config, batch_number, batch_inputs, dummy_inputs):
        # start_attack creates the shared run directory before launching threads.
        if not hasattr(self, "_reconstruction_run_dir"):
            self._reconstruction_run_dir = self._create_reconstruction_run(config)
        final_mapping = get_batch_sample_mapping(batch_inputs, dummy_inputs)
        output_path = self._reconstruction_run_dir / f"batch_{batch_number}.pt"
        torch.save(
            {
                "reconstructed_data": dummy_inputs[final_mapping].detach().cpu(),
                "original_data": batch_inputs.detach().cpu(),
            },
            output_path,
        )
        self._print_batch("Saved reconstructed data and original data:", output_path)

    def _print_batch(self, *args, **kwargs):
        batch = getattr(getattr(self, "_attack_context", None), "batch_number", None)
        prefix = f"[Batch {batch}]" if batch is not None else "[Attack]"
        with getattr(self, "_attack_print_lock", nullcontext()):
            kwargs.setdefault("flush", True)
            print(prefix, *args, **kwargs)

    def _log_metrics(self, metrics_dict, step):
        batch = getattr(getattr(self, "_attack_context", None), "batch_number", None)
        if batch is not None:
            metrics_dict = {f"batch_{batch}/{key}": value for key, value in metrics_dict.items()}
        with getattr(self, "_attack_logging_lock", nullcontext()):
            super()._log_metrics(metrics_dict, step)

    def _prepare_batch_inversion_model(self, model, config, batch_number, dummy_inputs, dummy_targets):
        if "num_learn_epochs" in config and config["num_learn_epochs"] <= 0:
            self._print_batch("Skipping inversion model training as num_learn_epochs <= 0")
            inversion_model = None
        else:
            _, aux_gi_t_dataloader = create_gradient_inversion_dataloader(
                self.auxiliary_train_dataloader, model, config, batch_number, dummy_inputs, dummy_targets, seed_generator=self.g
            )
            _, aux_gi_v_dataloader = create_gradient_inversion_dataloader(
                self.auxiliary_val_dataloader, model, config, batch_number, dummy_inputs, dummy_targets
            )

            self._print_batch("Created gradient inversion dataloaders")
            inversion_model, _, _ = self.initialize_inversion_model(config, dummy_inputs, dummy_targets)

            self._print_batch("Initialized Inversion Model")
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

                model_name = config["model"]

                defense_name = "" if config.get("defense_name", "none") == "none" else f'_{config.get("defense_name", "none")}'

                path_components = [
                    folder_path,
                    "grad_inputs_targets_model",
                    # str(batch_number),
                    dataset_name,
                    model_name,
                ]

                model_path = "_".join(path_components) + defense_name + ".pt"

                return model_path

            folder_path = "../data/_model_dataset_gradients/"
            # TODO: Make the checkpoint path specific to the experiment configuration
            # and victim model weights. For now, assume one fixed setup per dataset.
            model_path = generate_model_path(config, folder_path)
            if os.path.exists(model_path) and config["load_lti_model"]:
                self._print_batch(f"Loading inversion model from file: {model_path}")
                inversion_model.load_state_dict(torch.load(model_path))
                self._print_batch(f"Loaded inversion model from file: {model_path}")
            else:

                for epoch in tqdm(
                    range(config["num_learn_epochs"]),
                    disable=getattr(self, "_parallel_attacks", False),
                    desc=f"Batch {batch_number} inversion training",
                ):
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
                self._print_batch(f"Saved inversion model to file: {model_path}")

        return inversion_model

    @staticmethod
    def _uses_direct_property_inference(config):
        if config.get("fl_poisoning_enabled", False) and config.get("poison_feasibility_only", False):
            raise ValueError("FL poisoning does not support poison_feasibility_only=True.")
        mode = config.get("attack_mode", "gradient_inversion")
        if mode not in {"gradient_inversion", "poisoning_assisted_property_inference", "both"}:
            raise ValueError("attack_mode must be gradient_inversion, poisoning_assisted_property_inference, or both.")
        direct = mode in {"poisoning_assisted_property_inference", "both"}
        if mode == "both" and config.get("num_attack_steps", 0) <= 0:
            raise ValueError("attack_mode=both requires num_attack_steps > 0 to generate a reconstruction.")
        if direct:
            if not (config.get("poison_global_model", False) or config.get("fl_poisoning_enabled", False)) or config.get("dataset") not in ETT_DATASETS + CUSTOM_DATASETS:
                raise ValueError("Direct property inference requires centralized or FL poisoning and an ETT dataset.")
            if config.get("poison_feasibility_only", False):
                raise ValueError("Set poison_feasibility_only=False to evaluate recorded client gradients.")
            if config.get("update_model", False) or config.get("defense_name", "none") not in {None, "none"}:
                raise ValueError("Direct property inference currently requires update_model=False and defense_name='none'.")
        return direct

    def attack_batch(self, model, config, batch_number, original_dy_dx, dummy_inputs, dummy_targets, batch_inputs, batch_targets):
        if self._uses_direct_property_inference(config):
            row = self.poisoning_assisted_property_inference(
                model, config, batch_number, original_dy_dx, batch_inputs, batch_targets)
            if config.get("attack_mode") == "both":
                reconstruction = self.gradient_inversion_attack(
                    model, config, batch_number, original_dy_dx, dummy_inputs, dummy_targets, batch_inputs, batch_targets)
                if reconstruction is None:
                    raise RuntimeError(f"No completed reconstruction for comparison in batch {batch_number}.")
                from pi_eval.property_inference import run_fractions
                design = self.poison_feasibility_results["design"]
                values = reconstruction.detach()[..., design["property_channel"]]
                if design["threshold_units"] == "physical" and design["scale"]:
                    c = design["property_channel"]
                    values = values * design["scaler_scale"][c] + design["scaler_mean"][c]
                if not torch.isfinite(values).all():
                    raise ValueError("Non-finite reconstruction property inputs.")
                scores = run_fractions((values > design["threshold"]).cpu().numpy(), design["duration"])
                reconstructed_property = float(scores.mean())
                row.update({"reconstruction_estimate": reconstructed_property,
                            "reconstruction_per_window_property": scores.tolist(),
                            "reconstruction_signed_error": reconstructed_property - row["true_property"],
                            "reconstruction_absolute_error": abs(reconstructed_property - row["true_property"])})
            return row
        return self.gradient_inversion_attack(
            model, config, batch_number, original_dy_dx, dummy_inputs, dummy_targets, batch_inputs, batch_targets)

    def poisoning_assisted_property_inference(self, model, config, batch_number, original_dy_dx, batch_inputs, batch_targets):
        """Decode recorded client gradients; client data is used only for evaluation/identity checks."""
        from pi_eval.property_inference import run_fractions

        if not self._uses_direct_property_inference(config):
            raise ValueError("Select attack_mode='poisoning_assisted_property_inference'.")
        preparation = getattr(self, "poison_feasibility_results", None)
        if preparation is None or config.get("poison_method") != preparation["design"]["method"]:
            raise ValueError("Missing matching poisoned-model preparation metadata; prepare the poisoned model first.")
        design = preparation["design"]
        strategy = design.get("strategy", "constant_bias")
        if strategy not in {"constant_bias", "fixed_projection"}:
            raise ValueError(f"Unsupported poison strategy {strategy!r}.")
        if config.get("poison_strategy", strategy) != strategy:
            raise ValueError("Poison strategy differs from the prepared model.")
        return self._evaluate_property_observation(
            model, design, preparation["constant_property_baseline"], batch_number, original_dy_dx,
            batch_inputs, batch_targets, self.all_batch_input_marks[batch_number], self.all_batch_target_marks[batch_number])

    def _evaluate_property_observation(self, model, design, baseline, batch_number, original_dy_dx,
                                       batch_inputs, batch_targets, input_marks, target_marks):
        """Read-only decoding and analytical verification of an ordinary observed MSE gradient."""
        from pi_eval.property_inference import run_fractions
        strategy = design.get("strategy", "constant_bias")
        coordinates = design.get("coordinates", [design.get("coordinate")])
        names = [name for name, p in model.named_parameters() if p.requires_grad]
        name = design["weight_name"]
        if names != design["parameter_names"] or len(original_dy_dx) != len(names):
            raise ValueError("Recorded gradient ordering does not match poisoned-model parameters.")
        gradient = original_dy_dx[names.index(name)]
        weight = dict(model.named_parameters())[name]
        if gradient is None or gradient.shape != weight.shape or not torch.isfinite(gradient).all():
            raise ValueError(f"Missing, incompatible or non-finite recorded gradient for {name}.")
        K = 1 if design["features"] == "MS" else len(design["channels"])
        # The attack observation is only these recorded scalars. No client data,
        # recomputed gradient or ground-truth label enters this decoding step.
        scalars = np.array([gradient[k, j].item() for k, j in coordinates])
        if strategy == "constant_bias":
            individual = K * scalars / (2 * design["C"])
        else:
            individual = (scalars - design["beta"]) / design["alpha"]
        estimate = float(individual.mean())

        projection = dict(model.named_modules())[name.rsplit('.', 1)[0]]
        captured = []
        handle = projection.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                x = batch_inputs.to(device=weight.device, dtype=weight.dtype)
                y = batch_targets.to(device=weight.device, dtype=weight.dtype)
                horizon, history = design["pred_len"], design["label_len"]
                decoder = torch.cat([x[:, x.shape[1] - history:], x.new_zeros(len(x), horizon, x.shape[2])], dim=1)
                output = model(x, input_marks.to(weight.device), decoder,
                               target_marks.to(weight.device))
                output = output[0] if isinstance(output, (tuple, list)) else output
                if len(captured) != 1:
                    raise ValueError("Expected one final projection invocation for client evaluation.")
                hidden = captured[0][:, -horizon:]
                raw = output[:, -horizon:]
                if not torch.allclose(raw, F.linear(captured[0], projection.weight, projection.bias)[:, -horizon:], rtol=1e-5, atol=1e-6):
                    raise ValueError("Client forecast is not the raw final linear projection.")
                output = raw[..., -1:] if design["features"] == "MS" else raw
                if output.shape != y.shape:
                    raise ValueError("Recorded targets do not match client forecast loss slicing.")
                values = x[..., design["property_channel"]]
                if design["threshold_units"] == "physical" and design["scale"]:
                    c = design["property_channel"]
                    values = values * design["scaler_scale"][c] + design["scaler_mean"][c]
                if not torch.isfinite(values).all():
                    raise ValueError("Non-finite client property inputs.")
                phi = run_fractions((values > design["threshold"]).cpu().numpy(), design["duration"])
                truth = float(phi.mean())
                details = []
                for q, (k, j) in enumerate(coordinates):
                    ell = 0 if design["features"] == "MS" else k
                    h = hidden[..., j]
                    raw_analytic = ((2.0 / K) * ((output[..., ell] - y[..., ell]) * h).mean(dim=1)).mean().item()
                    if strategy == "constant_bias":
                        if torch.count_nonzero(projection.weight[k]).item() or projection.bias[k].item() != projection.bias.new_tensor(design["C"]).item():
                            raise ValueError("Client model does not retain the constant-bias measurement row.")
                        analytic = K * raw_analytic / (2 * design["C"])
                    else:
                        analytic = (raw_analytic - design["beta"]) / design["alpha"]
                    if not np.isclose(scalars[q], raw_analytic, rtol=1e-4, atol=1e-5) or not np.isclose(individual[q], analytic, rtol=1e-4, atol=1e-5):
                        raise ValueError(f"Recorded client gradient identity failed at [{k}, {j}] in batch {batch_number}; check model state, data and defenses.")
                    detail = {"coordinate": [k, j], "gradient_scalar": float(scalars[q]),
                              "estimate": float(individual[q]), "raw_analytic_gradient": raw_analytic,
                              "analytic_measurement": analytic, "feature_average": h.mean().item(),
                              "raw_identity_error": abs(float(scalars[q]) - raw_analytic),
                              "identity_error": abs(float(individual[q]) - analytic),
                              "signed_error": float(individual[q]) - truth}
                    if strategy == "constant_bias":
                        detail["target_correction"] = (y[..., ell] * h).mean().item() / design["C"]
                    details.append(detail)
        finally:
            handle.remove()
            model.train(was_training)
        row = {"batch": batch_number, "batch_size": len(batch_inputs), "estimate": estimate,
               "true_property": truth, "per_window_true_property": phi.tolist(),
               "signed_error": estimate - truth, "absolute_error": abs(estimate - truth),
               "constant_error": abs(baseline - truth),
               "measurements": details,
               "identity_error": max(d["identity_error"] for d in details)}
        if len(coordinates) > 1:
            row.update({"coordinates": coordinates, "gradient_scalars": scalars.tolist(),
                        "individual_estimates": individual.tolist()})
        else:
            row["gradient_scalar"] = float(scalars[0])
        self._print_batch(f"Client property batch {batch_number}: original={truth:.6f}, estimate={estimate:.6f}, "
                          f"absolute error={row['absolute_error']:.6f}, measurements={individual.tolist()}")
        return row

    def _save_client_property_inference_report(self, config):
        preparation = self.poison_feasibility_results
        design = preparation["design"]
        coordinates = design.get("coordinates", [design.get("coordinate")])
        rows = [self._property_inference_rows[k] for k in sorted(self._property_inference_rows)]
        report = {"evaluation": "recorded_client_gradients", "design": design, "config": config,
                  "constant_property_baseline": preparation["constant_property_baseline"], "results": []}
        for size in sorted({row["batch_size"] for row in rows}):
            batch_rows = [row for row in rows if row["batch_size"] == size]
            errors = np.array([row["signed_error"] for row in batch_rows])
            result = {"batch_size": size, "batches": len(batch_rows), "rows": batch_rows,
                      "mae": float(np.abs(errors).mean()), "rmse": float(np.sqrt(np.square(errors).mean())),
                      "bias": float(errors.mean()),
                      "constant_mae": float(np.mean([row["constant_error"] for row in batch_rows]))}
            if len(coordinates) > 1:
                errors_by_coordinate = np.array([[d["signed_error"] for d in row["measurements"]] for row in batch_rows])
                result["coordinate_summary"] = [
                    {"coordinate": q, "mae": float(np.abs(errors_by_coordinate[:, i]).mean()),
                     "rmse": float(np.sqrt(np.square(errors_by_coordinate[:, i]).mean())),
                     "bias": float(errors_by_coordinate[:, i].mean())} for i, q in enumerate(coordinates)]
                result["coordinate_summary"].append({"coordinate": "ensemble", **{key: result[key] for key in ['mae', 'rmse', 'bias']}})
                matrix = [[None for _ in coordinates] for _ in coordinates]
                for i in range(len(coordinates)):
                    for j in range(len(coordinates)):
                        if len(batch_rows) > 1 and np.ptp(errors_by_coordinate[:, i]) > 0 and np.ptp(errors_by_coordinate[:, j]) > 0:
                            value = float(np.corrcoef(errors_by_coordinate[:, i], errors_by_coordinate[:, j])[0, 1])
                            matrix[i][j] = value if np.isfinite(value) else None
                result["signed_error_correlation"] = {"coordinates": coordinates, "matrix": matrix}
            report["results"].append(result)
        if config.get("attack_mode") == "both":
            comparison = []
            for result in report["results"]:
                for method, key in [("direct_property_inference", "signed_error"),
                                    ("gradient_inversion", "reconstruction_signed_error")]:
                    errors = np.array([row[key] for row in result["rows"]])
                    comparison.append({"batch_size": result["batch_size"], "method": method,
                                       "batches": len(errors), "mae": float(np.abs(errors).mean()),
                                       "rmse": float(np.sqrt(np.square(errors).mean())), "bias": float(errors.mean())})
            report["method_comparison"] = comparison
            report["reconstruction_run_dir"] = str(self._reconstruction_run_dir)
        root = Path(config.get("property_inference_output_dir") or Path(__file__).resolve().parents[3] / "evaluation_results")
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_%f%z")
        path = root / f"{config['dataset']}_client_property_{stamp}.json"
        path.write_text(json.dumps(report, indent=2, default=str) + '\n')
        lines = [f"# Client property inference: {config['dataset']} ({design.get('strategy', 'constant_bias')})", "",
                 "Direct estimates use recorded client gradients only. Client inputs/targets provide ground truth and numerical identity checks; no client-data calibration is performed.", ""]
        if config.get("attack_mode") == "both":
            lines += ["## Direct inference versus gradient inversion", "",
                      "Both methods use the same recorded client batches and poisoned model states. The inversion estimate is the exact configured property measured in the reconstructed inputs. Errors are fractions, not time-series reconstruction MSE.", "",
                      "| Batch | Batch size | Original property | Direct estimate | Reconstruction property | Direct absolute error | Inversion absolute error |",
                      "|---:|---:|---:|---:|---:|---:|---:|"]
            for row in rows:
                lines.append(f"| {row['batch'] + 1} | {row['batch_size']} | {row['true_property']:.6f} | {row['estimate']:.6f} | {row['reconstruction_estimate']:.6f} | {row['absolute_error']:.6f} | {row['reconstruction_absolute_error']:.6f} |")
            lines += ["", "| Batch size | Method | Batches | Property MAE | Property RMSE | Signed bias |",
                      "|---:|:---|---:|---:|---:|---:|"]
            for result in report["method_comparison"]:
                lines.append(f"| {result['batch_size']} | {result['method']} | {result['batches']} | {result['mae']:.6f} | {result['rmse']:.6f} | {result['bias']:+.6f} |")
            lines.append("")
        tables = self._poison_feasibility_batch_tables(report)
        tables[0] = "## Recorded client batches: original property versus gradient estimate"
        tables = [line.replace('held-out batches', 'client batches') for line in tables]
        lines += tables
        lines += ["## Aggregate errors", "", "| Batch size | Batches | MAE | RMSE | Bias | Auxiliary constant MAE |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for result in report['results']:
            lines.append(f"| {result['batch_size']} | {result['batches']} | {result['mae']:.6f} | {result['rmse']:.6f} | {result['bias']:+.6f} | {result['constant_mae']:.6f} |")
        lines += ["", "## Measurement diagnostics", "", "| Batch | Coordinate | Raw gradient | Analytical gradient | Decoded estimate | Analytical estimate | Identity error |",
                  "|---:|:---|---:|---:|---:|---:|---:|"]
        for row in rows:
            for d in row['measurements']:
                lines.append(f"| {row['batch'] + 1} | {d['coordinate']} | {d['gradient_scalar']:.8g} | {d['raw_analytic_gradient']:.8g} | {d['estimate']:.6f} | {d['analytic_measurement']:.6f} | {d['identity_error']:.3e} |")
        lines += ["", "## Preparation and run settings", "", "```json", json.dumps({"design": design, "config": config}, indent=2, default=str), "```", ""]
        path.with_suffix('.md').write_text('\n'.join(lines))
        self.client_property_inference_results = report
        self._print_batch("Client property inference report:", path.with_suffix('.md'))
        return path

    def gradient_inversion_attack(self, model, config, batch_number, original_dy_dx, dummy_inputs, dummy_targets, batch_inputs, batch_targets):
        if not hasattr(self, "_attack_context"):
            self._attack_context = threading.local()
        self._attack_context.batch_number = batch_number
        # Shared auxiliary loaders and checkpoint paths must be accessed serially.
        # Only this setup phase is serialized; optimization runs on each stream.
        with getattr(self, "_inversion_setup_lock", nullcontext()):
            inversion_model = self._prepare_batch_inversion_model(
                model, config, batch_number, dummy_inputs, dummy_targets
            )
            if getattr(self, "_parallel_attacks", False):
                torch.cuda.current_stream(config["device"]).synchronize()


        if config["num_attack_steps"] == 0:
            return

        if (
            config['batch_size'] == 1 and config['one_shot_targets']
        ):
            with torch.no_grad():
                grad_w = original_dy_dx[-2]
                grad_b = original_dy_dx[-1].unsqueeze(-1)
                reconstructed_head_inputs = torch.mm(torch.pinverse(grad_b), grad_w)
                dummy_targets = model.fc(reconstructed_head_inputs) - (grad_b.T / (2 / batch_targets.shape[1]))
                dummy_targets.detach().clone()
                self._print_batch(dummy_inputs.shape, dummy_targets.shape)

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
                    if config["dataset"] in ETT_DATASETS + CUSTOM_DATASETS:
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
                # self._print_batch(regularization_inputs.shape, regularization_targets.shape, batch_inputs.shape, batch_targets.shape)
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

        perturbation_space = config.get("perturbation_space", "time")

        if (
            perturbation_space == "dct"
            and config.get("after_effect", "none") != "none"
        ):
            raise ValueError(
                "DCT optimization currently requires after_effect='none'. "
                "Time-domain clamping would require updating the "
                "optimized coefficients as well."
            )

        optimize_targets = config["attack_targets"] and (
            config["dataset"] in ETT_DATASETS + CUSTOM_DATASETS
            or config["batch_size"] > 1
            or not config["one_shot_targets"]
        )

        input_variable, decode_inputs = prepare_attack_variable(
            dummy_inputs, perturbation_space
        )
        optimization_space = [input_variable]

        target_variable = dummy_targets
        if optimize_targets:
            target_variable, decode_targets = prepare_attack_variable(
                dummy_targets, perturbation_space
            )
            optimization_space.append(target_variable)

        dummy_optimizer, dummy_schedular = self.set_attack_optimizer_and_schedular(
            optimization_space,
            config["attack_opti_optimizer"],
            config["optimization_learning_rate"],
            config["attack_opti_lr_decay"],
            config["num_attack_steps"],
        )

        sample_mapping = np.arange(0, batch_inputs.shape[0])
        if config['dataset'] not in ETT_DATASETS + CUSTOM_DATASETS:
            with getattr(self, "_attack_logging_lock", nullcontext()):
                plot_original_and_dummy_data(config, sample_mapping, dummy_inputs, dummy_targets, batch_inputs, batch_targets)

        for attack_step in tqdm(
            range(0, config["num_attack_steps"] + 1),
            disable=not config["verbose"] or getattr(self, "_parallel_attacks", False),
            desc=f"Batch {batch_number} attack",
        ):
            if getattr(self, "_attack_stop", None) is not None and self._attack_stop.is_set():
                return
            attack_metrics = {
                "step": attack_step + config["num_learn_epochs"],
                "sample_mapping": sample_mapping,
            }

            def closure():
                nonlocal dummy_inputs, dummy_targets

                dummy_optimizer.zero_grad()
                model.zero_grad()

                if perturbation_space == "dct":
                    dummy_inputs = decode_inputs(input_variable)
                    if optimize_targets:
                        dummy_targets = decode_targets(target_variable)

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

                if config["dataset"] in ETT_DATASETS + CUSTOM_DATASETS:
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

                    dummy_y = F.mse_loss(
                        dummy_out,
                        dummy_targets,
                    )
                dummy_dy_dx = torch.autograd.grad(
                    dummy_y,
                    [p for p in model.parameters()
                     if config["dataset"] not in ETT_DATASETS + CUSTOM_DATASETS or p.requires_grad],
                    create_graph=True,
                )

                if attack_step >= config["num_attack_steps"]:
                    with getattr(self, "_attack_logging_lock", nullcontext()):
                        grad_dict, fig = self.plot_gradients(dummy_dy_dx, original_dy_dx, config)
                        # attack_metrics.update(grad_dict)
                        self._log_matplotlib_figure(
                            fig, attack_step + config["num_learn_epochs"], log_name=f"gradients_batch_{batch_number}", matplotlib_only=True
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
                    config.get('total_variation_beta_targets', 0) > 0
                ):
                    if config["dataset"] in ETT_DATASETS + CUSTOM_DATASETS:
                        tv_targets = dummy_targets
                    else:
                        tv_targets = dummy_targets.unsqueeze(-1)

                    loss_components["total_variation_targets"] = (
                        config["total_variation_beta_targets"]
                        * total_variation_time_series(tv_targets)
                    )

                if config["dataset"] in ETT_DATASETS + CUSTOM_DATASETS:
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
                        if config["dataset"] in ETT_DATASETS + CUSTOM_DATASETS:
                            targets_for_filtering = dummy_targets
                        else:
                            targets_for_filtering = dummy_targets.unsqueeze(-1)

                        warped_targets = temporal_resolution_warping(
                            targets_for_filtering, 2
                        )
                        filtered_targets = interpolate(
                            warped_targets, dummy_targets.shape[1]
                        )

                        if config["dataset"] not in ETT_DATASETS + CUSTOM_DATASETS:
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

                    if config['dataset'] in ETT_DATASETS + CUSTOM_DATASETS:
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

                    if config["dataset"] in ETT_DATASETS + CUSTOM_DATASETS:
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
                    input_variable.grad.sign_()
                if config["dataset"] in ETT_DATASETS + CUSTOM_DATASETS:
                    if (
                        config["attack_targets"]
                        and config["grad_signs_for_targets"]
                    ):
                        if target_variable.grad is None:
                            raise RuntimeError(
                                "ETT target optimization is enabled, "
                                "but target_variable has no gradient."
                            )
                        target_variable.grad.sign_()
                elif config["grad_signs_for_targets"]:
                    target_variable.grad.sign_()
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

            if perturbation_space == "dct":
                with torch.no_grad():
                    dummy_inputs = decode_inputs(input_variable)
                    if optimize_targets:
                        dummy_targets = decode_targets(target_variable)

            self.after_effect(config, model, dummy_inputs, dummy_targets, attack_step)

            self.schedular_step(config["attack_opti_lr_decay"], dummy_schedular, attack_metrics, dy_dx_loss)

            if hasattr(self, "_attack_progress"):
                # Scalar loss reads above wait for the optimizer's CUDA work.
                self._set_attack_progress(batch_number, completed=attack_step + 1, status="running")

            # Should calcualte evalaution metrics and log them
            # but only if parallel is False, otherwise it will be done in the main thread after all batches are completed
            if attack_step % (config["num_attack_steps"] // min(config["num_attack_steps"], 200)) == 0 and not getattr(self, "_parallel_attacks", False):
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


        if config["dataset"] in ETT_DATASETS + CUSTOM_DATASETS:
            self._save_batch_reconstruction(config, batch_number, batch_inputs, dummy_inputs)
        return dummy_inputs.detach()

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

            # Activity labels are assumed known for MobiFall.
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

            if config["dataset"] in ETT_DATASETS + CUSTOM_DATASETS:
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

                if config['dataset'] in {'mobifall'} or (config['dataset'] in ETT_DATASETS + CUSTOM_DATASETS and predicted_dummy_quantile_targets is None):
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
                    if config["dataset"] in ETT_DATASETS + CUSTOM_DATASETS:
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

                if config["dataset"] in ETT_DATASETS + CUSTOM_DATASETS:
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
                    if config["dataset"] in ETT_DATASETS + CUSTOM_DATASETS:
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

            # self._print_batch(batch_size, batch_num, aux_grads.shape, aux_inputs.shape, aux_targets.shape)
            aux_grads, aux_inputs, aux_targets = aux_grads[:batch_num], aux_inputs[:batch_num], aux_targets[:batch_num]
            aux_grads = aux_grads.view(batch_size, config["batch_size"], aux_grads.shape[-1]).mean(
                1
            )  # gradients are always averaged over batch size
            aux_inputs = aux_inputs.view(batch_size, config["batch_size"], *aux_inputs.shape[-2:])

            if config["dataset"] in ETT_DATASETS + CUSTOM_DATASETS:
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
                config['dataset'] not in ETT_DATASETS + CUSTOM_DATASETS and (aux_inputs.min() < 0 or aux_inputs.max() > 1)
            ):
                self._print_batch("Aux inputs out of range:", aux_inputs.min(), aux_inputs.max())

            if (
                config['dataset'] not in ETT_DATASETS + CUSTOM_DATASETS and (aux_targets.min() < 0 or aux_targets.max() > 1)
            ):
                self._print_batch("Aux targets out of range:", aux_targets.min(), aux_targets.max())

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
                    self._print_batch(f'Epoch {epoch}/{config["num_learn_epochs"]}: Val Loss: {round(np.mean(epoch_loss), 5)}')
                else:
                    self._print_batch(
                        f'Epoch {epoch}/{config["num_learn_epochs"]}: Train Loss: {round(np.mean(epoch_loss), 5)} lr: {lr_schedular.get_last_lr()}',
                    )
        if config["verbose"]:
            self._print_batch()

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
                torch.rand_like(
                    batch_input_example,
                    device=config["device"],
                    requires_grad=True,
                )
                for _ in range(attack_number_of_batches)
            ]
            
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

    @staticmethod
    def _property_labels(x, design):
        from pi_eval.property_inference import run_fractions
        with torch.no_grad():
            c = design["property_channel"]
            values = x[..., c]
            if design["threshold_units"] == "physical" and design["scale"]:
                values = values * design["scaler_scale"][c] + design["scaler_mean"][c]
            if not torch.isfinite(values).all():
                raise ValueError("Non-finite property inputs.")
            scores = run_fractions((values > design["threshold"]).detach().cpu().numpy(), design["duration"])
            return torch.as_tensor(scores, device=x.device, dtype=x.dtype)

    def _property_measurement_batch(self, model, batch, setup, captured):
        projection = setup["projection"]
        channels = setup["channels"]
        dataset = setup["dataset"]
        mode = setup["mode"]
        multi = setup["multi"]
        j = setup["j"]
        k = setup["k"]
        poison_strategy = setup["poison_strategy"]
        C = setup["C"]
        alpha = setup["alpha"]
        beta = setup["beta"]
        K = 1 if mode == "MS" else len(channels)
        m = dataset.pred_len * K
        loss_channel = 0 if mode == "MS" else k
        if multi:
            hidden_indices = torch.tensor([q[1] for q in setup["coordinates"]], device=projection.weight.device)
            loss_indices = torch.tensor([0 if mode == "MS" else q[0] for q in setup["coordinates"]], device=projection.weight.device)
        x, y, xm, ym = (value.float().to(projection.weight.device) for value in batch)
        if x.shape[-1] != len(channels):
            raise ValueError("Input channel ordering/shape does not match the configured ETT dataset.")
        decoder = torch.cat([y[:, :dataset.label_len], torch.zeros_like(y[:, -dataset.pred_len:])], dim=1)
        captured.clear()
        output = model(x, xm, decoder, ym)
        output = output[0] if isinstance(output, (tuple, list)) else output
        if not captured or captured[0].ndim != 3:
            raise ValueError("Expected [batch, horizon, hidden] activation before the final projection.")
        projection_input = captured.pop()
        hidden = projection_input[:, -dataset.pred_len:]
        raw_output = output[:, -dataset.pred_len:]
        # Reject normalization, nonlinearities or other processing after the
        # projection: the raw linear identity must match the actual client.
        linear_output = F.linear(projection_input, projection.weight, projection.bias)[:, -dataset.pred_len:]
        if raw_output.shape != linear_output.shape or not torch.allclose(raw_output, linear_output, rtol=1e-5, atol=1e-6):
            raise ValueError("The configured projection is not the raw final forecast layer; measurement identity is unsupported.")
        f_dim = -1 if mode == "MS" else 0
        output, targets = raw_output[..., f_dim:], y[:, -dataset.pred_len:, f_dim:]
        if output.shape != targets.shape or output[0].numel() != m:
            raise ValueError("Forecast/target slicing does not match mean-MSE measurement coordinates.")
        h = hidden.index_select(2, hidden_indices) if multi else hidden[..., j]
        phi = self._property_labels(x, setup["metadata"])
        if poison_strategy == "constant_bias":
            s = ((1 - targets[..., loss_channel] / C) * h).mean(dim=1)
        else:
            if multi:
                residual = output.index_select(2, loss_indices) - targets.index_select(2, loss_indices)
            else:
                residual = output[..., loss_channel] - targets[..., loss_channel]
            gradient_contribution = (2.0 / K) * (residual * h).mean(dim=1)
            s = (gradient_contribution - beta) / alpha
        return output, targets, h, phi, s


    def _prepare_property_measurement(self, model, config):
        """Validate a measurement and construct purged auxiliary subsets; never train/load weights."""
        C = alpha = beta = rho = ensemble_weight = None

        poison_strategy = config.get("poison_strategy", "fixed_projection")
        if poison_strategy not in {"constant_bias", "fixed_projection"}:
            raise ValueError("poison_strategy must be 'constant_bias' or 'fixed_projection'.")
        if config["dataset"] not in ETT_DATASETS + CUSTOM_DATASETS or config.get("aux_dataset") is not None:
            raise ValueError("Weights-only measurement preparation requires the existing ETT auxiliary splits.")
        feasibility_only = config.get("poison_feasibility_only", False)
        if not feasibility_only and (config.get("update_model", False)
                                    or config.get("defense_name", "none") not in {None, "none"}):
            raise ValueError("This measurement supports raw gradients at the fixed model: use update_model=False "
                             "and defense_name='none', or poison_feasibility_only=True for an auxiliary-only check.")
        if config.get("forecast_loss_reduction", "mean") != "mean":
            raise ValueError("The measurement identity requires mean MSE over batch, horizon and selected channels.")
        dataset = self.train_datasets[0]
        mode = dataset.features
        if mode not in {"M", "MS", "S"}:
            raise ValueError(f"Unsupported ETT feature mode {mode!r}.")
        source = Path(dataset.root_path) / dataset.data_path
        channels = ([dataset.target] if mode == "S" else list(pd.read_csv(source, nrows=0).columns[1:]))

        def channel_index(value, key):
            if isinstance(value, str) and value in channels:
                return channels.index(value)
            if type(value) is int and 0 <= value < len(channels):
                return value
            raise ValueError(f"{key} must be a name in {channels} or a zero-based channel index.")

        c = channel_index(config.get("poison_target_channel"), "poison_target_channel")
        layer_name = config.get("poison_projection_layer", "decoder.projection")
        projection = dict(model.named_modules()).get(layer_name)
        if not isinstance(projection, nn.Linear) or projection.bias is None:
            raise ValueError(f"Expected an existing biased final Linear at {layer_name!r}. "
                             "Use the vanilla Transformer; iTransformer's output path is unsupported.")
        if projection.out_features != len(channels):
            raise ValueError("Final projection channels must match the dataset channel order.")
        if "poison_gradient_coordinates" in config:
            if poison_strategy != "fixed_projection":
                raise ValueError("poison_gradient_coordinates is supported only for fixed_projection.")
            requested_coordinates = config["poison_gradient_coordinates"]
            if not isinstance(requested_coordinates, list) or not requested_coordinates:
                raise ValueError("poison_gradient_coordinates must be a nonempty list.")
        else:
            requested_coordinates = [{
                "channel": config.get("poison_measurement_channel", channels[-1] if mode == "MS" else channels[c]),
                "hidden_coordinate": config.get("poison_hidden_coordinate", 0),
            }]
        coordinates = []
        for coordinate in requested_coordinates:
            if not isinstance(coordinate, dict) or "channel" not in coordinate or "hidden_coordinate" not in coordinate:
                raise ValueError("Each measurement coordinate requires channel and hidden_coordinate.")
            k = channel_index(coordinate["channel"], "measurement channel")
            j = coordinate["hidden_coordinate"]
            if type(j) is not int or not 0 <= j < projection.in_features:
                raise ValueError("Measurement hidden_coordinate is outside the projection's hidden dimension.")
            if mode == "MS" and k != len(channels) - 1:
                raise ValueError("MS measurement must use the last output channel selected by the client loss.")
            if [k, j] in coordinates:
                raise ValueError(f"Duplicate measurement coordinate [{k}, {j}].")
            coordinates.append([k, j])
        M = len(coordinates)
        multi = M > 1
        k, j = coordinates[0]  # Legacy scalar path is unchanged when M == 1.
        if poison_strategy == "constant_bias":
            C = config.get("poison_measurement_C", 10.0)
            if not np.isfinite(C) or C <= 0:
                raise ValueError("poison_measurement_C must be positive and finite.")
        else:
            alpha = config.get("poison_gradient_scale", 1.0)
            beta = config.get("poison_gradient_offset", 0.0)
            rho = config.get("poison_max_relative_weight_change")
            ensemble_weight = config.get("poison_ensemble_loss_weight", 0.5)
            if not np.isfinite(ensemble_weight) or not 0 <= ensemble_weight < 1:
                raise ValueError("poison_ensemble_loss_weight must be finite and in [0, 1).")
            if not np.isfinite(alpha) or alpha == 0 or not np.isfinite(beta):
                raise ValueError("poison_gradient_scale must be finite and nonzero; poison_gradient_offset must be finite.")
            if rho is not None and (not np.isfinite(rho) or rho < 0):
                raise ValueError("poison_max_relative_weight_change must be None or finite and nonnegative.")
        tau = config.get("poison_property_threshold", .5)
        duration = config.get("poison_property_duration", 1)
        units = config.get("poison_property_units", "standardized" if dataset.scale else "physical")
        if not np.isfinite(tau):
            raise ValueError("poison_property_threshold must be finite.")
        if type(duration) is not int or not 1 <= duration <= dataset.seq_len:
            raise ValueError("poison_property_duration must be between 1 and seq_len.")
        if units not in {"standardized", "physical"} or (units == "standardized" and not dataset.scale):
            raise ValueError("Use standardized property units only with scaled inputs, or choose physical units.")
        scaler = dataset.scaler if dataset.scale else None
        property_mean = float(scaler.mean_[c]) if scaler is not None and units == "physical" else 0.0
        property_scale = float(scaler.scale_[c]) if scaler is not None and units == "physical" else 1.0
        if poison_strategy == "constant_bias":
            feature_epochs = config.get("poison_feature_epochs", config.get("poison_pretrain_epochs", 5))
        else:
            feature_epochs = 0
        measurement_epochs = config.get("poison_train_epochs", 10)
        batch_size = config.get("poison_train_batch_size", config.get("model_train_batch_size", 32))
        lr = config.get("poison_train_learning_rate", config.get("model_train_learning_rate", 1e-4))
        seed = config.get("poison_seed", config["seed"])
        fraction = config.get("poison_holdout_fraction", .2)
        max_validation_batches = config.get("poison_validation_batches", 8)
        tolerance = config.get("poison_feasibility_tolerance", .05)
        if (type(feature_epochs) is not int or feature_epochs < 0 or type(measurement_epochs) is not int
                or measurement_epochs < 1 or type(batch_size) is not int or batch_size < 1):
            raise ValueError("Use nonnegative feature epochs and positive measurement epochs and batch size.")
        if not np.isfinite(lr) or lr <= 0 or not 0 < fraction < 1:
            raise ValueError("Use positive finite learning rate and 0 < poison_holdout_fraction < 1.")
        if type(max_validation_batches) is not int or max_validation_batches < 1:
            raise ValueError("poison_validation_batches must be positive.")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("poison_feasibility_tolerance must be finite and nonnegative.")
        requested_sizes = config.get("poison_validation_batch_sizes", [1, 8, 32, 128, config.get("batch_size", 32)])
        if not isinstance(requested_sizes, (list, tuple)) or not requested_sizes or any(type(n) is not int or n < 1 for n in requested_sizes):
            raise ValueError("poison_validation_batch_sizes must be a nonempty list of positive integers.")
        split = config.get("poison_auxiliary_split", "test")
        if split not in {"test", "val"}:
            raise ValueError("poison_auxiliary_split must be 'test' or 'val'.")
        auxiliary_sets = self.test_datasets if split == "test" else self.val_datasets
        preparation_parts, holdout_parts, partitions = [], [], []
        for part in auxiliary_sets:
            if (part.features, part.seq_len, part.label_len, part.pred_len) != (mode, dataset.seq_len, dataset.label_len, dataset.pred_len):
                raise ValueError("Auxiliary preprocessing/window lengths must match the benign dataset.")
            if ((Path(part.root_path) / part.data_path).resolve() != source.resolve()
                    or (part.target, part.scale, part.timeenc, part.freq)
                    != (dataset.target, dataset.scale, dataset.timeenc, dataset.freq)
                    or (scaler is not None and (
                        not np.array_equal(part.scaler.mean_, scaler.mean_)
                        or not np.array_equal(part.scaler.scale_, scaler.scale_)))):
                raise ValueError("Auxiliary channel order, source and scaling must match the benign dataset.")
            span = part.seq_len + part.pred_len
            raw_length = len(part) + span - 1
            start = part.seq_len  # Skip the ETT loader's preceding-split lookback.
            boundary = start + int((raw_length - start) * (1 - fraction))
            train_stop = boundary - span + 1
            if train_stop <= start or boundary >= len(part):
                raise ValueError("Auxiliary partition too short for a disjoint preparation/holdout split. "
                                 "Use more data or shorter windows.")
            preparation_parts.append(Subset(part, range(start, train_stop)))
            holdout_parts.append(Subset(part, range(boundary, len(part))))
            partitions.append({"preparation_start": start, "preparation_stop": train_stop,
                               "holdout_start": boundary, "holdout_stop": len(part), "window_span": span})
        if not preparation_parts:
            raise ValueError("No auxiliary datasets available.")
        preparation, holdout = ConcatDataset(preparation_parts), ConcatDataset(holdout_parts)
        validation_sizes = sorted({n for n in requested_sizes if n <= len(holdout)})
        if not validation_sizes:
            raise ValueError("No requested validation batch size fits the held-out partition.")
        decoding = f"C={C:g}" if poison_strategy == "constant_bias" else f"alpha={alpha:g}, beta={beta:g}"
        measurement_description = f"coordinates={coordinates}" if multi else f"output={channels[k]}, hidden={j}"
        print(f"[Poison] Weights-only measurement ({poison_strategy}): input={channels[c]}, {measurement_description}, "
              f"{decoding}, threshold>{tau:g} ({units}), duration={duration}", flush=True)
        print(f"[Poison] Auxiliary {split}: {len(preparation)} preparation windows, {len(holdout)} held-out windows. "
              f"Purged temporal partitions (inputs AND forecast labels): {partitions}", flush=True)
        if len(validation_sizes) != len(set(requested_sizes)):
            print(f"[Poison] Skipping batch sizes larger than holdout; evaluating {validation_sizes}.", flush=True)
        if config.get("defense_name", "none") not in {None, "none"}:
            print("[Poison] Feasibility check uses RAW mean-MSE gradients; configured client defenses are not applied.", flush=True)

        parameter_names = [name for name, p in model.named_parameters() if p.requires_grad]
        weight_name = next(name for name, p in model.named_parameters() if p is projection.weight)
        if weight_name not in parameter_names or not projection.bias.requires_grad:
            raise ValueError("The client's projection weight and bias must be trainable.")
        base_path = config.get("poison_base_checkpoint_path")
        base_identity = None
        if base_path:
            base_path = Path(base_path)
            base_identity = {"path": str(base_path.resolve()), "size": base_path.stat().st_size,
                             "mtime_ns": base_path.stat().st_mtime_ns}
        metadata = {
            "method": "weights_only_projection_measurement_v1", "dataset": config["dataset"],
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}", "architecture": repr(model),
            "parameter_names": parameter_names, "weight_name": weight_name, "coordinate": [k, j],
            "source_csv": str(source.resolve()), "source_size": source.stat().st_size,
            "source_mtime_ns": source.stat().st_mtime_ns,
            "channels": channels, "property_channel": c, "features": mode, "target": dataset.target,
            "seq_len": dataset.seq_len, "label_len": dataset.label_len, "pred_len": dataset.pred_len,
            "scale": dataset.scale, "timeenc": dataset.timeenc, "freq": dataset.freq,
            "scaler_mean": scaler.mean_.tolist() if scaler is not None else None,
            "scaler_scale": scaler.scale_.tolist() if scaler is not None else None,
            "threshold": tau, "threshold_units": units, "duration": duration,
            "auxiliary_split": split, "partitions": partitions, "base_checkpoint": base_identity,
            "feature_epochs": feature_epochs, "measurement_epochs": measurement_epochs,
            "batch_size": batch_size, "learning_rate": lr, "seed": seed, "initialization_seed": config["seed"],
        }
        if poison_strategy == "constant_bias":
            # Keep the legacy metadata and digest exactly: absent strategy means constant_bias.
            metadata["C"] = C
        else:
            metadata.update({"method": "weights_only_fixed_projection_measurement_v1",
                             "strategy": poison_strategy, "alpha": alpha, "beta": beta,
                             "training_stages": [["measurement", measurement_epochs]],
                             "max_relative_weight_change": rho})
            if multi:
                del metadata["coordinate"]
                metadata.update({"method": "weights_only_fixed_projection_multi_measurement_v1",
                                 "coordinates": coordinates, "ensemble_loss_weight": ensemble_weight,
                                 "aggregation": "arithmetic_mean"})
        return {
            "metadata": metadata,
            "decoding": decoding,
            "projection": projection,
            "poison_strategy": poison_strategy,
            "C": C,
            "alpha": alpha,
            "beta": beta,
            "rho": rho,
            "ensemble_weight": ensemble_weight,
            "c": c,
            "k": k,
            "j": j,
            "channels": channels,
            "units": units,
            "feature_epochs": feature_epochs,
            "measurement_epochs": measurement_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
            "tolerance": tolerance,
            "max_validation_batches": max_validation_batches,
            "validation_sizes": validation_sizes,
            "preparation": preparation,
            "holdout": holdout,
            "dataset": dataset,
            "mode": mode,
            "multi": multi,
            "M": M,
            "coordinates": coordinates,
            "base_path": base_path,
            "weight_name": weight_name,
            "parameter_names": parameter_names,
        }

    def load_or_train_poisoned_global_model(self, model, config):
        """Train an existing projection coordinate as a property measurement.

        No model architecture or client loss changes. constant_bias zeros a
        projection row and sets its bias to C; fixed_projection preserves the
        entire projection and learns the actual residual-times-feature measurement.
        Always validate actual raw mean-MSE gradients on a purged temporal holdout.
        This is centralized preparation; fl_aggregation selects the separate
        run_federated_training path, not this checkpoint-training path.
        """
        setup = self._prepare_property_measurement(model, config)
        metadata = setup["metadata"]
        decoding = setup["decoding"]
        projection = setup["projection"]
        poison_strategy = setup["poison_strategy"]
        C = setup["C"]
        alpha = setup["alpha"]
        beta = setup["beta"]
        rho = setup["rho"]
        ensemble_weight = setup["ensemble_weight"]
        c = setup["c"]
        k = setup["k"]
        j = setup["j"]
        channels = setup["channels"]
        units = setup["units"]
        feature_epochs = setup["feature_epochs"]
        measurement_epochs = setup["measurement_epochs"]
        batch_size = setup["batch_size"]
        lr = setup["lr"]
        seed = setup["seed"]
        tolerance = setup["tolerance"]
        max_validation_batches = setup["max_validation_batches"]
        validation_sizes = setup["validation_sizes"]
        preparation = setup["preparation"]
        holdout = setup["holdout"]
        dataset = setup["dataset"]
        mode = setup["mode"]
        multi = setup["multi"]
        M = setup["M"]
        coordinates = setup["coordinates"]
        base_path = setup["base_path"]
        weight_name = setup["weight_name"]
        parameter_names = setup["parameter_names"]
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
        checkpoint_path = Path(config.get("poison_checkpoint_path") or (
            Path(__file__).resolve().parents[3] / "models" / "poisoned_global"
            / f"{config['dataset']}_weights_only_{digest}.pt"))
        load_checkpoint = checkpoint_path.is_file() and not config.get("poison_force_retrain", False)
        model.to(config["device"])
        if load_checkpoint:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if checkpoint.get("poisoning") != metadata:
                raise ValueError(f"Incompatible weights-only checkpoint: {checkpoint_path}. "
                                 "Choose a new path or set poison_force_retrain=True.")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            if poison_strategy == "fixed_projection":
                projection_reference = checkpoint.get("projection_reference")
                if not isinstance(projection_reference, dict) or any(
                    not isinstance(projection_reference.get(name), torch.Tensor)
                    or not torch.equal(value.detach().cpu(), projection_reference[name])
                    for name, value in [("weight", projection.weight), ("bias", projection.bias)]
                ):
                    raise ValueError("Checkpoint does not preserve its original complete projection weight and bias.")
        else:
            if base_path:
                base = torch.load(base_path, map_location="cpu", weights_only=True)
                if "poisoning" in base or "measurement" in base:
                    raise ValueError("poison_base_checkpoint_path must be an unpoisoned model checkpoint.")
                model.load_state_dict(base.get("model_state_dict", base), strict=True)
            if poison_strategy == "constant_bias":
                with torch.no_grad():
                    projection.weight[k].zero_()
                    projection.bias[k].fill_(C)
        if poison_strategy == "constant_bias":
            if torch.count_nonzero(projection.weight[k]).item() or projection.bias[k].item() != projection.bias.new_tensor(C).item():
                raise ValueError("Checkpoint must have measurement projection row exactly zero and bias exactly C.")
        else:
            # On reload the saved original reference was checked above. On a new
            # run this snapshots the starting projection after optional base loading.
            projection_reference = {"weight": projection.weight.detach().clone(),
                                    "bias": projection.bias.detach().clone()}

        def loader(data, size, shuffle=False):
            return DataLoader(data, batch_size=size, shuffle=shuffle, worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(seed))

        def properties(x):
            return self._property_labels(x, metadata)

        captured = []

        def capture(module, inputs):
            captured[:] = [inputs[0]]  # Keep the graph for preparation, remove the hook before returning.

        handle = projection.register_forward_pre_hook(capture)
        original_flags = {name: p.requires_grad for name, p in model.named_parameters()}
        K = 1 if mode == "MS" else len(channels)
        loss_channel = 0 if mode == "MS" else k
        m = dataset.pred_len * K
        if multi:
            row_indices = torch.tensor([q[0] for q in coordinates], device=projection.weight.device)
            hidden_indices = torch.tensor([q[1] for q in coordinates], device=projection.weight.device)
            loss_indices = torch.zeros_like(row_indices) if mode == "MS" else row_indices

        def batch_values(batch):
            return self._property_measurement_batch(model, batch, setup, captured)

        device = torch.device(config["device"])
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        history = []
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                torch.random.default_generator.manual_seed(seed)
                if device.type == "cuda":
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(seed)
                # Eval mode matches the deterministic client's gradient observation.
                # eval() does not prevent gradient-based preparation.
                model.eval()
                if not load_checkpoint:
                    projection.weight.requires_grad_(False)
                    projection.bias.requires_grad_(False)
                    upstream_parameters = [p for p in model.parameters() if p.requires_grad]
                    optimizer = torch.optim.Adam(upstream_parameters, lr=lr)
                    weight_constraints = []
                    if poison_strategy == "fixed_projection" and rho is not None:
                        for p in upstream_parameters:
                            reference = p.detach().clone()
                            radius = rho * max(reference.norm().item(), 1e-12)
                            weight_constraints.append((p, reference, radius))
                    train_loader = loader(preparation, batch_size, shuffle=True)
                    if poison_strategy == "constant_bias":
                        stages = [("feature", feature_epochs), ("measurement", measurement_epochs)]
                    else:
                        stages = [("measurement", measurement_epochs)]
                    for stage, epochs in stages:
                        for epoch in range(epochs):
                            total, count = 0.0, 0
                            individual_total, ensemble_total = 0.0, 0.0
                            for batch in train_loader:
                                optimizer.zero_grad(set_to_none=True)
                                _, _, h, phi, s = batch_values(batch)
                                if multi:
                                    individual_loss = F.mse_loss(s, phi[:, None].expand_as(s))
                                    ensemble_loss = F.mse_loss(s.mean(dim=1), phi)
                                    loss = (1 - ensemble_weight) * individual_loss + ensemble_weight * ensemble_loss
                                    individual_total += individual_loss.item() * len(phi)
                                    ensemble_total += ensemble_loss.item() * len(phi)
                                else:
                                    prediction = h.mean(dim=1) if stage == "feature" else s
                                    loss = F.mse_loss(prediction, phi)
                                if not torch.isfinite(loss):
                                    raise RuntimeError("Non-finite property preparation loss.")
                                loss.backward()
                                optimizer.step()
                                with torch.no_grad():
                                    for parameter, reference, radius in weight_constraints:
                                        if radius == 0:
                                            parameter.copy_(reference)
                                        else:
                                            delta = parameter - reference
                                            norm = delta.norm().item()
                                            if norm > radius:
                                                parameter.copy_(reference + delta * (radius / norm))
                                total += loss.item() * len(phi)
                                count += len(phi)
                            history.append({"stage": stage, "epoch": epoch + 1, "mse": total / count})
                            loss_details = ""
                            if multi:
                                history[-1].update({"individual_mse": individual_total / count,
                                                    "ensemble_mse": ensemble_total / count})
                                loss_details = (f" | individual MSE={individual_total / count:.6f}"
                                                f" | ensemble MSE={ensemble_total / count:.6f}")
                            print(f"[Poison] {stage} epoch {epoch + 1}/{epochs} | per-window MSE={total / count:.6f}{loss_details}", flush=True)
                    del optimizer
                for name, p in model.named_parameters():
                    p.requires_grad_(original_flags[name])
                model.zero_grad(set_to_none=True)
                # The constant baseline uses PREPARATION labels only. No holdout
                # information is used for training, epoch selection, or baseline fitting.
                label_sum, label_count = 0.0, 0
                with torch.no_grad():
                    for batch in loader(preparation, batch_size):
                        phi = properties(batch[0].float().to(device))
                        label_sum += phi.double().sum().item()
                        label_count += len(phi)
                constant = label_sum / label_count
                ordering = torch.randperm(len(holdout), generator=torch.Generator().manual_seed(seed)).tolist()
                report = {"design": metadata, "checkpoint_source": "loaded" if load_checkpoint else "trained",
                          "preparation_history": history, "constant_property_baseline": constant,
                          "validation_batch_sizes": validation_sizes, "validation_seed": seed,
                          "validation_max_batches": max_validation_batches, "tolerance_fraction": tolerance,
                          "float32_matmul_precision": torch.get_float32_matmul_precision(),
                          "raw_gradient_only": True, "results": []}
                for size in validation_sizes:
                    n_batches = min(max_validation_batches, len(holdout) // size)
                    validation = loader(Subset(holdout, ordering[:n_batches * size]), size)
                    rows = []
                    for batch_index, batch in enumerate(validation):
                        model.zero_grad(set_to_none=True)
                        outputs, targets, h, phi, s = batch_values(batch)
                        loss = F.mse_loss(outputs, targets, reduction="mean")
                        loss.backward()  # Actual unchanged client loss, no optimizer step.
                        named = dict(model.named_parameters())
                        gradient = named[weight_name].grad
                        if gradient is None or not torch.isfinite(gradient).all():
                            raise RuntimeError(f"Missing or non-finite actual client gradient for {weight_name}.")
                        if multi:
                            gradients = gradient[row_indices, hidden_indices].detach().cpu().double().numpy()
                            individual_estimates = (gradients - beta) / alpha
                            estimate = float(individual_estimates.mean())
                            residual = outputs.index_select(2, loss_indices) - targets.index_select(2, loss_indices)
                            contributions = (2.0 / K) * (residual * h).mean(dim=1)
                            raw_analytics = contributions.mean(dim=0).detach().cpu().double().numpy()
                            analytics = s.mean(dim=0).detach().cpu().double().numpy()
                            for q, coordinate in enumerate(coordinates):
                                if not np.isclose(gradients[q], raw_analytics[q], rtol=1e-4, atol=1e-5):
                                    raise RuntimeError(f"Raw gradient identity failed at {coordinate}: "
                                                       f"gradient={gradients[q]}, analytic={raw_analytics[q]}.")
                                if not np.isclose(individual_estimates[q], analytics[q], rtol=1e-4, atol=1e-5):
                                    raise RuntimeError(f"Decoded gradient identity failed at {coordinate}: "
                                                       f"estimate={individual_estimates[q]}, analytic={analytics[q]}.")
                        else:
                            g = gradient[k, j].item()
                        if poison_strategy == "constant_bias":
                            estimate = K * g / (2 * C)
                        elif not multi:
                            estimate = (g - beta) / alpha
                            residual = outputs[..., loss_channel] - targets[..., loss_channel]
                            gradient_contribution = (2.0 / K) * (residual * h).mean(dim=1)
                            raw_analytic = gradient_contribution.mean().item()
                            if not np.isclose(g, raw_analytic, rtol=1e-4, atol=1e-5):
                                raise RuntimeError(f"Raw gradient identity failed: gradient={g}, analytic={raw_analytic}.")
                        analytic = s.mean().item()
                        if not np.isclose(estimate, analytic, rtol=1e-4, atol=1e-5):
                            raise RuntimeError(f"Gradient identity failed for {coordinates}: decoded={estimate}, analytic={analytic}. "
                                               "On CUDA, retry with torch.set_float32_matmul_precision('highest') "
                                               "to rule out reduced-precision matrix multiplication.")
                        truth = phi.mean().item()
                        rows.append({"batch": batch_index, "estimate": estimate,
                                     "true_property": truth, "analytic_measurement": analytic,
                                     "identity_error": abs(estimate - analytic),
                                     "absolute_error": abs(estimate - truth), "constant_error": abs(constant - truth)})
                        if multi:
                            rows[-1].update({"coordinates": coordinates, "gradient_scalars": gradients.tolist(),
                                             "individual_estimates": individual_estimates.tolist(),
                                             "raw_analytic_gradients": raw_analytics.tolist(),
                                             "analytic_measurements": analytics.tolist(),
                                             "raw_identity_errors": np.abs(gradients - raw_analytics).tolist(),
                                             "coordinate_identity_errors": np.abs(individual_estimates - analytics).tolist()})
                            gradient_description = f"gradients={gradients.tolist()}"
                        else:
                            rows[-1].update({"gradient_scalar": g, "feature_average": h.mean().item()})
                            gradient_description = f"gradient={g:.8g}"
                        if poison_strategy == "constant_bias":
                            rows[-1]["target_correction"] = (targets[..., loss_channel] * h).mean().item() / C
                        elif not multi:
                            rows[-1]["raw_analytic_gradient"] = raw_analytic
                            rows[-1]["raw_identity_error"] = abs(g - raw_analytic)
                        print(f"[Poison feasibility] B={size} | batch {batch_index + 1}/{n_batches} "
                              f"| original property={truth:.6f} | gradient estimate={estimate:.6f} "
                              f"| estimate - original={estimate - truth:+.6f} "
                              f"| absolute error={abs(estimate - truth):.6f} | {gradient_description}", flush=True)
                    errors = np.array([row["estimate"] - row["true_property"] for row in rows])
                    truths = np.array([row["true_property"] for row in rows])
                    estimates = np.array([row["estimate"] for row in rows])
                    baseline_errors = constant - truths
                    result = {"batch_size": size, "batches": n_batches,
                              "mae": float(np.abs(errors).mean()), "rmse": float(np.sqrt(np.square(errors).mean())),
                              "bias": float(errors.mean()), "true_property_std": float(truths.std()),
                              "estimate_std": float(estimates.std()),
                              "correlation": (float(np.corrcoef(estimates, truths)[0, 1])
                                              if estimates.std() > 1e-8 and truths.std() > 1e-8 else None),
                              "constant_mae": float(np.abs(baseline_errors).mean()),
                              "constant_rmse": float(np.sqrt(np.square(baseline_errors).mean())),
                              "max_identity_error": max(row["identity_error"] for row in rows), "rows": rows}
                    if multi:
                        coordinate_errors = np.array([row["individual_estimates"] for row in rows]) - truths[:, None]
                        summaries = []
                        for q, coordinate in enumerate(coordinates):
                            error = coordinate_errors[:, q]
                            summaries.append({"coordinate": coordinate, "mae": float(np.abs(error).mean()),
                                              "rmse": float(np.sqrt(np.square(error).mean())), "bias": float(error.mean())})
                        summaries.append({"coordinate": "ensemble", "mae": result["mae"],
                                          "rmse": result["rmse"], "bias": result["bias"]})
                        result["coordinate_summary"] = summaries
                        # Correlate signed ERRORS, not predictions. A constant-error
                        # coordinate (or fewer than two batches) has undefined correlation.
                        correlations = [[None for _ in coordinates] for _ in coordinates]
                        if n_batches >= 2:
                            for q in range(M):
                                for r in range(M):
                                    if np.ptp(coordinate_errors[:, q]) > 0 and np.ptp(coordinate_errors[:, r]) > 0:
                                        value = float(np.corrcoef(coordinate_errors[:, q], coordinate_errors[:, r])[0, 1])
                                        correlations[q][r] = value if np.isfinite(value) else None
                        result["signed_error_correlation"] = {"coordinates": coordinates, "matrix": correlations}
                    result["beats_constant"] = result["mae"] < result["constant_mae"]
                    result["within_tolerance"] = result["mae"] <= tolerance
                    report["results"].append(result)
                    print(f"[Poison feasibility] B={size} | batches={n_batches} | gradient MAE={result['mae']:.6f} "
                          f"RMSE={result['rmse']:.6f} | constant MAE={result['constant_mae']:.6f} "
                          f"| true property std={result['true_property_std']:.6f} "
                          f"| beats constant={result['beats_constant']} | within {tolerance:g}={result['within_tolerance']} "
                          f"| max identity error={result['max_identity_error']:.3e}", flush=True)
                    if result["true_property_std"] < 1e-6:
                        print("[Poison feasibility] Little/no held-out batch-property variation; low MAE alone is not evidence of informative leakage.", flush=True)
        finally:
            handle.remove()
            captured.clear()
            for name, p in model.named_parameters():
                p.requires_grad_(original_flags[name])
            model.zero_grad(set_to_none=True)
            model.eval()
        if poison_strategy == "constant_bias":
            if torch.count_nonzero(projection.weight[k]).item() or projection.bias[k].item() != projection.bias.new_tensor(C).item():
                raise RuntimeError("Measurement row or bias changed during preparation/validation.")
        else:
            if not torch.equal(projection.weight, projection_reference["weight"]) or not torch.equal(projection.bias, projection_reference["bias"]):
                raise RuntimeError("Complete projection weight or bias changed during preparation/validation.")
        if not load_checkpoint:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {"model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                          "poisoning": metadata}
            if poison_strategy == "fixed_projection":
                checkpoint["projection_reference"] = {name: value.cpu() for name, value in projection_reference.items()}
            torch.save(checkpoint, checkpoint_path)
        output_dir = Path(config.get("poison_feasibility_output_dir") or Path(__file__).resolve().parents[3] / "evaluation_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{checkpoint_path.stem}_feasibility.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        parameter_description = f"`{weight_name}` at {coordinates}" if multi else f"`{weight_name}[{k}, {j}]`"
        lines = [f"# Weights-only property measurement: {config['dataset']} ({poison_strategy})", "",
                 f"Parameter: {parameter_description}; K={K}, horizon={dataset.pred_len}, {decoding}.", "",
                 "Raw mean-MSE gradients; estimates are not clipped. This is an auxiliary-data feasibility test, not a guarantee of target-client leakage.", ""]
        lines += self._poison_feasibility_batch_tables(report)
        lines += ["## Aggregate diagnostics", "",
                 "| Batch size | Batches | Gradient MAE | Gradient RMSE | Bias | Constant MAE | Constant RMSE | True property SD | Estimate SD | Correlation | Beats constant | Within tolerance | Max identity error |",
                 "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|"]
        for row in report["results"]:
            correlation = "undefined" if row["correlation"] is None else f"{row['correlation']:.6f}"
            lines.append(f"| {row['batch_size']} | {row['batches']} | {row['mae']:.6f} | {row['rmse']:.6f} | {row['bias']:.6f} "
                         f"| {row['constant_mae']:.6f} | {row['constant_rmse']:.6f} | {row['true_property_std']:.6f} "
                         f"| {row['estimate_std']:.6f} | {correlation} "
                         f"| {row['beats_constant']} | {row['within_tolerance']} | {row['max_identity_error']:.3e} |")
        tuning_constants = "C" if poison_strategy == "constant_bias" else "alpha or beta"
        lines += ["", "All error metrics use fractions (multiply by 100 for percentage points). Windows within a partition may overlap; preparation and holdout share no input or target timesteps.",
                  f"The constant is fitted on preparation labels only. The holdout is used only after training; it is not used to tune {tuning_constants}, choose epochs, or select a checkpoint.", "",
                  "## Preparation settings", "", "```json", json.dumps(metadata, indent=2), "```", ""]
        report_path.with_suffix(".md").write_text("\n".join(lines))
        self.poison_feasibility_results = report
        config.update({"poison_method": metadata["method"], "poison_strategy": poison_strategy,
                       "poison_model_source": report["checkpoint_source"],
                       "poison_checkpoint_path": str(checkpoint_path.resolve()),
                       "poison_target_channel_name": channels[c], "poison_property_units": units,
                       "poison_measurement_count": M, "poison_gradient_parameter_name": weight_name,
                       "poison_gradient_parameter_index": parameter_names.index(weight_name),
                       "poison_output_channels_in_loss": K,
                       "poison_loss_elements_per_window": m, "poison_feature_epochs": feature_epochs,
                       "poison_train_epochs": measurement_epochs,
                       "poison_feasibility_report": str(report_path.resolve()),
                       "num_learn_epochs": 0, "load_lti_model": False, "one_shot_targets": False})
        if multi:
            for key in ["poison_measurement_channel", "poison_hidden_coordinate", "poison_gradient_coordinate"]:
                config.pop(key, None)
        else:
            config.update({"poison_measurement_channel": channels[k], "poison_hidden_coordinate": j,
                           "poison_gradient_coordinate": [k, j]})
        if poison_strategy == "constant_bias":
            config["poison_measurement_C"] = C
        else:
            config.update({"poison_gradient_scale": alpha, "poison_gradient_offset": beta,
                           "poison_max_relative_weight_change": rho,
                           "poison_gradient_coordinates": [{"channel": channels[row], "hidden_coordinate": hidden}
                                                           for row, hidden in coordinates]})
            if multi:
                config["poison_ensemble_loss_weight"] = ensemble_weight
        print(f"[Poison] Feasibility report: {report_path.with_suffix('.md')} | model checkpoint: {checkpoint_path}", flush=True)
        return model

    @staticmethod
    def _poison_feasibility_batch_tables(report):
        """Render individual held-out observations before aggregate diagnostics."""
        multi = "coordinates" in report["design"]
        if report["design"].get("strategy", "constant_bias") == "fixed_projection":
            formula = "(gradient - beta) / alpha"
        else:
            formula = "K × gradient / (2 × C)"
        if multi:
            estimate_description = ("The estimate is the arithmetic mean of the individually decoded gradients: "
                                    f"mean({formula}), in coordinate order {report['design']['coordinates']}.")
        else:
            estimate_description = f"The estimate uses only the selected gradient scalar: {formula}."
        lines = ["## Held-out batches: original property versus gradient estimate", "",
                 "Batch size is the number of input windows in one client gradient. Each row below is one separately computed batch gradient.", "",
                 "The original property is the mean of the exact per-window property values in that batch. "
                 + estimate_description, "",
                 "Property values and errors are fractions: 0.25 means 25%; an error of 0.03 means 3 percentage points. "
                 "A positive signed error means overestimation. Batch numbers start at 1 within each batch size.", ""]
        for result in report["results"]:
            gradient_column = "Gradient scalars (coordinate order)" if multi else "Gradient scalar"
            lines += [f"### Batch size {result['batch_size']}: {len(result['rows'])} held-out batches", "",
                      f"| Batch | Original property | Gradient estimate | Estimate − original | Absolute error | {gradient_column} |",
                      "|---:|---:|---:|---:|---:|---:|"]
            for row in result["rows"]:
                original, estimate = row["true_property"], row["estimate"]
                gradient_text = ("[" + ", ".join(f"{g:.8g}" for g in row["gradient_scalars"]) + "]"
                                 if multi else f"{row['gradient_scalar']:.8g}")
                lines.append(f"| {row['batch'] + 1} | {original:.6f} | {estimate:.6f} "
                             f"| {estimate - original:+.6f} | {abs(estimate - original):.6f} "
                             f"| {gradient_text} |")
            lines.append("")
            if multi:
                lines += ["Individual coordinates and ensemble (errors in property fractions):", "",
                          "| Coordinate | MAE | RMSE | Signed bias |", "|:---|---:|---:|---:|"]
                for row in result["coordinate_summary"]:
                    lines.append(f"| {row['coordinate']} | {row['mae']:.6f} | {row['rmse']:.6f} | {row['bias']:+.6f} |")
                correlation = result["signed_error_correlation"]
                labels = [str(q) for q in correlation["coordinates"]]
                lines += ["", "Pairwise signed estimation-error correlations (undefined when insufficient or constant):", "",
                          "| Coordinate | " + " | ".join(labels) + " |",
                          "|:---|" + "---:|" * len(labels)]
                for label, values in zip(labels, correlation["matrix"]):
                    cells = ["undefined" if value is None else f"{value:.6f}" for value in values]
                    lines.append(f"| {label} | " + " | ".join(cells) + " |")
                lines += ["", "Averaging is not assumed to improve accuracy; compare the individual and ensemble errors above.", ""]
        return lines

    def _property_percentile_attack_loader(self, tr_dataloader, config):
        """Select full candidate batches by their mean target-channel property.

        Candidates are a seeded random partition of the benign training windows.
        Only indices and scores are retained, then selected batches are replayed
        with their original inputs, forecast labels, and timestamps.
        """
        dataset = self.train_datasets[0]
        channels = ([dataset.target] if dataset.features == "S" else list(
            pd.read_csv(Path(dataset.root_path) / dataset.data_path, nrows=0).columns[1:]
        ))
        requested = config.get("attack_sampling_target_channel", config.get("poison_target_channel"))
        if isinstance(requested, str) and requested in channels:
            channel = channels.index(requested)
        elif type(requested) is int and 0 <= requested < len(channels):
            channel = requested
        else:
            raise ValueError(f"attack_sampling_target_channel must be a name in {channels} or a zero-based index.")
        threshold = config.get("attack_sampling_threshold", config.get("poison_property_threshold", 0.5))
        duration = config.get("attack_sampling_duration", config.get("poison_property_duration", 1))
        seed = config.get("attack_sampling_seed", config["seed"])
        if not np.isfinite(threshold):
            raise ValueError("attack_sampling_threshold must be finite.")
        if type(duration) is not int or not 1 <= duration <= dataset.seq_len:
            raise ValueError("attack_sampling_duration must be between 1 and seq_len.")
        batch_size = tr_dataloader.batch_size
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Property sampling requires a DataLoader with a fixed batch_size.")
        record_count = config["number_of_batches"]
        attack_count = config.get("attack_number_of_batches", record_count)
        if (type(record_count) is not int or type(attack_count) is not int
                or not 1 <= attack_count <= record_count):
            raise ValueError("Property sampling requires 1 <= attack_number_of_batches <= number_of_batches.")
        available = len(tr_dataloader.dataset) // batch_size
        candidate_count = config.get("attack_sampling_candidate_batches")
        if candidate_count is None:
            candidate_count = available
        elif type(candidate_count) is not int or candidate_count < 1:
            raise ValueError("attack_sampling_candidate_batches must be a positive integer or None.")
        candidate_count = min(candidate_count, available)
        if candidate_count < record_count:
            raise ValueError(
                f"Property sampling has {candidate_count} full candidate batches, but needs {record_count}. "
                "Increase attack_sampling_candidate_batches or reduce number_of_batches/batch_size."
            )
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(len(tr_dataloader.dataset), generator=generator)
        candidates = permutation[:candidate_count * batch_size].reshape(candidate_count, batch_size).tolist()
        loader = DataLoader(
            tr_dataloader.dataset, batch_sampler=candidates,
            collate_fn=tr_dataloader.collate_fn,
            generator=torch.Generator().manual_seed(seed),
        )
        print(f"[Attack sampling] Scoring {candidate_count}/{available} full candidate batches "
              f"of {batch_size} windows | seed={seed} | channel={channels[channel]} "
              f"| threshold>{threshold:g} | minimum duration={duration}", flush=True)
        print(f"[Attack sampling] Property is the mean per-window elevated fraction in input windows; "
              f"units are preprocessed. Excluding {len(tr_dataloader.dataset) % batch_size} remainder windows.", flush=True)
        scores = []
        report_every = max(1, candidate_count // 10)
        with torch.no_grad():
            for index, batch in enumerate(loader):
                inputs = batch[0]
                if inputs.ndim != 3 or inputs.shape[-1] != len(channels):
                    raise ValueError("Candidate input shape does not match the dataset channels.")
                values = inputs[..., channel]
                if not torch.isfinite(values).all():
                    raise ValueError(f"Candidate batch {index} has non-finite target-channel values.")
                elevated = values > threshold
                if duration > 1:
                    qualifying = elevated.unfold(1, duration, 1).all(dim=-1)
                    elevated = F.conv_transpose1d(
                        qualifying.float().unsqueeze(1),
                        torch.ones(1, 1, duration, device=values.device),
                    ).squeeze(1) > 0
                score = elevated.double().mean(dim=1).mean().item()
                scores.append(score)
                if (index + 1) % report_every == 0 or index + 1 == candidate_count:
                    print(f"[Attack sampling] Scored {index + 1}/{candidate_count} batches "
                          f"| latest score={score:.6f} | range so far=[{min(scores):.6f}, {max(scores):.6f}]", flush=True)
        scores = np.asarray(scores)
        order = np.argsort(scores, kind="stable")
        percentiles = np.arange(0, 101, 10)
        percentile_scores = np.percentile(scores, percentiles)
        print(f"[Attack sampling] Score distribution: min={scores.min():.6f}, max={scores.max():.6f}, "
              f"mean={scores.mean():.6f}, std={scores.std():.6f}, unique={len(np.unique(scores))}", flush=True)
        print("[Attack sampling] Percentile | property score | fraction chart (0 to 1)", flush=True)
        for percentile, score in zip(percentiles, percentile_scores):
            print(f"[Attack sampling] P{percentile:03d} | {score:.6f} | {'#' * int(round(score * 40))}", flush=True)
        if scores.min() == scores.max():
            print("[Attack sampling] WARNING: all candidate scores are identical; percentile selection "
                  "cannot add property variation. Consider another threshold/duration or candidate pool.", flush=True)

        def spread_indices(size, count):
            return np.rint(np.linspace(0, size - 1, count)).astype(int) if count > 1 else np.array([(size - 1) // 2])

        # The first attack_count entries are the batches actually attacked by
        # start_attack. Spread those across the FULL pool even if extra batches
        # were requested just for gradient recording.
        attack_ranks = spread_indices(candidate_count, attack_count).tolist()
        ranks = list(attack_ranks)
        if record_count > attack_count:
            chosen_ranks = set(attack_ranks)
            remaining = [rank for rank in range(candidate_count) if rank not in chosen_ranks]
            ranks += [remaining[i] for i in spread_indices(len(remaining), record_count - attack_count)]
        selected = []
        print(f"[Attack sampling] Selected {attack_count} attack batches in ascending score order "
              f"and {record_count - attack_count} additional recording batches. Rank percentiles may share tied scores.", flush=True)
        for batch_number, rank in enumerate(ranks):
            candidate = int(order[rank])
            rank_percentile = 100 * rank / (candidate_count - 1) if candidate_count > 1 else 50.0
            entry = {
                "batch_index": batch_number, "candidate_index": candidate,
                "rank_percentile": rank_percentile, "property_score": float(scores[candidate]),
                "window_indices": candidates[candidate], "attacked": batch_number < attack_count,
            }
            selected.append(entry)
            print(f"[Attack sampling] Batch {batch_number} ({'attack' if entry['attacked'] else 'record only'}) "
                  f"<- candidate {candidate} | rank P{rank_percentile:.2f} | score={scores[candidate]:.6f} "
                  f"| window indices={candidates[candidate]}", flush=True)
        config.update({
            "attack_sampling_target_channel": channels[channel], "attack_sampling_threshold": threshold,
            "attack_sampling_duration": duration, "attack_sampling_seed": seed,
            "attack_sampling_candidate_batches": candidate_count,
            "attack_sampling_percentiles": [
                {"percentile": int(p), "property_score": float(v)} for p, v in zip(percentiles, percentile_scores)
            ],
            "attack_sampling_selected_batches": selected,
        })
        return DataLoader(
            tr_dataloader.dataset, batch_sampler=[entry["window_indices"] for entry in selected],
            collate_fn=tr_dataloader.collate_fn, generator=torch.Generator().manual_seed(seed),
        )

    def run_federated_training(self, model, model_train_loader, model_val_loader, config,
                               num_rounds, malicious_poisoning=False):
        """Shared-loader FL with FedAvg or attachment-style FLAME on model deltas.

        Model states are submissions. Subsequent ordinary client gradients are
        separately observed; model deltas are never decoded as gradients.
        """
        if config.get('fl_training_defense', 'none') not in {None, 'none'}:
            raise ValueError('FedAvg training defenses are unsupported; fl_training_defense must be none. Recorded-gradient defenses remain separate.')
        aggregation = config.get('fl_aggregation', 'fedavg')
        if aggregation not in {'fedavg', 'flame'}:
            raise ValueError('fl_aggregation must be fedavg or flame.')
        aggregation_label = 'FedAvg' if aggregation == 'fedavg' else 'FLAME'
        flame_sigma = config.get('flame_sigma', 0.0001)
        if aggregation == 'flame':
            if not np.isfinite(flame_sigma) or flame_sigma < 0:
                raise ValueError('flame_sigma must be finite and nonnegative.')
            try:
                import hdbscan
                from sklearn.metrics.pairwise import cosine_distances
            except ImportError as exc:
                raise ImportError('FLAME requires hdbscan and scikit-learn; install ts-inverse[flame].') from exc
        clients = config.get('num_clients', 10)
        steps, lr = config.get('local_steps', 1), config.get('local_lr', 1e-3)
        if type(clients) is not int or clients < 1 or type(num_rounds) is not int or num_rounds < 0:
            raise ValueError('Use positive num_clients and nonnegative integer model_train_epochs.')
        if type(steps) is not int or steps < 1 or not np.isfinite(lr) or lr <= 0:
            raise ValueError('Use positive local_steps and finite positive local_lr.')
        if aggregation == 'flame' and clients < 2:
            raise ValueError('FLAME requires at least two clients.')
        device = next(model.parameters()).device
        dataset = self.train_datasets[0]
        pred_len, label_len = dataset.pred_len, dataset.label_len
        f_dim = -1 if dataset.features == 'MS' else 0
        names = [name for name, p in model.named_parameters() if p.requires_grad]
        flags = {name: p.requires_grad for name, p in model.named_parameters()}
        print(f'[{aggregation_label}] Shared-loader simulation; no server optimizer. '
              f'Aggregation={aggregation}; recorded-gradient defense={config.get("defense_name", "none")}.', flush=True)

        def flame_weights(submissions, start):
            # Match the supplied implementation's normalized clipping weights,
            # using model deltas (the FL payload here), not observed gradients.
            flat = np.stack([
                torch.cat([(state[name] - start[name]).reshape(-1).double().cpu()
                           for name in names]).numpy()
                for state in submissions
            ])
            distances = cosine_distances(flat).astype(np.float64)
            labels = hdbscan.HDBSCAN(
                min_cluster_size=clients // 2 + 1, min_samples=1,
                allow_single_cluster=True, metric='precomputed',
            ).fit_predict(distances)
            norms = np.linalg.norm(flat, axis=1) + 1e-12
            median_norm = float(np.median(norms))
            clip_factors = np.minimum(1., median_norm / norms)
            weights = np.where(labels != -1, clip_factors, 0.)
            fallback = bool(weights.sum() <= 0)
            if fallback:
                weights = np.ones(clients) / clients
            else:
                weights /= weights.sum()
            # Deliberately retain the attachment's squared noise multiplier.
            noise_std = float((flame_sigma * median_norm) ** 2)
            diagnostics = {'cluster_labels': labels.tolist(), 'update_norms': norms.tolist(),
                           'median_norm': median_norm, 'clip_factors': clip_factors.tolist(),
                           'all_rejected_fallback': fallback, 'noise_std': noise_std,
                           'accepted_client_ids': np.flatnonzero(weights > 0).tolist()}
            return weights.tolist(), diagnostics

        def forward_loss(local, batch):
            x, y, xm, ym = (v.float().to(device) for v in batch)
            decoder = torch.cat([y[:, :label_len], torch.zeros_like(y[:, -pred_len:])], dim=1)
            output = local(x, xm, decoder, ym)
            output = output[0] if isinstance(output, (tuple, list)) else output
            output, target = output[:, -pred_len:, f_dim:], y[:, -pred_len:, f_dim:]
            if output.shape != target.shape:
                raise ValueError('ETT forecast and target shapes differ in FedAvg.')
            return F.mse_loss(output, target), target.numel()

        def checked_gradients(local):
            for name, p in local.named_parameters():
                if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
                    raise RuntimeError(f'Missing or non-finite local gradient for {name!r}.')

        malicious_ids = set()
        report = {'preparation_origin': f'shared_loader_{aggregation}', 'aggregation': aggregation,
                  'flame_sigma': flame_sigma if aggregation == 'flame' else None, 'training_defense': 'none',
                  'recorded_gradient_defense': config.get('defense_name', 'none'), 'rounds': []}
        if malicious_poisoning:
            if config.get('poison_feasibility_only', False):
                raise ValueError('FL poisoning does not support poison_feasibility_only=True.')
            if config.get('poison_strategy', 'fixed_projection') != 'fixed_projection':
                raise ValueError('FL poisoning supports fixed_projection only; constant_bias remains centralized.')
            if config.get('poison_base_checkpoint_path') or config.get('poison_checkpoint_path'):
                raise ValueError('FL poisoning does not load centralized base/poison checkpoints; remove checkpoint paths.')
            fraction = config.get('fl_malicious_fraction', .1)
            if not np.isfinite(fraction) or not 0 < fraction < 1:
                raise ValueError('fl_malicious_fraction must be finite and strictly between 0 and 1.')
            count = int(np.ceil(clients * fraction))
            if not 1 <= count < clients:
                raise ValueError('FL poisoning needs at least one malicious and one honest client.')
            seed = config.get('fl_poison_seed', config.get('seed', 0))
            malicious_ids = set(np.random.default_rng(seed).choice(clients, count, replace=False).tolist())
            poison_steps = config.get('fl_poison_local_steps', config.get('local_steps', 1))
            poison_optimizer = config.get('fl_poison_optimizer', 'sgd')
            poison_start_round = config.get('fl_poison_start_round', 1)
            if poison_optimizer not in {'sgd', 'adam'}:
                raise ValueError('fl_poison_optimizer must be sgd or adam.')
            if type(poison_start_round) is not int or poison_start_round < 1:
                raise ValueError('fl_poison_start_round must be a positive integer (rounds start at 1).')
            poison_lr = config.get('fl_poison_learning_rate', config.get('poison_train_learning_rate', config.get('local_lr', 1e-3)))
            poison_size = config.get('fl_poison_batch_size', config.get('poison_train_batch_size', config.get('model_train_batch_size', 32)))
            eval_batches = config.get('fl_poison_eval_batches', config.get('poison_validation_batches', 8))
            if any(type(n) is not int or n < 1 for n in [poison_steps, poison_size, eval_batches]) or not np.isfinite(poison_lr) or poison_lr <= 0:
                raise ValueError('FL poison steps/batch sizes/evaluation batches and learning rate must be positive.')
            channel_names = ([dataset.target] if dataset.features == 'S' else list(pd.read_csv(Path(dataset.root_path) / dataset.data_path, nrows=0).columns[1:]))
            setup_config = dict(config)
            setup_config.setdefault('seed', 0)
            setup_config.setdefault('device', str(device))
            setup_config.setdefault('poison_target_channel', 'HUFL' if 'HUFL' in channel_names else channel_names[0])
            # This is validation/subset construction only, never centralized preparation.
            setup_config.update(poison_feasibility_only=True, poison_train_batch_size=poison_size,
                                poison_train_learning_rate=poison_lr, poison_seed=seed,
                                poison_feature_epochs=0, poison_train_epochs=1)
            setup = self._prepare_property_measurement(model, setup_config)
            design = dict(setup['metadata'], preparation_origin=f'malicious_client_{aggregation}', fl_aggregation=aggregation,
                          method='fl_fixed_projection_measurement_v1')
            design.update(fl_malicious_fraction=fraction, malicious_client_ids=sorted(malicious_ids),
                          fl_poison_local_steps=poison_steps, fl_poison_learning_rate=poison_lr,
                          fl_poison_optimizer=poison_optimizer, fl_poison_start_round=poison_start_round,
                          training_stages=[["local_property_steps", poison_steps]])
            for key in ['feature_epochs', 'measurement_epochs']:
                design.pop(key, None)
            setup['metadata'] = design
            auxiliary = DataLoader(setup['preparation'], batch_size=poison_size, shuffle=True,
                                   generator=torch.Generator().manual_seed(seed))
            auxiliary_iterator = iter(auxiliary)
            holdout = DataLoader(setup['holdout'], batch_size=poison_size, shuffle=True,
                                 generator=torch.Generator().manual_seed(seed))
            # Fixed holdout batches, never consumed by local training.
            evaluation_batches = []
            for batch in holdout:
                evaluation_batches.append(batch)
                if len(evaluation_batches) >= eval_batches:
                    break
            baseline_loader = DataLoader(setup['preparation'], batch_size=poison_size,
                                         generator=torch.Generator().manual_seed(seed))
            labels = [self._property_labels(batch[0].float(), design).double() for batch in baseline_loader]
            baseline = torch.cat(labels).mean().item()
            config.update(poison_method=design['method'], poison_strategy='fixed_projection',
                          poison_gradient_parameter_name=design['weight_name'],
                          poison_gradient_parameter_index=names.index(design['weight_name']),
                          poison_output_channels_in_loss=1 if dataset.features == 'MS' else len(channel_names),
                          preparation_origin=f'malicious_client_{aggregation}', fl_aggregation=aggregation)
            report.update(design=design, requested_malicious_fraction=fraction, malicious_count=count,
                          realized_malicious_fraction=count / clients, malicious_client_ids=sorted(malicious_ids),
                          constant_property_baseline=baseline)
            evaluator_client = min(malicious_ids)
            report.update(evaluator_client_id=evaluator_client, requested_rounds=num_rounds)
            root = Path(config.get('property_inference_output_dir') or Path(__file__).resolve().parents[3] / 'evaluation_results')
            root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d_%H%M%S_%f%z')
            path = root / f'{config.get("dataset", "ETT")}_fedavg_{stamp}.json'
            config['fl_poison_report'] = str(path)
            print(f'[{aggregation_label} poison] requested fraction={fraction:g}, count={count}/{clients}, realized={count / clients:g}, IDs={sorted(malicious_ids)}', flush=True)
            print(f'[{aggregation_label} poison] local optimizer={poison_optimizer}; starts at round {poison_start_round}. '
                  'Before that, all clients use honest forecasting SGD. Local optimizer state resets for each client each round.', flush=True)

        def evaluate_property(round_index):
            # One designated malicious client observes ordinary MSE gradients on
            # its held-out auxiliary data using a fresh copy of the global model.
            # This copy is never trained or submitted to aggregation.
            evaluation_model = deepcopy(model)
            evaluation_model.eval()
            observations = []
            print(f'[{aggregation_label} property] round={round_index}/{num_rounds} evaluator_client={evaluator_client} '
                  f'held-out batches={len(evaluation_batches)}; ordinary mean-MSE gradients, no local updates.', flush=True)
            for batch_number, batch in enumerate(evaluation_batches):
                loss, _ = forward_loss(evaluation_model, batch)
                gradients = torch.autograd.grad(loss, [p for p in evaluation_model.parameters() if p.requires_grad])
                x, y, xm, ym = (v.float().to(device) for v in batch)
                observations.append(self._evaluate_property_observation(
                    evaluation_model, design, baseline, batch_number, gradients, x, y[:, -pred_len:, f_dim:], xm, ym))
            errors = np.array([r['signed_error'] for r in observations])
            coordinates = design.get('coordinates', [design.get('coordinate')])
            per_coordinate = []
            for q, coordinate in enumerate(coordinates):
                e = np.array([r['measurements'][q]['signed_error'] for r in observations])
                per_coordinate.append({'coordinate': coordinate, 'mae': float(np.abs(e).mean()),
                                       'rmse': float(np.sqrt(np.square(e).mean())), 'bias': float(e.mean())})
            return {'round': round_index, 'evaluator_client_id': evaluator_client,
                    'mae': float(np.abs(errors).mean()), 'rmse': float(np.sqrt(np.square(errors).mean())),
                    'beats_constant': float(np.abs(errors).mean()) < float(np.mean([r['constant_error'] for r in observations])),
                    'constant_mae': float(np.mean([r['constant_error'] for r in observations])),
                    'coordinate_errors': per_coordinate, 'rows': observations}

        if malicious_poisoning:
            report['initial_property_evaluation'] = evaluate_property(0)
            path.write_text(json.dumps(report, indent=2) + '\n')
        iterator = iter(model_train_loader)
        for round_number in range(num_rounds):
            poison_active = malicious_poisoning and round_number + 1 >= poison_start_round
            start = {name: value.detach().clone() for name, value in model.state_dict().items()}
            submissions, sizes, honest_losses, poison_losses = [], [], [], []
            for client in range(clients):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(model_train_loader)
                    batch = next(iterator)
                sizes.append(len(batch[0]))  # Auxiliary volume and repeated steps never affect weight.
                local = deepcopy(model)
                local.load_state_dict(start, strict=True)
                if [name for name, p in local.named_parameters() if p.requires_grad] != names:
                    raise ValueError('Client trainable parameter ordering differs from global model.')
                if poison_active and client in malicious_ids:
                    local.eval()
                    projection = dict(local.named_modules())[design['weight_name'].rsplit('.', 1)[0]]
                    reference = {name: value.detach().clone() for name, value in projection.state_dict().items()}
                    captured = []
                    handle = projection.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
                    try:
                        projection.weight.requires_grad_(False)
                        projection.bias.requires_grad_(False)
                        parameters = [p for p in local.parameters() if p.requires_grad]
                        if poison_optimizer == 'adam':
                            local_optimizer = torch.optim.Adam(parameters, lr=poison_lr)
                        else:
                            local_optimizer = torch.optim.SGD(parameters, lr=poison_lr)
                        local_setup = dict(setup, projection=projection)
                        budget = setup['rho']
                        constraints = [(p, p.detach().clone(), budget * max(p.norm().item(), 1e-12)) for p in parameters] if budget is not None else []
                        for _ in range(poison_steps):
                            try:
                                poison_batch = next(auxiliary_iterator)
                            except StopIteration:
                                auxiliary_iterator = iter(auxiliary)
                                poison_batch = next(auxiliary_iterator)
                            local_optimizer.zero_grad(set_to_none=True)
                            _, _, _, phi, s = self._property_measurement_batch(local, poison_batch, local_setup, captured)
                            if setup['multi']:
                                loss = ((1 - setup['ensemble_weight']) * F.mse_loss(s, phi[:, None].expand_as(s))
                                        + setup['ensemble_weight'] * F.mse_loss(s.mean(1), phi))
                            else:
                                loss = F.mse_loss(s, phi)
                            if not torch.isfinite(loss):
                                raise RuntimeError('Non-finite malicious local objective.')
                            loss.backward()
                            checked_gradients(local)
                            local_optimizer.step()
                            with torch.no_grad():
                                for p, center, radius in constraints:
                                    delta = p - center
                                    norm = delta.norm().item()
                                    if radius == 0:
                                        p.copy_(center)
                                    elif norm > radius:
                                        p.copy_(center + delta * (radius / norm))
                            poison_losses.append(loss.item())
                        if any(not torch.equal(value, reference[name]) for name, value in projection.state_dict().items()):
                            raise RuntimeError('Malicious local projection changed.')
                    finally:
                        handle.remove()
                        captured.clear()
                        for name, p in local.named_parameters():
                            p.requires_grad_(flags[name])
                else:
                    local.train()
                    local_optimizer = torch.optim.SGD(local.parameters(), lr=lr)
                    for _ in range(steps):
                        local_optimizer.zero_grad(set_to_none=True)
                        loss, _ = forward_loss(local, batch)
                        loss.backward()
                        checked_gradients(local)
                        local_optimizer.step()
                    honest_losses.append(loss.item())
                submission = {name: value.detach().clone() for name, value in local.state_dict().items()}
                if submission.keys() != start.keys():
                    raise ValueError('Client state names differ from round-start state.')
                for name, value in submission.items():
                    if value.shape != start[name].shape or value.dtype != start[name].dtype or not torch.isfinite(value).all():
                        raise ValueError(f'Invalid FedAvg state entry {name!r}.')
                    if name not in names and not torch.equal(value, start[name]):
                        raise ValueError(f'Unsupported mutable buffer/frozen parameter {name!r}; no aggregation policy defined.')
                    if name in names and not value.is_floating_point():
                        raise ValueError(f'FedAvg requires floating trainable parameter {name!r}.')
                submissions.append(submission)
                if config.get('debug_fl', False):
                    print(f'[{aggregation_label} debug] round={round_number + 1} client={client} assigned_size={sizes[-1]} poisoning={poison_active and client in malicious_ids}')
            if any(size <= 0 for size in sizes):
                raise ValueError('FedAvg assigned client batches must be nonempty.')
            if aggregation == 'fedavg':
                weights = [size / sum(sizes) for size in sizes]
            else:
                weights, flame_diagnostics = flame_weights(submissions, start)
                # Before the poisoning start round, selected clients train honestly.
                malicious_weights = [{'client_id': client, 'weight': weights[client]}
                                     for client in sorted(malicious_ids) if poison_active]
                malicious_accepted = sum(entry['weight'] > 0 for entry in malicious_weights)
                malicious_weight_total = sum(entry['weight'] for entry in malicious_weights)
                flame_diagnostics.update(malicious_update_weights=malicious_weights,
                                         malicious_updates_aggregated=malicious_accepted,
                                         malicious_aggregation_weight=malicious_weight_total)
                print(f'[FLAME] round={round_number + 1}/{num_rounds} '
                      f'accepted={flame_diagnostics["accepted_client_ids"]} '
                      f'malicious_aggregated={malicious_accepted}/{len(malicious_weights)} '
                      f'malicious_weights={malicious_weights} '
                      f'malicious_total_weight={malicious_weight_total:.6g} '
                      f'fallback={flame_diagnostics["all_rejected_fallback"]} '
                      f'median_norm={flame_diagnostics["median_norm"]:.6g} '
                      f'noise_std={flame_diagnostics["noise_std"]:.6g}', flush=True)
            with torch.no_grad():
                averaged = dict(start)
                for name in names:
                    if aggregation == 'fedavg':
                        averaged[name] = sum(submission[name] * weight for submission, weight in zip(submissions, weights))
                    else:
                        delta = sum((state[name] - start[name]) * weight
                                    for state, weight in zip(submissions, weights))
                        averaged[name] = start[name] + delta
                        if flame_sigma > 0:
                            averaged[name] += torch.randn_like(averaged[name]) * flame_diagnostics['noise_std']
                    if not torch.isfinite(averaged[name]).all():
                        raise ValueError(f'Non-finite FedAvg aggregate {name!r}.')
                model.load_state_dict(averaged, strict=True)
            if malicious_poisoning:
                # Evaluate immediately after every global update, before any
                # forecasting validation or the next round's local training.
                property_evaluation = evaluate_property(round_number + 1)
            model.eval()
            total, elements = 0., 0
            with torch.no_grad():
                for batch in model_val_loader:
                    loss, count_elements = forward_loss(model, batch)
                    total += loss.item() * count_elements
                    elements += count_elements
            metrics = {'round': round_number + 1, 'client_weights': weights, 'assigned_batch_sizes': sizes,
                       'poisoning_active': poison_active,
                       'honest_forecasting_loss': float(np.mean(honest_losses)) if honest_losses else None,
                       'malicious_poisoning_objective': float(np.mean(poison_losses)) if poison_losses else None,
                       'validation_forecasting_mse': total / elements if elements else None}
            if aggregation == 'flame':
                metrics['flame'] = flame_diagnostics
            if malicious_poisoning:
                metrics['property_evaluation'] = property_evaluation
                previous = report['rounds'][-1]['property_evaluation'] if report['rounds'] else report['initial_property_evaluation']
                metrics['property_mae_improved_over_previous'] = metrics['property_evaluation']['mae'] < previous['mae']
                print(f'[{aggregation_label} property] round={round_number + 1}/{num_rounds} client={evaluator_client} MAE={metrics["property_evaluation"]["mae"]:.6f}, RMSE={metrics["property_evaluation"]["rmse"]:.6f}, improved={metrics["property_mae_improved_over_previous"]}', flush=True)
            report['rounds'].append(metrics)
            if malicious_poisoning:
                path.write_text(json.dumps(report, indent=2) + '\n')
            print(f'[{aggregation_label}] round={round_number + 1}/{num_rounds} poisoning_active={poison_active} honest forecasting loss={metrics["honest_forecasting_loss"]}, malicious objective={metrics["malicious_poisoning_objective"]}, validation MSE={metrics["validation_forecasting_mse"]}', flush=True)
        if num_rounds == 0:
            print(f'[{aggregation_label}] Zero rounds: no FL training or poisoning updates occurred.', flush=True)
        elif malicious_poisoning and poison_start_round > num_rounds:
            print(f'[{aggregation_label}] Poisoning start round was not reached; all rounds used honest training.', flush=True)
        if malicious_poisoning:
            final = report['rounds'][-1]['property_evaluation'] if report['rounds'] else report['initial_property_evaluation']
            # Actual final-global evaluation supplies direct inference metadata, not a fabricated central report.
            self.poison_feasibility_results = {'design': design, 'constant_property_baseline': baseline,
                                             'preparation_origin': f'malicious_client_{aggregation}', 'evaluation': final}
            path.write_text(json.dumps(report, indent=2) + '\n')
            lines = [f'# Shared-loader malicious-client {aggregation_label}', '',
                     f'Aggregation: {aggregation}. FedAvg uses assigned batch-size weights; FLAME uses cosine clustering and normalized clipping weights on model deltas, plus noise. Individual benign gradients are observed separately; they are not model submissions.', '',
                     f'Requested malicious fraction: {fraction}; count: {count}/{clients}; realized: {count / clients}.', '',
                     f'Evaluator client: {evaluator_client}. Held-out auxiliary gradients are measured after every global update using a fresh global-model copy.', '',
                     f'Poisoning optimizer: {poison_optimizer}; first poisoning round: {poison_start_round}. Earlier rounds use honest training for every client.', '',
                     f'Training defense: none. Recorded-gradient defense: {config.get("defense_name", "none")}.', '',
                     '| Round | Honest forecast loss | Malicious objective | Validation MSE | Property MAE | Property RMSE | Improved over previous |',
                     '|---:|---:|---:|---:|---:|---:|:---|']
            for r in report['rounds']:
                e = r['property_evaluation']
                lines.append(f'| {r["round"]} | {r["honest_forecasting_loss"]} | {r["malicious_poisoning_objective"]} | {r["validation_forecasting_mse"]} | {e["mae"]} | {e["rmse"]} | {r["property_mae_improved_over_previous"]} |')
            if num_rounds == 0:
                lines += ['', 'Zero rounds: no FL poisoning updates occurred.']
            lines += ['', '| Round | Coordinate / ensemble | Property MAE | Property RMSE | Auxiliary constant MAE |',
                      '|---:|:---|---:|---:|---:|']
            for number, evaluation in [(0, report['initial_property_evaluation'])] + [(r['round'], r['property_evaluation']) for r in report['rounds']]:
                for result in evaluation['coordinate_errors'] + [dict(evaluation, coordinate='ensemble')]:
                    lines.append(f'| {number} | {result["coordinate"]} | {result["mae"]} | {result["rmse"]} | {evaluation["constant_mae"]} |')
            lines += ['', 'Round 0 is the initial global model. Low malicious local objective does not establish successful global poisoning.',
                      '', 'Initial and post-aggregation measurements and settings:', '', '```json', json.dumps(report, indent=2), '```', '']
            path.with_suffix('.md').write_text('\n'.join(lines))
            config['fl_poison_report'] = str(path)
        self.federated_training_results = report
        return report

    def train_model_and_record(self, model, tr_dataloader, config):
        if config.get("fl_poisoning_enabled", False) and (config.get("poison_feasibility_only", False) or config.get("dataset") not in ETT_DATASETS + CUSTOM_DATASETS):
            raise ValueError("FL poisoning requires ETT and poison_feasibility_only=False.")
        dataset = config["dataset"]
        sampling = config.get("attack_batch_sampling", "random")
        if sampling not in {"random", "property_percentiles"}:
            raise ValueError("attack_batch_sampling must be 'random' or 'property_percentiles'.")
        if sampling == "property_percentiles" and dataset not in ETT_DATASETS + CUSTOM_DATASETS:
            raise ValueError("Property-percentile attack sampling currently supports ETT datasets only.")
        if config.get("poison_global_model", False) and dataset not in ETT_DATASETS + CUSTOM_DATASETS:
            raise ValueError("Poisoned global-model training currently supports ETT datasets only.")
        model.to(config["device"])
        all_batch_inputs, all_batch_targets, all_model_state_dicts, all_model_gradients, all_model_updates = [], [], [], [], []
        if dataset in ETT_DATASETS + CUSTOM_DATASETS:
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

            if config.get("fl_poisoning_enabled", False):
                print("[Preparation] Malicious-client FL selected (takes precedence over centralized poisoning).")
            elif config.get("poison_global_model", False):
                print("[Preparation] Centralized poisoning selected.")
                self.load_or_train_poisoned_global_model(model, config)
                # Begin benign gradient collection from this poisoned state;
                # clean FL training here would undo the experimental intervention.
                model_train_epochs = 0
            else:
                print("[Preparation] Clean shared-loader FL selected.")

            model_train_loader = DataLoader(
                tr_dataloader.dataset,
                batch_size=config.get("model_train_batch_size", 32),
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=self.g,
            )

            model_val_loader = self.val_dataloader

            # Recording optimizer only (used later if update_model=True).
            # FedAvg never receives or steps this optimizer.
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.get("model_train_learning_rate", 1e-4),
            )

            loss_function = nn.MSELoss()

            if config.get("fl_poisoning_enabled", False) or not config.get("poison_global_model", False):
                self.run_federated_training(model, model_train_loader, model_val_loader, config,
                                            model_train_epochs, malicious_poisoning=config.get("fl_poisoning_enabled", False))
            if config.get("fl_poisoning_enabled", False):
                print("[FL] Recording separately observed benign mean-MSE gradients at the final global model.")

            # Record deterministic gradients, with dropout disabled.
            # Reconstruction must also use model.eval().
            model.eval()

            self.all_batch_input_marks = []
            self.all_batch_target_marks = []

            attack_loader = (
                self._property_percentile_attack_loader(tr_dataloader, config)
                if sampling == "property_percentiles" else tr_dataloader
            )
            for batch_number, batch in enumerate(attack_loader):
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
                # Bookmark: gradient collection code
                gradients = []
                for name, parameter in model.named_parameters():
                    # Fixed timestamp embeddings are model state, not gradient entries.
                    if not parameter.requires_grad:
                        continue
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
                        (p for p in model.parameters() if p.requires_grad), gradients
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

        Target metrics are also aggregated across all attacked batches.
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
            target_element_count > 0
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

    def evaluate_and_log_reconstruction(self, *args, **kwargs):
        with getattr(self, "_attack_logging_lock", nullcontext()):
            try:
                return self._evaluate_and_log_reconstruction(*args, **kwargs)
            finally:
                if getattr(self, "_parallel_attacks", False):
                    # The shared evaluator may cache tensors created on this
                    # batch's stream; make them ready before another uses it.
                    torch.cuda.current_stream().synchronize()

    def _evaluate_and_log_reconstruction(
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
            target_sample_mapping = get_batch_sample_mapping(batch_targets, dummy_targets)

            sample_mapping = np.arange(0, batch_inputs.shape[0])
            if not (standard_mapping == input_sample_mapping).all():
                sample_mapping = input_sample_mapping
            if (
                (standard_mapping == input_sample_mapping).all() and (not (standard_mapping == target_sample_mapping).all())
            ):
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

            # Optional evaluation only; existing datasets/configs remain unchanged.
            if dataset_name in ETT_DATASETS + CUSTOM_DATASETS and (
                config.get("eval_ts2vec", False) or config.get("eval_catch22", False)
            ):
                interval = int(config.get("extra_eval_interval", 0))
                if interval < 0:
                    raise ValueError("extra_eval_interval must be nonnegative.")
                if attack_step >= num_attack_steps or (interval > 0 and attack_step % interval == 0):
                    from .reconstruction_metrics import ETTReconstructionEvaluator
                    if not hasattr(self, "_ett_reconstruction_evaluator"):
                        self._ett_reconstruction_evaluator = ETTReconstructionEvaluator(
                            config, self.val_datasets, self.test_datasets
                        )
                    attack_metrics.update(self._ett_reconstruction_evaluator.evaluate(
                        batch_inputs, dummy_inputs[sample_mapping], prefix="inputs"
                    ))


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

            if dataset_name not in ETT_DATASETS + CUSTOM_DATASETS and attack_step % (num_attack_steps // log_plots_n_times) == 0:
                df, fig = plot_original_and_dummy_data(
                    config, sample_mapping, dummy_inputs, dummy_targets, batch_inputs, batch_targets
                )
                self._log_dataframe(df, attack_step + attack_step_offset, log_name=f"_batch_{batch_number}")
                self._log_matplotlib_figure(fig, step=attack_step + attack_step_offset, log_name=f"_batch_{batch_number}")

            elif attack_step % (num_attack_steps // log_plots_n_times) == 0 or attack_step == num_attack_steps - 1:
                plot_targets_separate = config.get("plot_targets_separate", dataset_name in ETT_DATASETS + CUSTOM_DATASETS)
                reconstruction_plot_styles = config.get("reconstruction_plot_styles", ["overlay", "channel_grid"])
                for style in reconstruction_plot_styles:
                    for sample_idx in range(batch_inputs.shape[0]):
                        df, fig = plot_reconstruction(
                            config,
                            sample_mapping,
                            dummy_inputs,
                            batch_inputs,
                            batch_number,
                            sample_idx=sample_idx,
                            plot_targets_separate=plot_targets_separate,
                            dummy_targets=dummy_targets if not plot_targets_separate else None,
                            batch_targets=batch_targets if not plot_targets_separate else None,
                            reconstruction_plot_style=style
                        )
                        self._log_dataframe(df, attack_step + attack_step_offset, log_name=f"_batch_{batch_number}_sample_{sample_idx}")
                        self._log_matplotlib_figure(
                            fig, step=attack_step + attack_step_offset, log_name=f"_batch_{batch_number}_sample_{sample_idx}_style_{style}", matplotlib_only=True
                        )
                        if plot_targets_separate:
                            df, fig = plot_reconstruction(
                                config,
                                sample_mapping,
                                dummy_targets,
                                batch_targets,
                                batch_number,
                                sample_idx=sample_idx,
                                dummy_targets=None,
                                batch_targets=None,
                                reconstruction_plot_style=style,
                                plot_targets_separate=True
                            )
                            self._log_dataframe(df, attack_step + attack_step_offset, log_name=f"_batch_{batch_number}_sample_{sample_idx}_targets")
                            self._log_matplotlib_figure(
                                fig, step=attack_step + attack_step_offset, log_name=f"_batch_{batch_number}_sample_{sample_idx}_targets_style_{style}", matplotlib_only=True
                            )
            self._log_metrics(attack_metrics, step=attack_step + attack_step_offset)

            if config["verbose"]:
                self._print_batch(f"Attack Step {attack_step + attack_step_offset}: {attack_metrics}")

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

                    dummy_targets.data = torch.clamp(
                        dummy_targets,
                        0,
                        1,
                    )
            if "clamp_min0_2" in config["after_effect"]:
                if attack_step % 2 == 0:
                    dummy_inputs.data = torch.clamp(dummy_inputs, 0)
                    dummy_targets.data = torch.clamp(dummy_targets, 0)

def prepare_attack_variable(x, space="time"):
    """Return an optimizer variable and a differentiable decoder.

    x: [batch, time] or [batch, time, features].
    The transform acts only along the time dimension.
    """
    if space == "time":
        # Preserve the existing tensor and optimization behavior.
        return x, lambda value: value

    if space != "dct":
        raise ValueError(f"Unknown perturbation_space: {space!r}")

    if x.ndim not in (2, 3) or not x.is_floating_point():
        raise ValueError(
            "DCT requires floating-point data shaped "
            "[batch, time] or [batch, time, features]."
        )

    length = x.shape[1]
    if length == 0:
        raise ValueError("Cannot transform an empty time dimension.")

    # Orthonormal DCT-II matrix: basis[frequency, time].
    frequencies = torch.arange(
        length, device=x.device, dtype=x.dtype
    ).unsqueeze(1)
    times = torch.arange(
        length, device=x.device, dtype=x.dtype
    ).unsqueeze(0)

    basis = torch.cos(
        (torch.pi / length) * frequencies * (times + 0.5)
    )
    basis = basis * (2.0 / length) ** 0.5
    basis[0] = basis[0] / (2.0 ** 0.5)

    # Initialize coefficients from the existing dummy initialization.
    coefficients = (
        x.detach().movedim(1, -1) @ basis.T
    ).movedim(-1, 1).contiguous().requires_grad_(True)

    def decode(value):
        return (
            value.movedim(1, -1) @ basis
        ).movedim(-1, 1).contiguous()

    return coefficients, decode

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


def plot_channel_grid_original_vs_reconstructed(config, sample_mapping, dummy_inputs, batch_inputs, batch_number, sample_idx=0, **unused_kwargs):
    # Only the input window is drawn. dummy_targets / batch_targets are accepted
    # through **unused_kwargs so both plot functions share one call signature.
    
    original = batch_inputs[sample_idx].detach().cpu().numpy()  # [seq_len, n_features]
    reconstructed = dummy_inputs[sample_mapping[sample_idx]].detach().cpu().numpy()

    num_features = original.shape[-1]
    names = get_feature_names(config, num_features)

    fig, axes = plt.subplots(
        2, num_features, figsize=(2.5 * num_features, 5), sharex=True, squeeze=False
    )

    for feature_idx in range(num_features):
        ax_original = axes[0, feature_idx]
        ax_reconstructed = axes[1, feature_idx]

        ax_original.plot(original[:, feature_idx], color="tab:blue", linewidth=1.2)
        ax_reconstructed.plot(reconstructed[:, feature_idx], color="tab:orange", linewidth=1.2)

        ax_original.set_title(names[feature_idx], fontsize=9)

        if feature_idx == 0:
            ax_original.set_ylabel("Original", fontsize=11)
            ax_reconstructed.set_ylabel("Reconstructed", fontsize=11)

        ax_reconstructed.set_xlabel("Time step", fontsize=8)

    fig.suptitle(
        f'{config["dataset"]}: Batch {batch_number} \u2014 Original vs. Reconstructed by Channel',
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    data = {}
    for feature_idx in range(num_features):
        data[f"original_{names[feature_idx]}"] = original[:, feature_idx]
        data[f"reconstructed_{names[feature_idx]}"] = reconstructed[:, feature_idx]
    df = pd.DataFrame(data)

    if config["verbose"]:
        plt.show()
    else:
        plt.close(fig)

    return df, fig

def get_feature_names(config, num_features):
    if config["dataset"] in ETT_DATASETS + CUSTOM_DATASETS:
        if config.get("features", "M") == "S":
            names = [config.get("target", "OT")]
        elif config["dataset"] in CUSTOM_DATASETS:
            names = WEATHER_FEATURE_NAMES
        else:
            names = ETT_FEATURE_NAMES
    else:
        names = FEATURE_NAMES
    return [
        names[i] if i < len(names) else f"feature_{i}"
        for i in range(num_features)
    ]


def plot_overlay_original_vs_reconstructed(
    config,
    sample_mapping,
    dummy_inputs,
    batch_inputs,
    batch_number,
    sample_idx=0,
    dummy_targets=None,
    batch_targets=None,
    max_cols=4,
    **unused_kwargs
):
    """One subplot per channel, original and reconstruction overlaid.

    For forecasting data (ETT), the target horizon is drawn to the right of
    a dotted vertical line, so history and forecast share one time axis.
    """
    j = sample_mapping[sample_idx]
    original_in = batch_inputs[sample_idx].detach().cpu().numpy()  # [L, F]
    recon_in = dummy_inputs[j].detach().cpu().numpy()
    seq_len, num_features = original_in.shape

    has_targets = (
        batch_targets is not None
        and dummy_targets is not None
        and batch_targets.ndim == 3
    )
    if has_targets:
        original_tg = batch_targets[sample_idx].detach().cpu().numpy()  # [P, T]
        recon_tg = dummy_targets[j].detach().cpu().numpy()
        pred_len = original_tg.shape[0]
        # For features="MS" the targets only contain the last channel(s).
        target_offset = num_features - original_tg.shape[-1]
    else:
        pred_len = 0
        target_offset = 0

    names = get_feature_names(config, num_features)
    ncols = min(max_cols, num_features)
    nrows = int(np.ceil(num_features / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 2.8 * nrows), squeeze=False
    )

    input_x = np.arange(seq_len)
    target_x = np.arange(seq_len, seq_len + pred_len)

    for f in range(num_features):
        ax = axes[f // ncols][f % ncols]
        ax.plot(input_x, original_in[:, f], color="tab:blue",
                linewidth=1.3, label="Original")
        ax.plot(input_x, recon_in[:, f], color="tab:orange",
                linewidth=1.3, linestyle="--", label="Reconstructed")

        t = f - target_offset
        if has_targets and t >= 0:
            ax.plot(target_x, original_tg[:, t], color="tab:blue", linewidth=1.3)
            ax.plot(target_x, recon_tg[:, t], color="tab:orange",
                    linewidth=1.3, linestyle="--")
            ax.axvline(seq_len - 0.5, color="gray", linestyle=":", linewidth=0.8)

        ax.set_title(names[f], fontsize=9)
        ax.set_xlabel("Time step", fontsize=8)
        if f == 0:
            ax.legend(fontsize=7)

    # Hide unused axes in the last row.
    for k in range(num_features, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    suffix = " (dotted line = forecast start)" if has_targets else ""
    fig.suptitle(
        f'{config["dataset"]}: Batch {batch_number}, Sample {sample_idx}'
        f" \u2014 Original vs. Reconstructed{suffix}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # DataFrame for wandb: one original/reconstructed column pair per channel.
    total_len = seq_len + pred_len

    def padded(inputs_col, targets_col):
        col = np.full(total_len, np.nan)
        col[:seq_len] = inputs_col
        if targets_col is not None:
            col[seq_len:] = targets_col
        return col

    data = {}
    for f in range(num_features):
        t = f - target_offset
        use_t = has_targets and t >= 0
        data[f"original_{names[f]}"] = padded(
            original_in[:, f], original_tg[:, t] if use_t else None
        )
        data[f"reconstructed_{names[f]}"] = padded(
            recon_in[:, f], recon_tg[:, t] if use_t else None
        )
    df = pd.DataFrame(data)

    if config["verbose"]:
        plt.show()
    else:
        plt.close(fig)

    return df, fig

RECONSTRUCTION_PLOT_FUNCTIONS = {
    "channel_grid": plot_channel_grid_original_vs_reconstructed,
    "overlay": plot_overlay_original_vs_reconstructed,
}


def plot_reconstruction(config, *args, **kwargs):
    """Call the plot function selected by config['reconstruction_plot_style']."""
    # style = config.get("reconstruction_plot_style", "overlay")
    style = kwargs.get("reconstruction_plot_style", "overlay")
    if style not in RECONSTRUCTION_PLOT_FUNCTIONS:
        raise ValueError(
            f"Unknown reconstruction_plot_style {style!r}. "
            f"Choose from {sorted(RECONSTRUCTION_PLOT_FUNCTIONS)}."
        )
    return RECONSTRUCTION_PLOT_FUNCTIONS[style](config, *args, **kwargs)

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
    gradient_parameters = [
        p for p in model.parameters()
        if config["dataset"] not in ETT_DATASETS + CUSTOM_DATASETS or p.requires_grad
    ]

    folder_path = "../data/_model_dataset_gradients/"
    task_kind = "forecasting_mse_v1"
    # Path where the dataset will be saved or loaded from
    if config["dataset"] in ETT_DATASETS + CUSTOM_DATASETS:
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
        print(f"[Batch {batch_number}] Loaded gradient to inputs targets dataset:", dataset_path)
        grad_inputs_targets_dataset = torch.load(dataset_path, weights_only=False)
        config["loaded_grad_to_inputs_targets_dataset_from_file"] = True
    else:
        # If the file does not exist, proceed with creating the dataset
        print(f"[Batch {batch_number}] Creating gradient to inputs targets dataset:", dataset_path)
        config["loaded_grad_to_inputs_targets_dataset_from_file"] = False
        aux_dy_dx_inputs, aux_inputs_targets, aux_targets_targets = [], [], []
        for i, batch in enumerate(aux_dataloader):
            model.zero_grad(set_to_none=True)

            if config["dataset"] in ETT_DATASETS + CUSTOM_DATASETS:
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
                        "Auxiliary inputs must have shape "
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
                if config["dataset"] in ETT_DATASETS + CUSTOM_DATASETS:
                    for name, parameter in model.named_parameters():
                        if not parameter.requires_grad:
                            continue
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
                gradients = [param.grad for param in gradient_parameters]
                if config.get("sign", False):
                    gradients = apply_sign_transformation(gradients)
                if config.get("prune_rate") is not None:
                    gradients = apply_pruning(gradients, config["prune_rate"])
                if config.get("dp_epsilon", 0) > 0:
                    gradients = add_gaussian_noise(gradients, config["dp_epsilon"])

                # Update model parameters with modified gradients
                for param, grad in zip(gradient_parameters, gradients):
                    param.grad = grad

            flattened_aux_dy_dx = torch.cat([p.grad.detach().view(-1) for p in gradient_parameters]).unsqueeze(
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
            if config["dataset"] in ETT_DATASETS + CUSTOM_DATASETS:
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
    if config["dataset"] in ETT_DATASETS + CUSTOM_DATASETS:
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
