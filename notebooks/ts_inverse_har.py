import pandas as pd
import dotenv
dotenv.load_dotenv()

import torch
import torch.nn as nn
import psutil

class MotionSenseActivityCNN(nn.Module):
    """
    Activity-only PyTorch version of the MTCNN architecture used in
    1_MotionSense_Trial.ipynb and Table 2 of the paper.

    The shared CNN trunk is preserved. The gender head is removed,
    leaving only the four-class activity head.
    """

    def __init__(
        self,
        num_features=12,
        window_size=50,
        num_classes=4,
    ):
        super().__init__()

        self.name = "MotionSenseActivityCNN"
        self.features = list(range(num_features))

        self.feature_extractor = nn.Sequential(
            # Keras Conv2D(50, (1, 5), padding="valid")
            nn.Conv2d(
                in_channels=1,
                out_channels=50,
                kernel_size=(1, 5),
            ),
            nn.ReLU(),

            # Keras Conv2D(50, (1, 3), padding="same")
            nn.Conv2d(
                in_channels=50,
                out_channels=50,
                kernel_size=(1, 3),
                padding=(0, 1),
            ),
            nn.ReLU(),

            # Keras Dense(50), applied independently at each
            # feature/time position.
            nn.Conv2d(
                in_channels=50,
                out_channels=50,
                kernel_size=1,
            ),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.2),

            # Keras Conv2D(40, (1, 5), padding="valid")
            nn.Conv2d(
                in_channels=50,
                out_channels=40,
                kernel_size=(1, 5),
            ),
            nn.ReLU(),

            # Keras Dense(40)
            nn.Conv2d(
                in_channels=40,
                out_channels=40,
                kernel_size=1,
            ),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Dropout(0.2),

            # Keras Conv2D(20, (1, 3), padding="valid")
            nn.Conv2d(
                in_channels=40,
                out_channels=20,
                kernel_size=(1, 3),
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Determine the flatten size without hard-coding it.
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                1,
                num_features,
                window_size,
            )
            flattened_size = self.feature_extractor(dummy).numel()

        self.activity_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 400),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(400, num_classes),
        )

    def forward(self, x):
        # Worker supplies [batch, time, features].
        x = x.transpose(1, 2)

        # Conv2d expects [batch, channel, feature, time].
        x = x.unsqueeze(1)

        x = self.feature_extractor(x)
        return self.activity_head(x)
    
# Check if CUDA is available
if torch.cuda.is_available():
    # Get the current device
    torch.set_float32_matmul_precision('high')
    device = torch.cuda.current_device()
    print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
else:
    device = 'cpu'
    print("CUDA is not available")

print("Physical cores:", psutil.cpu_count(logical=False))
print("Total cores:", psutil.cpu_count(logical=True))
cpu_freq = psutil.cpu_freq()
print(f"Current Frequency: {cpu_freq.current:.2f}Mhz")

# RAM Information
svmem = psutil.virtual_memory()
print(f"Total: {svmem.total / (1024 ** 3):.2f} GB")
print(f"Available: {svmem.available / (1024 ** 3):.2f} GB")
print(f"Used: {svmem.used / (1024 ** 3):.2f} GB")
print(f"Percentage: {svmem.percent}%")

print("Distributed PyTorch available:", torch.distributed.is_available())

from copy import deepcopy
from ts_inverse.models import FCN_Predictor, CNN_Predictor, GRU_Predictor, JitGRU_Predictor, CNNJitGRU_Predictor, TCN_Predictor, JitSeq2Seq_Predictor, STMAE_Pre, STMAE_Finetune, RealWorldCNN
from ts_inverse.utils import grid_search_params
from ts_inverse.workers import AttackTSInverseWorker


def start_multi_process(g_config, a_config, d_config, m_config, pool_size):
    search_args = []
    search_configs = list(grid_search_params(g_config))
    search_attack_configs = list(grid_search_params(a_config))
    search_dataset_settings = list(grid_search_params(d_config))
    search_model_settings = list(grid_search_params(m_config))
    for original_g_config in search_configs:
        for a_config in search_attack_configs:
            g_config = deepcopy(original_g_config)
            g_config.update(a_config)
            for m_config in search_model_settings:
                for d_config in search_dataset_settings:
                    if d_config["dataset"] == "motionsense":
                        fa_models_config = {
                            "features": [list(range(12))],
                            "input_size": d_config["seq_len"],
                            # The target is one scalar activity class.
                            "output_size": 1,
                        }
                    else:
                        fa_models_config = {
                            "features": [[0]],
                            "input_size": d_config["observation_days"],
                            "output_size": d_config["future_days"],
                        }
                    search_for_all_models_settings = list(grid_search_params(fa_models_config))
                    for fa_models_config in search_for_all_models_settings:
                        g_config['run_number'] = len(search_args)
                        args = (g_config, d_config, m_config, fa_models_config, None)
                        search_args.append(deepcopy(args))

    print(f"Starting {len(search_args)} processes")
    if pool_size == 1:
        for args in search_args:
            AttackTSInverseWorker(args[0]['run_number']).worker_process(*args)


global_config = {
    'logger_service': 'wandb',
    # 'logger_service': 'none',
    'experiment_name': 'ts-inverse_batch1_without_target_reconst_12-6-2024',
    # 'seed': [10, 43, 28, 80, 71], # 28, 80, 71],
    'seed': [43], # 28, 80, 71],
    'batch_size': 1,
    'device': 0,
    'verbose': False,
    'pool_size': 1,
    'run_number': -1,
    'total_variation_alpha_inputs': 0, 
    'total_variation_beta_targets': 0,
    'after_effect': 'none',
    'warmup_number_of_batches': 0,
    'number_of_batches': 100,
    'update_model': False, # Update the model in generating gradients from training data
    'model_evaluation_during_attack': False, # Baselines do not consider this
    'load_lti_model': False,

    'dropout': 0,
    'optimize_dropout': False,
    'dropout_probability_regularizer': 0,
    'dummy_init_method': 'rand',

    "model_train_epochs": 20,
    "model_train_batch_size": 64,
    "model_train_learning_rate": 1e-3,

    "reconstruction_output_dir": "/scratch/ejk5818/ts-inverse/reconstructions/",
}

attack_config = [
    {
        'attack_method': 'TS-Inverse',
        # invert attack
        'num_learn_epochs': 0,
        'learn_learning_rate': 1e-3, 
        'attack_batch_size': 32,
        'inversion_batch_size': 1, # global_config['batch_size'],
        'attack_hidden_size': [[768, 512]],
        'quantiles': [[0.1, 0.3, 0.7, 0.9]],
        'attack_loss': ['quantile'],
        'inversion_model': 'ImprovedGradToInputNN_Quantile', 
        'attack_targets': False,
        'learn_optimizer': 'adamW',
        'learn_lr_decay': ['on_plateau'],
        'aux_dataset': None,
        'one_shot_targets': False,

        ## Inversion regularization in optimization attack
        'inversion_regularization_term_inputs': [0], 
        'inversion_regularization_term_targets': [0],
        'inversion_regularization_loss': ['quantile'],

        'lower_res_term': [0],
        'trend_term': [0],
        'trend_loss': ['l1_mean'],
        'trend_reduce_lr': [False],
        'periodicity_term': [0],
        'periodicity_loss': ['l1_mean'],
        'periodicity_reduce_lr': [False],


        ## Optimization attack
        'gradient_loss': ['l1'], 
        'base_num_attack_steps': 500,
        # 'after_effect': 'clamp_2',
        'after_effect': 'none',
        'optimization_learning_rate': 0.01, 
        'attack_opti_optimizer': ['adam'],
        'attack_opti_lr_decay': ['on_plateau_10'],
        'optimize_dropout': True,
        'clamp_dropout': 1,
        'clamp_dropout_min_max': [[0.0, 1.0]],
        'dropout_probability_regularizer': [1e-5], #1e-6, 1e-7, 1e-8],
        'dropout_mask_init_type': 'halves', #['bernoulli', 'halves', 'uniform', 'p', '1-p'], #['halves', 'bernoulli'],
        'grad_signs_for_inputs': True,
        'grad_signs_for_targets': False,
        'grad_signs_for_dropouts': True, #[False, True],

        'attack_number_of_batches': 100,
    },
]

dataset_config = [
    {
        "dataset": "motionsense",
        "data_path": "/scratch/ejk5818/motion-sense/data/",
        "seq_len": 50,
        "stride": 10,
        "num_features": 12,
        "num_classes": 4,
        "normalize": "standard",
    },
]

model_config = [
    {
        "_model": MotionSenseActivityCNN,
        "_attack_step_multiplier": 10,
    }
]

start_multi_process(global_config, attack_config, dataset_config, model_config, global_config['pool_size'])