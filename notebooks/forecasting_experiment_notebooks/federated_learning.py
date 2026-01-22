from ts_inverse import datahandler
import ts_inverse.models as models
from ts_inverse.models.fcn import FCN_Predictor
import ts_inverse.utils as utils
import ts_inverse.server as server
import ts_inverse.client as client
import flwr as fl
import torch
import wandb

NUM_CLIENTS = 10
FL_ROUNDS = 1
DEVICE = 'cuda'
DATASET_CONFIG = {
    'dataset': 'london_smartmeter',
    'train_stride': 1,
    'validation_stride': 24,
    'observation_days': 1,
    'future_days': 1,
    'normalize': 'minmax',
}

MODEL_CONFIG = {
        '_model': FCN_Predictor,
        'hidden_size': 64,
        '_attack_step_multiplier': 1,
}

trainsets, valsets, testsets = datahandler.get_datasets(**DATASET_CONFIG, columns=0)
trainset, valset, testset = trainsets[0], valsets[0], testsets[0]

client_resources = {"num_cpus": 1, "num_gpus": 0.1}

model = MODEL_CONFIG['_model'](features=[0], hidden_size=MODEL_CONFIG['hidden_size'], input_size=trainset.freq_in_day*DATASET_CONFIG['observation_days'], output_size=trainset.freq_in_day*DATASET_CONFIG['future_days'])
model_parameters = utils.get_model_parameters(model)

# Create strategy
strategy = server.CustomStrategy(
    on_fit_config_fn=server.fit_config,
    evaluate_metrics_aggregation_fn=server.evaluate_metrics_aggregation,
    initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
)

run_id = "your_run_id"  # Replace with your actual run_id
wandb.init(project="ts-inverse-fl", name=f"{DATASET_CONFIG['dataset']}_{run_id}", config={
    "dataset": DATASET_CONFIG,
    "model": {k: (v.__name__ if k == "_model" else v) for k, v in MODEL_CONFIG.items()},
    "num_clients": NUM_CLIENTS,
    "fl_rounds": FL_ROUNDS,
    "device": DEVICE,
})

history = fl.simulation.start_simulation(
    client_fn=client.client_factory(d_config=DATASET_CONFIG, m_config=MODEL_CONFIG, device=DEVICE, num_clients=NUM_CLIENTS),
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=FL_ROUNDS),  
    strategy=strategy,
    client_resources=client_resources,
    ray_init_args={"num_cpus": 64, "num_gpus": 1, "include_dashboard": False},
)

# Log losses
for item in history["losses_distributed"]:
    wandb.log({"loss_distributed": item["value"]}, step=item["round"])
for item in history["losses_centralized"]:
    wandb.log({"loss_centralized": item["value"]}, step=item["round"])

# Log metrics
for name, series in history["metrics_distributed"].items():
    for item in series:
        wandb.log({f"{name}_distributed": item["value"]}, step=item["round"])
for name, series in history["metrics_centralized"].items():
    for item in series:
        wandb.log({f"{name}_centralized": item["value"]}, step=item["round"])

wandb.save("path_to_your_json_file")  # attach the JSON artifact
wandb.finish()