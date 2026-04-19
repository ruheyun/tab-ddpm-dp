import tempfile
import subprocess
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
import lib
import optuna
import argparse
from pathlib import Path
from train_sample_ctgan import train_ctgan, sample_ctgan
from scripts.eval_catboost import train_catboost

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/adult')
parser.add_argument('--train_size', type=int, default=26048)
parser.add_argument('--eval_type', type=str, default='synthetic')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--epsilon', type=float, default=10)
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument('--max_grad_norm', type=float, default=1)
parser.add_argument('--n_trials', type=int, default=50)


args = parser.parse_args()
real_data_path = args.data_path
eval_type = args.eval_type
train_size = args.train_size
device = args.device
epsilon = args.epsilon
delta = args.delta
max_grad_norm = args.max_grad_norm
n_trials = args.n_trials
assert eval_type in ('merged', 'synthetic')
best_seed = 0


def objective(trial):
    global best_seed
    
    lr = trial.suggest_loguniform('lr', 0.00001, 0.003)

    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t
    
    # construct model
    min_n_layers, max_n_layers, d_min, d_max = 1, 3, 6, 9
    n_layers = 2 * trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    ####

    steps = trial.suggest_categorical('steps', [50, 100, 300, 500])
    batch_size = trial.suggest_categorical('batch_size', [128, 256])

    num_samples = int(train_size * (2 ** trial.suggest_int('frac_samples', -1, 2)))
    embedding_dim = 2 ** trial.suggest_int('embedding_dim', 6, 9)

    train_params = {
        "generator_lr": lr,
        "discriminator_lr": lr,
        "epochs": steps,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "generator_dims": d_layers,
        "discriminator_dims": d_layers
    }

    trial.set_user_attr("train_params", train_params)
    trial.set_user_attr("num_samples", num_samples)

    score = 0.0
    with tempfile.TemporaryDirectory() as dir_:
        dir_ = Path(dir_)
        ctgan = train_ctgan(
            parent_dir=dir_,
            real_data_path=real_data_path,
            train_params=train_params,
            change_val=True,
            device=device,
            epsilon=epsilon,
            delta=delta,
            max_grad_norm=max_grad_norm,
        )

        for sample_seed in range(5):
            sample_ctgan(
                ctgan,
                parent_dir=dir_,
                real_data_path=real_data_path,
                num_samples=num_samples,
                train_params=train_params,
                change_val=False,
                seed=sample_seed,
                device=device
            )

            T_dict = {
                "seed": 0,
                "normalization": None,
                "num_nan_policy": None,
                "cat_nan_policy": None,
                "cat_min_frequency": None,
                "cat_encoding": None,
                "y_policy": "default"
            }
            metrics = train_catboost(
                parent_dir=dir_,
                real_data_path=real_data_path, 
                eval_type=eval_type,
                T_dict=T_dict,
                change_val=False,
                seed=0
            )
            if score <= metrics.get_dp_score():
                score = metrics.get_dp_score()
                best_seed = sample_seed
            # score += metrics.get_dp_score()
    return score


study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)

study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

os.makedirs(f"exp/{Path(real_data_path).name}/ctgan/", exist_ok=True)
config = {
    "parent_dir": f"exp/{Path(real_data_path).name}/ctgan/",
    "real_data_path": real_data_path,
    "seed": 0,
    "device": args.device,
    "train_params": study.best_trial.user_attrs["train_params"],
    "sample": {"seed": best_seed, "num_samples": study.best_trial.user_attrs["num_samples"]},
    "eval": {
        "type": {"eval_model": "catboost", "eval_type": eval_type},
        "T": {
            "seed": 0,
            "normalization": None,
            "num_nan_policy": None,
            "cat_nan_policy": None,
            "cat_min_frequency": None,
            "cat_encoding": None,
            "y_policy": "default"
        },
    },
    "dp": {"epsilon": epsilon, "delta": delta, "max_grad_norm": max_grad_norm}
}

lib.dump_config(config, config["parent_dir"]+"config.toml")

python_exec = sys.executable
subprocess.run([python_exec, "CTGAN/pipeline_ctgan.py",
                '--config', f'{config["parent_dir"]+"config.toml"}',
                '--train',
                '--sample',
               '--eval',], check=True)
