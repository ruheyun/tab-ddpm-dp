import subprocess
import optuna
import shutil
import argparse
from pathlib import Path
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
import lib  # noqa


parser = argparse.ArgumentParser()
parser.add_argument('--ds_name', type=str, default='wilt')
parser.add_argument('--train_size', type=int, default=3096)
parser.add_argument('--eval_type', type=str, default='synthetic')
parser.add_argument('--eval_model', type=str, default='cb')
parser.add_argument('--prefix', type=str, default='ddpm')
parser.add_argument('--eval_seeds', action='store_true',  default=False)
parser.add_argument('--epsilon', type=float, default=10)
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument('--max_grad_norm', type=float, default=1)
parser.add_argument('--n_trials', type=int,  default=25)

args = parser.parse_args()
train_size = args.train_size
ds_name = args.ds_name
eval_type = args.eval_type
eval_model = args.eval_model
epsilon = args.epsilon
delta = args.delta
max_grad_norm = args.max_grad_norm
n_trials = args.n_trials
assert eval_type in ('merged', 'synthetic')
prefix = str(args.prefix + '_' + eval_model)

pipeline = f'scripts/pipeline.py'
base_config_path = f'exp/{ds_name}/config.toml'
parent_path = Path(f'exp/{ds_name}/')
exps_path = Path(f'exp/{ds_name}/many-exps/') # temporary dir. maybe will be replaced with tempdiвdr
eval_seeds = f'scripts/eval_seeds.py'

os.makedirs(exps_path, exist_ok=True)
best_seed = 0
python_exec = sys.executable


def _suggest_mlp_layers(trial):
    def suggest_dim(name):
        t = trial.suggest_int(name, d_min, d_max)
        return 2 ** t
    min_n_layers, max_n_layers, d_min, d_max = 1, 4, 7, 10
    n_layers = 2 * trial.suggest_int('n_layers', min_n_layers, max_n_layers)
    d_first = [suggest_dim('d_first')] if n_layers else []
    d_middle = (
        [suggest_dim('d_middle')] * (n_layers - 2)
        if n_layers > 2
        else []
    )
    d_last = [suggest_dim('d_last')] if n_layers > 1 else []
    d_layers = d_first + d_middle + d_last
    return d_layers


def objective(trial):
    global best_seed
    
    lr = trial.suggest_loguniform('lr', 0.00001, 0.003)
    d_layers = _suggest_mlp_layers(trial)
    weight_decay = 0.0    
    # batch_size = trial.suggest_categorical('batch_size', [256, 4096])
    # steps = trial.suggest_categorical('steps', [5000, 20000, 30000])
    # steps = trial.suggest_categorical('steps', [500]) # for debug
    # gaussian_loss_type = 'mse'
    # scheduler = trial.suggest_categorical('scheduler', ['cosine', 'linear'])
    num_timesteps = trial.suggest_categorical('num_timesteps', [100, 500, 1000])
    num_samples = int(train_size * (2 ** trial.suggest_int('num_samples', -2, 1)))

    base_config = lib.load_config(base_config_path)

    base_config['train']['main']['lr'] = lr
    # base_config['train']['main']['steps'] = steps
    # base_config['train']['main']['batch_size'] = batch_size
    base_config['train']['main']['weight_decay'] = weight_decay
    base_config['model_params']['rtdl_params']['d_layers'] = d_layers
    base_config['eval']['type']['eval_type'] = eval_type
    base_config['sample']['num_samples'] = num_samples
    # base_config['diffusion_params']['gaussian_loss_type'] = gaussian_loss_type
    base_config['diffusion_params']['num_timesteps'] = num_timesteps
    # base_config['diffusion_params']['scheduler'] = scheduler

    base_config['parent_dir'] = str(exps_path / f"{trial.number}")
    base_config['eval']['type']['eval_model'] = args.eval_model
    if args.eval_model == "mlp":
        base_config['eval']['T']['normalization'] = "quantile"
        base_config['eval']['T']['cat_encoding'] = "one-hot"

    # trial.set_user_attr("config", base_config)

    lib.dump_config(base_config, exps_path / 'config.toml')

    subprocess.run([python_exec, f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--train'], check=True)

    n_datasets = 5
    score = float('-inf')

    for sample_seed in range(n_datasets):
        base_config['sample']['seed'] = sample_seed
        lib.dump_config(base_config, exps_path / 'config.toml')
        
        subprocess.run([python_exec, f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--sample', '--eval', '--change_val'], check=True)

        report_path = str(Path(base_config['parent_dir']) / f'results_{args.eval_model}.json')
        report = lib.load_json(report_path)

        if 'r2' in report['metrics']['val']:
            if score <= report['metrics']['val']['r2']:
                score = report['metrics']['val']['r2']
                best_seed = sample_seed
        else:
            if score <= report['metrics']['val']['roc_auc']:
                score = report['metrics']['val']['roc_auc']
                best_seed = sample_seed

    base_config['sample']['seed'] = best_seed
    trial.set_user_attr("config", base_config)

    shutil.rmtree(exps_path / f"{trial.number}")

    return score


study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)

study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

best_config_path = parent_path / f'{prefix}_best/config.toml'
best_config = study.best_trial.user_attrs['config']
best_config["parent_dir"] = str(parent_path / f'{prefix}_best/')

os.makedirs(parent_path / f'{prefix}_best', exist_ok=True)
lib.dump_config(best_config, best_config_path)
lib.dump_json(optuna.importance.get_param_importances(study), parent_path / f'{prefix}_best/importance.json')

subprocess.run([python_exec, f'{pipeline}', '--config', f'{best_config_path}', '--train', '--sample', '--eval'], check=True)

if args.eval_seeds:
    best_exp = str(parent_path / f'{prefix}_best/config.toml')
    subprocess.run([python_exec, f'{eval_seeds}', '--config', f'{best_exp}', '10', "ddpm", eval_type, args.eval_model, '5'], check=True)