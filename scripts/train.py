from copy import deepcopy
import torch
import os
from opacus import PrivacyEngine
import numpy as np
import pandas as pd
import zero
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from tab_ddpm import GaussianMultinomialDiffusion  # noqa: E402
from utils_train import get_model, make_dataset, update_ema  # noqa: E402
import lib  # noqa: E402


def print_grad_stats(model, name="before opacus"):
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1))
    if grads:
        grads = torch.cat(grads)
        print(f'[{name}] Gradients - Mean: {grads.mean().item(): .6f}, Std: {grads.std().item(): .6f}, '
              f'Max: {grads.max().item(): .6f}, Min: {grads.min().item(): .6f}')


class Trainer:
    def __init__(
            self,
            diffusion,
            train_iter,
            lr,
            weight_decay,
            steps,
            delta,
            epsilon,
            max_grad_norm,
            device=torch.device('cuda:0'),
    ):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000

        self.delta = delta
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.privacy_engine = PrivacyEngine()
        self.diffusion, self.optimizer, self.train_iter = self.privacy_engine.make_private_with_epsilon(
            module=self.diffusion,
            optimizer=self.optimizer,
            data_loader=self.train_iter,
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            epochs=None,
            max_grad_norm=self.max_grad_norm,
            steps=self.steps
        )

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        while step < self.steps:
            for x, out_dict in self.train_iter:
                out_dict = {'y': out_dict}
                batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

                self._anneal_lr(step)

                curr_count += len(x)
                curr_loss_multi += batch_loss_multi.item() * len(x)
                curr_loss_gauss += batch_loss_gauss.item() * len(x)

                if (step + 1) % self.log_every == 0:
                    mloss = np.around(curr_loss_multi / curr_count, 4)
                    gloss = np.around(curr_loss_gauss / curr_count, 4)
                    if (step + 1) % self.print_every == 0:
                        epsilon = self.privacy_engine.get_epsilon(delta=self.delta)
                        print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss} '
                              f'Epsilon: {epsilon: .4f}')
                    self.loss_history.loc[len(self.loss_history)] = [step + 1, mloss, gloss, mloss + gloss]
                    curr_count = 0
                    curr_loss_gauss = 0.0
                    curr_loss_multi = 0.0

                update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

                step += 1
                if step >= self.steps:
                    break


def train(
    parent_dir,
    real_data_path='data/higgs-small',
    steps=1000,
    lr=0.002,
    weight_decay=1e-4,
    batch_size=1024,
    model_type='mlp',
    model_params=None,
    num_timesteps=1000,
    gaussian_loss_type='mse',
    scheduler='cosine',
    T_dict=None,
    device=torch.device('cuda:0'),
    seed=0,
    change_val=False,
    delta=1e-5,
    epsilon=10,
    max_grad_norm=1,
):
    real_data_path = os.path.normpath(real_data_path)
    parent_dir = os.path.normpath(parent_dir)

    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)

    dataset = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])
    print(K)

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in
    print(d_in)
    
    print(model_params)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    )
    model.to(device)

    train_loader = lib.prepare_fast_dp_dataloader(dataset, split='train', batch_size=batch_size)

    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device
    )
    diffusion.to(device)
    diffusion.train()

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device,
        delta=delta,
        epsilon=epsilon,
        max_grad_norm=max_grad_norm,
    )
    trainer.run_loop()

    trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))
