# coding:UTF-8
# RuHe  2026/3/13 14:37
"""
DP Mechanisms Calculator for TVAE
"""
import argparse
import os.path
from typing import Tuple, Dict
from opacus.accountants.utils import get_noise_multiplier
from opacus import PrivacyEngine
from lib import load_json, load_config


def calculate_noise_multiplier(
        target_epsilon: float,
        target_delta: float,
        sample_size: int,
        batch_size: int,
        epochs: int,
        accountant: str = "gdp"
) -> float:
    sample_rate = batch_size / sample_size

    noise_multiplier = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        epochs=epochs,
        accountant=accountant,
    )

    return noise_multiplier


def predict_final_epsilon(
        noise_multiplier: float,
        target_delta: float,
        sample_size: int,
        batch_size: int,
        epochs: int,
) -> float:
    sample_rate = batch_size / sample_size

    privacy_engine = PrivacyEngine()

    # 创建一个虚拟的隐私账本来计算
    accountant_obj = privacy_engine.accountant
    accountant_obj.history = []  # 重置历史

    # 模拟 epochs 轮的隐私消耗
    steps_per_epoch = max(1, sample_size // batch_size)
    total_steps = steps_per_epoch * epochs

    for _ in range(total_steps):
        accountant_obj.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

    final_epsilon = accountant_obj.get_epsilon(delta=target_delta)

    return final_epsilon


def quick_epsilon_estimate(
        target_epsilon: float,
        target_delta: float,
        sample_size: int,
        batch_size: int,
        epochs: int,
) -> Dict[str, float]:
    # 计算噪声乘数
    noise_multiplier = calculate_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_size=sample_size,
        batch_size=batch_size,
        epochs=epochs,
    )

    # 预测实际隐私预算
    final_epsilon = predict_final_epsilon(
        noise_multiplier=noise_multiplier,
        target_delta=target_delta,
        sample_size=sample_size,
        batch_size=batch_size,
        epochs=epochs,
    )

    # 计算效率
    efficiency = final_epsilon / target_epsilon if target_epsilon > 0 else 0

    return {
        "target_epsilon": target_epsilon,
        "final_epsilon": final_epsilon,
        "noise_multiplier": noise_multiplier,
        "efficiency": efficiency,
        "sample_rate": batch_size / sample_size,
        "total_steps": (sample_size // batch_size) * epochs,
    }


def print_dp_summary(
        sample_size: int,
        batch_size: int,
        epochs: int,
        target_epsilon: float,
        target_delta: float = 1e-5,
) -> None:
    """
    打印差分隐私参数摘要

    Args:
        sample_size: 样本总数
        batch_size: 批次大小
        epochs: 训练轮次
        target_epsilon: 目标隐私预算
        target_delta: 隐私失败率
    """
    result = quick_epsilon_estimate(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_size=sample_size,
        batch_size=batch_size,
        epochs=epochs,
    )

    print("\n" + "=" * 60)
    print("DIFFERENTIAL PRIVACY PARAMETERS SUMMARY")
    print("=" * 60)
    print(f"Dataset Size:        {sample_size:,} samples")
    print(f"Batch Size:          {batch_size}")
    print(f"Epochs:              {epochs}")
    print(f"Sample Rate:         {result['sample_rate']:.4f}")
    print(f"Total Steps:         {result['total_steps']:,}")
    print("-" * 60)
    print(f"Target Epsilon:      {result['target_epsilon']:.2f}")
    print(f"Target Delta:        {target_delta:.2e}")
    print(f"Expected Final ε:    {result['final_epsilon']:.2f}")
    print(f"Noise Multiplier:    {result['noise_multiplier']:.4f}")
    print(f"Budget Efficiency:   {result['efficiency']:.2%}")
    print("=" * 60)

    if result['efficiency'] < 0.8:
        print("WARNING: Low efficiency! Consider:")
        print("   • Increase epochs")
        print("   • Decrease batch_size")
        print("   • Decrease max_grad_norm")
    elif result['efficiency'] > 1.1:
        print("WARNING: Budget exceeded! Consider:")
        print("   • Decrease epochs")
        print("   • Increase batch_size")
        print("   • Increase noise_multiplier")
    else:
        print("Privacy budget is well-utilized!")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='file', default='exp/wilt/tvae/config.toml')
    args, _ = parser.parse_known_args()
    raw_config = load_config(args.config)
    raw_info = load_json(os.path.join(raw_config['real_data_path'], 'info.json'))

    print_dp_summary(
        sample_size=raw_info['train_size'],
        batch_size=raw_config['train_params']['batch_size'],
        epochs=raw_config['train_params']['epochs'],
        target_epsilon=raw_config['dp']['epsilon'],
        target_delta=raw_config['dp']['delta'],
    )


if __name__ == '__main__':
    main()
