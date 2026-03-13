# coding:UTF-8
# RuHe  2026/3/13 14:37
"""
DP Mechanisms Calculator for TVAE
在训练前快速计算噪声参数和隐私预算，无需运行完整训练
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
        accountant: str = "rdp"
) -> float:
    """
    计算达到目标隐私预算所需的噪声乘数

    Args:
        target_epsilon: 目标隐私预算 ε
        target_delta: 目标隐私失败率 δ
        sample_size: 训练样本总数
        batch_size: 批次大小
        epochs: 训练轮次
        accountant: 隐私会计类型 ("prv" 或 "rdp")

    Returns:
        noise_multiplier: 需要的高斯噪声乘数
    """
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
    """
    预测给定噪声参数下的最终隐私预算

    Args:
        noise_multiplier: 噪声乘数
        target_delta: 隐私失败率 δ
        sample_size: 训练样本总数
        batch_size: 批次大小
        epochs: 训练轮次
        accountant: 隐私会计类型

    Returns:
        final_epsilon: 预测的最终隐私预算 ε
    """
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
    """
    快速估算隐私参数（推荐使用）

    Args:
        target_epsilon: 目标隐私预算
        target_delta: 隐私失败率
        sample_size: 样本总数
        batch_size: 批次大小
        epochs: 训练轮次

    Returns:
        dict: 包含 noise_multiplier 和预计 final_epsilon 的字典
    """
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


def search_optimal_params(
        target_epsilon: float,
        target_delta: float,
        sample_size: int,
        batch_size_range: Tuple[int, int, int] = (128, 512, 128),
        epochs_range: Tuple[int, int, int] = (100, 500, 100),
        efficiency_threshold: float = 0.9,
) -> Dict:
    """
    搜索最优参数组合，使 final_epsilon 接近 target_epsilon

    Args:
        target_epsilon: 目标隐私预算
        target_delta: 隐私失败率
        sample_size: 样本总数
        batch_size_range: (min, max, step) 批次大小搜索范围
        epochs_range: (min, max, step) 训练轮次搜索范围
        efficiency_threshold: 效率阈值 (final/target)

    Returns:
        dict: 最优参数组合
    """
    best_params = None
    best_efficiency = 0

    print(f"Searching optimal params for ε={target_epsilon}...")
    print(f"   Sample size: {sample_size}")
    print(f"   Batch size range: {batch_size_range}")
    print(f"   Epochs range: {epochs_range}")
    print("-" * 60)

    for batch_size in range(batch_size_range[0], batch_size_range[1] + 1, batch_size_range[2]):
        for epochs in range(epochs_range[0], epochs_range[1] + 1, epochs_range[2]):
            result = quick_epsilon_estimate(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                sample_size=sample_size,
                batch_size=batch_size,
                epochs=epochs,
            )

            efficiency = result["efficiency"]

            # 打印部分结果
            if efficiency >= 0.85:
                print(f"   BS={batch_size:4d}, Epochs={epochs:3d} → "
                      f"ε={result['final_epsilon']:.2f}, "
                      f"noise={result['noise_multiplier']:.4f}, "
                      f"eff={efficiency:.2%}")

            # 寻找最接近 100% 效率的组合
            if efficiency >= efficiency_threshold and efficiency <= 1.1:
                if abs(efficiency - 1.0) < abs(best_efficiency - 1.0):
                    best_efficiency = efficiency
                    best_params = {
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "noise_multiplier": result["noise_multiplier"],
                        "final_epsilon": result["final_epsilon"],
                        "efficiency": efficiency,
                    }

    print("-" * 60)

    if best_params:
        print(f"Found optimal params:")
        print(f"   Batch Size: {best_params['batch_size']}")
        print(f"   Epochs: {best_params['epochs']}")
        print(f"   Noise Multiplier: {best_params['noise_multiplier']:.4f}")
        print(f"   Expected Final ε: {best_params['final_epsilon']:.2f}")
        print(f"   Efficiency: {best_params['efficiency']:.2%}")
    else:
        print("No params found within efficiency threshold. Try widening the search range.")

    return best_params


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
