import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import Union, List, Tuple
import random


def plot_rewards(*args):
    """
    Plot rewards for multiple sequences.

    Args:
        *args: Variable number of reward sequences to plot.

    Returns:
        None

    Displays a plot showing the total rewards for each sequence over episodes.

    """
    for i, rewards in enumerate(args):
        plt.plot(rewards, label=f'Sequence {i+1}')

    plt.title('Total Rewards Over Episodes')  # 圖表標題：情節總獎勵
    plt.xlabel('Episodes')  # X 軸標籤：情節
    plt.ylabel('Total Reward')  # Y 軸標籤：總獎勵
    plt.grid(True)  # 顯示網格
    plt.legend()  # 顯示圖例
    plt.show()  # 顯示圖表


def fix(env, seed):
    """
    Fix random seed for reproducibility.

    Args:
        env: Environment object.
        seed (int): Seed value for random number generation.

    Notes:
        This function fixes the random seed for various libraries and modules
        to ensure reproducibility of results in environments with randomness.
    """
    env.reset(seed=seed)  # 重置環境，使用指定的種子
    env.action_space.seed(seed)  # 設置動作空間的種子
    torch.manual_seed(seed)  # 設置 PyTorch 的種子
    torch.cuda.manual_seed(seed)  # 設置 PyTorch CUDA 的種子
    torch.cuda.manual_seed_all(seed)  # 設置所有 CUDA 設備的種子
    np.random.seed(seed)  # 設置 NumPy 的種子
    random.seed = seed  # 設置 Python 內建的隨機種子
    torch.use_deterministic_algorithms = True  # 使用確定性算法
    torch.are_deterministic_algorithms_enabled = True  # 啟用確定性算法
    torch.backends.cudnn.benchmark = False  # 禁用 CuDNN 的基準測試
    torch.backends.cudnn.deterministic = True  # 啟用 CuDNN 的確定性模式

def avg(sequence: Union[List, Tuple, np.ndarray]) -> float:
    if isinstance(sequence, np.ndarray):
        return sequence.reshape((-1,)).mean()
    else:
        return sum(sequence)/len(sequence)
