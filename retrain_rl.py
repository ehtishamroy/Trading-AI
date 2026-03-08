"""
Reinforcement Learning Training Script (Phase 10)
Trains a PPO agent on historical market data using the custom TradingEnv.
"""

import os
import sys
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Callable

# Suppress TensorFlow/PyTorch warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from config.settings import DATA_DIR
from models.rl_env import TradingEnv

def load_data(market: str = "BTCUSD") -> pd.DataFrame:
    # First try exact match
    files = list(Path(DATA_DIR).glob(f"{market}.parquet"))
    if not files:
        # Fallback to wildcard
        files = list(Path(DATA_DIR).glob(f"{market}*.parquet"))
    
    if not files:
        logger.error(f"No parquet data found for {market}")
        return pd.DataFrame()
    
    # Load the most recent one
    latest_file = max(files, key=os.path.getctime)
    logger.info(f"Loading data from {latest_file.name}")
    df = pd.read_parquet(latest_file)
    df = df.dropna().reset_index(drop=True)
    return df

def train_rl_agent(market: str = "BTCUSD", total_timesteps: int = 100000, df: pd.DataFrame = None):
    logger.info(f"🚀 Starting RL Agent Training for {market}")
    if df is None:
        df = load_data(market)
    
    if df.empty or len(df) < 1000:
        logger.error("Not enough data to train RL agent.")
        return

    # Split data: 80% train, 20% validation
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    val_df = df.iloc[split_idx:].copy()

    logger.info(f"Training on {len(train_df)} rows. Validating on {len(val_df)} rows.")

    # Create vectorized environments
    train_env = make_vec_env(lambda: TradingEnv(train_df), n_envs=4)
    val_env = make_vec_env(lambda: TradingEnv(val_df), n_envs=1)

    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value
        return func

    # Deep Neural Network for Policy and Value Function
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])
    )

    # Initialize Advanced PPO Agent with Entropy Bonus
    model = PPO(
        "MlpPolicy", 
        train_env, 
        learning_rate=linear_schedule(0.0003),
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Forces agent to explore trades instead of just holding forever
        policy_kwargs=policy_kwargs,
        verbose=1, 
        tensorboard_log="./logs/rl_tensorboard/"
    )  # Train
    logger.info(f"Training PPO for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    # Evaluate
    logger.info("Evaluating trained agent...")
    mean_reward, std_reward = evaluate_policy(model, val_env, n_eval_episodes=10)
    logger.info(f"Validation Result: Mean Reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    # Save Model
    save_dir = Path("models/saved")
    save_dir.mkdir(exist_ok=True, parents=True)
    model_path = save_dir / f"ppo_{market}.zip"
    model.save(str(model_path))
    logger.success(f"🏁 RL Agent saved to {model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", type=str, default="BTCUSD")
    parser.add_argument("--steps", type=int, default=100000)
    args = parser.parse_args()
    
    train_rl_agent(args.market, args.steps)
