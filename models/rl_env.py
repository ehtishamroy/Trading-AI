"""
Reinforcement Learning Environment (Phase 10)
Custom Gymnasium environment for training PPO trading agents.
The agent learns to dynamically Buy/Sell/Hold based on 60+ indicators to maximize PnL.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from loguru import logger

class TradingEnv(gym.Env):
    """
    OpenAI Gym environment for trading.
    State: Current market indicators.
    Action: 0 (Hold), 1 (Buy), 2 (Sell).
    Reward: Realized PnL - Drawdown Penalty.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 500.0, transaction_cost: float = 0.0001, window_size: int = 5):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        
        # Determine feature columns (exclude time, close, unscaled prices)
        self.feature_cols = [c for c in df.columns if c not in ['time', 'open', 'high', 'low', 'close', 'volume'] and not df[c].dtype == "object"]
        
        # Action Space: 0 = Flat, 1 = Long, 2 = Short
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: The current indicators + Position state + Unrealized PnL (stacked over window_size)
        self.obs_shape = (len(self.feature_cols) + 2) * self.window_size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.obs_shape,), 
            dtype=np.float32
        )
        
        # State tracking
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0      # 1 for Long, -1 for Short, 0 for Flat
        self.entry_price = 0.0
        self.max_net_worth = initial_balance
        self.history = []

    def _get_net_worth(self, price: float) -> float:
        if self.position == 1:
            pnl_pct = (price - self.entry_price) / self.entry_price
            return self.balance * (1 + pnl_pct)
        elif self.position == -1:
            pnl_pct = (self.entry_price - price) / self.entry_price
            return self.balance * (1 + pnl_pct)
        return self.balance

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 50 + self.window_size  # Start a bit in to ensure moving averages are populated
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.max_net_worth = self.initial_balance
        self.history = []
        
        obs = self._get_observation()
        info = self._get_info(self.initial_balance)
        return obs, info

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'close']
        done = False
        
        # Calculate net worth before taking any new actions
        prev_net_worth = self._get_net_worth(current_price)
        
        # ── Execute Target Portfolio Action ──
        # Action meanings: 0 = Flat, 1 = Long, 2 = Short
        if action == 1 and self.position != 1:
            # Switch to long
            self.balance = prev_net_worth * (1 - self.transaction_cost)
            self.position = 1
            self.entry_price = current_price
            
        elif action == 2 and self.position != -1:
            # Switch to short
            self.balance = prev_net_worth * (1 - self.transaction_cost)
            self.position = -1
            self.entry_price = current_price
            
        elif action == 0 and self.position != 0:
            # Close positions (go flat)
            self.balance = prev_net_worth * (1 - self.transaction_cost)
            self.position = 0
            self.entry_price = 0.0
            
        # Advance time
        self.current_step += 1
        
        if self.current_step >= len(self.df) - 1:
            done = True
            
        next_price = self.df.loc[self.current_step, 'close']
        current_net_worth = self._get_net_worth(next_price)
        
        # Standard RL trading reward: Percentage change in net worth over this step
        reward = ((current_net_worth - prev_net_worth) / prev_net_worth) * 100.0
        
        # Drawdown calculation
        if current_net_worth > self.max_net_worth:
            self.max_net_worth = current_net_worth
            
        drawdown = (self.max_net_worth - current_net_worth) / self.max_net_worth
        
        # Penalty for deep drawdown
        if drawdown > 0.15:
            reward -= 5.0  # Big penalty
            done = True    # Stop episode if drawdown exceeds 15%
                 
        obs = self._get_observation()
        info = self._get_info(current_net_worth)
        
        return obs, reward, done, False, info

    def _get_observation(self):
        # Feature array window
        start_idx = self.current_step - self.window_size + 1
        obs_window = self.df.loc[start_idx:self.current_step, self.feature_cols].values
        
        # Calculate unrealized PnL to pass to agent
        unrealized_pnl = 0.0
        if self.position != 0:
            current_price = self.df.loc[self.current_step, 'close']
            if self.position == 1:
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            else:
                unrealized_pnl = (self.entry_price - current_price) / self.entry_price
                
        # Append internal state so agent isn't trading blind
        additional_states = np.array([self.position, unrealized_pnl], dtype=np.float32)
        add_window = np.tile(additional_states, (self.window_size, 1))
        
        full_obs = np.concatenate((obs_window, add_window), axis=1)
        return full_obs.flatten().astype(np.float32)

    def _get_info(self, net_worth):
        return {
            "step": self.current_step,
            "net_worth": net_worth,
            "position": self.position
        }
