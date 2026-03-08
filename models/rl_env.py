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

    def __init__(self, df: pd.DataFrame, initial_balance: float = 500.0, transaction_cost: float = 0.0001):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # Determine feature columns (exclude time, close, unscaled prices)
        self.feature_cols = [c for c in df.columns if c not in ['time', 'open', 'high', 'low', 'close', 'volume'] and not df[c].dtype == "object"]
        
        # Action Space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation Space: The current indicators
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(self.feature_cols),), 
            dtype=np.float32
        )
        
        # State tracking
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0      # 1 for Long, -1 for Short, 0 for Flat
        self.entry_price = 0.0
        self.max_balance = initial_balance
        self.history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 50  # Start a bit in to ensure moving averages are populated
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.max_balance = self.initial_balance
        self.history = []
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'close']
        reward = 0
        done = False
        trade_executed = False
        step_pnl = 0
        
        # Calculate running PnL if holding a position
        if self.position == 1:
            step_pnl = (current_price - self.entry_price) / self.entry_price
        elif self.position == -1:
            step_pnl = (self.entry_price - current_price) / self.entry_price
            
        # ── Execute Actions ──
        if action == 1: # Buy Signal
            if self.position == -1:  # Close Short, Open Long
                reward += step_pnl - self.transaction_cost
                self.balance *= (1 + step_pnl - self.transaction_cost)
                self.position = 1
                self.entry_price = current_price
                trade_executed = True
            elif self.position == 0: # Open Long
                self.position = 1
                self.entry_price = current_price
                reward -= self.transaction_cost
                trade_executed = True
                
        elif action == 2: # Sell Signal
            if self.position == 1:   # Close Long, Open Short
                reward += step_pnl - self.transaction_cost
                self.balance *= (1 + step_pnl - self.transaction_cost)
                self.position = -1
                self.entry_price = current_price
                trade_executed = True
            elif self.position == 0: # Open Short
                self.position = -1
                self.entry_price = current_price
                reward -= self.transaction_cost
                trade_executed = True
                
        elif action == 0: # Hold
            # Small penalty for holding over long periods to encourage trading
            reward = 0  
        
        # Update running max balance for drawdown calculation
        if self.balance > self.max_balance:
            self.max_balance = self.balance
            
        # Drawdown calculation
        drawdown = (self.max_balance - self.balance) / self.max_balance
        
        # ── Reward Shaping ──
        # Provide step reward based on unrealized PnL to guide the agent
        if not trade_executed and self.position != 0:
             reward += (step_pnl * 0.1) # Soft reward for holding a winning position
             
        # Heavy penalty for deep drawdown
        if drawdown > 0.15:
            reward -= 1.0  # Big penalty
            done = True    # Stop episode if drawdown exceeds 15%
            
        # Advance time
        self.current_step += 1
        
        if self.current_step >= len(self.df) - 1:
            done = True
            # Force close position at end of data
            if self.position != 0:
                 self.balance *= (1 + step_pnl - self.transaction_cost)
                 reward += step_pnl
                 
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, done, False, info

    def _get_observation(self):
        # Return the feature array for the current step
        obs = self.df.loc[self.current_step, self.feature_cols].values
        return obs.astype(np.float32)

    def _get_info(self):
        return {
            "step": self.current_step,
            "balance": self.balance,
            "position": self.position
        }
