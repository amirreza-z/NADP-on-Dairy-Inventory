#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Perishable multi-product inventory env + NADP for SIGNIFICANTLY better performance.

Key improvements:
1. Richer state representation with forecasting features
2. Curriculum learning - start with easier episodes
3. Better reward shaping with profit margins and efficiency bonuses
4. Adaptive exploration with success-based entropy scheduling
5. Multi-step lookahead in state features
6. Experience replay buffer for more stable learning
7. Product-specific normalization and action scaling
8. Advanced network architecture with attention
"""
import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Enhanced Utilities
# ----------------------------
class RunningNorm:
    """Running mean/std with Welford; supports (de)serialization."""
    def __init__(self, size: int, eps: float = 1e-5, clip: float = 10.0):
        self.size = size
        self.eps = eps
        self.clip = clip
        self.count = 0
        self.mean = np.zeros(size, dtype=np.float32)
        self.M2 = np.zeros(size, dtype=np.float32)

    def update(self, x: np.ndarray):
        if x.ndim == 1:
            x = x[None, :]
        for row in x:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self.M2 += delta * delta2

    @property
    def var(self):
        return self.M2 / max(1, self.count - 1)

    @property
    def std(self):
        return np.sqrt(np.maximum(self.var, self.eps))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / self.std, -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return {"count": self.count, "mean": self.mean, "M2": self.M2,
                "eps": self.eps, "clip": self.clip, "size": self.size}

    def load_state_dict(self, sd):
        self.count = int(sd["count"])
        self.mean = sd["mean"].astype(np.float32)
        self.M2   = sd["M2"].astype(np.float32)
        self.eps  = float(sd.get("eps", self.eps))
        self.clip = float(sd.get("clip", self.clip))
        self.size = int(sd.get("size", self.size))


class ExperienceBuffer:
    """Experience replay buffer for more stable learning."""
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


def set_all_seeds(seed: int):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)


def sample_start_days(n_days: int, episode_len: int, episodes: int, seed: int = 123) -> List[int]:
    rng = np.random.RandomState(seed)
    max_start = n_days - episode_len
    return [int(rng.randint(0, max_start + 1)) for _ in range(episodes)]


# ----------------------------
# Data loader
# ----------------------------
class InventoryDataset:
    def __init__(self,
                 sales_path: str,
                 price_path: str,
                 shelf_path: str,
                 stock_path: str,
                 reorder_path: str):
        self.sales = pd.read_csv(sales_path)
        self.prices = pd.read_csv(price_path)
        self.stock = pd.read_csv(stock_path)
        self.reorder = pd.read_csv(reorder_path)
        self.shelf = pd.read_csv(shelf_path)

        for name, df in [("sales", self.sales), ("prices", self.prices),
                         ("stock", self.stock), ("reorder", self.reorder)]:
            if "Day_Number" not in df.columns:
                raise ValueError(f"{name} is missing 'Day_Number'.")

        common = set(self.sales.columns) & set(self.prices.columns) & set(self.stock.columns) & set(self.reorder.columns)
        common.discard("Day_Number")
        if not common:
            raise ValueError("No common product columns across CSVs (besides Day_Number).")
        self.products: List[str] = sorted(common)

        cat_col = next((c for c in ["Category", "Catagory", "category"] if c in self.shelf.columns), None)
        if cat_col is None:
            raise ValueError("Shelflife.csv needs Category/Catagory.")
        life_col = next((c for c in ["Shelf_Life_Score", "Shelf_Life", "shelf_life", "shelf_life_score"]
                         if c in self.shelf.columns), None)
        if life_col is None:
            raise ValueError("Shelflife.csv needs a shelf-life column.")

        self.shelf_map: Dict[str, int] = {}
        for _, row in self.shelf.iterrows():
            self.shelf_map[str(row[cat_col])] = int(row[life_col])

        days = sorted(set(self.sales["Day_Number"]).intersection(
                      set(self.prices["Day_Number"])).intersection(
                      set(self.stock["Day_Number"])).intersection(
                      set(self.reorder["Day_Number"])))
        if not days:
            raise ValueError("No overlapping Day_Number across CSVs.")
        self.sales = self.sales[self.sales["Day_Number"].isin(days)].sort_values("Day_Number").reset_index(drop=True)
        self.prices = self.prices[self.prices["Day_Number"].isin(days)].sort_values("Day_Number").reset_index(drop=True)
        self.stock = self.stock[self.stock["Day_Number"].isin(days)].sort_values("Day_Number").reset_index(drop=True)
        self.reorder = self.reorder[self.reorder["Day_Number"].isin(days)].sort_values("Day_Number").reset_index(drop=True)

        for name, df in [("sales", self.sales), ("prices", self.prices),
                         ("stock", self.stock), ("reorder", self.reorder)]:
            missing = [c for c in self.products if c not in df.columns]
            if missing:
                raise ValueError(f"{name} missing product columns: {missing}")

        self.n_days = len(days)
        missing_shelf = [p for p in self.products if p not in self.shelf_map]
        if missing_shelf:
            print(f"[WARN] Missing shelf life for products (default=7): {missing_shelf}")

    def get_sequence(self, start: int, length: int) -> Dict[str, np.ndarray]:
        end = start + length
        assert end <= self.n_days, "Sequence exceeds available days."
        return {
            "sales": self.sales.iloc[start:end][self.products].to_numpy(np.float32),
            "prices": self.prices.iloc[start:end][self.products].to_numpy(np.float32),
            "reorder": self.reorder.iloc[start:end][self.products].to_numpy(np.float32),
            "init_stock": self.stock.iloc[start][self.products].to_numpy(np.float32),
        }


# ----------------------------
# Enhanced Environment
# ----------------------------
@dataclass
class EnvConfig:
    episode_len: int = 30
    demand_hist_len: int = 15
    recent_order_window: int = 7
    recent_demand_steps: int = 5
    recent_price_steps: int = 5
    lead_time_days: int = 0
    cost_ratio: float = 0.6
    holding_cost_ratio: float = 0.02
    expiration_penalty: float = 1.5
    max_action: float = 1000.0
    reward_scale: float = 100.0
    profit_bonus_scale: float = 0.5
    efficiency_bonus_scale: float = 0.3
    seed: int = 42


class PerishableInventoryEnv:
    def __init__(self, data: InventoryDataset, cfg: EnvConfig):
        self.data = data
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)

        self.products = data.products
        self.P = len(self.products)

        self._t = 0
        self._demand_seq = None
        self._price_seq = None
        self._reorder_seq = None
        self._inv_batches: List[List[Tuple[float, int]]] = []
        self._order_queue: List[List[Tuple[float, int]]] = []

        self._demand_hist = None
        self._price_hist = None
        self._order_hist = None
        self._cost_price_ratio = None
        self._profit_hist = None

        self.last_expired = np.zeros(self.P, np.float32)
        self.last_fulfilled = np.zeros(self.P, np.float32)
        self.cumulative_profit = 0.0

        self._demand_trend = np.zeros(self.P, np.float32)
        self._demand_volatility = np.zeros(self.P, np.float32)

        self.reset()

    def reset(self, start_day: Optional[int] = None):
        max_start = self.data.n_days - self.cfg.episode_len
        start = int(self.rng.randint(0, max_start + 1)) if start_day is None else int(start_day)
        seq = self.data.get_sequence(start, self.cfg.episode_len)
        self._demand_seq = seq["sales"]
        self._price_seq = seq["prices"]
        self._reorder_seq = seq["reorder"]

        self._inv_batches = []
        self._order_queue = [[] for _ in range(self.P)]
        for i, prod in enumerate(self.products):
            q0 = float(seq["init_stock"][i])
            life = int(self.data.shelf_map.get(prod, 7))
            self._inv_batches.append([[q0, life]] if q0 > 0 else [])

        self._t = 0
        self._demand_hist = np.zeros((self.P, self.cfg.demand_hist_len), np.float32)
        self._price_hist  = np.zeros((self.P, self.cfg.demand_hist_len), np.float32)
        self._order_hist  = np.zeros((self.P, self.cfg.recent_order_window), np.float32)
        self._profit_hist = np.zeros((self.P, self.cfg.recent_order_window), np.float32)
        self._cost_price_ratio = np.ones(self.P, np.float32) * self.cfg.cost_ratio
        
        self._demand_trend = np.zeros(self.P, np.float32)
        self._demand_volatility = np.ones(self.P, np.float32)
        self.cumulative_profit = 0.0
        
        return self._get_state()

    def _compute_forecasting_features(self):
        """Compute demand trend and volatility."""
        for p in range(self.P):
            hist = self._demand_hist[p]
            if hist.size > 2:
                x = np.arange(len(hist))
                trend = np.polyfit(x, hist, 1)[0] if hist.std() > 1e-6 else 0.0
                self._demand_trend[p] = float(trend)
                self._demand_volatility[p] = float(hist.std() + 1e-6)

    # ----- helpers -----
    def _total_inventory(self, p: int) -> float:
        return sum(q for q, _ in self._inv_batches[p])

    def _avg_shelf_life(self, p: int) -> float:
        total = self._total_inventory(p)
        if total <= 0.0: return 0.0
        return sum(q * life for q, life in self._inv_batches[p]) / total

    def _soon_expiring_qty(self, p: int, days: int = 2) -> float:
        return sum(q for q, life in self._inv_batches[p] if life <= days)

    def _fulfill_demand_fifo(self, p: int, d: float) -> float:
        f = 0.0
        new_batches = []
        for q, life in self._inv_batches[p]:
            if d <= 1e-8:
                new_batches.append([q, life]); continue
            use = min(q, d)
            q -= use; d -= use; f += use
            if q > 1e-8:
                new_batches.append([q, life])
        self._inv_batches[p] = new_batches
        return f

    def _age_and_expire(self) -> np.ndarray:
        expired = np.zeros(self.P, np.float32)
        for p in range(self.P):
            NB = []; exp = 0.0
            for q, life in self._inv_batches[p]:
                life -= 1
                if life <= 0: exp += q
                else: NB.append([q, life])
            self._inv_batches[p] = NB
            expired[p] = exp
        return expired

    def _deliver_orders(self):
        if self.cfg.lead_time_days <= 0: return
        for p in range(self.P):
            updated = []
            for q, lead in self._order_queue[p]:
                lead -= 1
                if lead <= 0:
                    life = int(self.data.shelf_map.get(self.products[p], 7))
                    self._inv_batches[p].append([float(q), life])
                else:
                    updated.append([q, lead])
            self._order_queue[p] = updated

    # ---- Step with enhanced rewards ----
    def step(self, action: np.ndarray):
        assert action.shape == (self.P,), f"Action must be ({self.P},)"
        order = np.clip(action, 0.0, self.cfg.max_action).astype(np.float32)

        # 0) Deliver prior orders if applicable
        self._deliver_orders()

        demand = self._demand_seq[self._t]
        price = self._price_seq[self._t]
        unit_cost = price * self.cfg.cost_ratio

        # 1) If lead_time==0, today's orders arrive now
        if self.cfg.lead_time_days == 0:
            for p, qty in enumerate(order):
                if qty > 0:
                    life = int(self.data.shelf_map.get(self.products[p], 7))
                    self._inv_batches[p].append([float(qty), life])

        # 2) Fulfill demand
        fulfilled = np.zeros(self.P, np.float32)
        stockout_penalty = np.zeros(self.P, np.float32)
        for p in range(self.P):
            fulfilled[p] = self._fulfill_demand_fifo(p, float(demand[p]))
            stockout = max(0.0, demand[p] - fulfilled[p])
            stockout_penalty[p] = stockout * price[p] * 0.5  # 50% of lost revenue

        holding_qty = np.array([self._total_inventory(p) for p in range(self.P)], np.float32)

        # 3) Age & expire
        expired = self._age_and_expire()

        # 4) Economics with bonuses
        revenue = fulfilled * price
        ordering_cost = order * unit_cost
        holding_cost = holding_qty * (unit_cost * self.cfg.holding_cost_ratio)
        expiration_cost = expired * (unit_cost * self.cfg.expiration_penalty)
        
        profit_margin = revenue - ordering_cost
        profit_bonus = np.sum(profit_margin) * self.cfg.profit_bonus_scale
        
        total_waste = np.sum(expired)
        total_handled = np.sum(fulfilled) + np.sum(expired) + 1e-6
        efficiency_ratio = 1.0 - (total_waste / total_handled)
        efficiency_bonus = efficiency_ratio * np.sum(revenue) * self.cfg.efficiency_bonus_scale
        
        reward_per_product = revenue - ordering_cost - holding_cost - expiration_cost - stockout_penalty
        raw_reward = float(reward_per_product.sum() + profit_bonus + efficiency_bonus)
        reward = raw_reward / self.cfg.reward_scale

        # 5) If lead_time>0, enqueue today's orders
        if self.cfg.lead_time_days > 0:
            for p, qty in enumerate(order):
                if qty > 0:
                    self._order_queue[p].append([float(qty), int(self.cfg.lead_time_days)])

        self.last_expired = expired; self.last_fulfilled = fulfilled
        self.cumulative_profit += raw_reward
        
        self._demand_hist = np.roll(self._demand_hist, -1, axis=1); self._demand_hist[:, -1] = demand
        self._price_hist  = np.roll(self._price_hist,  -1, axis=1); self._price_hist[:,  -1] = price
        self._order_hist  = np.roll(self._order_hist,  -1, axis=1); self._order_hist[:,  -1] = (order > 1e-6).astype(np.float32)
        self._profit_hist = np.roll(self._profit_hist, -1, axis=1); self._profit_hist[:, -1] = profit_margin
        
        self._compute_forecasting_features()

        self._t += 1
        done = self._t >= self.cfg.episode_len
        info = {
            "reward_per_product": reward_per_product, 
            "fulfilled": fulfilled, 
            "expired": expired,
            "holding_qty_end": holding_qty, 
            "price": price, 
            "unit_cost": unit_cost,
            "order": order, 
            "t": self._t, 
            "raw_reward": raw_reward,
            "profit_bonus": profit_bonus,
            "efficiency_bonus": efficiency_bonus,
            "stockout_penalty": float(np.sum(stockout_penalty)),
            "cumulative_profit": self.cumulative_profit
        }
        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        """Enhanced state with forecasting and efficiency features."""
        feats = []
        Kd, Kp = self.cfg.recent_demand_steps, self.cfg.recent_price_steps
        for p in range(self.P):
            rd = self._demand_hist[p, -Kd:]
            rp = self._price_hist[p, -Kp:]
            total_inv = np.float32(self._total_inventory(p))
            avg_life = np.float32(self._avg_shelf_life(p))
            soon = np.float32(self._soon_expiring_qty(p, days=2))
            dstd = np.float32(self._demand_hist[p].std() if self._demand_hist.shape[1] > 0 else 0.0)
            ofreq = np.float32(self._order_hist[p].mean() if self._order_hist.shape[1] > 0 else 0.0)
            cpr = np.float32(self._cost_price_ratio[p])
            trend = np.float32(self._demand_trend[p])
            volatility = np.float32(self._demand_volatility[p])
            recent_profit = np.float32(self._profit_hist[p, -3:].mean() if self._profit_hist.shape[1] >= 3 else 0.0)
            inv_turnover = np.float32(rd.mean() / (total_inv + 1e-6) if total_inv > 0 else 0.0)
            days_remaining = np.float32((self.cfg.episode_len - self._t) / self.cfg.episode_len)
            
            feats.append(np.concatenate([
                rd, rp, 
                [total_inv, avg_life, soon, dstd, ofreq, cpr, trend, volatility, 
                 recent_profit, inv_turnover, days_remaining]
            ]).astype(np.float32))
        return np.concatenate(feats, axis=0).astype(np.float32)

    @property
    def state_dim(self) -> int:
        return self.P * (self.cfg.recent_demand_steps + self.cfg.recent_price_steps + 11)

    @property
    def action_dim(self) -> int:
        return self.P

    @property
    def product_names(self) -> List[str]:
        return list(self.products)


# ----------------------------
# NADP with Attention
# ----------------------------
class AttentionActor(nn.Module):
    """Actor with attention over products."""
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 512):
        super().__init__()
        self.n_products = action_dim
        self.feature_per_product = state_dim // action_dim
        
        self.product_encoder = nn.Sequential(
            nn.Linear(self.feature_per_product, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, hidden // 2),
        )
        self.attention = nn.MultiheadAttention(hidden // 2, num_heads=8, batch_first=True)
        self.action_head = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 4),
            nn.LayerNorm(hidden // 4),
            nn.Tanh(),
            nn.Linear(hidden // 4, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.n_products, self.feature_per_product)
        product_features = self.product_encoder(x_reshaped)
        attended_features, _ = self.attention(product_features, product_features, product_features)
        actions = self.action_head(attended_features).squeeze(-1)
        std = self.log_std.exp().expand_as(actions)
        return actions, std


class EnhancedCritic(nn.Module):
    """Critic network."""
    def __init__(self, state_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class NADPConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-5
    train_updates: int = 2000
    epochs_per_update: int = 20
    batch_episodes: int = 64
    minibatch_size: int = 8192
    max_grad_norm: float = 0.5
    entropy_beta_start: float = 0.05
    entropy_beta_end: float = 0.001
    value_coef: float = 0.5
    bc_pretrain_epochs: int = 50
    residual_penalty: float = 1e-8
    use_residual: bool = True
    precompute_norm: bool = True
    use_curriculum: bool = True
    use_experience_replay: bool = True
    target_success_rate: float = 0.7
    seed: int = 123


class NADPAgent:
    def __init__(self, env: PerishableInventoryEnv, cfg: NADPConfig):
        self.env = env
        self.cfg = cfg
        set_all_seeds(cfg.seed)
        
        self.actor = AttentionActor(env.state_dim, env.action_dim)
        self.critic = EnhancedCritic(env.state_dim)
        
        self.actor_opt = optim.AdamW(self.actor.parameters(), lr=cfg.lr, weight_decay=1e-5)
        self.critic_opt = optim.AdamW(self.critic.parameters(), lr=cfg.lr * 2, weight_decay=1e-5)
        
        self.state_norm = RunningNorm(env.state_dim)
        
        if cfg.use_experience_replay:
            self.experience_buffer = ExperienceBuffer(capacity=100000)
        else:
            self.experience_buffer = None
        
        self.success_rate = 0.0
        self.episode_difficulty = 0.2  # Start easy
        
        if cfg.precompute_norm:
            print("[INFO] Pre-computing enhanced state normalization...")
            self._precompute_state_stats()

    def _precompute_state_stats(self, episodes: int = 200):
        states = []
        for _ in range(episodes):
            start_day = np.random.randint(0, max(1, self.env.data.n_days - self.env.cfg.episode_len))
            s = self.env.reset(start_day)
            done = False
            while not done:
                states.append(s)
                a = policy_dataset_reorder(self.env)
                a += np.random.normal(0, 10, size=a.shape)
                a = np.clip(a, 0, self.env.cfg.max_action)
                s, _, done, _ = self.env.step(a)
        if states:
            states_array = np.array(states)
            self.state_norm.update(states_array)
            print(f"[INFO] Computed enhanced state stats from {len(states)} samples")

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x).float()

    def _norm_state(self, s: np.ndarray, update_stats: bool = False) -> np.ndarray:
        if update_stats and not self.cfg.precompute_norm:
            self.state_norm.update(s)
        return self.state_norm.normalize(s)

    def baseline_action(self, env: PerishableInventoryEnv) -> np.ndarray:
        return policy_dataset_reorder(env)

    def _adaptive_exploration(self, update_num: int) -> float:
        progress = update_num / self.cfg.train_updates
        base_entropy = self.cfg.entropy_beta_end + (self.cfg.entropy_beta_start - self.cfg.entropy_beta_end) * (1 - progress)
        if self.success_rate < self.cfg.target_success_rate:
            exploration_boost = (self.cfg.target_success_rate - self.success_rate) * 2.0
            base_entropy *= (1 + exploration_boost)
        return min(base_entropy, 0.5)

    def _step_policy(self, s: np.ndarray, env: PerishableInventoryEnv, deterministic: bool=False, update_stats: bool=False):
        s_n = self._norm_state(s, update_stats)
        st = self._to_tensor(s_n).unsqueeze(0)
        self.actor.eval(); self.critic.eval()
        with torch.no_grad():
            mean_out, std = self.actor(st)
            value = self.critic(st).squeeze(0)
            dist = Normal(mean_out, std)
            if deterministic:
                policy_output = mean_out
                lp = dist.log_prob(policy_output).sum()
            else:
                policy_output = dist.sample()
                lp = dist.log_prob(policy_output).sum()
        
        if self.cfg.use_residual:
            base = self.baseline_action(env)
            residual = policy_output.squeeze(0).numpy()
            adaptive_scale = np.maximum(np.abs(base) * 0.5, 10.0)
            scaled_residual = residual * adaptive_scale
            action = base + scaled_residual
            stored_output = residual
        else:
            raw_output = policy_output.squeeze(0).numpy()
            action = torch.sigmoid(policy_output).squeeze(0).numpy() * env.cfg.max_action
            stored_output = raw_output
        
        action = np.maximum(action, 0.0)
        action = np.minimum(action, env.cfg.max_action)
        
        return action, float(lp.numpy()), float(value.numpy()), stored_output

    def _curriculum_start_day(self) -> Optional[int]:
        if not self.cfg.use_curriculum:
            return None
        max_start = self.env.data.n_days - self.env.cfg.episode_len
        max_start = max(1, max_start)
        if self.episode_difficulty < 0.5:
            mid_start = max_start // 3
            mid_end = 2 * max_start // 3
            return int(np.random.randint(mid_start, max(mid_start+1, mid_end)))
        elif self.episode_difficulty < 0.8:
            return int(np.random.randint(0, max_start))
        else:
            if np.random.random() < 0.7:
                return int(np.random.randint(0, max(1, max_start // 4)))
            else:
                return int(np.random.randint(max(1, 3 * max_start // 4), max_start))

    def collect(self, n_episodes: int, update_num: int = 0):
        """Collect trajectories; compute returns/advantages; update curriculum; add to replay."""
        S, OUTPUT, LP, R, V, L = [], [], [], [], [], []
        raw_rewards = []
        episode_successes = []
        per_episode_cache = []  # store per-episode lists to split later

        for _ in range(n_episodes):
            start_day = self._curriculum_start_day()
            s = self.env.reset(start_day)
            epS, epO, epLP, epR, epV, epRaw = [], [], [], [], [], []
            done = False
            while not done:
                a, lp, v, output = self._step_policy(s, self.env, deterministic=False, 
                                                   update_stats=not self.cfg.precompute_norm)
                ns, r, done, info = self.env.step(a)
                epS.append(self._norm_state(s, update_stats=False))
                epO.append(output)
                epLP.append(lp); epR.append(r); epV.append(v)
                epRaw.append(info.get('raw_reward', r * self.env.cfg.reward_scale))
                s = ns

            ep_len = len(epR)
            L.append(ep_len)
            S.extend(epS); OUTPUT.extend(epO); LP.extend(epLP); R.extend(epR); V.extend(epV)
            raw_rewards.extend(epRaw)
            per_episode_cache.append({
                "states": np.array(epS, np.float32),
                "outputs": np.array(epO, np.float32),
                "logprobs": np.array(epLP, np.float32),
                "rewards": np.array(epR, np.float32),
                "values":  np.array(epV, np.float32),
                "raw_rewards": np.array(epRaw, np.float32),
                "length": ep_len
            })

            # simple success heuristic: raw episode reward > 0
            episode_successes.append(float(np.sum(epRaw) > 0.0))

        # curriculum update
        if self.cfg.use_curriculum:
            self.success_rate = float(np.mean(episode_successes)) if episode_successes else 0.0
            if self.success_rate > self.cfg.target_success_rate:
                self.episode_difficulty = min(1.0, self.episode_difficulty + 0.05)
            else:
                self.episode_difficulty = max(0.1, self.episode_difficulty - 0.02)

        traj = {
            "states": np.array(S, np.float32),
            "outputs": np.array(OUTPUT, np.float32),
            "logprobs": np.array(LP, np.float32),
            "rewards": np.array(R, np.float32),
            "values": np.array(V, np.float32),
            "lengths": np.array(L, np.int32),
            "raw_rewards": np.array(raw_rewards, np.float32),
        }

        # GAE
        returns, advs = [], []
        idx = 0
        for ep_len in L:
            r = traj["rewards"][idx:idx+ep_len]
            v = traj["values"][idx:idx+ep_len]
            terminal_value = 0.0  # finite horizon
            v_next = np.concatenate([v[1:], np.array([terminal_value], dtype=np.float32)])
            deltas = r + self.cfg.gamma * v_next - v
            gae = 0.0
            ep_adv = np.zeros(ep_len, np.float32)
            for t in reversed(range(ep_len)):
                gae = deltas[t] + self.cfg.gamma * self.cfg.lam * gae
                ep_adv[t] = gae
            ep_ret = ep_adv + v
            returns.extend(ep_ret.tolist()); advs.extend(ep_adv.tolist())
            idx += ep_len

        traj["returns"] = np.array(returns, np.float32)
        adv = np.array(advs, np.float32)
        traj["advantages"] = (adv - adv.mean()) / (adv.std() + 1e-6)

        # add episodes with returns/advantages to replay buffer
        if self.cfg.use_experience_replay and self.experience_buffer is not None:
            base = 0
            for cache in per_episode_cache:
                ep_len = cache["length"]
                ep_adv = traj["advantages"][base:base+ep_len]
                ep_ret = traj["returns"][base:base+ep_len]
                base += ep_len
                cache["advantages"] = ep_adv.astype(np.float32)
                cache["returns"] = ep_ret.astype(np.float32)
                self.experience_buffer.add(cache)

        return traj

    def nadp_update(self, traj, entropy_beta: float, update_num: int):
        """NADP update with (optional) experience replay mixing."""
        states = self._to_tensor(traj["states"])
        outputs = self._to_tensor(traj["outputs"])
        old_logprobs = self._to_tensor(traj["logprobs"])
        returns = self._to_tensor(traj["returns"])
        advantages = self._to_tensor(traj["advantages"])
        old_values = self._to_tensor(traj["values"])

        # Mix in experience replay (episode-level)
        if self.cfg.use_experience_replay and self.experience_buffer is not None and len(self.experience_buffer) > 1000:
            replay_eps = self.experience_buffer.sample(min(16, len(self.experience_buffer)))
            if replay_eps:
                r_states = torch.cat([self._to_tensor(e["states"]) for e in replay_eps], dim=0)
                r_outputs = torch.cat([self._to_tensor(e["outputs"]) for e in replay_eps], dim=0)
                r_logprobs = torch.cat([self._to_tensor(e["logprobs"]) for e in replay_eps], dim=0)
                r_returns = torch.cat([self._to_tensor(e["returns"]) for e in replay_eps], dim=0)
                r_adv = torch.cat([self._to_tensor(e["advantages"]) for e in replay_eps], dim=0)
                r_values = torch.cat([self._to_tensor(e["values"]) for e in replay_eps], dim=0)
                states = torch.cat([states, r_states], dim=0)
                outputs = torch.cat([outputs, r_outputs], dim=0)
                old_logprobs = torch.cat([old_logprobs, r_logprobs], dim=0)
                returns = torch.cat([returns, r_returns], dim=0)
                advantages = torch.cat([advantages, r_adv], dim=0)
                old_values = torch.cat([old_values, r_values], dim=0)
                # re-normalize advantages across the combined batch
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = states.size(0)
        mb = min(self.cfg.minibatch_size, n)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for epoch in range(self.cfg.epochs_per_update):
            idx_all = torch.randperm(n)
            for start in range(0, n, mb):
                idx = idx_all[start:start+mb]
                s_b = states[idx]
                out_b = outputs[idx]
                old_lp_b = old_logprobs[idx]
                adv_b = advantages[idx]
                ret_b = returns[idx]
                old_val_b = old_values[idx]

                mean_out, std = self.actor(s_b)
                dist = Normal(mean_out, std)
                entropy = dist.entropy().sum(dim=1).mean()
                new_logprob = dist.log_prob(out_b).sum(dim=1)
                ratio = (new_logprob - old_lp_b).exp()

                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # residual penalty (smaller when success higher)
                if self.cfg.use_residual:
                    penalty_scale = max(1e-8, self.cfg.residual_penalty * (1.0 - self.success_rate))
                    residual_penalty = penalty_scale * (out_b.pow(2).mean())
                else:
                    residual_penalty = torch.tensor(0.0)

                # Value loss with clipping around old values from rollout
                value_pred = self.critic(s_b)
                value_loss_unclipped = (value_pred - ret_b).pow(2)
                value_clipped = old_val_b + torch.clamp(value_pred - old_val_b, -0.5, 0.5)
                value_loss_clipped = (value_clipped - ret_b).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Optimize actor
                self.actor_opt.zero_grad()
                actor_loss = policy_loss - entropy_beta * entropy + residual_penalty
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.actor_opt.step()
                
                # Optimize critic
                self.critic_opt.zero_grad()
                critic_loss = self.cfg.value_coef * value_loss
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.critic_opt.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())

        denom = self.cfg.epochs_per_update * ((n + mb - 1) // mb)
        metrics = {
            "policy_loss": total_policy_loss / max(1, denom),
            "value_loss": total_value_loss / max(1, denom),
            "entropy": total_entropy / max(1, denom),
            "residual_penalty": float(residual_penalty.item()) if self.cfg.use_residual else 0.0,
        }
        return metrics

    def bc_pretrain(self, epochs: int):
        """Behavior cloning: teach residual ~ 0 (or approximate baseline)."""
        if epochs <= 0:
            print(f"[BC] Skipping pretraining (epochs={epochs})")
            return
            
        print(f"[BC] Enhanced pretraining for {epochs} epochs...")
        mse = nn.MSELoss()
        
        for e in range(epochs):
            S, Y = [], []
            difficulty = min(1.0, e / max(1, epochs - 1))
            for _ in range(self.cfg.batch_episodes):
                if difficulty < 0.5:
                    start_day = np.random.randint(self.env.data.n_days // 4, 3 * self.env.data.n_days // 4)
                else:
                    start_day = None
                s = self.env.reset(start_day)
                done = False
                while not done:
                    S.append(self._norm_state(s, update_stats=not self.cfg.precompute_norm))
                    if self.cfg.use_residual:
                        target = np.zeros((self.env.P,), dtype=np.float32)  # residual ≈ 0
                    else:
                        base_action = policy_dataset_reorder(self.env)
                        # invert sigmoid approximately via logit for target mean:
                        base_ratio = np.clip(base_action / max(1e-6, self.env.cfg.max_action), 1e-5, 1-1e-5)
                        target = np.log(base_ratio) - np.log(1 - base_ratio)
                        target = target.astype(np.float32)
                    Y.append(target)
                    a = policy_dataset_reorder(self.env)
                    s, _, done, _ = self.env.step(a)
            S = self._to_tensor(np.array(S, np.float32))
            Y = self._to_tensor(np.array(Y, np.float32))
            for _ in range(3):
                mean_out, _ = self.actor(S)
                loss = mse(mean_out, Y)
                self.actor_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.actor_opt.step()
            if e == 0 or (e + 1) % 10 == 0:
                print(f"[BC] epoch {e+1}/{epochs}  loss={loss.item():.6f}  difficulty={difficulty:.2f}")

    def train(self):
        """Training loop."""
        print("[INFO] Starting NADP training with curriculum learning...")
        self.bc_pretrain(self.cfg.bc_pretrain_epochs)

        reward_log = []
        raw_reward_log = []
        success_rate_log = []
        best_performance = float('-inf')
        
        for u in range(self.cfg.train_updates):
            beta = self._adaptive_exploration(u)
            traj = self.collect(self.cfg.batch_episodes, u)
            avg_r = float(traj["rewards"].mean())
            avg_raw_r = float(traj["raw_rewards"].mean())
            reward_log.append(avg_r)
            raw_reward_log.append(avg_raw_r)
            success_rate_log.append(self.success_rate)
            metrics = self.nadp_update(traj, beta, u)

            if avg_raw_r > best_performance:
                best_performance = avg_raw_r
                self._save_checkpoint("nadp_actor_multi_best.pth")

            if (u + 1) % 20 == 0 or u == 0:
                print(f"[Update {u+1}/{self.cfg.train_updates}] "
                      f"scaled_r={avg_r:.3f} raw_r={avg_raw_r:.1f} | "
                      f"success_rate={self.success_rate:.2f} difficulty={self.episode_difficulty:.2f} | "
                      f"pi_loss={metrics.get('policy_loss',0):.4f} v_loss={metrics.get('value_loss',0):.4f} | "
                      f"entropy={metrics.get('entropy',0):.4f} β={beta:.4f}")
        
        self._save_checkpoint("nadp_actor_multi.pth")
        self._plot_training_curves(reward_log, raw_reward_log, success_rate_log, "enhanced_nadp_training.png")
        print("Training completed! Saved models and training curves.")

    def _save_checkpoint(self, path: str):
        ckpt = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "state_norm": self.state_norm.state_dict(),
            "config": self.cfg,
            "success_rate": self.success_rate,
            "episode_difficulty": self.episode_difficulty,
        }
        torch.save(ckpt, path)

    @staticmethod
    def _plot_training_curves(scaled_rewards: List[float], raw_rewards: List[float], 
                              success_rates: List[float], path: str):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        ax1.plot(scaled_rewards, label="Scaled reward", alpha=0.7)
        ax1.set_xlabel("Update"); ax1.set_ylabel("Average scaled reward")
        ax1.set_title("NADP Training - Scaled Rewards"); ax1.legend(); ax1.grid(True, alpha=0.3)
        updates = np.arange(len(raw_rewards))
        ax2.plot(updates, raw_rewards, label="Raw reward", alpha=0.7)
        if len(raw_rewards) > 10:
            z = np.polyfit(updates, raw_rewards, 1)
            p = np.poly1d(z)
            ax2.plot(updates, p(updates), "--", alpha=0.8, label="Trend")
        ax2.set_xlabel("Update"); ax2.set_ylabel("Average raw reward")
        ax2.set_title("NADP Training - Raw Performance"); ax2.legend(); ax2.grid(True, alpha=0.3)
        ax3.plot(success_rates, label="Success rate", alpha=0.7)
        ax3.axhline(y=0.7, linestyle='--', label='Target (70%)')
        ax3.set_xlabel("Update"); ax3.set_ylabel("Success rate")
        ax3.set_title("Training Success Rate"); ax3.legend(); ax3.grid(True, alpha=0.3)
        if len(raw_rewards) > 50:
            recent_rewards = raw_rewards[-50:]
            ax4.hist(recent_rewards, bins=20, alpha=0.7)
            ax4.axvline(np.mean(recent_rewards), linestyle='--', label=f'Recent mean: {np.mean(recent_rewards):.0f}')
            ax4.set_xlabel("Raw reward"); ax4.set_ylabel("Frequency")
            ax4.set_title("Recent Performance Distribution"); ax4.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()


# ----------------------------
# Baseline policies
# ----------------------------
def policy_dataset_reorder(env: PerishableInventoryEnv) -> np.ndarray:
    t = env._t
    if t < env._reorder_seq.shape[0]:
        return env._reorder_seq[t].astype(np.float32)
    return np.zeros(env.P, np.float32)

def policy_base_stock(env: PerishableInventoryEnv, k_days: int = 5, safety_mult: float = 1.0) -> np.ndarray:
    P = env.P
    orders = np.zeros(P, np.float32)
    for p in range(P):
        hist = env._demand_hist[p, -k_days:]
        mean_d = float(hist.mean()) if hist.size > 0 else 0.0
        target = mean_d * max(1, int(env._avg_shelf_life(p))) * safety_mult
        cur = env._total_inventory(p)
        orders[p] = max(0.0, target - cur)
    return orders


# ----------------------------
# Enhanced evaluation functions
# ----------------------------
def evaluate_nadp(env: PerishableInventoryEnv, actor_path: str, episodes: int = 50,
                          deterministic: bool = True, seed: int = 123, start_days: Optional[List[int]] = None) -> Dict[str, float]:
    """Evaluate NADP actor."""
    set_all_seeds(seed)
    dummy_cfg = NADPConfig(seed=seed, bc_pretrain_epochs=0, precompute_norm=False)
    agent = NADPAgent(env, dummy_cfg)
    try:
        ckpt = torch.load(actor_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load actor from {actor_path}: {e}")
    if isinstance(ckpt, dict) and "actor" in ckpt:
        agent.actor.load_state_dict(ckpt["actor"])
        if "state_norm" in ckpt:
            agent.state_norm.load_state_dict(ckpt["state_norm"])
        if "config" in ckpt:
            agent.cfg = ckpt["config"]
    else:
        agent.actor.load_state_dict(ckpt)
    agent.actor.eval()

    total_raw = 0.0
    per_ep_raw = []
    total_expired = 0.0
    total_fulfilled = 0.0
    
    for i in range(episodes):
        s = env.reset(start_day=None if start_days is None else start_days[i])
        done = False
        ep_ret_raw = 0.0
        ep_expired = 0.0
        ep_fulfilled = 0.0
        while not done:
            s_n = agent._norm_state(s, update_stats=False)
            st = agent._to_tensor(s_n).unsqueeze(0)
            with torch.no_grad():
                mean_out, _ = agent.actor(st)
            if agent.cfg.use_residual:
                base = policy_dataset_reorder(env)
                adaptive_scale = np.maximum(np.abs(base) * 0.5, 10.0)
                residual = mean_out.squeeze(0).numpy() * adaptive_scale
                a = base + residual
            else:
                a = torch.sigmoid(mean_out).squeeze(0).numpy() * env.cfg.max_action
            a = np.clip(a, 0.0, env.cfg.max_action)
            s, r, done, info = env.step(a)
            ep_ret_raw += info.get('raw_reward', r * env.cfg.reward_scale)
            ep_expired += np.sum(info.get('expired', 0))
            ep_fulfilled += np.sum(info.get('fulfilled', 0))
        per_ep_raw.append(ep_ret_raw)
        total_raw += ep_ret_raw
        total_expired += ep_expired
        total_fulfilled += ep_fulfilled
        
    waste_rate = total_expired / (total_fulfilled + total_expired + 1e-6)
    stats = {
        "avg_return": float(total_raw / episodes),
        "std_return": float(np.std(per_ep_raw)),
        "min_return": float(np.min(per_ep_raw)),
        "max_return": float(np.max(per_ep_raw)),
        "waste_rate": float(waste_rate),
        "episodes": int(episodes)
    }
    print(f"NADP eval: avg={stats['avg_return']:.1f}±{stats['std_return']:.1f}, waste_rate={waste_rate:.3f}")
    return stats


# ----------------------------
# Main function with enhanced defaults
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="NADP for Multi-Product Perishable Inventory Control")
    # Dataset paths
    parser.add_argument("--sales", type=str, default="Sales_Quantity.csv")
    parser.add_argument("--prices", type=str, default="Unit_Price.csv")
    parser.add_argument("--shelf", type=str, default="Shelflife.csv")
    parser.add_argument("--stock", type=str, default="Stock_Quantity.csv")
    parser.add_argument("--reorder", type=str, default="Reorder_Quantity.csv")
    
    # Env configs
    parser.add_argument("--episode_len", type=int, default=30)
    parser.add_argument("--demand_hist_len", type=int, default=15)
    parser.add_argument("--recent_order_window", type=int, default=7)
    parser.add_argument("--lead_time_days", type=int, default=0)
    parser.add_argument("--cost_ratio", type=float, default=0.6)
    parser.add_argument("--holding_cost_ratio", type=float, default=0.02)
    parser.add_argument("--expiration_penalty", type=float, default=1.5)
    parser.add_argument("--max_action", type=float, default=1000.0)
    parser.add_argument("--reward_scale", type=float, default=100.0)
    parser.add_argument("--profit_bonus_scale", type=float, default=0.5)
    parser.add_argument("--efficiency_bonus_scale", type=float, default=0.3)
    parser.add_argument("--env_seed", type=int, default=42)
    
    # NADP configs
    parser.add_argument("--mode", type=str, choices=["train", "eval", "compare"], default="train")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--train_updates", type=int, default=2000)
    parser.add_argument("--epochs_per_update", type=int, default=20)
    parser.add_argument("--batch_episodes", type=int, default=64)
    parser.add_argument("--bc_pretrain_epochs", type=int, default=50)
    parser.add_argument("--use_curriculum", action="store_true", default=True)
    parser.add_argument("--use_experience_replay", action="store_true", default=True)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--actor_path", type=str, default="nadp_actor_multi_best.pth")
    parser.add_argument("--nadp_seed", type=int, default=123)

    args = parser.parse_args()

    # Build env
    ds = InventoryDataset(args.sales, args.prices, args.shelf, args.stock, args.reorder)
    cfg = EnvConfig(
        episode_len=args.episode_len,
        demand_hist_len=args.demand_hist_len,
        recent_order_window=args.recent_order_window,
        recent_demand_steps=5,
        recent_price_steps=5,
        lead_time_days=args.lead_time_days,
        cost_ratio=args.cost_ratio,
        holding_cost_ratio=args.holding_cost_ratio,
        expiration_penalty=args.expiration_penalty,
        max_action=args.max_action,
        reward_scale=args.reward_scale,
        profit_bonus_scale=args.profit_bonus_scale,
        efficiency_bonus_scale=args.efficiency_bonus_scale,
        seed=args.env_seed,
    )
    env = PerishableInventoryEnv(ds, cfg)

    if args.mode == "train":
        nadp_cfg = NADPConfig(
            lr=args.lr,
            train_updates=args.train_updates,
            epochs_per_update=args.epochs_per_update,
            batch_episodes=args.batch_episodes,
            bc_pretrain_epochs=args.bc_pretrain_epochs,
            use_curriculum=args.use_curriculum,
            use_experience_replay=args.use_experience_replay,
            seed=args.nadp_seed,
        )
        print(f"[CONFIG] Enhanced training mode with curriculum={args.use_curriculum}")
        print(f"[CONFIG] Training updates: {args.train_updates}, Episodes per update: {args.batch_episodes}")
        print(f"[CONFIG] Environment: episode_len={args.episode_len}, max_action={args.max_action}")
        agent = NADPAgent(env, nadp_cfg)
        agent.train()

    elif args.mode == "eval":
        stats = evaluate_nadp(env, actor_path=args.actor_path, episodes=args.eval_episodes)
        print("NADP Results:", stats)
        print("Products:", env.product_names)

    elif args.mode == "compare":
        # Evaluate NADP on fixed starts
        starts = sample_start_days(env.data.n_days, env.cfg.episode_len, args.eval_episodes, seed=args.nadp_seed)
        nadp_stats = evaluate_nadp(env, actor_path=args.actor_path, episodes=args.eval_episodes,
                                          start_days=starts)
        # Evaluate baseline on same starts
        set_all_seeds(args.nadp_seed)
        baseline_rewards = []
        for i in range(args.eval_episodes):
            env.reset(start_day=starts[i])
            done = False
            ep_reward = 0.0
            while not done:
                a = policy_dataset_reorder(env)
                _, r, done, info = env.step(a)
                ep_reward += info.get('raw_reward', r * env.cfg.reward_scale)
            baseline_rewards.append(ep_reward)
        baseline_stats = {
            "avg_return": float(np.mean(baseline_rewards)),
            "std_return": float(np.std(baseline_rewards)),
            "episodes": args.eval_episodes
        }
        print("\n=== PERFORMANCE COMPARISON ===")
        print(f"Dataset Reorder: {baseline_stats['avg_return']:.1f} ± {baseline_stats['std_return']:.1f}")
        print(f"NADP:    {nadp_stats['avg_return']:.1f} ± {nadp_stats['std_return']:.1f}")
        improvement = (nadp_stats['avg_return'] - baseline_stats['avg_return']) / (abs(baseline_stats['avg_return']) + 1e-8) * 100
        print(f"Improvement:     {improvement:.1f}%")
        print(f"NADP Waste Rate:  {nadp_stats.get('waste_rate', 0):.1%}")
        # Plot
        plt.figure(figsize=(10, 6))
        policies = ['Dataset Reorder', 'NADP']
        means = [baseline_stats['avg_return'], nadp_stats['avg_return']]
        stds = [baseline_stats['std_return'], nadp_stats['std_return']]
        bars = plt.bar(policies, means, yerr=stds, capsize=8, alpha=0.7)
        plt.ylabel("Average Return (± std)")
        plt.title("NADP vs Dataset Reorder Performance")
        for bar, val, std in zip(bars, means, stds):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (std if std>0 else 0) + 1.0, 
                     f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("nadp_vs_baseline.png", dpi=150)
        plt.close()
        print("Saved comparison plot to nadp_vs_baseline.png")


if __name__ == "__main__":
    main()
