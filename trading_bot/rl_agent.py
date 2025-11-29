import logging
import math
import os
import random
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from . import config

logger = logging.getLogger(__name__)

_STATE_FEATURES = (
    "risk_reward",
    "orig_prob",
    "rsi",
    "macd",
    "atr",
    "sentiment",
    "order_book_imbalance",
    "volume_factor",
    "volatility",
)


@dataclass
class Experience:
    state: np.ndarray
    action: np.ndarray
    reward: float


class _OfflineTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action):  # type: ignore[override]
        del action
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        terminated = True
        truncated = False
        return obs, reward, terminated, truncated, {}


class _ReplayTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        experiences: Sequence[Experience],
        observation_space: gym.Space,
        action_space: gym.Space,
    ):
        super().__init__()
        self._experiences = list(experiences)
        self.observation_space = observation_space
        self.action_space = action_space
        self._current: Experience | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        if not self._experiences:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            self._current = None
            return obs, {}
        self._current = random.choice(self._experiences)
        return self._current.state.astype(np.float32), {}

    def step(self, action):  # type: ignore[override]
        if self._current is None:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, {}

        expected = self._current.action
        reward = float(self._current.reward)
        penalty = 0.0
        try:
            delta = np.asarray(action, dtype=float) - np.asarray(expected, dtype=float)
            penalty = float(np.linalg.norm(delta))
        except Exception:
            penalty = 0.0
        reward = reward - penalty
        terminated = True
        truncated = False
        info = {"expected_action": expected}
        obs = self._current.state.astype(np.float32)
        self._current = None
        return obs, reward, terminated, truncated, info


class RLTradingAgent:
    def __init__(self) -> None:
        agent_control_enabled = bool(getattr(config, "AGENT_CONTROL_ENABLED", False))
        env_overrides_rl = "RL_AGENT_ENABLED" in os.environ
        self.enabled = bool(getattr(config, "RL_AGENT_ENABLED", False))
        if agent_control_enabled and not env_overrides_rl and not self.enabled:
            logger.info("Enabling RL agent to mirror AGENT_CONTROL_ENABLED setting")
            self.enabled = True
        tp_low, tp_high = getattr(config, "RL_TP_MULT_RANGE", (0.5, 3.0))
        sl_low, sl_high = getattr(config, "RL_SL_MULT_RANGE", (0.5, 2.5))
        self.tp_range = (float(tp_low), float(tp_high))
        self.sl_range = (float(sl_low), float(sl_high))
        self._algo = getattr(config, "RL_ALGO", "ppo").strip().lower() or "ppo"
        self._policy_path = Path(getattr(config, "RL_POLICY_PATH", "models/rl_policy.zip"))
        self._buffer = deque(maxlen=int(getattr(config, "RL_BUFFER_CAPACITY", 200)))
        self._learn_interval = int(getattr(config, "RL_LEARN_INTERVAL", 20))
        self._min_samples = int(getattr(config, "RL_MIN_TRAINING_SAMPLES", 25))
        self._training_steps = int(getattr(config, "RL_LEARN_STEPS", 200))
        self._discrete_tp_bins = int(getattr(config, "RL_DISCRETE_TP_BINS", 4))
        self._discrete_sl_bins = int(getattr(config, "RL_DISCRETE_SL_BINS", 3))
        self._max_tp_pct = float(getattr(config, "RL_MAX_TP_PCT", 0.0))
        self._max_stop_pct = float(getattr(config, "RL_MAX_STOP_LOSS_PCT", 0.0))
        self._lock = threading.RLock()
        self._learn_calls = 0

        self._discrete_actions = self._build_discrete_actions()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(_STATE_FEATURES) + 3,),
            dtype=np.float32,
        )
        if self._algo == "dqn":
            self.action_space = gym.spaces.Discrete(len(self._discrete_actions))
        else:
            self.action_space = gym.spaces.Box(
                low=np.array([self.tp_range[0], self.sl_range[0]], dtype=np.float32),
                high=np.array([self.tp_range[1], self.sl_range[1]], dtype=np.float32),
                dtype=np.float32,
        )

        self._model = self._load_model()

    # --- Nuevos métodos para el agente maestro ---

    def encode_state(self, observation: dict | None) -> np.ndarray:
        """Vectoriza una observación de alto nivel.

        Incluye señales básicas de riesgo (trades abiertos, drawdown) y
        características de mercado derivadas de ``observation``.
        """

        observation = observation or {}
        market = observation.get("market") or {}
        closes = market.get("close") or []
        high = market.get("high") or []
        low = market.get("low") or []
        open_trades = observation.get("open_trades") or []
        account = observation.get("account") or {}
        signal = observation.get("candidate_signal") or {}

        last_close = float(closes[-1]) if closes else 0.0
        last_high = float(high[-1]) if high else last_close
        last_low = float(low[-1]) if low else last_close
        price_range = last_high - last_low
        price_range = price_range / max(last_close, 1e-6)
        open_count = float(len(open_trades))
        daily_dd_pct = float(account.get("daily_drawdown_pct") or 0.0)
        prob_success = float(signal.get("prob_success") or 0.5)

        state_fields = [
            last_close,
            price_range,
            open_count,
            daily_dd_pct,
            prob_success,
        ]

        # Asegurar tamaño fijo con padding si faltan datos técnicos
        return np.asarray(state_fields, dtype=np.float32)

    def decide_action(
        self,
        *,
        state_vec: np.ndarray | None,
        direction: str,
        confidence: float,
        has_open_trades: bool,
    ) -> dict[str, float | str | None]:
        """Elegir acción de alto nivel combinando política RL y señal base."""

        confidence = max(0.0, min(1.0, float(confidence)))
        direction = direction.lower()

        action = "HOLD"

        # Inferir tipo de acción
        if direction == "long" and confidence >= 0.55:
            action = "OPEN_LONG"
        elif direction == "short" and confidence >= 0.55:
            action = "OPEN_SHORT"
        elif has_open_trades and confidence <= 0.35:
            action = "CLOSE_TRADE"

        # Obtener multiplicadores TP/SL desde la política entrenada
        action_vector = None
        action_idx = None
        if self.enabled and state_vec is not None:
            try:
                action_vector, action_idx = self._predict_action(np.asarray(state_vec, dtype=np.float32))
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("RL predict failed, fallback multipliers: %s", exc)

        size_mult = 1.0 + (confidence - 0.5) * 1.5
        size_mult = float(np.clip(size_mult, 0.25, 2.0))
        tp_mult, sl_mult = 1.0, 1.0
        if action_vector is not None and action_vector.size >= 2:
            tp_mult = float(action_vector[0])
            sl_mult = float(action_vector[1])

        decision = {
            "action": action,
            "size_mult": size_mult,
            "tp_mult": tp_mult,
            "sl_mult": sl_mult,
            "action_index": action_idx,
        }

        logger.info(
            "RL agent decision",
            extra={
                "rl_enabled": self.enabled,
                "direction": direction,
                "confidence": confidence,
                "has_open_trades": has_open_trades,
                "decision": decision,
            },
        )

        return decision

    def _build_discrete_actions(self) -> List[tuple[float, float]]:
        tp_vals = np.linspace(self.tp_range[0], self.tp_range[1], self._discrete_tp_bins)
        sl_vals = np.linspace(self.sl_range[0], self.sl_range[1], self._discrete_sl_bins)
        combos: List[tuple[float, float]] = []
        for tp_val in tp_vals:
            for sl_val in sl_vals:
                combos.append((float(tp_val), float(sl_val)))
        return combos

    def _make_env(self) -> gym.Env:
        return _OfflineTradingEnv(self.observation_space, self.action_space)

    def _load_model(self):
        if self._algo == "dqn":
            model_cls: Callable[..., DQN | PPO] = DQN
        else:
            model_cls = PPO

        env = self._make_env()
        model = None
        if self._policy_path.exists():
            try:
                if self._algo == "dqn":
                    model = model_cls.load(self._policy_path, env=env)
                else:
                    model = model_cls.load(
                        self._policy_path, env=DummyVecEnv([self._make_env])
                    )
                logger.info("Loaded RL policy from %s", self._policy_path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to load RL policy from %s: %s", self._policy_path, exc)
                model = None

        if model is None:
            if self._algo == "dqn":
                model = DQN("MlpPolicy", env, verbose=0, learning_starts=1)
            else:
                model = PPO("MlpPolicy", DummyVecEnv([self._make_env]), verbose=0)
        return model

    def _save_policy(self) -> None:
        try:
            self._policy_path.parent.mkdir(parents=True, exist_ok=True)
            self._model.save(self._policy_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to persist RL policy to %s: %s", self._policy_path, exc)

    def _state_vector(
        self,
        feature_snapshot: dict,
        entry_price: float,
        take_profit: float,
        stop_loss: float,
    ) -> np.ndarray:
        side_raw = feature_snapshot.get("side") or ""
        side = 1.0 if str(side_raw).lower().startswith("long") or side_raw == "BUY" else -1.0
        entry = float(entry_price) if entry_price else 0.0
        tp_dist = (float(take_profit) - entry) / max(entry, 1e-6)
        sl_dist = (entry - float(stop_loss)) / max(entry, 1e-6)
        fields: list[float] = []
        for name in _STATE_FEATURES:
            value = feature_snapshot.get(name)
            if value is None:
                fields.append(0.0)
                continue
            try:
                fields.append(float(value))
            except (TypeError, ValueError):
                fields.append(0.0)
        fields.extend([side, tp_dist, sl_dist])
        return np.asarray(fields, dtype=np.float32)

    def _predict_action(self, obs: np.ndarray) -> tuple[np.ndarray, int | None]:
        if self._algo == "dqn":
            action_idx, _ = self._model.predict(obs, deterministic=True)
            idx_int = int(action_idx)
            idx_int = max(0, min(idx_int, len(self._discrete_actions) - 1))
            tp_mult, sl_mult = self._discrete_actions[idx_int]
            return np.array([tp_mult, sl_mult], dtype=np.float32), idx_int

        action, _ = self._model.predict(obs, deterministic=True)
        action_arr = np.asarray(action, dtype=float).reshape(-1)
        if action_arr.size < 2:
            action_arr = np.pad(action_arr, (0, 2 - action_arr.size), constant_values=1.0)
        tp_mult, sl_mult = action_arr[:2]
        tp_mult = float(np.clip(tp_mult, self.tp_range[0], self.tp_range[1]))
        sl_mult = float(np.clip(sl_mult, self.sl_range[0], self.sl_range[1]))
        return np.array([tp_mult, sl_mult], dtype=np.float32), None

    def _apply_safety_limits(
        self, entry_price: float, take_profit: float, stop_loss: float
    ) -> tuple[float, float]:
        entry = float(entry_price)
        tp_val = float(take_profit)
        sl_val = float(stop_loss)
        if not np.isfinite(entry) or entry <= 0:
            return tp_val, sl_val

        max_stop_pct = max(0.0, self._max_stop_pct)
        if max_stop_pct > 0:
            if tp_val >= entry:
                min_stop = entry * (1 - max_stop_pct)
                sl_val = max(sl_val, min_stop)
                sl_val = max(sl_val, 0.0)
            else:
                max_stop = entry * (1 + max_stop_pct)
                sl_val = min(sl_val, max_stop)

        max_tp_pct = max(0.0, self._max_tp_pct)
        if max_tp_pct > 0:
            if tp_val >= entry:
                tp_cap = entry * (1 + max_tp_pct)
                tp_val = min(tp_val, tp_cap)
            else:
                tp_floor = entry * (1 - max_tp_pct)
                tp_val = max(tp_val, tp_floor)

        return tp_val, sl_val

    def suggest_adjustments(
        self,
        feature_snapshot: dict,
        entry_price: float,
        take_profit: float,
        stop_loss: float,
    ) -> dict | None:
        if not self.enabled:
            return None

        obs = self._state_vector(feature_snapshot, entry_price, take_profit, stop_loss)
        try:
            action, action_idx = self._predict_action(obs)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("RL agent failed to predict: %s", exc)
            return None

        if not np.isfinite(action).all():
            return None

        tp_dist = abs(float(take_profit) - float(entry_price))
        sl_dist = abs(float(entry_price) - float(stop_loss))
        new_tp = float(entry_price) + math.copysign(tp_dist * float(action[0]), take_profit - entry_price)
        new_sl = float(entry_price) - math.copysign(sl_dist * float(action[1]), entry_price - stop_loss)

        new_tp, new_sl = self._apply_safety_limits(entry_price, new_tp, new_sl)

        return {
            "tp_multiplier": float(action[0]),
            "sl_multiplier": float(action[1]),
            "take_profit": new_tp,
            "stop_loss": new_sl,
            "state": obs,
            "action_index": action_idx,
        }

    def record_experience(self, state: np.ndarray, action: np.ndarray, reward: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._buffer.append(Experience(state=state, action=action, reward=reward))
            buffer_len = len(self._buffer)
            if buffer_len >= self._min_samples and buffer_len % self._learn_interval == 0:
                self._learn_from_buffer()

    def _vectorized_env(self, experiences: Sequence[Experience]):
        if self._algo == "dqn":
            return _ReplayTradingEnv(experiences, self.observation_space, self.action_space)
        return DummyVecEnv(
            [lambda: _ReplayTradingEnv(experiences, self.observation_space, self.action_space)]
        )

    def _learn_from_buffer(self) -> None:
        experiences = list(self._buffer)
        if not experiences:
            return
        env = self._vectorized_env(experiences)
        try:
            self._model.set_env(env)
            self._model.learn(total_timesteps=self._training_steps)
            self._learn_calls += 1
            if self._learn_calls % 3 == 0:
                self._save_policy()
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("RL training failed: %s", exc)
        finally:
            self._model.set_env(self._make_env() if self._algo == "dqn" else DummyVecEnv([self._make_env]))

    def persist_policy(self) -> None:
        if not self.enabled:
            return
        self._save_policy()


_AGENT: RLTradingAgent | None = None
_AGENT_LOCK = threading.RLock()


def _get_agent() -> RLTradingAgent:
    global _AGENT
    with _AGENT_LOCK:
        if _AGENT is None:
            _AGENT = RLTradingAgent()
    return _AGENT


def adjust_targets(
    feature_snapshot: dict,
    entry_price: float,
    take_profit: float,
    stop_loss: float,
) -> dict | None:
    agent = _get_agent()
    return agent.suggest_adjustments(feature_snapshot, entry_price, take_profit, stop_loss)


def record_trade_outcome(trade: dict) -> None:
    agent = _get_agent()
    state = trade.get("rl_state")
    action = trade.get("rl_action")
    reward = trade.get("realized_pnl") or trade.get("profit")

    if not agent.enabled or state is None or action is None:
        return

    try:
        state_arr = np.asarray(state, dtype=np.float32)
        action_mults = np.asarray(
            [action.get("tp_multiplier"), action.get("sl_multiplier")], dtype=np.float32
        )
        reward_val = float(reward or 0.0)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to record RL experience: %s", exc)
        return

    agent.record_experience(state_arr, action_mults, reward_val)
    if getattr(config, "RL_PERSIST_AFTER_TRADE", True):
        agent.persist_policy()
