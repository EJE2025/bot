"""Runtime mode selection helpers for the trading bot."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModeProfile:
    """Describe how a given runtime mode tweaks bot behaviour."""

    name: str
    description: str
    enable_trading: bool
    shadow_mode: bool
    enable_model: bool
    model_weight: float | None = None
    dry_run_testnet: bool = False
    maintenance: bool = False
    run_backtest: bool = False


MODES: Dict[str, ModeProfile] = {
    "normal": ModeProfile(
        name="normal",
        description="Trading en vivo (heurística + modelo)",
        enable_trading=True,
        shadow_mode=False,
        enable_model=True,
    ),
    "hybrid": ModeProfile(
        name="hybrid",
        description="Alias de normal (heurística + modelo)",
        enable_trading=True,
        shadow_mode=False,
        enable_model=True,
    ),
    "heuristic": ModeProfile(
        name="heuristic",
        description="Solo heurística sin modelo predictivo",
        enable_trading=True,
        shadow_mode=False,
        enable_model=False,
        model_weight=0.0,
    ),
    "shadow": ModeProfile(
        name="shadow",
        description="Ejecución shadow-mode A/B sin enviar órdenes",
        enable_trading=False,
        shadow_mode=True,
        enable_model=True,
    ),
    "testnet": ModeProfile(
        name="testnet",
        description="Operativa en testnet/dry-run",
        enable_trading=True,
        shadow_mode=False,
        enable_model=True,
        dry_run_testnet=True,
    ),
    "backtest": ModeProfile(
        name="backtest",
        description="Ejecuta el runner de backtest y sale",
        enable_trading=False,
        shadow_mode=False,
        enable_model=True,
        run_backtest=True,
    ),
    "maintenance": ModeProfile(
        name="maintenance",
        description="Modo mantenimiento (sin aperturas)",
        enable_trading=False,
        shadow_mode=False,
        enable_model=False,
        maintenance=True,
    ),
}

NUMERIC_MENU = [
    ("1", "normal"),
    ("2", "shadow"),
    ("3", "heuristic"),
    ("4", "hybrid"),
    ("5", "testnet"),
    ("6", "backtest"),
    ("7", "maintenance"),
]


def resolve_mode(cli_mode: str | None, env_mode: str | None) -> str:
    """Resolve the runtime mode honouring CLI over environment defaults."""

    mode = (cli_mode or env_mode or "hybrid").strip().lower()
    if mode in MODES:
        return mode
    logger.warning("BOT_MODE desconocido '%s'. Usando 'hybrid' por defecto.", mode)
    return "hybrid"


def interactive_pick() -> str:
    """Offer a simple interactive selection menu when a TTY is available."""

    stdin = sys.stdin
    if stdin is None or not stdin.isatty():
        return "hybrid"

    print("\n=== Selecciona modo de ejecución ===")
    for num, key in NUMERIC_MENU:
        profile = MODES[key]
        print(f"{num}) {key:10s} — {profile.description}")
    choice = input("Elige número (por defecto 1): ").strip() or "1"
    for num, key in NUMERIC_MENU:
        if choice == num:
            return key
    print("Selección no válida. Usando 'hybrid'.")
    return "hybrid"


def apply_mode_to_config(mode_key: str, config_module) -> None:
    """Mutate the global config module to match the selected mode."""

    profile = MODES[mode_key]
    config_module.ENABLE_TRADING = profile.enable_trading and not profile.maintenance
    config_module.SHADOW_MODE = profile.shadow_mode
    config_module.ENABLE_MODEL = profile.enable_model
    if profile.model_weight is not None:
        config_module.MODEL_WEIGHT = profile.model_weight
    config_module.DRY_RUN = getattr(config_module, "DRY_RUN", False) or profile.dry_run_testnet
    config_module.MAINTENANCE = profile.maintenance
    config_module.RUN_BACKTEST_ON_START = profile.run_backtest
    logger.info(
        "Modo aplicado: %s | trading=%s shadow=%s model=%s weight=%s dry_run=%s maint=%s backtest=%s",
        mode_key,
        config_module.ENABLE_TRADING,
        config_module.SHADOW_MODE,
        config_module.ENABLE_MODEL,
        getattr(config_module, "MODEL_WEIGHT", None),
        getattr(config_module, "DRY_RUN", False),
        config_module.MAINTENANCE,
        config_module.RUN_BACKTEST_ON_START,
    )


__all__ = [
    "ModeProfile",
    "MODES",
    "NUMERIC_MENU",
    "resolve_mode",
    "interactive_pick",
    "apply_mode_to_config",
]
