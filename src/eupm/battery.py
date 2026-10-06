"""Daily battery dispatch LP for European day-ahead prices.

Decide charge/discharge from forecast prices, then settle at actual prices. Revenue at the
settled prices is the score that matters to a trader."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from eupm.config import BatteryConfig

DT_H = 1.0  # hourly day-ahead MTU (use 0.25 for 15-minute SDAC prices)


@dataclass(frozen=True)
class Schedule:
    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    soc_mwh: np.ndarray

    @property
    def net_mw(self) -> np.ndarray:
        return self.discharge_mw - self.charge_mw


def optimise_day(prices: np.ndarray, cfg: BatteryConfig, dt_h: float = DT_H) -> Schedule:
    """Maximise sum_t [p_t (d_t - c_t) - k d_t] dt, where k is the degradation cost,
    subject to power, energy, cycle and end-of-day SoC constraints.
    ``dt_h`` is the period length in hours (1.0 for hourly, 0.25 for 15-minute MTUs).
    Variables are [c_0..c_T-1, d_0..d_T-1, s_0..s_T-1]."""
    t = len(prices)
    eta = cfg.one_way_eff
    s0 = cfg.initial_soc_frac * cfg.energy_mwh
    deg = cfg.degradation_eur_per_mwh
    # minimise -(revenue - degradation)
    cost = np.concatenate([prices * dt_h, (deg - prices) * dt_h, np.zeros(t)])

    # SoC dynamics: s_t - s_{t-1} - eta*c_t*dt + d_t*dt/eta = 0 (s_{-1} = s0)
    a_eq = np.zeros((t + 1, 3 * t))
    b_eq = np.zeros(t + 1)
    for i in range(t):
        a_eq[i, i] = -eta * dt_h
        a_eq[i, t + i] = dt_h / eta
        a_eq[i, 2 * t + i] = 1.0
        if i > 0:
            a_eq[i, 2 * t + i - 1] = -1.0
        else:
            b_eq[i] = s0
    a_eq[t, 3 * t - 1] = 1.0  # finish the day where it started
    b_eq[t] = s0

    # Cycle limit: total discharged energy <= cycles * capacity
    a_ub = np.zeros((1, 3 * t))
    a_ub[0, t : 2 * t] = dt_h
    b_ub = np.array([cfg.max_cycles_per_day * cfg.energy_mwh])

    bounds = [(0, cfg.power_mw)] * (2 * t) + [(0, cfg.energy_mwh)] * t
    res = linprog(cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"dispatch LP failed: {res.message}")
    x = res.x
    return Schedule(x[:t], x[t : 2 * t], x[2 * t :])


def settle(
    schedule: Schedule, actual_prices: np.ndarray, cfg: BatteryConfig, dt_h: float = DT_H
) -> float:
    """Net margin (EUR) of the schedule at actual prices, after degradation cost."""
    cash = np.sum(schedule.net_mw * actual_prices * dt_h)
    wear = cfg.degradation_eur_per_mwh * np.sum(schedule.discharge_mw) * dt_h
    return float(cash - wear)
