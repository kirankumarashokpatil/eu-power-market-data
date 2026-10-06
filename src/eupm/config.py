"""Typed configuration for models, the battery and the walk-forward back-test."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    quantiles: list[float] = [0.1, 0.5, 0.9]
    n_estimators: int = Field(default=600, ge=10)
    learning_rate: float = Field(default=0.03, gt=0, le=1)
    num_leaves: int = 63
    max_depth: int = 8
    random_state: int = 42


class BatteryConfig(BaseModel):
    power_mw: float = Field(default=1.0, gt=0)
    energy_mwh: float = Field(default=2.0, gt=0)
    round_trip_efficiency: float = Field(default=0.88, gt=0, le=1)
    max_cycles_per_day: float = Field(default=1.5, gt=0)
    initial_soc_frac: float = Field(default=0.5, ge=0, le=1)
    degradation_eur_per_mwh: float = Field(default=4.0, ge=0, description="cost per MWh discharged")

    @property
    def one_way_eff(self) -> float:
        return float(self.round_trip_efficiency**0.5)


class BacktestConfig(BaseModel):
    train_days: int = Field(default=180, ge=14)
    test_days: int = Field(default=90, ge=1)
    retrain_every_days: int = Field(default=7, ge=1)


class Settings(BaseModel):
    model: ModelConfig = ModelConfig()
    battery: BatteryConfig = BatteryConfig()
    backtest: BacktestConfig = BacktestConfig()
