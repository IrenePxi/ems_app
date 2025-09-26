from __future__ import annotations
from dataclasses import dataclass
from datetime import time
import numpy as np
import pandas as pd

def _window_mask(idx: pd.DatetimeIndex, start: time, end: time) -> np.ndarray:
    mins = idx.hour*60 + idx.minute
    s = start.hour*60 + start.minute
    e = end.hour*60 + end.minute
    if s <= e:
        mask = (mins >= s) & (mins < e)
    else:
        mask = (mins >= s) | (mins < e)
    return mask

# ---------- Fixed window devices ----------
@dataclass
class FixedDevice:
    name: str
    power_w: float
    count: int
    start: time
    end: time
    duty_cycle: float = 1.0
    def series_kw(self, idx: pd.DatetimeIndex) -> pd.Series:
        kw = np.zeros(len(idx), dtype=float)
        mask = _window_mask(idx, self.start, self.end)
        kw[mask] = self.power_w/1000.0 * self.count * self.duty_cycle
        return pd.Series(kw, index=idx, name=self.name)

# ---------- Cycled within a window ----------
@dataclass
class CycledWindowDevice:
    name: str
    power_w: float
    count: int
    start: time
    end: time
    on_min: int
    off_min: int
    def series_kw(self, idx: pd.DatetimeIndex) -> pd.Series:
        kw = np.zeros(len(idx), dtype=float)
        mask = _window_mask(idx, self.start, self.end)
        if self.on_min <= 0:
            kw[mask] = self.power_w/1000.0 * self.count
            return pd.Series(kw, index=idx, name=self.name)
        cycle = self.on_min + max(self.off_min, 0)
        mins_of_day = idx.hour*60 + idx.minute
        s0 = self.start.hour*60 + self.start.minute
        phase = (mins_of_day - s0) % cycle
        on_state = phase < self.on_min
        active = mask & on_state
        kw[active] = self.power_w/1000.0 * self.count
        return pd.Series(kw, index=idx, name=self.name)

# ---------- Always-on/cycling (e.g., refrigerator) ----------
@dataclass
class CyclingDevice:
    name: str
    power_w: float        # on-power
    period_min: int       # full cycle length
    duty: float           # fraction 0..1
    def series_kw(self, idx: pd.DatetimeIndex) -> pd.Series:
        kw = np.zeros(len(idx), dtype=float)
        on_min = int(round(self.period_min * self.duty))
        mins = (idx.hour*60 + idx.minute).astype(int)
        on_state = (mins % self.period_min) < max(on_min,1)
        kw[on_state] = self.power_w/1000.0
        return pd.Series(kw, index=idx, name=self.name)

# ---------- Block loads ----------
@dataclass
class BlockDevice:
    name: str
    power_w: float
    duration_min: int
    def block_kw(self, idx: pd.DatetimeIndex, start_pos: int) -> pd.Series:
        kw = np.zeros(len(idx), dtype=float)
        length = len(idx)
        s = int(start_pos) % length
        e = s + int(self.duration_min)
        if e <= length:
            kw[s:e] = self.power_w/1000.0
        else:
            kw[s:length] = self.power_w/1000.0
            kw[0:e-length] = self.power_w/1000.0
        return pd.Series(kw, index=idx, name=self.name)

from datetime import time as _t

@dataclass
class WashingMachine:
    name: str = "washing_machine"
    power_w: float = 1200.0
    duration_min: int = 90
    window_start: _t = _t(8,0)
    window_end: _t = _t(22,0)
    def feasible_mask(self, idx: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(_window_mask(idx, self.window_start, self.window_end), index=idx)
    def block_kw(self, idx: pd.DatetimeIndex, start_pos: int) -> pd.Series:
        return BlockDevice(self.name, self.power_w, self.duration_min).block_kw(idx, start_pos)

@dataclass
class Dishwasher:
    name: str = "dishwasher"
    power_w: float = 1500.0
    duration_min: int = 90
    window_start: _t = _t(19,0)
    window_end: _t = _t(7,0)
    def feasible_mask(self, idx: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(_window_mask(idx, self.window_start, self.window_end), index=idx)
    def block_kw(self, idx: pd.DatetimeIndex, start_pos: int) -> pd.Series:
        return BlockDevice(self.name, self.power_w, self.duration_min).block_kw(idx, start_pos)

@dataclass
class Dryer:
    name: str = "dryer"
    power_w: float = 1000.0
    duration_min: int = 90
    window_start: _t = _t(8,0)
    window_end: _t = _t(22,0)
    def feasible_mask(self, idx: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(_window_mask(idx, self.window_start, self.window_end), index=idx)
    def block_kw(self, idx: pd.DatetimeIndex, start_pos: int) -> pd.Series:
        return BlockDevice(self.name, self.power_w, self.duration_min).block_kw(idx, start_pos)

# ---------- Range hood with lunch/dinner blocks ----------
@dataclass
class RangeHood:
    name: str = "range_hood"
    power_w: float = 150.0
    lunch_start: _t = _t(12,0)
    lunch_duration_min: int = 20
    dinner_start: _t = _t(18,0)
    dinner_duration_min: int = 30
    lunch_enabled: bool = True
    dinner_enabled: bool = True
    def series_kw(self, idx: pd.DatetimeIndex) -> pd.Series:
        s = np.zeros(len(idx), dtype=float)
        mins = idx.hour*60 + idx.minute
        if self.lunch_enabled:
            ls = self.lunch_start.hour*60 + self.lunch_start.minute
            le = ls + self.lunch_duration_min
            s[(mins >= ls) & (mins < le)] += self.power_w/1000.0
        if self.dinner_enabled:
            ds = self.dinner_start.hour*60 + self.dinner_start.minute
            de = ds + self.dinner_duration_min
            s[(mins >= ds) & (mins < de)] += self.power_w/1000.0
        return pd.Series(s, index=idx, name=self.name)

# ---------- EV charging ----------
@dataclass
class EVCharger:
    name: str = "ev_charger"
    power_kw: float = 11.0
    energy_target_kwh: float = 20.0
    window_start: _t = _t(22,0)
    window_end: _t = _t(6,0)
    def feasible_mask(self, idx: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(_window_mask(idx, self.window_start, self.window_end), index=idx)
    def duration_minutes(self, dt_h: float) -> int:
        if self.power_kw <= 0:
            return 0
        minutes = int(np.ceil(self.energy_target_kwh / self.power_kw / dt_h))
        return max(minutes, 0)
    def block_kw(self, idx: pd.DatetimeIndex, start_pos: int, dt_h: float) -> pd.Series:
        duration_min = self.duration_minutes(dt_h)
        return BlockDevice(self.name, self.power_kw*1000.0, duration_min).block_kw(idx, start_pos)

# ---------- Baseload (with fridge option) ----------
@dataclass
class BaseloadSpec:
    name: str = "baseload"
    router_w: float = 12.0                      # Router + modem
    ventilation_w: float = 40.0                 # HRV/ERV or continuous fan
    standby_w: float = 60.0                     # TV standby, alarm, hubs, etc.
    dhw_recirc_w: float = 0.0                   # Hot-water recirculation pump
    fridge_avg_w: float = 45.0                  # Refrigerator average
    other1_w: float = 0.0                       # User-defined
    other2_w: float = 0.0                       # User-defined

    def series_kw(self, idx: pd.DatetimeIndex) -> pd.Series:
        total_w = (
            float(self.router_w)
            + float(self.ventilation_w)
            + float(self.standby_w)
            + float(self.dhw_recirc_w)
            + float(self.fridge_avg_w)
            + float(self.other1_w)
            + float(self.other2_w)
        )
        return pd.Series(total_w / 1000.0, index=idx, name=self.name)

# ---------- Weather-aware HP ----------
@dataclass
class WeatherHP:
    name: str = "heat_pump_weather"
    # simple inputs
    ua_kw_per_c: float = 0.25
    t_set_c: float = 21.0
    q_rated_kw: float = 6.0
    cop_at_7c: float = 3.2            # simple-mode input

    # advanced (optional override)
    cop_a: float | None = None
    cop_b: float | None = None        # per °C
    cop_min: float = 1.6
    cop_max: float = 4.2
    defrost: bool = True

    def _cop_params(self):
        # If a,b given → use them; else derive from cop_at_7c with a gentle slope
        if self.cop_a is not None and self.cop_b is not None:
            return float(self.cop_a), float(self.cop_b)
        b = 0.05   # per °C (sensible default)
        a = self.cop_at_7c - b*7.0
        return a, b

    def series_kw(self, idx: pd.DatetimeIndex, tout_c: pd.Series) -> pd.Series:
        tout = pd.Series(tout_c, index=idx).astype(float)
        q_heat = self.ua_kw_per_c * np.maximum(self.t_set_c - tout.values, 0.0)
        a, b = self._cop_params()
        cop = np.clip(a + b * tout.values, self.cop_min, self.cop_max)
        if self.defrost:
            cop = cop * np.where(tout.values < 3.0, 0.92, 1.0)
        q_served = np.minimum(q_heat, self.q_rated_kw)
        p = q_served / np.maximum(cop, 1e-6)
        return pd.Series(p, index=idx, name=self.name)

