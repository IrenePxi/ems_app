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
    # ---- names / plotting ----
    name: str = "heat_pump_weather"

    # ---- envelope + setpoint (base) ----
    ua_kw_per_c: float = 0.25            # kW/°C, base heat loss coeff
    t_set_c: float = 21.0                # °C, fallback constant setpoint
    C_th_kwh_per_c: float = 0.20         # kWh/°C, building thermal mass

    # Allow time-varying inputs (Series, optional)
    tset_series_c: float | pd.Series | None = None   # °C vs time
    internal_gains_kw: float | pd.Series = 0.0       # kW vs time (solar + people)
    wind_ms: float | pd.Series = 0.0                 # m/s vs time
    ua_wind_factor: float = 0.03                     # +3% UA per 1 m/s (tune 0–0.05)

    # ---- machine ----
    q_rated_kw: float = 6.0             # thermal capacity at ON
    cop_at_7c: float = 3.0              # simple COP anchor (if no (a,b))
    cop_a: float | None = None          # COP = a + b * Tout
    cop_b: float | None = None
    cop_min: float = 1.6
    cop_max: float = 4.2
    defrost: bool = True                # small COP penalty below 3°C
    defrost_mult: float = 0.92

    # ---- thermostat behavior ----
    hyst_band_c: float = 0.8            # total band (±0.4°C)
    min_on_min: int = 4                 # compressor protection
    min_off_min: int = 10
    debounce_steps: int = 2             # require N consecutive steps beyond band to switch

    # ---- power when OFF ----
    p_off_kw: float = 0.05              # standby/crankcase/pumps

    # ---- integration ----
    Ti0_c: float = 21.0                 # initial indoor temp

    # ---- optional DHW/defrost spikes ----
    dhw_windows: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None
    dhw_boost_factor: float = 1.4       # Q_hp boost during DHW
    dhw_ignore_thermostat: bool = True  # heat regardless of room band during DHW

    # ------------------------------------------------------------------
    def _cop_params(self) -> tuple[float, float]:
        if self.cop_a is not None and self.cop_b is not None:
            return float(self.cop_a), float(self.cop_b)
        # derive a,b from cop_at_7c with a gentle slope
        b = 0.05
        a = float(self.cop_at_7c) - b * 7.0
        return a, b

    def _as_series(self, x, idx, name=None) -> pd.Series:
        if isinstance(x, pd.Series):
            s = x.reindex(idx).interpolate().bfill().ffill()
        else:
            s = pd.Series(float(x), index=idx)
        if name:
            s.name = name
        return s.astype(float)

    def _in_any_window(self, t: pd.Timestamp) -> bool:
        if not self.dhw_windows:
            return False
        for s, e in self.dhw_windows:
            if s <= t < e:
                return True
        return False

    def series_kw(self, idx: pd.DatetimeIndex, tout_c: pd.Series) -> pd.Series:
        """Return electrical power (kW), incl. standby when OFF."""
        idx = pd.DatetimeIndex(idx)  # ensure proper index
        dt_h = (idx[1] - idx[0]).total_seconds()/3600.0 if len(idx) > 1 else 1.0/60.0

        # Inputs as time series
        Tout = self._as_series(tout_c, idx, "Tout_C").values
        Gains = self._as_series(self.internal_gains_kw, idx, "Qg_kw").values
        Wind = self._as_series(self.wind_ms, idx, "wind_ms").values
        Tset_series = self._as_series(self.tset_series_c if self.tset_series_c is not None else self.t_set_c,
                                      idx, "Tset_C").values

        # Effective UA with wind
        UAeff = float(self.ua_kw_per_c) * (1.0 + float(self.ua_wind_factor) * np.maximum(Wind, 0.0))

        # COP(Tout)
        a, b = self._cop_params()
        COP = np.clip(a + b * Tout, self.cop_min, self.cop_max)
        if self.defrost:
            COP = COP * np.where(Tout < 3.0, self.defrost_mult, 1.0)

        # Thermostat thresholds
        half = float(self.hyst_band_c) / 2.0
        low = Tset_series - half
        high = Tset_series + half

        # Arrays to fill
        n = len(idx)
        Ti = np.empty(n, dtype=float)
        Ti[0] = float(self.Ti0_c)
        P = np.empty(n, dtype=float)   # electric power
        on = np.zeros(n, dtype=bool)

        # Guards
        min_on_steps = int(round(self.min_on_min / (dt_h*60.0))) if self.min_on_min > 0 else 0
        min_off_steps = int(round(self.min_off_min / (dt_h*60.0))) if self.min_off_min > 0 else 0
        on_timer = 0
        off_timer = 0
        above_cnt = 0
        below_cnt = 0

        for k in range(n):
            tnow = idx[k]

            # During DHW/defrost windows, optionally force heating regardless of thermostat
            dhw_now = self._in_any_window(tnow)
            if k == 0:
                # decide initial state based on band
                if Ti[0] < low[0]:
                    on[0] = True
                    on_timer = 1
                    off_timer = 0
                else:
                    on[0] = False
                    on_timer = 0
                    off_timer = 1

            if k > 0:
                desired_on = on[k-1]
                # Debounced crossings only when not in DHW
                if not dhw_now:
                    if Ti[k-1] < low[k-1]:
                        below_cnt += 1
                        above_cnt = 0
                    elif Ti[k-1] > high[k-1]:
                        above_cnt += 1
                        below_cnt = 0
                    else:
                        above_cnt = below_cnt = 0

                    if not on[k-1] and below_cnt >= self.debounce_steps and off_timer >= min_off_steps:
                        desired_on = True
                        below_cnt = 0
                        off_timer = 0
                    elif on[k-1] and above_cnt >= self.debounce_steps and on_timer >= min_on_steps:
                        desired_on = False
                        above_cnt = 0
                        on_timer = 0
                else:
                    # DHW ignores thermostat (if requested)
                    desired_on = True if self.dhw_ignore_thermostat else on[k-1]

                on[k] = desired_on
                # update timers
                if on[k]:
                    on_timer += 1
                    off_timer = 0
                else:
                    off_timer += 1
                    on_timer = 0

            # Heat pump thermal output this step
            Q_hp = self.q_rated_kw if on[k] else 0.0
            if dhw_now and on[k]:
                Q_hp *= float(self.dhw_boost_factor)

            # Indoor temperature update (first-order)
            heat_loss = UAeff[k] * (Ti[k-1 if k>0 else 0] - Tout[k])
            dTi = (Q_hp + Gains[k] - heat_loss) / max(self.C_th_kwh_per_c, 1e-6) * dt_h
            Ti[k] = Ti[k-1 if k>0 else 0] + dTi

            # Electrical power
            P[k] = (Q_hp / max(COP[k], 1e-6)) if on[k] else float(self.p_off_kw)

        return pd.Series(P, index=idx, name=self.name)
