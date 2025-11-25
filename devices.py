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
    name: str = "ev_charging"

    power_kw: float = 11.0        # Charger rating
    ev_capacity_kwh: float = 75.0 # EV battery size
    soc_arrive_pct: float = 20.0
    soc_target_pct: float = 80.0

    window_start: time = time(1,0)
    window_end: time = time(6,0)

    # Manual control (optional)
    manual_start: time | None = None
    scheduling_mode: str = "Auto"

    def energy_needed_kwh(self) -> float:
        soc_diff = max(self.soc_target_pct - self.soc_arrive_pct, 0.0) / 100
        return soc_diff * self.ev_capacity_kwh

    def feasible_mask(self, idx: pd.DatetimeIndex) -> np.ndarray:
        rel_min = idx.hour * 60 + idx.minute
        s = self.window_start.hour * 60 + self.window_start.minute
        e = self.window_end.hour   * 60 + self.window_end.minute
        if e <= s:
            e += 24 * 60
        mins = rel_min.copy()
        mins[mins < s] += 24 * 60
        return (mins >= s) & (mins < e)

    def duration_minutes(self, dt_h: float) -> int:
        need = self.energy_needed_kwh()
        if need <= 0:
            return 0
        return int(np.ceil(need / self.power_kw / dt_h))

    def scheduled_profile(self, idx, dt_h):
        """Only preview. Auto/manual scheduling produces a block."""
        mask = self.feasible_mask(idx)
        dur_min = self.duration_minutes(dt_h)
        if dur_min == 0:
            return pd.Series(0.0, index=idx, name=self.name)

        # Decide start
        if self.scheduling_mode == "Manual" and self.manual_start:
            # convert manual start -> index position
            target = self.manual_start.hour * 60 + self.manual_start.minute
            arr = idx.hour * 60 + idx.minute
            # find nearest feasible index >= target
            candidates = np.where((arr >= target) & mask)[0]
            if len(candidates) == 0:
                return pd.Series(0.0, index=idx, name=self.name)
            start_pos = candidates[0]
        else:
            # Auto: earliest feasible start
            feasible_positions = np.where(mask)[0]
            if len(feasible_positions) == 0:
                return pd.Series(0.0, index=idx, name=self.name)
            start_pos = feasible_positions[0]

        end_pos = min(start_pos + dur_min, len(idx))
        power = np.zeros(len(idx))
        power[start_pos:end_pos] = self.power_kw
        return pd.Series(power, index=idx, name=self.name)

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

    # Building + HP parameters
    ua_kw_per_c: float = 0.25          # heat loss coefficient [kW/°C]
    t_set_c: float = 21.0              # thermostat setpoint [°C]
    q_rated_kw: float = 6.0            # HP rated thermal output [kW]

    # COP parameters
    cop_at_7c: float = 3.2
    cop_a: float | None = None         # if both a,b given -> COP = a + b * Tout
    cop_b: float | None = None
    cop_min: float = 1.6
    cop_max: float = 4.2
    defrost: bool = True               # simple penalty below ~3°C

    # Thermostat + building dynamics
    hyst_band_c: float = 0.6           # thermostat hysteresis width [°C] (±0.3°C)
    C_th_kwh_per_c: float = 3.0        # thermal capacitance of building [kWh/°C]
    Ti0_c: float = 21.0                # initial indoor temp [°C]
    internal_gains_kw: float = 0.0     # constant internal gains (optional) [kW]
    
    p_off_kw: float = 0.05   # 50 W standby when OFF


    
    # Optional: minimum ON/OFF time (set both to 0 to disable)
    min_on_min: int = 0
    min_off_min: int = 0

    # ---- helpers ----
    def _cop_params(self):
        if self.cop_a is not None and self.cop_b is not None:
            return float(self.cop_a), float(self.cop_b)
        b = 0.05
        a = self.cop_at_7c - b * 7.0
        return a, b

    def _cop(self, Tout: np.ndarray) -> np.ndarray:
        a, b = self._cop_params()
        cop = np.clip(a + b * Tout, self.cop_min, self.cop_max)
        if self.defrost:
            cop = cop * np.where(Tout < 3.0, 0.92, 1.0)
        return np.maximum(cop, 1e-6)

    def _min_period_guard(self, state_hist: np.ndarray, state: bool, t: int, dt_min: float) -> bool:
        """Return True if we must keep current 'state' to respect min on/off time."""
        if self.min_on_min <= 0 and self.min_off_min <= 0:
            return False
        # how long (minutes) we've been in current state?
        run = 0
        i = t - 1
        while i >= 0 and state_hist[i] == state:
            run += 1
            i -= 1
        held_min = run * dt_min
        if state and self.min_on_min > 0:
            return held_min < self.min_on_min
        if (not state) and self.min_off_min > 0:
            return held_min < self.min_off_min
        return False

    # ---- main ----
    def series_kw(self, idx: pd.DatetimeIndex, tout_c: pd.Series) -> pd.Series:
        # Align inputs
        tout = pd.Series(tout_c, index=idx).astype(float)
        n = len(idx)
        if n == 0:
            return pd.Series(dtype=float, index=idx, name=self.name)

        # time step (minutes / hours)
        if n > 1:
            dt_min = (idx[1] - idx[0]).total_seconds() / 60.0
        else:
            dt_min = 1.0
        dt_h = dt_min / 60.0

        T_out = tout.values
        cop = self._cop(T_out)

        # Storage for results
        Ti = np.zeros(n, dtype=float)
        P  = np.zeros(n, dtype=float)        # electrical power [kW]
        state_hist = np.zeros(n, dtype=bool) # ON/OFF for min-period guard

        # initial indoor temp near setpoint
        Ti[0] = float(self.Ti0_c)

        # thermostat thresholds
        low  = self.t_set_c - self.hyst_band_c/2.0
        high = self.t_set_c + self.hyst_band_c/2.0

        hp_on = Ti[0] < low

        # just before the loop in WeatherHP.series_kw(...)
        # Build a small daytime internal-gains profile (0.2 kW at night → 0.4 kW mid-day)
        hours = pd.Index(idx).hour.values if isinstance(idx, pd.DatetimeIndex) else np.zeros(n)
        G = 0.2 + 0.2 * ( (hours >= 9) & (hours <= 20) ).astype(float)  # 0.4 kW from 09–20


        # step through time
        for k in range(1, n):
            # thermostat with optional min-on/off guard
            desired_on = hp_on
            if Ti[k-1] < low:
                desired_on = True
            elif Ti[k-1] > high:
                desired_on = False

            # Enforce minimum ON/OFF if requested
            if self._min_period_guard(state_hist, hp_on, k, dt_min):
                desired_on = hp_on

            hp_on = desired_on
            state_hist[k] = hp_on

            # then in the loop, replace self.internal_gains_kw with G[k]
            Q_hp = self.q_rated_kw if hp_on else 0.0
            P[k]  = (Q_hp / cop[k]) if hp_on else self.p_off_kw
            heat_loss = self.ua_kw_per_c * (Ti[k-1] - T_out[k])
            dTi = (Q_hp + G[k] - heat_loss) / max(self.C_th_kwh_per_c, 1e-6) * dt_h
            Ti[k] = Ti[k-1] + dTi

        return pd.Series(P, index=idx, name=self.name)

    # (Optional) helper if you want the indoor temperature trace for debugging/plots
    def simulate_with_Ti(self, idx: pd.DatetimeIndex, tout_c: pd.Series) -> tuple[pd.Series, pd.Series]:
        sP = self.series_kw(idx, tout_c)
        # Re-run quickly to extract Ti (kept simple: call again with a tiny change)
        # If you want, you can refactor to compute Ti & P in one pass and return both.
        return sP, pd.Series([], dtype=float)  # stub to keep interface minimal



# ---------- Weather-aware HP ----------

@dataclass
class WeatherELheater:
    name: str = "EL_heater_weather"

    # Building + HP parameters
    ua_kw_per_c: float = 0.25          # heat loss coefficient [kW/°C]
    t_set_c: float = 21.0              # thermostat setpoint [°C]
    q_rated_kw: float = 6.0            # HP rated thermal output [kW]


    # Thermostat + building dynamics
    hyst_band_c: float = 0.6           # thermostat hysteresis width [°C] (±0.3°C)
    C_th_kwh_per_c: float = 3.0        # thermal capacitance of building [kWh/°C]
    Ti0_c: float = 21.0                # initial indoor temp [°C]
    internal_gains_kw: float = 0.0     # constant internal gains (optional) [kW]
    
    p_off_kw: float = 0.00   # 0 W standby when OFF


    
    # Optional: minimum ON/OFF time (set both to 0 to disable)
    min_on_min: int = 0
    min_off_min: int = 0

    # ---- helpers ----


    def _min_period_guard(self, state_hist: np.ndarray, state: bool, t: int, dt_min: float) -> bool:
        """Return True if we must keep current 'state' to respect min on/off time."""
        if self.min_on_min <= 0 and self.min_off_min <= 0:
            return False
        # how long (minutes) we've been in current state?
        run = 0
        i = t - 1
        while i >= 0 and state_hist[i] == state:
            run += 1
            i -= 1
        held_min = run * dt_min
        if state and self.min_on_min > 0:
            return held_min < self.min_on_min
        if (not state) and self.min_off_min > 0:
            return held_min < self.min_off_min
        return False

    # ---- main ----
    def series_kw(self, idx: pd.DatetimeIndex, tout_c: pd.Series) -> pd.Series:
        # Align inputs
        tout = pd.Series(tout_c, index=idx).astype(float)
        n = len(idx)
        if n == 0:
            return pd.Series(dtype=float, index=idx, name=self.name)

        # time step (minutes / hours)
        if n > 1:
            dt_min = (idx[1] - idx[0]).total_seconds() / 60.0
        else:
            dt_min = 1.0
        dt_h = dt_min / 60.0

        T_out = tout.values
        cop = 1

        # Storage for results
        Ti = np.zeros(n, dtype=float)
        P  = np.zeros(n, dtype=float)        # electrical power [kW]
        state_hist = np.zeros(n, dtype=bool) # ON/OFF for min-period guard

        # initial indoor temp near setpoint
        Ti[0] = float(self.Ti0_c)

        # thermostat thresholds
        low  = self.t_set_c - self.hyst_band_c/2.0
        high = self.t_set_c + self.hyst_band_c/2.0

        hp_on = Ti[0] < low

        # just before the loop in WeatherHP.series_kw(...)
        # Build a small daytime internal-gains profile (0.2 kW at night → 0.4 kW mid-day)
        hours = pd.Index(idx).hour.values if isinstance(idx, pd.DatetimeIndex) else np.zeros(n)
        G = 0.2 + 0.2 * ( (hours >= 9) & (hours <= 20) ).astype(float)  # 0.4 kW from 09–20


        # step through time
        for k in range(1, n):
            # thermostat with optional min-on/off guard
            desired_on = hp_on
            if Ti[k-1] < low:
                desired_on = True
            elif Ti[k-1] > high:
                desired_on = False

            # Enforce minimum ON/OFF if requested
            if self._min_period_guard(state_hist, hp_on, k, dt_min):
                desired_on = hp_on

            hp_on = desired_on
            state_hist[k] = hp_on

            # then in the loop, replace self.internal_gains_kw with G[k]
            Q_hp = self.q_rated_kw if hp_on else 0.0
            P[k]  = (Q_hp) if hp_on else self.p_off_kw
            heat_loss = self.ua_kw_per_c * (Ti[k-1] - T_out[k])
            dTi = (Q_hp + G[k] - heat_loss) / max(self.C_th_kwh_per_c, 1e-6) * dt_h
            Ti[k] = Ti[k-1] + dTi

        return pd.Series(P, index=idx, name=self.name)

    # (Optional) helper if you want the indoor temperature trace for debugging/plots
    def simulate_with_Ti(self, idx: pd.DatetimeIndex, tout_c: pd.Series) -> tuple[pd.Series, pd.Series]:
        sP = self.series_kw(idx, tout_c)
        # Re-run quickly to extract Ti (kept simple: call again with a tiny change)
        # If you want, you can refactor to compute Ti & P in one pass and return both.
        return sP, pd.Series([], dtype=float)  # stub to keep interface minimal

@dataclass
class WeatherHotTub:
    name: str = "hot_tub_weather"

    target_c: float = 38.0       # water temp during use
    idle_c: float = 32.0         # keep-warm temperature
    heater_kw: float = 3.0       # heater power
    water_l: float = 800.0       # typical 600–1200 L
    ua_kw_per_c: float = 0.02    # heat loss coefficient

    sessions: list = None        # [{ "start": time, "duration_min": int }]

    def series_kw(self, idx: pd.DatetimeIndex, tout_minute: pd.Series):
        """Simulate hot-tub heater power for one day."""
        if self.sessions is None:
            self.sessions = []

        if len(idx) == 0:
            return pd.Series(dtype=float, index=idx, name="P_hot_tub_kW")

        # Align temperature
        tout = pd.Series(tout_minute, index=idx).astype(float)
        n = len(idx)

        # time step
        if n > 1:
            dt_min = (idx[1] - idx[0]).total_seconds() / 60.0
        else:
            dt_min = 1.0
        dt_h = dt_min / 60.0

        # thermal capacity
        C_kwh_per_c = max(self.water_l * 1.16 / 1000.0, 1e-6)

        # usage mask
        rel_min = idx.hour * 60 + idx.minute
        in_use = np.zeros(n, dtype=bool)
        for sess in self.sessions:
            start = sess.get("start")
            dur = sess.get("duration_min", 0)
            if start is None or dur <= 0:
                continue
            s_min = start.hour * 60 + start.minute
            e_min = min(s_min + dur, 1440)
            mask = (rel_min >= s_min) & (rel_min < e_min)
            in_use |= mask

        # simulate
        T = np.zeros(n, dtype=float)
        P = np.zeros(n, dtype=float)
        T[0] = self.idle_c
        heater_on = False
        hyst = 0.4

        for k in range(1, n):
            setpoint = self.target_c if in_use[k] else self.idle_c

            # hysteresis
            if T[k - 1] < setpoint - hyst:
                heater_on = True
            elif T[k - 1] > setpoint + hyst:
                heater_on = False

            q_in = self.heater_kw if heater_on else 0.0
            P[k] = q_in

            # thermal balance
            dT = (q_in - self.ua_kw_per_c * (T[k - 1] - tout.iloc[k])) * dt_h / C_kwh_per_c
            T[k] = T[k - 1] + dT

        return pd.Series(P, index=idx, name="P_hot_tub_kW")



from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class DHWTank:
    """
    Very simple domestic hot water (DHW) tank model.
    - Single water node at T_tank
    - Electric heater with thermostat
    - Daily draw events (showers etc.) based on usage_level
    """

    name: str = "dhw_tank"

    volume_l: float = 200.0          # tank volume [L]
    t_set_c: float = 50.0            # nominal setpoint [°C]
    hyst_band_c: float = 5.0         # hysteresis width [°C] (±2.5)
    ua_kw_per_c: float = 0.02        # tank losses [kW/°C]
    p_el_kw: float = 2.0             # heater power [kW]
    p_off_kw: float = 0.01           # standby power [kW]

    T_cold_c: float = 10.0           # cold water temperature [°C]
    T_amb_c: float = 20.0            # ambient around tank [°C]
    Ti0_c: float = 50.0              # initial tank temperature [°C]

    min_on_min: int = 0
    min_off_min: int = 0

    usage_level: str = "Medium"      # "Low" / "Medium" / "High"

    def _C_kwh_per_c(self) -> float:
        """
        Thermal capacity of the tank [kWh/°C].
        1 L water ≈ 0.001163 kWh/°C.
        """
        return max(self.volume_l * 0.001163, 1e-6)

    def _min_period_guard(
        self,
        state_hist: np.ndarray,
        state: bool,
        t: int,
        dt_min: float
    ) -> bool:
        """Return True if we must keep current 'state' to respect min on/off time."""
        if self.min_on_min <= 0 and self.min_off_min <= 0:
            return False
        run = 0
        i = t - 1
        while i >= 0 and state_hist[i] == state:
            run += 1
            i -= 1
        held_min = run * dt_min
        if state and self.min_on_min > 0:
            return held_min < self.min_on_min
        if (not state) and self.min_off_min > 0:
            return held_min < self.min_off_min
        return False

    def _build_draw_profile_lpm(
        self,
        idx: pd.DatetimeIndex
    ) -> np.ndarray:
        """
        Build a hot-water draw profile [L/min] for the given index
        based on self.usage_level. Extremely simplified:
        - A few 30-minute windows with constant flow (showers, dishwashing).
        """
        n = len(idx)
        draw_lpm = np.zeros(n, dtype=float)
        if n == 0:
            return draw_lpm

        # Define simple daily events: (start_hour, duration_min, total_volume_L)
        if self.usage_level == "Low":
            events = [
                (7,  30, 40.0),   # morning shower
                (21, 30, 40.0),   # evening shower
            ]
        elif self.usage_level == "High":
            events = [
                (7,  45, 60.0),   # longer shower / two people
                (12, 20, 30.0),   # mid-day draw
                (19, 45, 80.0),   # evening
            ]
        else:  # "Medium"
            events = [
                (7,  30, 50.0),   # morning
                (19, 30, 60.0),   # evening
            ]

        minutes_of_day = idx.hour * 60 + idx.minute

        for start_hour, dur_min, vol_L in events:
            start_min = start_hour * 60
            end_min = start_min + dur_min
            mask = (minutes_of_day >= start_min) & (minutes_of_day < end_min)
            if mask.any():
                # distribute volume uniformly over the window
                n_steps = int(mask.sum())
                if n_steps > 0:
                    draw_per_step = vol_L / float(n_steps)
                    draw_lpm[mask] += draw_per_step

        return draw_lpm

    def series_kw(
        self,
        idx: pd.DatetimeIndex,
        tout_c: pd.Series | None = None
    ) -> pd.Series:
        """
        Simulate one day of DHW tank.
        - idx: DatetimeIndex (ideally 1-min)
        - tout_c: optional outdoor temperature. If given, we can
          slightly adjust ambient around tank, but it's not required.
        Returns electric power [kW] as a Series.
        """
        n = len(idx)
        if n == 0:
            return pd.Series(dtype=float, index=idx, name=self.name)

        # Time step
        if n > 1:
            dt_min = (idx[1] - idx[0]).total_seconds() / 60.0
        else:
            dt_min = 1.0
        dt_h = dt_min / 60.0

        # Ambient: keep simple (could be e.g. 0.5*(Tout + 20))
        T_amb = np.full(n, self.T_amb_c, dtype=float)

        # Draw profile [L/min]
        draw_lpm = self._build_draw_profile_lpm(idx)

        C = self._C_kwh_per_c()

        # State storage
        T = np.zeros(n, dtype=float)
        P = np.zeros(n, dtype=float)
        state_hist = np.zeros(n, dtype=bool)

        # Initial tank temperature
        T[0] = float(self.Ti0_c)

        # Thermostat thresholds
        low = self.t_set_c - self.hyst_band_c / 2.0
        high = self.t_set_c + self.hyst_band_c / 2.0
        heater_on = T[0] < low

        for k in range(1, n):
            desired_on = heater_on
            if T[k - 1] < low:
                desired_on = True
            elif T[k - 1] > high:
                desired_on = False

            if self._min_period_guard(state_hist, heater_on, k, dt_min):
                desired_on = heater_on

            heater_on = desired_on
            state_hist[k] = heater_on

            # Heater power
            Q_in = self.p_el_kw if heater_on else self.p_off_kw

            # Thermal losses
            Q_loss = self.ua_kw_per_c * (T[k - 1] - T_amb[k])

            # Temperature change from heat in/out
            dT = (Q_in - Q_loss) / C * dt_h
            T_pre = T[k - 1] + dT

            # Apply hot-water draw as mixing with cold water
            if draw_lpm[k] > 0.0:
                # Volume drawn during this step
                V_draw = draw_lpm[k] * dt_min  # [L]
                f = min(V_draw / max(self.volume_l, 1e-6), 0.9)  # replaced fraction
                T[k] = (1.0 - f) * T_pre + f * self.T_cold_c
            else:
                T[k] = T_pre

            P[k] = Q_in

        return pd.Series(P, index=idx, name=self.name)
