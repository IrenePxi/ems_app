from __future__ import annotations
from dataclasses import dataclass
from datetime import time
import numpy as np
import pandas as pd



# ---------- Fixed window devices ----------

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

    # NEW: operation mode & parameters
    mode: str = "onoff"        # "onoff" or "modulating"
    mod_kp: float = 1.0        # kW/°C proportional gain
    mod_min_frac: float = 0.0  # minimum modulation fraction (0..1)


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
        tout = pd.Series(tout_c, index=idx).astype(float)
        n = len(idx)
        if n == 0:
            return pd.Series(dtype=float, index=idx, name=self.name)

        if n > 1:
            dt_min = (idx[1] - idx[0]).total_seconds() / 60.0
        else:
            dt_min = 1.0
        dt_h = dt_min / 60.0

        T_out = tout.values
        cop   = self._cop(T_out)

        Ti = np.zeros(n, dtype=float)
        P  = np.zeros(n, dtype=float)
        state_hist = np.zeros(n, dtype=bool)

        Ti[0] = float(self.Ti0_c)

        low  = self.t_set_c - self.hyst_band_c / 2.0
        high = self.t_set_c + self.hyst_band_c / 2.0

        hp_on = Ti[0] < low

        # simple internal gains profile
        hours = pd.Index(idx).hour.values if isinstance(idx, pd.DatetimeIndex) else np.zeros(n)
        G = 0.2 + 0.2 * ((hours >= 9) & (hours <= 20)).astype(float)

        for k in range(1, n):
            heat_loss = self.ua_kw_per_c * (Ti[k-1] - T_out[k])
            if self.mode == "onoff":
                # --- your original thermostat logic ---
                desired_on = hp_on
                if Ti[k-1] < low:
                    desired_on = True
                elif Ti[k-1] > high:
                    desired_on = False

                if self._min_period_guard(state_hist, hp_on, k, dt_min):
                    desired_on = hp_on

                hp_on = desired_on
                state_hist[k] = hp_on

                Q_hp = self.q_rated_kw if hp_on else 0.0

            else:  # --- modulating mode ---
                Q_base = heat_loss - G[k]          # hold temperature
                err    = self.t_set_c - Ti[k-1]    # °C

                Q_req = Q_base + self.mod_kp * err
                Q_hp  = np.clip(Q_req, 0.0, self.q_rated_kw)

                if Q_hp > 0.0 and self.mod_min_frac > 0.0:
                    Q_hp = max(Q_hp, self.mod_min_frac * self.q_rated_kw)

                hp_on = Q_hp > 0.0
                state_hist[k] = hp_on

            # common part: power + temperature update
            P[k] = (Q_hp / cop[k]) if Q_hp > 0.0 else self.p_off_kw
            dTi = (Q_hp + G[k] - heat_loss) / max(self.C_th_kwh_per_c, 1e-6) * dt_h
            Ti[k] = Ti[k-1] + dTi

        sP  = pd.Series(P,  index=idx, name="P_HP_kW")
        sTi = pd.Series(Ti, index=idx, name="Ti_C")

        return sP, sTi

  


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

        sP  = pd.Series(P,  index=idx, name="P_EH_kW")
        sTi = pd.Series(Ti, index=idx, name="Ti_C")

        return sP, sTi


@dataclass
class WeatherHotTub:
    name: str = "hot_tub_weather"

    target_c: float = 38.0       # water temp during use
    idle_c: float = 32.0         # keep-warm temperature
    heater_kw: float = 3.0       # heater power
    water_l: float = 800.0       # typical 600–1200 L
    ua_kw_per_c: float = 0.02    # heat loss coefficient

    # NEW: ambient handling
    indoor_ambient_c: float = 21.0
    use_outdoor_for_ambient: bool = False  # False for hot tub, True for pool

    sessions: list | None = None  # [{ "start": time, "duration_min": int }]

    def series_kw(self, idx: pd.DatetimeIndex, tout_minute: pd.Series):
        """Simulate hot-tub / pool heater power for one day."""
        if self.sessions is None:
            self.sessions = []

        if len(idx) == 0:
            return (
                pd.Series(dtype=float, index=idx, name="P_hot_tub_kW"),
                pd.Series(dtype=float, index=idx, name="T_water_C"),
            )

        # time base
        n = len(idx)
        if n > 1:
            dt_min = (idx[1] - idx[0]).total_seconds() / 60.0
        else:
            dt_min = 1.0
        dt_h = dt_min / 60.0

        # ambient (indoor or outdoor)
        if self.use_outdoor_for_ambient and tout_minute is not None:
            tout = pd.Series(tout_minute, index=idx).astype(float)
            T_amb = tout.values
        else:
            T_amb = np.full(n, float(self.indoor_ambient_c), dtype=float)

        # thermal capacity [kWh/°C]
        C_kwh_per_c = max(self.water_l * 1.16 / 1000.0, 1e-6)

        # minutes from midnight
        rel_min = idx.hour * 60 + idx.minute

        # --------------------------------------------------
        # 1) Build in_use mask (as before)
        # --------------------------------------------------
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

        # --------------------------------------------------
        # 2) Estimate preheat time from idle → target
        # --------------------------------------------------
        deltaT = max(self.target_c - self.idle_c, 0.0)
        T_amb_ref = float(np.mean(T_amb))
        if deltaT <= 0:
            preheat_min = 0.0
        else:
            # net heating power near idle
            q_net = self.heater_kw - self.ua_kw_per_c * (self.idle_c - T_amb_ref)
            if q_net <= 0.0:
                # heater too weak → no meaningful preheat estimate
                preheat_min = 0.0
            else:
                dTdt_h = q_net / C_kwh_per_c            # °C per hour
                preheat_min = 60.0 * deltaT / max(dTdt_h, 1e-6)  # minutes

        # --------------------------------------------------
        # 3) Preheat mask before each session
        # --------------------------------------------------
        preheat_mask = np.zeros(n, dtype=bool)
        for sess in self.sessions:
            start = sess.get("start")
            dur = sess.get("duration_min", 0)
            if start is None or dur <= 0 or preheat_min <= 0:
                continue
            s_min = start.hour * 60 + start.minute
            # start preheat this many minutes before session
            start_ph = max(int(round(s_min - preheat_min)), 0)
            mask_ph = (rel_min >= start_ph) & (rel_min < s_min)
            preheat_mask |= mask_ph

        # --------------------------------------------------
        # 4) Simulate
        # --------------------------------------------------
        T = np.zeros(n, dtype=float)
        P = np.zeros(n, dtype=float)
        T[0] = self.idle_c
        heater_on = False
        hyst = 0.4  # hysteresis around setpoint

        for k in range(1, n):
            # decide setpoint
            if in_use[k] or preheat_mask[k]:
                setpoint = self.target_c
            else:
                setpoint = self.idle_c

            # hysteresis around setpoint
            if T[k - 1] < setpoint - hyst:
                heater_on = True
            elif T[k - 1] > setpoint + hyst:
                heater_on = False

            q_in = self.heater_kw if heater_on else 0.0
            P[k] = q_in

            # thermal balance vs ambient
            dT = (q_in - self.ua_kw_per_c * (T[k - 1] - T_amb[k])) * dt_h / C_kwh_per_c
            T[k] = T[k - 1] + dT

        sP = pd.Series(P, index=idx, name="P_tub_kW")
        sT = pd.Series(T, index=idx, name="T_water_C")
        return sP, sT



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
        
        sP  = pd.Series(P, index=idx, name="P_DHW_kW")
        sT  = pd.Series(T, index=idx, name="T_tank_C")
        return sP, sT

