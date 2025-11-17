from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, date, time


def minute_index(day: date, step_min: int = 1) -> pd.DatetimeIndex:
    start = datetime.combine(day, time(0,0))
    periods = (24*60)//step_min
    return pd.date_range(start=start, periods=periods, freq=f"{step_min}min")

def default_price_profile(idx: pd.DatetimeIndex) -> pd.Series:
    minutes = (idx.view('i8') - idx[0].to_datetime64().astype('datetime64[ns]').astype('int64')) // (60*10**9)
    t = (minutes % 1440) / 1440.0 * 2*np.pi
    base = 1.5 + 0.4*np.sin(t - 0.5) + 0.8*np.maximum(0, np.sin(2*t))
    hour = idx.hour + idx.minute/60.0
    peak = 0.9*np.exp(-0.5*((hour-8.0)/1.5)**2) + 1.0*np.exp(-0.5*((hour-19.0)/1.8)**2)
    price = base + peak
    return pd.Series(price, index=idx, name="price_dkk_per_kwh")

def default_co2_profile(idx: pd.DatetimeIndex) -> pd.Series:
    hour = idx.hour + idx.minute/60.0
    co2 = 250 + 80*np.cos((hour-12)/12*np.pi) + 40*np.exp(-0.5*((hour-19)/1.5)**2)
    return pd.Series(co2, index=idx, name="co2_g_per_kwh")

def simple_pv_profile(idx: pd.DatetimeIndex, kwp: float = 3.0) -> pd.Series:
    hour = idx.hour + idx.minute/60.0
    pv = np.zeros(len(idx), dtype=float)
    mask = (hour >= 6.0) & (hour <= 18.0)
    x = (hour[mask]-6.0)/12.0 * np.pi
    pv[mask] = kwp * np.sin(x)
    return pd.Series(pv, index=idx, name="pv_kw")

def synthetic_outdoor_temp(idx: pd.DatetimeIndex, mean_c: float = 6.0, swing_c: float = 4.0, phase_hours: float = 15.0) -> pd.Series:
    """Very light diurnal outdoor temperature (°C). Peak around 'phase_hours' local time by default."""
    h = idx.hour + idx.minute/60.0
    t = mean_c + swing_c * np.sin((h - phase_hours)/24.0 * 2*np.pi)
    return pd.Series(t, index=idx, name="Tout_C")
