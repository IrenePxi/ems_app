#%%
from __future__ import annotations
import streamlit as st
from datetime import datetime, date, time
import pandas as pd  
import numpy as np

from datetime import date, datetime, timedelta
import json
import requests
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import time as _time, date as _date
from profiles import minute_index, default_price_profile, default_co2_profile, simple_pv_profile, synthetic_outdoor_temp
from devices import  WeatherHP,WeatherELheater,WeatherHotTub, DHWTank
import pvlib
from pvlib.location import Location
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from scipy.optimize import minimize
from ems import rule_power_share
from Optimization_based import generate_smart_time_slots, assign_data_to_time_slots_single, mpc_opt_single, mpc_opt_multi, format_results_single

#%% helper for page 1
# -------- EnergiDataService endpoints --------
EDS_PRICE_URL_OLD = "https://api.energidataservice.dk/dataset/Elspotprices"
EDS_PRICE_URL_NEW = "https://api.energidataservice.dk/dataset/DayAheadPrices"
EDS_CO2_HIST_URL  = "https://api.energidataservice.dk/dataset/CO2Emis"
EDS_CO2_PROG_URL  = "https://api.energidataservice.dk/dataset/CO2EmisProg"
TZ_DK = "Europe/Copenhagen"

if "day" not in st.session_state:
    st.session_state["day"] = date.today()

step_min=1

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_dayahead_prices_latest(area: str = "DK1") -> pd.DataFrame:
    r = requests.get(f"{EDS_PRICE_URL_NEW}?limit=200000", timeout=40)
    r.raise_for_status()
    recs = r.json().get("records", [])
    if not recs:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(recs)

    # Normalize new -> old column names
    if "TimeDK" in df.columns:
        df = df.rename(columns={"TimeDK": "HourDK"})
    if "DayAheadPriceDKK" in df.columns:
        df = df.rename(columns={"DayAheadPriceDKK": "SpotPriceDKK"})
    if "DayAheadPriceEUR" in df.columns:
        df = df.rename(columns={"DayAheadPriceEUR": "SpotPriceEUR"})

    if "HourDK" not in df.columns or "PriceArea" not in df.columns:
        return pd.DataFrame()

    # Filter area
    df = df[df["PriceArea"] == area].copy()
    if df.empty:
        return pd.DataFrame()

    # Clean time axis first
    df["HourDK"] = pd.to_datetime(df["HourDK"], errors="coerce")
    df = df.dropna(subset=["HourDK"]).sort_values("HourDK")
    df = df[~df["HourDK"].duplicated(keep="first")]  # handle DST/dups

    # NOW build price column so its length matches the cleaned index
    if "SpotPriceDKK" in df.columns and df["SpotPriceDKK"].notna().any():
        df["price_dkk_per_kwh"] = df["SpotPriceDKK"].astype(float) / 1000.0  # DKK/MWh -> DKK/kWh
    elif "SpotPriceEUR" in df.columns and df["SpotPriceEUR"].notna().any():
        eur_to_dkk = 7.45
        df["price_dkk_per_kwh"] = df["SpotPriceEUR"].astype(float) * eur_to_dkk / 1000.0
    else:
        return pd.DataFrame()

    return df.set_index("HourDK")[["price_dkk_per_kwh"]]


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_elspot_prices(area: str = "DK1") -> pd.DataFrame:
    r = requests.get(f"{EDS_PRICE_URL_OLD}?limit=200000", timeout=40); r.raise_for_status()
    df = pd.DataFrame.from_records(r.json().get("records", []))
    if df.empty or "HourDK" not in df or "PriceArea" not in df or "SpotPriceDKK" not in df:
        return pd.DataFrame()
    df = df[df["PriceArea"] == area][["HourDK","SpotPriceDKK"]].copy()
    df["price_dkk_per_kwh"] = df["SpotPriceDKK"].astype(float) / 1000.0
    return (df.assign(HourDK=pd.to_datetime(df["HourDK"], errors="coerce"))
              .dropna(subset=["HourDK"])
              .sort_values("HourDK")
              .set_index("HourDK")[["price_dkk_per_kwh"]])


def step_hold_to_minutes(s_native, idx_min):
    s = s_native.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()]
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)

    # Slice to the day window (optional but tidy)
    start, end = idx_min[0], idx_min[-1]
    s = s[(s.index >= start) & (s.index <= end)]

    # Direct step-hold upsample to minutes
    s_min = (
        s.reindex(idx_min)   # put values on the exact minute grid
         .ffill()            # hold-forward within each 15-min bin and past the last stamp
         .bfill()            # fill the first few minutes before the first stamp, if any
         .astype(float)
    )
    return s_min.rename("price_dkk_per_kwh")



def daily_price_dual(idx_min: pd.DatetimeIndex, period_start:date, period_end:date, area):
    """
    Returns:
      - price_plot: native-cadence series aligned to idx_min (for charts)
      - price_hourly: clean hourly series (for EMS/optimization)
      - note: optional note for the UI
    """
    tz = "Europe/Copenhagen"
    day_start = pd.Timestamp(period_start).tz_localize(tz).tz_localize(None)
    end   = datetime.combine(period_end, time(23, 59))
    day_end   = pd.Timestamp(end).tz_localize(tz).tz_localize(None)
    note = None

    # Try new dataset (15-min)
    df_new = _fetch_dayahead_prices_latest(area)
    if not df_new.empty:
        s_native = df_new["price_dkk_per_kwh"].loc[(df_new.index >= day_start) & (df_new.index < day_end)]
    else:
        s_native = pd.Series(dtype=float)

    # Fallback to old dataset (hourly)
    if s_native.empty:
        df_old = _fetch_elspot_prices(area)
        if not df_old.empty:
            s_native = df_old["price_dkk_per_kwh"].loc[(df_old.index >= day_start) & (df_old.index < day_end)]

    # If still nothing → placeholder
    if s_native.empty:
        hrs = (idx_min - idx_min[0]).total_seconds()/3600.0
        price_plot = pd.Series(2.0 + 0.8*np.sin(2*np.pi*(hrs-17)/24.0), index=idx_min, name="price_dkk_per_kwh")
        note = "No day-ahead price data available for this day. Showing a smooth placeholder curve."
        return price_plot, note

    # Build plotting series at native resolution → align to minute index for display only
    price_plot = step_hold_to_minutes(s_native, idx_min)


    # Optional note if we had gaps
    miss = int(s_native.isna().sum()) if hasattr(s_native, "isna") else 0
    if miss > 0:
        note = f"Filled {miss} missing price points by interpolation."

    return price_plot, note


@st.cache_data(ttl=300, show_spinner=False)
def fetch_co2_for_day(period_start:date, period_end:date, area) -> pd.Series:
    """Return local-naive 5-min gCO2/kWh series for the given calendar day. May contain NaNs."""
    tz = "Europe/Copenhagen"
    start_naive = datetime.combine(period_start, time(0, 0))
    end_naive   = datetime.combine(period_end + timedelta(days=1), time(0, 0))
    idx5 = pd.date_range(start=start_naive, end=end_naive, freq="5min", inclusive="left")


    url = "https://api.energidataservice.dk/dataset/CO2EmisProg?limit=200000"
    r = requests.get(url, timeout=40); r.raise_for_status()
    recs = r.json().get("records", [])
    if not recs:
        return pd.Series(index=idx5, dtype=float, name="gCO2_per_kWh")

    df = pd.DataFrame.from_records(recs)
    need = {"Minutes5UTC", "PriceArea", "CO2Emission"}
    if not need.issubset(df.columns):
        return pd.Series(index=idx5, dtype=float, name="gCO2_per_kWh")

    df = df.loc[df["PriceArea"] == area].copy()
    df["Time"] = (
        pd.to_datetime(df["Minutes5UTC"], utc=True)
          .dt.tz_convert(tz)
          .dt.tz_localize(None)
    )

    s = (df.rename(columns={"CO2Emission":"gCO2_per_kWh"})
           .set_index("Time")["gCO2_per_kWh"]
           .sort_index())
    
    s = s.groupby(level=0).mean()


    # Keep only that day and align to exact 5-min grid
    s = s.loc[(s.index >= idx5[0]) & (s.index <= idx5[-1])].reindex(idx5)
    return s.rename("gCO2_per_kWh")

def daily_co2_with_note(idx_min: pd.DatetimeIndex, period_start:date, period_end:date, area) -> tuple[pd.Series, str|None]:
    """
    Return minute-level CO₂ (g/kWh) where each 5-min value is held constant
    through its 5-minute block. Also reports how many 5-min points were missing.
    """
    # 5-min local-naive CO₂ for the calendar day (may contain NaNs at 5-min stamps)
    s5 = fetch_co2_for_day(period_start, period_end, area).rename("gCO2_per_kWh")  # expected 288 rows

    # Build a 5-min grid that covers the minute range
    start5 = idx_min[0].floor("5min")
    end5   = idx_min[-1].ceil("5min")
    idx5   = pd.date_range(start=start5, end=end5, freq="5min", inclusive="left")


    # Align to the 5-min grid
    s5_aligned = s5.reindex(idx5)
    miss5 = int(s5_aligned.isna().sum())
    note = None

    if s5_aligned.isna().all():
        # No API data at all → synthesize (then step-hold)
        hrs = (idx_min - idx_min[0]).total_seconds() / 3600.0
        s_min = pd.Series(250.0 + 100.0*np.sin(2*np.pi*(hrs - 15.0)/24.0),
                          index=idx_min, name="gCO2_per_kWh")
        note = "No CO₂ data from EnergiDataService for this day. Showing a smooth placeholder curve."
        return s_min, note

    # Fill only the missing *5-min* stamps (no interpolation within blocks)
    if miss5 > 0:
        s5_aligned = s5_aligned.ffill().bfill()
        note = f"Filled {miss5} missing CO₂ points by forward/backward fill on the 5-min grid."

    # Upsample to minutes with step-hold (constant within each 5-min slot)
    s_min = s5_aligned.reindex(idx_min).ffill().astype(float)
    return s_min.rename("gCO2_per_kWh"), note



def fetch_weather_open_meteo(lat: float, lon: float, start_date, end_date,
                             tz: str = "Europe/Copenhagen") -> pd.DataFrame:
    """
    Hourly weather (GHI/DNI/DHI + temp + wind) from Open-Meteo (archive+forecast).
    Columns: ['ghi','dni','dhi','temp','wind'] indexed by local time (tz-naive).
    """
    def _get(base_url, s_date, e_date):
        url = (
            f"{base_url}?latitude={lat}&longitude={lon}"
            f"&hourly=shortwave_radiation,direct_normal_irradiance,"
            f"diffuse_radiation,temperature_2m,wind_speed_10m"
            f"&start_date={s_date.strftime('%Y-%m-%d')}"
            f"&end_date={e_date.strftime('%Y-%m-%d')}"
            f"&timezone={tz.replace('/', '%2F')}"
        )
        r = requests.get(url, timeout=40); r.raise_for_status()
        h = r.json().get("hourly", {})
        if not h: return pd.DataFrame()
        df = pd.DataFrame(h).rename(columns={
            "shortwave_radiation": "ghi",
            "direct_normal_irradiance": "dni",
            "diffuse_radiation": "dhi",
            "temperature_2m": "temp",
            "wind_speed_10m": "wind",
        })
        df["time"] = pd.to_datetime(df["time"])
        return df.set_index("time").sort_index()

    today = pd.Timestamp.now(tz).normalize()
    s_dt  = pd.Timestamp(start_date, tz=tz)
    e_dt  = pd.Timestamp(end_date,   tz=tz)

    past = _get("https://archive-api.open-meteo.com/v1/archive", s_dt, min(e_dt, today - pd.Timedelta(hours=1))) if s_dt < today else pd.DataFrame()
    futr = _get("https://api.open-meteo.com/v1/forecast", max(s_dt, today), e_dt) if e_dt >= today else pd.DataFrame()

    parts = [p for p in (past, futr) if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["ghi","dni","dhi","temp","wind"])
    return pd.concat(parts).sort_index()


def _clean_hourly_index(s: pd.Series) -> pd.Series:
    """Make hourly series safe for reindex: tz-naive, on-the-hour, unique, sorted."""
    s = s.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()]
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)   # drop tz
    s.index = s.index.floor("h")              # snap to hour
    # if duplicates remain (DST, API dup rows), keep the first (or .mean() if you prefer)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    return s
def daily_temperature_with_note(idx_min: pd.DatetimeIndex, weather_hr: pd.DataFrame) -> tuple[pd.Series, str|None]:
    note = None
    if weather_hr is None or weather_hr.empty or "temp" not in weather_hr:
        hrs = (idx_min - idx_min[0]).total_seconds()/3600.0
        placeholder = 10.0 + 6.0*np.sin(2*np.pi*(hrs-15)/24.0)
        return pd.Series(placeholder, index=idx_min, name="Tout_C"), \
               "No temperature data available. Showing a smooth placeholder curve."

    s_h = _clean_hourly_index(weather_hr["temp"].astype(float))

    start_h = idx_min[0].replace(minute=0, second=0, microsecond=0)
    end_h   = idx_min[-1].floor("h")
    idx_h   = pd.date_range(start=start_h, end=end_h, freq="h")

    s_h_aligned = s_h.reindex(idx_h)
    miss_h = int(s_h_aligned.isna().sum())
    if miss_h > 0:
        s_h_aligned = s_h_aligned.interpolate(limit_direction="both").bfill().ffill()
        note = f"Filled {miss_h} missing temperature points by interpolation."

    s_min = s_h_aligned.reindex(idx_min).interpolate().bfill().ffill().astype(float)
    return s_min.rename("Tout_C"), note


def plot_period_minute(
    series: pd.Series,
    selected_day: date | None,
    title: str,
    ytitle: str
) -> go.Figure:
    """
    Plot a minute-level time series for an arbitrary period and highlight one selected day
    (if it lies inside the period).

    - series: minute-level Series with a DateTimeIndex (tz-naive)
    - selected_day: date chosen in sidebar (can be None)
    """
    fig = go.Figure()

    if series is None or series.empty:
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title=ytitle,
            hovermode="x unified",
        )
        return fig

    s = series.copy()
    s.index = pd.to_datetime(s.index)

    # 1) Full-period line
    fig.add_scatter(
        x=s.index,
        y=s.values,
        mode="lines",
        name="Full period",
        line=dict(color="rgba(100,100,100,0.7)", width=1),
    )

    # 2) Highlight selected day (if given and within range)
    if selected_day is not None:
        day_start = pd.Timestamp(selected_day)
        day_end   = day_start + timedelta(days=1)

        mask = (s.index >= day_start) & (s.index < day_end)
        if mask.any():
            s_sel = s[mask]

            # Background stripe for that day
            fig.add_vrect(
                x0=day_start,
                x1=day_end,
                fillcolor="rgba(200, 30, 30, 0.05)",
                line_width=0,
                layer="below",
            )

            # Thicker overlay line on that day
            fig.add_scatter(
                x=s_sel.index,
                y=s_sel.values,
                mode="lines",
                name="Selected day",
                line=dict(color="crimson", width=3),
            )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=ytitle,
        hovermode="x unified",
    )

    return fig

def plot_period_bar(
    series: pd.Series,
    selected_day: date | None,
    title: str,
    ytitle: str,
    bar_opacity: float = 0.8,
) -> go.Figure:
    """
    Plot a bar chart over a raw time series (minute/5-min/15-min/hourly).
    Highlight selected day with a different color and a background band.
    """
    fig = go.Figure()

    if series is None or series.empty:
        fig.update_layout(title=title, xaxis_title="Time", yaxis_title=ytitle)
        return fig

    s = series.copy()
    s.index = pd.to_datetime(s.index)

    # Detect step automatically (for bar width)
    if len(s) > 1:
        step_seconds = (s.index[1] - s.index[0]).total_seconds()
        bar_width_ms = step_seconds * 1000  # milliseconds
    else:
        bar_width_ms = 60000  # fallback: 1 min

    # 1) Entire period bars
    fig.add_bar(
        x=s.index,
        y=s.values,
        name="Full period",
        marker=dict(color="lightgray"),
        opacity=bar_opacity,
        width=bar_width_ms,
    )

    # 2) Highlight selected day
    if selected_day is not None:
        day_start = pd.Timestamp(selected_day)
        day_end = day_start + timedelta(days=1)
        mask = (s.index >= day_start) & (s.index < day_end)

        if mask.any():
            s_sel = s[mask]

            # Background shading for the day
            fig.add_vrect(
                x0=day_start,
                x1=day_end,
                fillcolor="rgba(200, 30, 30, 0.08)",
                line_width=0,
                layer="below",
            )

            # Overlay bars for selected day
            fig.add_bar(
                x=s_sel.index,
                y=s_sel.values,
                name="Selected day",
                marker=dict(color="crimson"),
                opacity=1.0,
                width=bar_width_ms,
            )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=ytitle,
        barmode="overlay",
        hovermode="x unified",
    )

    return fig

#%% helper for FC
def fc_cost_fun(Pfc_W, Price_ch3oh, Elec_price, Mini_Pfc_W, Maxi_Pfc_W):
    """
    Your original cost function, slightly cleaned:
    Pfc_W: scalar (W), numpy array shape (1,) from scipy is also ok.
    Price_ch3oh: methanol price (same units as before).
    Elec_price: electricity price at this slot (DKK/kWh or same as before).
    Returns scalar objective value.
    """
    # Ensure scalar
    if isinstance(Pfc_W, (np.ndarray, list, tuple)):
        Pfc = float(Pfc_W[0])
    else:
        Pfc = float(Pfc_W)

    # Polynomial coefficients (same as your MATLAB-based fit)
    p = [0.001812, 0.003538, -0.004421, -0.009001,
         0.003244, 0.007644, 0.02274, 0.3901]

    # Normalization (same as before)
    x = (Pfc - 3557.0) / 890.0
    y = np.polyval(p, x)  # dimensionless factor

    # If outside bounds, just return 0 (though bounds should prevent this)
    if not (Mini_Pfc_W <= Pfc <= Maxi_Pfc_W):
        return 0.0

    # Objective (same structure as before)
    return (y * Price_ch3oh - Elec_price) * Pfc

def solve_fc_schedule_minute(prices_minute, Price_ch3oh, Pmin_W, Prated_W):
    """
    Solve FC optimal response at 1-minute resolution.
    prices_minute: np.array of size 1440 for one day.
    """
    prices = np.asarray(prices_minute, dtype=float)
    n = len(prices)
    opti_fc = np.zeros(n, dtype=float)
    fvals = np.zeros(n, dtype=float)

    # 1-minute threshold scaling
    dt_min = 1.0
    threshold = -50 * (dt_min / 60.0)   # = -0.8333

    for i, Elec_price in enumerate(prices):
        x0 = Pmin_W
        bounds = [(Pmin_W, Prated_W)]

        res = minimize(
            fc_cost_fun,
            x0=[x0],
            args=(Price_ch3oh, Elec_price, Pmin_W, Prated_W),
            method="SLSQP",
            bounds=bounds,
            options={'maxiter': 200, 'disp': False},
        )

        fval = float(res.fun)
        Popt = float(res.x[0])

        # Scaled threshold
        if fval >= threshold:
            opti_fc[i] = 0.0
        else:
            opti_fc[i] = Popt

        fvals[i] = fval

    return opti_fc, fvals

def smooth_fc_schedule(p_fc: pd.Series,
                       dt_min: int = 1,
                       min_on_min: int = 60,
                       min_off_min: int = 30) -> pd.Series:
    """
    Post-process FC power profile to enforce:
      - minimum ON time
      - minimum OFF time
    and remove short OFF gaps between ON periods.

    p_fc: minute-level FC power [kW] (index = DatetimeIndex)
    """
    if p_fc.empty:
        return p_fc

    on = p_fc > 0  # boolean

    # Group consecutive equal values (runs)
    groups = (on != on.shift(fill_value=on.iloc[0])).cumsum()

    on2 = on.copy()

    # First pass: kill too-short ON periods
    for g, idx_g in on.groupby(groups).groups.items():
        mask = groups == g
        is_on = on[mask].iloc[0]
        length_min = mask.sum() * dt_min

        if is_on and length_min < min_on_min:
            # too short ON → force OFF
            on2[mask] = False

    # Recompute groups after first pass
    groups2 = (on2 != on2.shift(fill_value=on2.iloc[0])).cumsum()

    # Second pass: fill too-short OFF gaps between ON periods
    for g, idx_g in on2.groupby(groups2).groups.items():
        mask = groups2 == g
        is_on = on2[mask].iloc[0]
        length_min = mask.sum() * dt_min

        if (not is_on) and length_min < min_off_min:
            # OFF gap, check neighbours
            left_on  = (g - 1 in groups2.values) and on2[groups2 == (g - 1)].iloc[0]
            right_on = (g + 1 in groups2.values) and on2[groups2 == (g + 1)].iloc[0]
            if left_on and right_on:
                # short OFF between two ON blocks → fill it
                on2[mask] = True

    # Build new power profile:
    p_new = p_fc.copy()

    # Force OFF where on2 is False
    p_new[~on2] = 0.0

    # For points we turned ON (gap-fill) but original was 0,
    # just use the previous non-zero value (or a constant).
    turned_on = on2 & (~on)
    p_new[turned_on] = p_new.where(p_new > 0).ffill()[turned_on]

    return p_new

#%% helper for page 2    
import streamlit as st
import plotly.graph_objects as go
def normalize_to_dummy_day(s: pd.Series,
                           dummy_date: pd.Timestamp = pd.Timestamp("2000-01-01")) -> pd.Series:
    """
    Take a Series with a DatetimeIndex and map it to a single dummy date,
    keeping only hour/minute information.
    """
    idx = s.index
    if not isinstance(idx, pd.DatetimeIndex):
        return s

    minutes = idx.hour * 60 + idx.minute
    new_idx = dummy_date + pd.to_timedelta(minutes, unit="m")

    s2 = s.copy()
    s2.index = new_idx
    return s2

def pv_from_weather_modelchain_from_df(
    idx_min: pd.DatetimeIndex,
    dfh: pd.DataFrame,             # hourly weather: ghi, dni, dhi, temp, wind
    lat: float, lon: float, kwp: float,
    tilt_deg: float = 30.0, az_deg: float = 180.0,
    sys_loss_frac: float = 0.14,
    tz: str = "Europe/Copenhagen",
) -> pd.Series:
    if kwp <= 0:
        return pd.Series(0.0, index=idx_min, name="pv_kw")

    # ---- 1) Normalize index tz for ALL inputs (use tz-aware consistently) ----
    if not isinstance(dfh.index, pd.DatetimeIndex):
        dfh = dfh.copy()
        dfh.index = pd.to_datetime(dfh.index)

    if dfh.index.tz is None:
        dfh.index = dfh.index.tz_localize(tz)
    else:
        dfh.index = dfh.index.tz_convert(tz)

    times_h = dfh.index  # tz-aware hourly index

    # ---- 2) Solar position with the SAME tz-aware index ----
    loc    = pvlib.location.Location(lat, lon, tz=tz)
    solpos = loc.get_solarposition(times_h)
    zen    = solpos["apparent_zenith"].clip(0, 90)

    # ---- 3) Fill DNI/DHI from GHI if missing (ERBS) using the SAME index ----
    ghi = dfh["ghi"].astype(float)
    dni = dfh["dni"] if "dni" in dfh.columns else None
    dhi = dfh["dhi"] if "dhi" in dfh.columns else None

    needs_fill = (dni is None) or (dhi is None) \
                or (dni.isna().any() if dni is not None else False) \
                or (dhi.isna().any() if dhi is not None else False)

    if needs_fill:
        split = pvlib.irradiance.erbs(
            ghi=ghi.values, zenith=zen.values, datetime_or_doy=times_h
        )

        # DNI
        if dni is None:
            dfh["dni"] = pd.Series(split["dni"].values, index=times_h, dtype=float)
        else:
            dfh["dni"] = dni.astype(float).fillna(pd.Series(split["dni"].values, index=times_h))

        # DHI
        if dhi is None:
            dfh["dhi"] = pd.Series(split["dhi"].values, index=times_h, dtype=float)
        else:
            dfh["dhi"] = dhi.astype(float).fillna(pd.Series(split["dhi"].values, index=times_h))

    # ---- 4) POA with matched indices (no tz mix) ----
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt_deg, surface_azimuth=az_deg,
        dni=dfh["dni"].astype(float), ghi=ghi, dhi=dfh["dhi"].astype(float),
        solar_zenith=zen, solar_azimuth=solpos["azimuth"]
    )
    poa_global = poa["poa_global"].clip(lower=0.0)

    # ---- 5) ModelChain from POA; keep times tz-aware then drop tz at the end ----
    weather_poa = pd.DataFrame({
        "poa_global":  poa["poa_global"].clip(lower=0.0).astype(float),
        "poa_direct":  poa["poa_direct"].clip(lower=0.0).astype(float),
        "poa_diffuse": poa["poa_diffuse"].clip(lower=0.0).astype(float),
        "temp_air":    dfh["temp"].fillna(15.0).astype(float),
        "wind_speed":  dfh["wind"].fillna(2.0).astype(float),
    }, index=times_h)

    pdc0_w = float(kwp) * 1000.0
    temp_params = TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
    system = pvlib.pvsystem.PVSystem(
        arrays=[pvlib.pvsystem.Array(
            mount=pvlib.pvsystem.FixedMount(surface_tilt=tilt_deg, surface_azimuth=az_deg),
            module_parameters={"pdc0": pdc0_w, "gamma_pdc": -0.0045},
            temperature_model_parameters=temp_params,
        )],
        inverter_parameters={"pdc0": pdc0_w}
    )

    mc = pvlib.modelchain.ModelChain(
        system, loc,
        dc_model="pvwatts", ac_model="pvwatts",
        aoi_model="physical", spectral_model="no_loss",
    )
    mc.run_model_from_poa(weather_poa)

    # AC (W) → kW, apply system loss, then drop tz and interpolate to minutes
    ac_kw_h = (mc.results.ac / 1000.0) * (1.0 - float(sys_loss_frac))
    ac_kw_h = ac_kw_h.tz_convert(None)               # now tz-naive
    ac_kw_m = ac_kw_h.reindex(idx_min).interpolate().bfill().ffill()
    return ac_kw_m.rename("pv_kw")



def build_minute_profile(power_w: float,
                            intervals: list[dict],
                            step_min: int = 1) -> pd.Series:
    """
    Build a 24h minutely (or step_min) profile for one device.
    power_w: device power when ON (constant)
    intervals: list of dicts with 'start'/'end' as datetime.time
    Returns Series [kW] indexed from 00:00–24:00 (dummy date).
    """
    if step_min <= 0:
        step_min = 1

    # Use a dummy date (just for plotting)
    dummy_day = date(2025, 1, 10)
    start_dt = pd.Timestamp(dummy_day)
    periods = (24 * 60) // step_min
    idx = pd.date_range(start=start_dt, periods=periods, freq=f"{step_min}min")

    power = np.zeros(len(idx), dtype=float)  # [kW]

    # Precompute interval masks
    for it in intervals:
        s: time = it["start"]
        e: time = it["end"]

        # Map to minutes since midnight
        s_min = s.hour * 60 + s.minute
        e_min = e.hour * 60 + e.minute

        # Handle "wrap around" (if end < start, we treat it as crossing midnight)
        if e_min <= s_min:
            ranges = [(s_min, 24 * 60), (0, e_min)]
        else:
            ranges = [(s_min, e_min)]

        for (m0, m1) in ranges:
            # indices of idx where minutes since midnight in [m0, m1)
            rel_min = ((idx - idx[0]).total_seconds() / 60.0).astype(int)
            mask = (rel_min >= m0) & (rel_min < m1)
            power[mask] = power_w / 1000.0  # convert W → kW

    return pd.Series(power, index=idx, name="P_device_kW")

#%% -------------------------------- Page 2 main code----------------------------------------
def get_selected_day_data(input_series):
    sel_day = st.session_state.get("day") 

    day_start = pd.Timestamp(sel_day)
    day_end   = day_start + pd.Timedelta(days=1)
    df=input_series.copy()

    df = df.loc[(df.index >= day_start) & (df.index < day_end)]

    return df
def render_devices_page_house():
    # --- 1) Init device on/off state once ---
    if "devices_enabled" not in st.session_state:
        st.session_state["devices_enabled"] = {
            # Electrical – fixed
            "lights": True,
            "hood": True,
            "fridge": True,
            # Electrical – flexible
            "wm": True,
            "dw": True,
            "dryer": False,
            # Thermal
            "hp": True,
            "e_heater": False,
            "hot_tub": False,
            # Generation & storage
            "pv": True,
            "battery": True,
            "fuel_cell": False,
            "diesel": False,
            # EV
            "ev": True,
        }

    if "devices" not in st.session_state:
        st.session_state["devices"] = {
            "elec_fixed": [
            ],
            "elec_flex": [

            ],
            "thermal": [

            ],
            "gen_store": [
 
            ],
            "ev": [
                {"id": "ev_1", "type": "ev", "name": "EV"},
            ],
        }

    devices = st.session_state["devices"]

    # --- 2) Init per-device configuration (power, schedule) ---
    if "device_configs" not in st.session_state:
        st.session_state["device_configs"] = {}
    device_configs = st.session_state["device_configs"]

    # Max devices per category (tuned so icons stay inside the colored areas)
    MAX_PER_CATEGORY = {
        "elec_fixed": 20,   # 5 x 4 grid
        "elec_flex":  20,   # 5 x 4 grid
        "thermal":    16,   # 4 x 4 grid
        "gen_store":  12,   # triangular arrangement (2 + 4 + 6)
        "ev":         2,    # up to 2 EV chargers stacked
    }

    TYPE_ICONS = {
        "lights":    "💡",
        "hood":      "🍳",
        "fridge":    "🧊",
        "wm":        "🧺",
        "dw":        "🍽",
        "dryer":     "🌀",
        "hp":        "❄️",
        "e_heater":  "🔥",
        "hot_tub":   "🛁",
        "dhw":       "🚿",   
        "pv":        "☀️",
        "battery":   "🔋",
        "fuel_cell": "🧪",
        "diesel":    "⛽",
        "ev":        "🚗",
        "other":     "🔌",
    }

    # ---- default config for a device type (used when first opening settings) ----
    def get_default_device_config(dev_type: str) -> dict:
        from datetime import time as _time
        if dev_type == "lights":
            return {
                "num_devices": 1,
                "power_w": 20.0,
                "intervals": [{"start": _time(18, 0), "end": _time(23, 0)}],
            }
        if dev_type == "hood":
            return {
                "num_devices": 1,
                "power_w": 150.0,
                "intervals": [
                    {"start": _time(12, 0), "end": _time(12, 30)},
                    {"start": _time(18, 0), "end": _time(18, 30)},
                ],
            }
        if dev_type == "fridge":
            return {
                "num_devices": 1,
                "power_w": 60.0,
                "intervals": [{"start": _time(0, 0), "end": _time(23, 59)}],
            }
        
        if dev_type == "ev":
            return {
                "intervals": [{"start": _time(1, 0), "end": _time(6, 0)}],
            }
        
        if dev_type == "hp":
            # User-friendly HP config: comfort + qualitative house info
            return {
                "t_min_c": 20.0,
                "t_max_c": 22.0,
                "house_size": "Medium house",      # "Small apartment" / "Medium house" / "Large house"
                "insulation": "Average",           # "Poor" / "Average" / "Good"
                "q_rated_kw": 6.0,                 # default capacity (kW)
            }
        
        if dev_type == "e_heater":
            # User-friendly HP config: comfort + qualitative house info
            return {
                "t_min_c": 20.0,
                "t_max_c": 22.0,
                "house_size": "Medium house",      # "Small apartment" / "Medium house" / "Large house"
                "insulation": "Average",           # "Poor" / "Average" / "Good"
                "q_rated_kw": 6.0,                 # default capacity (kW)
            }

        if dev_type == "pv":
            # defaults similar to your old UI: 16 × 400 W, 30° tilt, 180° az, 14% losses
            return {
                "module_wp": 400.0,
                "n_panels": 16,
                "tilt": 30.0,
                "azimuth": 180.0,
                "loss_frac": 0.14,
            }
        
        if dev_type == "battery":
            return {
                "capacity_kwh": 10.0,
                "power_kw": 5.0,
                "soc_init_pct": 50.0,
                "soc_min_pct": 15.0,
                "control_mode": "Auto",
                "manual_slots": [],   # list of 6-slot dicts, empty initially
                "rule_priority": 2,   # 2=Load-first, 1=Battery-first
            }

        
        # generic fallback
        return {
            "num_devices": 1,
            "power_w": 100.0,
            "intervals": [{"start": _time(8, 0), "end": _time(9, 30)}],
        }

    # ---- helper to suggest new intervals for flexible devices ----
    # ---- helper: suggest ONE continuous interval for a flexible device ----
        # ---- helper: suggest ONE continuous interval for a flexible device ----
    def suggest_best_interval_for_day(duration_min: int,
                                      w_cost: float = 0.5) -> dict | None:
        """
        Returns {"start": time, "end": time} for the chosen day.
        Priority:
          1) st.session_state["selected_day"]  (if valid)
          2) period_range[1]  (end of selected period)
        Returns None if no day or price/CO₂ data is available.
        """
    
        price = st.session_state.get("price_daily")
        co2   = st.session_state.get("co2_daily")

        # --- find the day to use ---
        sel_day = st.session_state.get("day")

        # fallback: use end of selected period if selected_day is missing
        if not isinstance(sel_day, _date):
            pr = st.session_state.get("period_range")
            if pr and len(pr) == 2 and isinstance(pr[1], _date):
                sel_day = pr[1]

        # if still nothing usable → bail
        if price is None or co2 is None or len(price) == 0 or not isinstance(sel_day, _date):
            return None

        # --- build unified dataframe over that day ---
        df = pd.DataFrame(index=price.index.copy())
        df["price"] = np.asarray(price, dtype=float)
        df["co2"]   = co2.reindex(df.index).interpolate().bfill().ffill()

        day_start = pd.Timestamp(sel_day)
        day_end   = day_start + pd.Timedelta(days=1)

        df = df.loc[(df.index >= day_start) & (df.index < day_end)]
        if df.empty:
            return None

        # normalize to [0,1]
        for col in ["price", "co2"]:
            x = df[col].values.astype(float)
            mn, mx = np.nanmin(x), np.nanmax(x)
            if mx > mn:
                df[col] = (x - mn) / (mx - mn)
            else:
                df[col] = 0.5

        w_cost = float(np.clip(w_cost, 0.0, 1.0))
        df["score"] = w_cost * df["price"] + (1.0 - w_cost) * df["co2"]

        n = len(df)
        dur = int(duration_min)
        if dur <= 0:
            dur = 30
        if dur > n:
            dur = n

        best_score = None
        best_t0 = None

        # simple sliding window
        for t0 in range(0, n - dur + 1):
            sc = float(df["score"].iloc[t0:t0 + dur].mean())
            if best_score is None or sc < best_score:
                best_score = sc
                best_t0 = t0

        if best_t0 is None:
            return None

        t_start = df.index[best_t0]
        t_end   = df.index[best_t0 + dur - 1] + pd.Timedelta(minutes=1)

        # clamp to valid 0..23:59 (avoid hour=24 problem)
        start_min = t_start.hour * 60 + t_start.minute
        end_min   = t_end.hour * 60 + t_end.minute
        start_min = max(0, min(start_min, 24 * 60 - 1))
        end_min   = max(1, min(end_min, 24 * 60 - 1))

        return {
            "start": _time(start_min // 60, start_min % 60),
            "end":   _time(end_min   // 60, end_min   % 60),
        }
    

    
    

    def suggest_best_interval_for_ev(duration_min: int,
                                 w_cost: float,
                                 window_start_min: int = 60,
                                 window_end_min: int = 360) -> dict | None:
        """
        Find best continuous interval of given length within [01:00, 06:00)
        using selected-day price & CO2 (minute series).
        Returns {"start": time, "end": time} or None.
        """
        import numpy as np
        import pandas as pd

        price_all = st.session_state.get("price_daily")
        co2_all   = st.session_state.get("co2_daily")

        if not isinstance(price_all, pd.Series) or price_all.empty:
            return None
        if not isinstance(co2_all, pd.Series) or co2_all.empty:
            return None

        # Slice to selected day (same helper you already use elsewhere)
        price = get_selected_day_data(price_all)
        co2   = get_selected_day_data(co2_all)

        if price is None or co2 is None or price.empty or co2.empty:
            return None

        # Make sure aligned indices
        price, co2 = price.align(co2, join="inner")
        idx = price.index
        n = len(idx)
        if n == 0 or duration_min <= 0 or duration_min > n:
            return None

        rel_min = idx.hour * 60 + idx.minute
        prices  = price.values.astype(float)
        co2v    = co2.values.astype(float)

        w_c   = float(w_cost)
        w_co2 = 1.0 - w_c

        best_score = None
        best_start_pos = None

        # candidate starts only inside [window_start_min, window_end_min)
        candidates = np.where(
            (rel_min >= window_start_min) & (rel_min < window_end_min)
        )[0]

        for start_pos in candidates:
            end_pos = start_pos + duration_min
            if end_pos > n:
                break
            # ensure the *end* of block is still inside window
            if rel_min[end_pos - 1] >= window_end_min:
                continue

            p_seg = prices[start_pos:end_pos]
            c_seg = co2v[start_pos:end_pos]
            score = w_c * p_seg.mean() + w_co2 * c_seg.mean()

            if best_score is None or score < best_score:
                best_score = score
                best_start_pos = start_pos

        if best_start_pos is None:
            return None

        start_ts = idx[best_start_pos]
        end_ts   = idx[min(best_start_pos + duration_min, n - 1)] + pd.Timedelta(minutes=1)

        return {
            "start": start_ts.time(),
            "end":   end_ts.time(),
        }

    



    # Helper to render one category list on the left
    def render_category_ui(cat_key, title, type_choices):
        """
        type_choices: dict label -> internal_type (e.g. {"Lights": "lights", ...})
        """
        from datetime import time as _time

        st.markdown(f"**{title}**")
        dev_list = devices[cat_key]

        # Existing devices
        if dev_list:
            for i, dev in enumerate(dev_list):
                settings_id = f"{cat_key}_{dev['id']}"
                row = st.container()
                with row:
                    c0, c1, c2 = st.columns([0.15, 0.65, 0.20])
                    with c0:
                        st.markdown(TYPE_ICONS.get(dev["type"], "🔌"))
                    with c1:
                        new_name = st.text_input(
                            "Device name",
                            value=dev["name"],
                            key=f"{cat_key}_name_{dev['id']}",
                            label_visibility="collapsed",
                        )
                        dev["name"] = new_name
                    with c2:
                        col_set, col_del = st.columns(2)
                        open_key = f"settings_open_{settings_id}"
                        current_open = st.session_state.get(open_key, False)
                        with col_set:
                            if st.button("⚙️", key=f"{settings_id}_cfg", help="Settings"):
                                st.session_state[open_key] = not current_open
                                st.rerun()
                        with col_del:
                            if st.button("🗑", key=f"{settings_id}_del"):
                                dev_list.pop(i)
                                device_configs.pop(settings_id, None)
                                st.rerun()

                # ---- SETTINGS PANEL (under each row) ----
                if st.session_state.get(f"settings_open_{settings_id}", False):
                    cfg = device_configs.get(settings_id)
                    if cfg is None:
                        cfg = get_default_device_config(dev["type"])
                        device_configs[settings_id] = cfg

                    # top dashed line
                    st.markdown(
                        "<hr style='border-top: 1px dashed #bbb;'/>",
                        unsafe_allow_html=True,
                    )

                    # ---------- FIXED ELECTRICAL LOADS ----------
                    if cat_key == "elec_fixed":
                        # Number of identical devices
                        cfg["num_devices"] = int(
                            st.number_input(
                                "Number of devices",
                                min_value=1,
                                max_value=10,
                                step=1,
                                value=int(cfg.get("num_devices", 1)),
                                key=f"{settings_id}_numdev",
                            )
                        )
                        # Power
                        cfg["power_w"] = st.number_input(
                            "Power (W)",
                            min_value=0.0,
                            max_value=5000.0,
                            step=10.0,
                            value=float(cfg.get("power_w", 100.0)),
                            key=f"{settings_id}_power",
                        )

                        # Intervals (multiple allowed)
                        st.caption("On/off intervals (you can add multiple):")
                        intervals = cfg.setdefault("intervals", [])
                        if not intervals:
                            intervals.append({"start": _time(18, 0), "end": _time(23, 0)})

                        to_delete = None
                        for j, iv in enumerate(intervals):
                            c_a, c_b, c_c = st.columns([0.4, 0.4, 0.2])
                            with c_a:
                                s_t = c_a.time_input(
                                    "Start",
                                    value=iv.get("start", _time(18, 0)),
                                    key=f"{settings_id}_start_{j}",
                                )
                            with c_b:
                                e_t = c_b.time_input(
                                    "End",
                                    value=iv.get("end", _time(23, 0)),
                                    key=f"{settings_id}_end_{j}",
                                )
                            with c_c:
                                if c_c.button("🗑", key=f"{settings_id}_ivdel_{j}"):
                                    to_delete = j
                            iv["start"], iv["end"] = s_t, e_t

                        if to_delete is not None:
                            intervals.pop(to_delete)
                            st.rerun()

                        if st.button("➕ Add interval", key=f"{settings_id}_add_interval"):
                            intervals.append({"start": _time(18, 0), "end": _time(23, 0)})
                            st.rerun()

                        # small gap
                        st.markdown("<div style='height:0.75rem'></div>",
                                    unsafe_allow_html=True)
                        
                        # Profile (scaled by number of devices)
                        st.markdown("**Daily load profile (preview)**")
                        prof = build_minute_profile(
                            power_w=cfg["power_w"] * cfg["num_devices"],
                            intervals=intervals,
                            step_min=1,
                        )
                        cfg["profile_index"] = prof.index.astype(str).tolist()
                        cfg["profile_kw"]    = prof.values.tolist()

                        fig_p = go.Figure()
                        fig_p.add_scatter(
                            x=prof.index,
                            y=prof.values,
                            mode="lines",
                            name="P_flex_total_kW",
                        )
                        fig_p.update_layout(
                            height=180,
                            margin=dict(l=10, r=10, t=10, b=8),
                            xaxis_title="Time",
                            yaxis_title="kW",
                            showlegend=False,
                        )
                        st.plotly_chart(fig_p, use_container_width=True)

                        if st.button("▲ Hide details", key=f"{settings_id}_hide_flex"):
                            st.session_state[f"settings_open_{settings_id}"] = False
                            st.rerun()

                        

                    # ---------- FLEXIBLE ELECTRICAL LOADS ----------
                    elif cat_key == "elec_flex":
                        # Number of identical devices
                        cfg["num_devices"] = int(
                            st.number_input(
                                "Number of devices",
                                min_value=1,
                                max_value=10,
                                step=1,
                                value=int(cfg.get("num_devices", 1)),
                                key=f"{settings_id}_numdev",
                            )
                        )

                        # Power when ON
                        cfg["power_w"] = st.number_input(
                            "Power per device (W)",
                            min_value=0.0,
                            max_value=5000.0,
                            step=50.0,
                            value=float(cfg.get("power_w", 1200.0)),
                            key=f"{settings_id}_power_flex",
                        )

                        # Total ON duration
                        cfg["duration_min"] = int(
                            st.number_input(
                                "Operation duration (minutes)",
                                min_value=15,
                                max_value=600,
                                step=15,
                                value=int(cfg.get("duration_min", 90)),
                                key=f"{settings_id}_dur",
                            )
                        )

                        # cost vs CO2 preference
                        cfg["w_cost"] = float(
                            st.slider(
                                "Preference (0 = CO₂ only, 1 = cost only)",
                                min_value=0.0,
                                max_value=1.0,
                                step=0.05,
                                value=float(cfg.get("w_cost", 0.5)),
                                key=f"{settings_id}_w_cost",
                            )
                        )

                        # Make sure we have exactly one interval in cfg
                        intervals = cfg.setdefault("intervals", [])
                        if not intervals:
                            intervals.append(
                                {"start": _time(20, 0), "end": _time(21, 30)}
                            )
                        elif len(intervals) > 1:
                            intervals[:] = intervals[:1]

                        current_iv = intervals[0]

                        st.caption("Current scheduled interval (kept as one continuous block):")
                        c_a, c_b = st.columns(2)
                        with c_a:
                            new_start = c_a.time_input(
                                "Start",
                                value=current_iv.get("start", _time(20, 0)),
                                key=f"{settings_id}_flex_start",
                            )
                        with c_b:
                            new_end = c_b.time_input(
                                "End",
                                value=current_iv.get("end", _time(21, 30)),
                                key=f"{settings_id}_flex_end",
                            )
                        current_iv["start"], current_iv["end"] = new_start, new_end

                        # Suggest button
                        if st.button("💡 Suggest schedule from price/CO₂",
                                key=f"{settings_id}_suggest"):
                            interval = suggest_best_interval_for_day(
                                duration_min=cfg["duration_min"],
                                w_cost=cfg["w_cost"],
                            )
                            if interval is None:
                                st.warning(
                                    "No price/CO₂ data or no selected day. "
                                    "Please select a day and fetch data first."
                                )
                            else:
                                intervals[0] = interval
                                st.success(
                                    f"Suggested interval: {interval['start'].strftime('%H:%M')}–"
                                    f"{interval['end'].strftime('%H:%M')}"
                                )
                                st.rerun()


                        # Profile (scaled by number of devices)
                        st.markdown("**Daily load profile (preview)**")
                        prof = build_minute_profile(
                            power_w=cfg["power_w"] * cfg["num_devices"],
                            intervals=intervals,
                            step_min=1,
                        )
                        cfg["profile_index"] = prof.index.astype(str).tolist()
                        cfg["profile_kw"]    = prof.values.tolist()

                        fig_p = go.Figure()
                        fig_p.add_scatter(
                            x=prof.index,
                            y=prof.values,
                            mode="lines",
                            name="P_flex_total_kW",
                        )
                        fig_p.update_layout(
                            height=180,
                            margin=dict(l=10, r=10, t=10, b=8),
                            xaxis_title="Time",
                            yaxis_title="kW",
                            showlegend=False,
                        )
                        st.plotly_chart(fig_p, use_container_width=True)

                        if st.button("▲ Hide details", key=f"{settings_id}_hide_flex"):
                            st.session_state[f"settings_open_{settings_id}"] = False
                            st.rerun()
                                        
                    # 2) Heat pump in thermal category: user-friendly inputs only
                    elif cat_key == "thermal" and dev["type"] == "hp":
                        st.markdown(
                            """
                            <hr style="border: 0; border-top: 1px dotted #fecaca; margin: 0.4rem 0;" />
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown("**Heat pump comfort & house settings**")

                        # --- Comfort band ---
                        c_tmin, c_tmax = st.columns(2)
                        cfg["t_min_c"] = c_tmin.number_input(
                            "Min indoor temperature (°C)",
                            min_value=5.0,
                            max_value=30.0,
                            step=0.5,
                            value=float(cfg.get("t_min_c", 20.0)),
                            key=f"{settings_id}_tmin",
                        )
                        cfg["t_max_c"] = c_tmax.number_input(
                            "Max indoor temperature (°C)",
                            min_value=5.0,
                            max_value=30.0,
                            step=0.5,
                            value=float(cfg.get("t_max_c", 22.0)),
                            key=f"{settings_id}_tmax",
                        )
                        if cfg["t_max_c"] < cfg["t_min_c"]:
                            cfg["t_max_c"] = cfg["t_min_c"]

                        # --- Qualitative building info ---
                        c_size, c_ins = st.columns(2)
                        cfg["house_size"] = c_size.selectbox(
                            "House size",
                            ["Small apartment", "Medium house", "Large house"],
                            index=["Small apartment", "Medium house", "Large house"].index(
                                cfg.get("house_size", "Medium house")
                            ),
                            key=f"{settings_id}_hsize",
                        )
                        
                        cfg["insulation"] = c_ins.selectbox(
                            "Insulation level",
                            ["Poor", "Average", "Good"],
                            index=["Poor", "Average", "Good"].index(
                                cfg.get("insulation", "Average")
                            ),
                            key=f"{settings_id}_insul",
                        )

                        # --- Map qualitative choices to ua & q_rated ---
                        # --- Map qualitative choices to ua & suggested q_rated ---
                        size = cfg["house_size"]
                        ins  = cfg["insulation"]

                        # base UA & capacity guess by size
                        if size == "Small apartment":
                            ua_base = 0.10   # kW/°C
                            q_guess = 4.0    # kW
                            Cth_guess = 0.50   # kWh/°C
                        elif size == "Large house":
                            ua_base = 0.14
                            q_guess = 8.0
                            Cth_guess = 0.75  # kWh/°C
                        else:  # "Medium house"
                            ua_base = 0.12
                            q_guess = 6.0
                            Cth_guess = 0.60

                        # adjust UA by insulation
                        if ins == "Poor":
                            ua = ua_base * 1.3
                        elif ins == "Good":
                            ua = ua_base * 0.7
                        else:
                            ua = ua_base

                        # --- Let the user override the capacity ---
                        cfg["q_rated_kw"] = st.number_input(
                            "Heat pump capacity (kW)",
                            min_value=1.0,
                            max_value=30.0,
                            step=0.5,
                            value=float(cfg.get("q_rated_kw", q_guess)),
                            key=f"{settings_id}_q_rated",
                        )

                        q_rated = float(cfg["q_rated_kw"])


                        # thermostat parameters
                        t_min = float(cfg["t_min_c"])
                        t_max = float(cfg["t_max_c"])
                        t_set = 0.5 * (t_min + t_max)
                        hyst  = max(t_max - t_min, 0.5)

                        # --- Get outdoor temperature from page 1, or fallback ---
                        if (
                            "temp_daily" in st.session_state
                            and isinstance(st.session_state["temp_daily"], pd.Series)
                            and not st.session_state["temp_daily"].empty
                        ):
                            tout_tot = st.session_state["temp_daily"]
                            tout_minute=get_selected_day_data(tout_tot)
                            
                            idx_hp = tout_minute.index
                            st.caption("Using fetched outdoor temperature for this preview.")
                        else:
                            idx_hp = pd.date_range(
                                "2025-01-10 00:00", periods=24 * 60, freq="min"
                            )
                            hours = idx_hp.hour + idx_hp.minute / 60.0
                            tout_minute = pd.Series(
                                5.0 + 5.0 * np.sin(2 * np.pi * (hours - 15) / 24.0),
                                index=idx_hp,
                                name="Tout_C",
                            )
                            st.caption(
                                "No weather data found, using a synthetic outdoor temperature profile."
                            )

                        # --- Build HP model with derived parameters ---
                        hp = WeatherHP(
                            ua_kw_per_c=float(ua),
                            t_set_c=t_set,
                            q_rated_kw=float(q_rated),
                            cop_at_7c=3.2,
                            cop_min=1.6,
                            cop_max=4.2,
                            C_th_kwh_per_c=float(Cth_guess),
                            hyst_band_c=hyst,
                            p_off_kw=0.05,     # 50 W standby
                            defrost=True,
                            min_on_min=0,
                            min_off_min=0,
                            Ti0_c=t_set,
                        )
                        p_hp = hp.series_kw(idx_hp, tout_minute)
                        cfg["profile_index"] = p_hp.index.astype(str).tolist()
                        cfg["profile_kw"]    = p_hp.values.tolist()


                        

                        # --- Plot HP electrical power ---
                        fig_hp = go.Figure()
                        fig_hp.add_scatter(
                            x=p_hp.index,
                            y=p_hp.values,
                            mode="lines",
                            name="P_HP_kW",
                        )
                        # small gap
                        st.markdown("<div style='height:0.75rem'></div>",
                                    unsafe_allow_html=True)
                        st.markdown("**Heat pump electrical power (preview)**")
                        fig_hp.update_layout(
                            height=180,
                            margin=dict(l=10, r=10, t=10, b=8),
                            xaxis_title="Time",
                            yaxis_title="kW",
                            showlegend=False,
                        )
                        st.plotly_chart(fig_hp, use_container_width=True)

                        if st.button("▲ Hide details", key=f"{settings_id}_hide_hp"):
                            st.session_state[f"settings_open_{settings_id}"] = False
                            st.rerun()

                    # 3) EL heater in thermal category: user-friendly inputs only
                    elif cat_key == "thermal" and dev["type"] == "e_heater":
                        st.markdown(
                            """
                            <hr style="border: 0; border-top: 1px dotted #fecaca; margin: 0.4rem 0;" />
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown("**E Heater comfort & house settings**")

                        # --- Comfort band ---
                        c_tmin, c_tmax = st.columns(2)
                        cfg["t_min_c"] = c_tmin.number_input(
                            "Min indoor temperature (°C)",
                            min_value=5.0,
                            max_value=30.0,
                            step=0.5,
                            value=float(cfg.get("t_min_c", 20.0)),
                            key=f"{settings_id}_tmin_eh",   # <<< changed key (added _eh)
                        )
                        cfg["t_max_c"] = c_tmax.number_input(
                            "Max indoor temperature (°C)",
                            min_value=5.0,
                            max_value=30.0,
                            step=0.5,
                            value=float(cfg.get("t_max_c", 22.0)),
                            key=f"{settings_id}_tmax_eh",   # <<< changed key (added _eh)
                        )
                        if cfg["t_max_c"] < cfg["t_min_c"]:
                            cfg["t_max_c"] = cfg["t_min_c"]

                        # --- Qualitative building info ---
                        c_size, c_ins = st.columns(2)
                        cfg["house_size"] = c_size.selectbox(
                            "House size",
                            ["Small apartment", "Medium house", "Large house"],
                            index=["Small apartment", "Medium house", "Large house"].index(
                                cfg.get("house_size", "Medium house")
                            ),
                            key=f"{settings_id}_hsize_eh",   # <<< changed key (added _eh)
                        )
                        cfg["insulation"] = c_ins.selectbox(
                            "Insulation level",
                            ["Poor", "Average", "Good"],
                            index=["Poor", "Average", "Good"].index(
                                cfg.get("insulation", "Average")
                            ),
                            key=f"{settings_id}_insul_eh",   # <<< changed key (added _eh)
                        )

                        # --- Map qualitative choices to ua & suggested q_rated ---
                        size = cfg["house_size"]
                        ins  = cfg["insulation"]

                        # base UA & capacity guess by size
                        if size == "Small apartment":
                            ua_base = 0.10   # kW/°C
                            q_guess = 4.0    # kW
                            Cth_guess = 0.50   # kWh/°C
                        elif size == "Large house":
                            ua_base = 0.14
                            q_guess = 8.0
                            Cth_guess = 0.75  # kWh/°C
                        else:  # "Medium house"
                            ua_base = 0.12
                            q_guess = 6.0
                            Cth_guess = 0.60

                        # adjust UA by insulation
                        if ins == "Poor":
                            ua = ua_base * 1.3
                        elif ins == "Good":
                            ua = ua_base * 0.7
                        else:
                            ua = ua_base

                        # --- Let the user override the capacity ---
                        cfg["q_rated_kw"] = st.number_input(
                            "E Heater capacity (kW)",
                            min_value=1.0,
                            max_value=30.0,
                            step=0.5,
                            value=float(cfg.get("q_rated_kw", q_guess)),
                            key=f"{settings_id}_q_rated_eh",   # <<< changed key (added _eh)
                        )

                        q_rated = float(cfg["q_rated_kw"])

                        # thermostat parameters
                        t_min = float(cfg["t_min_c"])
                        t_max = float(cfg["t_max_c"])
                        t_set = 0.5 * (t_min + t_max)
                        hyst  = max(t_max - t_min, 0.5)

                        # --- Get outdoor temperature from page 1, or fallback ---
                        if (
                            "temp_daily" in st.session_state
                            and isinstance(st.session_state["temp_daily"], pd.Series)
                            and not st.session_state["temp_daily"].empty
                        ):
                            tout_tot = st.session_state["temp_daily"]
                            tout_minute=get_selected_day_data(tout_tot)
                            idx_hp = tout_minute.index
                            st.caption("Using fetched outdoor temperature for this preview.")
                        else:
                            idx_hp = pd.date_range(
                                "2025-01-10 00:00", periods=24 * 60, freq="min"
                            )
                            hours = idx_hp.hour + idx_hp.minute / 60.0
                            tout_minute = pd.Series(
                                5.0 + 5.0 * np.sin(2 * np.pi * (hours - 15) / 24.0),
                                index=idx_hp,
                                name="Tout_C",
                            )
                            st.caption(
                                "No weather data found, using a synthetic outdoor temperature profile."
                            )

                        # --- Build EL-heater model with derived parameters ---
                        eh = WeatherELheater(
                            ua_kw_per_c=float(ua),
                            t_set_c=t_set,
                            q_rated_kw=float(q_rated),
                            C_th_kwh_per_c=float(Cth_guess),
                            hyst_band_c=hyst,
                            p_off_kw=0.05,     # 50 W standby
                            min_on_min=0,
                            min_off_min=0,
                            Ti0_c=t_set,
                        )
                        p_eh = eh.series_kw(idx_hp, tout_minute)
                        cfg["profile_index"] = p_eh.index.astype(str).tolist()
                        cfg["profile_kw"]    = p_eh.values.tolist()

                        # --- Plot E-heater electrical power ---
                        fig_Ehp = go.Figure()
                        fig_Ehp.add_scatter(
                            x=p_eh.index,
                            y=p_eh.values,
                            mode="lines",
                            name="P_EH_kW",
                        )
                        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
                        st.markdown("**E Heater electrical power (preview)**")
                        fig_Ehp.update_layout(
                            height=180,
                            margin=dict(l=10, r=10, t=10, b=8),
                            xaxis_title="Time",
                            yaxis_title="kW",
                            showlegend=False,
                        )
                        st.plotly_chart(fig_Ehp, use_container_width=True)

                        if st.button("▲ Hide details", key=f"{settings_id}_hide_eh"):  # <<< changed key
                            st.session_state[f"settings_open_{settings_id}"] = False
                            st.rerun()

                    # 4) Hot tub in thermal category
                    elif cat_key == "thermal" and dev["type"] == "hot_tub":
                        st.markdown(
                            """
                            <hr style="border: 0; border-top: 1px dotted #fecaca; margin: 0.4rem 0;" />
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown("**Hot tub comfort & usage settings**")

                        # ---- Basic temperature settings ----
                        c_tgt, c_idle = st.columns(2)
                        cfg["target_c"] = c_tgt.number_input(
                            "Target water temperature (°C)",
                            min_value=25.0,
                            max_value=45.0,
                            step=0.5,
                            value=float(cfg.get("target_c", 40.0)),
                            key=f"{settings_id}_ht_Ttarget",
                        )
                        cfg["idle_c"] = c_idle.number_input(
                            "Idle temperature (°C)",
                            min_value=10.0,
                            max_value=40.0,
                            step=0.5,
                            value=float(cfg.get("idle_c", 30.0)),
                            key=f"{settings_id}_ht_Tidle",
                        )
                        if cfg["idle_c"] > cfg["target_c"]:
                            cfg["idle_c"] = cfg["target_c"]

                        # ---- Water volume & heater size ----
                        c_vol, c_pow = st.columns(2)
                        cfg["water_l"] = c_vol.number_input(
                            "Water volume (L)",
                            min_value=400.0,
                            max_value=3000.0,
                            step=50.0,
                            value=float(cfg.get("water_l", 1200.0)),
                            key=f"{settings_id}_ht_vol",
                        )
                        cfg["heater_kw"] = c_pow.number_input(
                            "Heater capacity (kW)",
                            min_value=1.0,
                            max_value=12.0,
                            step=0.5,
                            value=float(cfg.get("heater_kw", 3.0)),
                            key=f"{settings_id}_ht_kw",
                        )

                        # Optional: number of identical hot tubs (very rarely >1, but keep general)
                        cfg["num_units"] = st.number_input(
                            "Number of identical hot tubs",
                            min_value=1,
                            max_value=5,
                            step=1,
                            value=int(cfg.get("num_units", 1)),
                            key=f"{settings_id}_ht_nunits",
                        )

                        # ---- Insulation level → UA ----
                        ins_levels = ["Good cover", "Average", "Poor"]
                        cfg["insulation_ht"] = st.selectbox(
                            "Cover / insulation level",
                            ins_levels,
                            index=ins_levels.index(cfg.get("insulation_ht", "Average")),
                            key=f"{settings_id}_ht_ins",
                        )

                        ins = cfg["insulation_ht"]
                        ua_base = 0.07  # kW/°C for "Average"
                        if ins == "Good cover":
                            ua = ua_base * 0.6
                        elif ins == "Poor":
                            ua = ua_base * 1.4
                        else:
                            ua = ua_base

                        # ---- Use sessions (start + duration) ----
                        st.markdown("**Use sessions (hot tub in use)**")
                        sessions = cfg.setdefault(
                            "sessions",
                            [{"start": _time(20, 0), "duration_min": 60}],
                        )

                        del_idx = None
                        for j, sess in enumerate(sessions):
                            c_s, c_d, c_del = st.columns([0.4, 0.4, 0.2])
                            with c_s:
                                s_t = c_s.time_input(
                                    "Start",
                                    value=sess.get("start", _time(20, 0)),
                                    key=f"{settings_id}_ht_s_{j}",
                                )
                            with c_d:
                                dur = c_d.number_input(
                                    "Duration (min)",
                                    min_value=15,
                                    max_value=600,
                                    step=15,
                                    value=int(sess.get("duration_min", 60)),
                                    key=f"{settings_id}_ht_d_{j}",
                                )
                            with c_del:
                                if c_del.button("🗑", key=f"{settings_id}_ht_del_{j}"):
                                    del_idx = j
                            sess["start"] = s_t
                            sess["duration_min"] = dur

                        if del_idx is not None:
                            sessions.pop(del_idx)
                            st.rerun()

                        if st.button("➕ Add use session", key=f"{settings_id}_ht_add"):
                            sessions.append(
                                {"start": _time(19, 0), "duration_min": 60}
                            )
                            st.rerun()

                        # ---- Get outdoor temperature for this day (same logic as HP) ----
                        if (
                            "temp_daily" in st.session_state
                            and isinstance(st.session_state["temp_daily"], pd.Series)
                            and not st.session_state["temp_daily"].empty
                        ):
                            tout_tot = st.session_state["temp_daily"]
                            # You already have this helper in the HP code:
                            tout_minute = get_selected_day_data(tout_tot)
                            idx_ht = tout_minute.index
                            st.caption("Using fetched outdoor temperature for this preview.")
                        else:
                            idx_ht = pd.date_range(
                                "2025-01-10 00:00", periods=24 * 60, freq="min"
                            )
                            hours = idx_ht.hour + idx_ht.minute / 60.0
                            tout_minute = pd.Series(
                                5.0 + 5.0 * np.sin(2 * np.pi * (hours - 15) / 24.0),
                                index=idx_ht,
                                name="Tout_C",
                            )
                            st.caption(
                                "No weather data found, using a synthetic outdoor temperature profile."
                            )

                        # ---- Simulate hot tub electrical power ----
                        ht = WeatherHotTub(
                            target_c=float(cfg["target_c"]),
                            idle_c=float(cfg["idle_c"]),
                            heater_kw=float(cfg["heater_kw"]),
                            water_l=float(cfg["water_l"]),
                            ua_kw_per_c=float(ua),
                            sessions=sessions,
                        )
                        p_ht_single= ht.series_kw(idx_ht, tout_minute)
                        p_ht = cfg["num_units"] * p_ht_single
                        cfg["profile_index"] = p_ht.index.astype(str).tolist()
                        cfg["profile_kw"]    = p_ht.values.tolist()

                        # ---- Plot ----
                        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                        st.markdown("**Hot tub electrical power (preview)**")

                        fig_ht = go.Figure()
                        fig_ht.add_scatter(
                            x=p_ht.index,
                            y=p_ht.values,
                            mode="lines",
                            name="P_hot_tub_kW",
                        )
                        fig_ht.update_layout(
                            height=180,
                            margin=dict(l=10, r=10, t=10, b=8),
                            xaxis_title="Time",
                            yaxis_title="kW",
                            showlegend=False,
                        )
                        st.plotly_chart(fig_ht, use_container_width=True)

                        if st.button("▲ Hide details", key=f"{settings_id}_hide_ht"):
                            st.session_state[f"settings_open_{settings_id}"] = False
                            st.rerun()

                    # 4) DHW tank in thermal category
                    elif cat_key == "thermal" and dev["type"] == "dhw":
                        st.markdown(
                            """
                            <hr style="border: 0; border-top: 1px dotted #fecaca; margin: 0.4rem 0;" />
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown("**DHW tank settings**")

                        # --- Tank volume & usage ---
                        c_vol, c_use = st.columns(2)
                        cfg["volume_l"] = c_vol.number_input(
                            "Tank volume (L)",
                            min_value=50.0,
                            max_value=500.0,
                            step=25.0,
                            value=float(cfg.get("volume_l", 200.0)),
                            key=f"{settings_id}_dhw_vol",
                        )
                      

                        cfg["usage_level"] = c_use.selectbox(
                            "Usage level",
                            {
                                "Low – 1–2 persons, short showers": "Low",
                                "Medium – 3–4 persons, normal daily use": "Medium",
                                "High – 5+ persons or long/frequent showers": "High",
                            }.keys(),
                            index=list({
                                "Low – 1–2 persons, short showers": "Low",
                                "Medium – 3–4 persons, normal daily use": "Medium",
                                "High – 5+ persons or long/frequent showers": "High",
                            }.values()).index(cfg.get("usage_level", "Medium")),
                            key=f"{settings_id}_dhw_use",
                            help="Select typical hot-water usage for your household. This changes the daily DHW energy consumption.",
                        )

                        # Internally store only 'Low' / 'Medium' / 'High'
                        selected_label = cfg["usage_level"]
                        label_to_value = {
                            "Low – 1–2 persons, short showers": "Low",
                            "Medium – 3–4 persons, normal daily use": "Medium",
                            "High – 5+ persons or long/frequent showers": "High",
                        }
                        cfg["usage_level"] = label_to_value[selected_label]


                        # --- Temperature band for tank ---
                        c_tmin, c_tmax = st.columns(2)
                        cfg["t_min_c"] = c_tmin.number_input(
                            "Min tank temperature (°C)",
                            min_value=30.0,
                            max_value=70.0,
                            step=1.0,
                            value=float(cfg.get("t_min_c", 45.0)),
                            key=f"{settings_id}_dhw_tmin",
                        )
                        cfg["t_max_c"] = c_tmax.number_input(
                            "Max tank temperature (°C)",
                            min_value=30.0,
                            max_value=70.0,
                            step=1.0,
                            value=float(cfg.get("t_max_c", 55.0)),
                            key=f"{settings_id}_dhw_tmax",
                        )
                        if cfg["t_max_c"] < cfg["t_min_c"]:
                            cfg["t_max_c"] = cfg["t_min_c"]

                        # --- Heater power ---
                        cfg["p_el_kw"] = st.number_input(
                            "Heater power (kW)",
                            min_value=0.5,
                            max_value=10.0,
                            step=0.5,
                            value=float(cfg.get("p_el_kw", 2.0)),
                            key=f"{settings_id}_dhw_pel",
                        )

                        # thermostat parameters
                        t_min = float(cfg["t_min_c"])
                        t_max = float(cfg["t_max_c"])
                        t_set = 0.5 * (t_min + t_max)
                        hyst  = max(t_max - t_min, 1.0)

                        # --- Get outdoor temperature from page 1, or fallback ---
                        if (
                            "temp_daily" in st.session_state
                            and isinstance(st.session_state["temp_daily"], pd.Series)
                            and not st.session_state["temp_daily"].empty
                        ):
                            tout_tot = st.session_state["temp_daily"]
                            tout_minute = get_selected_day_data(tout_tot)
                            idx_dhw = tout_minute.index
                            st.caption("Using fetched outdoor temperature for this preview.")
                        else:
                            idx_dhw = pd.date_range(
                                "2025-01-10 00:00", periods=24 * 60, freq="min"
                            )
                            hours = idx_dhw.hour + idx_dhw.minute / 60.0
                            tout_minute = pd.Series(
                                5.0 + 5.0 * np.sin(2 * np.pi * (hours - 15) / 24.0),
                                index=idx_dhw,
                                name="Tout_C",
                            )
                            st.caption(
                                "No weather data found, using a synthetic outdoor temperature profile."
                            )

                        # --- Build DHW tank model ---
                        dhw = DHWTank(
                            volume_l=float(cfg["volume_l"]),
                            t_set_c=t_set,
                            hyst_band_c=hyst,
                            ua_kw_per_c=0.02,  # you can expose later if you want
                            p_el_kw=float(cfg["p_el_kw"]),
                            p_off_kw=0.01,
                            T_cold_c=10.0,
                            T_amb_c=20.0,
                            Ti0_c=t_set,
                            usage_level=cfg["usage_level"],
                        )
                        p_dhw = dhw.series_kw(idx_dhw, tout_minute)
                        cfg["profile_index"] = p_dhw.index.astype(str).tolist()
                        cfg["profile_kw"]    = p_dhw.values.tolist()

                        # --- Plot DHW electric power ---
                        st.markdown("<div style='height:0.75rem'></div>",
                                    unsafe_allow_html=True)
                        st.markdown("**DHW heater electrical power (preview)**")

                        fig_dhw = go.Figure()
                        fig_dhw.add_scatter(
                            x=p_dhw.index,
                            y=p_dhw.values,
                            mode="lines",
                            name="P_DHW_kW",
                        )
                        fig_dhw.update_layout(
                            height=180,
                            margin=dict(l=10, r=10, t=10, b=8),
                            xaxis_title="Time",
                            yaxis_title="kW",
                            showlegend=False,
                        )
                        st.plotly_chart(fig_dhw, use_container_width=True)

                        if st.button("▲ Hide details", key=f"{settings_id}_hide_dhw"):
                            st.session_state[f"settings_open_{settings_id}"] = False
                            st.rerun()
                    # 3) (Generation & Storage) =========================================================
                    # PV 
                    # =========================================================
                    elif cat_key == "gen_store" and dev["type"] == "pv":
                        st.markdown(
                            """
                            <hr style="border: 0; border-top: 1px dotted #bbf7d0; margin: 0.4rem 0;" />
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown("**PV system settings**")

                        # --- Basic sizing ---
                        c_wp, c_np = st.columns(2)
                        cfg["module_wp"] = c_wp.number_input(
                            "Module nameplate (Wp)",
                            min_value=50.0,
                            max_value=1000.0,
                            step=10.0,
                            value=float(cfg.get("module_wp", 400.0)),
                            key=f"{settings_id}_pv_mod_wp",
                        )
                        cfg["n_panels"] = c_np.number_input(
                            "Number of panels",
                            min_value=0,
                            max_value=200,
                            step=1,
                            value=int(cfg.get("n_panels", 16)),
                            key=f"{settings_id}_pv_n_panels",
                        )
                        kwp = (cfg["module_wp"] * cfg["n_panels"]) / 1000.0
                        st.caption(f"Total DC size: **{kwp:.2f} kWp**")

                        # --- Orientation & losses ---
                        c_tilt, c_az = st.columns(2)
                        cfg["tilt"] = c_tilt.number_input(
                            "Tilt (°)",
                            min_value=0.0,
                            max_value=90.0,
                            step=1.0,
                            value=float(cfg.get("tilt", 30.0)),
                            key=f"{settings_id}_pv_tilt",
                        )
                        cfg["azimuth"] = c_az.number_input(
                            "Azimuth (°; 180 = South)",
                            min_value=0.0,
                            max_value=360.0,
                            step=1.0,
                            value=float(cfg.get("azimuth", 180.0)),
                            key=f"{settings_id}_pv_az",
                        )

                        cfg["loss_frac"] = st.number_input(
                            "System losses (fraction)",
                            min_value=0.0,
                            max_value=0.5,
                            step=0.01,
                            value=float(cfg.get("loss_frac", 0.14)),
                            key=f"{settings_id}_pv_loss",
                        )

                        # --- Build minute index for selected day ---
                        # We only need the selected day from Page 1
                        from datetime import timedelta as _td

                        sel_day = st.session_state.get("day")
                        if sel_day is not None:
                            day_start = pd.Timestamp(sel_day)
                            day_end = day_start + _td(days=1)
                            idx_pv = pd.date_range(
                                day_start,
                                day_end,
                                freq="min",
                                inclusive="left",
                            )
                        else:
                            # fallback: synthetic day
                            idx_pv = pd.date_range(
                                "2025-01-10 00:00",
                                periods=24 * 60,
                                freq="min",
                            )

                        # --- Use real weather if available, otherwise synthetic ---
                        pv_series = None                        
                        if (
                            "weather_hr" in st.session_state
                            and isinstance(st.session_state["weather_hr"], pd.DataFrame)
                            and not st.session_state["weather_hr"].empty
                            and kwp > 0
                        ):
                            weather_hr = st.session_state["weather_hr"]
                            try:
                                pv_series = pv_from_weather_modelchain_from_df(
                                    idx_min=idx_pv,
                                    dfh=weather_hr,
                                    lat=float(st.session_state.get("geo_lat", 57.0488)),
                                    lon=float(st.session_state.get("geo_lon", 9.9217)),
                                    kwp=kwp,
                                    tilt_deg=float(cfg["tilt"]),
                                    az_deg=float(cfg["azimuth"]),
                                    sys_loss_frac=float(cfg["loss_frac"]),
                                )
                                st.caption("Using fetched weather for PV preview.")
                            except Exception as e:
                                st.warning(f"PV preview using weather failed: {e}. Falling back to synthetic curve.")
                                pv_series = None

                        if pv_series is None:
                            # Simple synthetic bell-shaped PV profile for preview only
                            hours = idx_pv.hour + idx_pv.minute / 60.0
                            shape = np.maximum(0.0, np.sin(np.pi * (hours - 6.0) / 12.0))
                            pv_series = pd.Series(kwp * shape, index=idx_pv, name="pv_kw")
                            st.caption("No usable weather data – showing a synthetic PV curve for preview.")

                        cfg["profile_index"] = pv_series.index.astype(str).tolist()
                        cfg["profile_kw"]    = pv_series.values.tolist()

                        # --- Plot PV preview ---
                        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
                        st.markdown("**PV power for selected day (preview)**")

                        fig_pv = go.Figure()
                        fig_pv.add_scatter(
                            x=pv_series.index,
                            y=pv_series.values,
                            mode="lines",
                            name="pv_kw",
                        )
                        fig_pv.update_layout(
                            height=180,
                            margin=dict(l=10, r=10, t=10, b=8),
                            xaxis_title="Time",
                            yaxis_title="kW",
                            showlegend=False,
                        )
                        st.plotly_chart(fig_pv, use_container_width=True)

                        # Hide button
                        if st.button("▲ Hide details", key=f"{settings_id}_hide_pv"):
                            st.session_state[f"settings_open_{settings_id}"] = False
                            st.rerun()

                    # =========================================================
                    # BATTERY (Generation & Storage)
                    # =========================================================
                    elif cat_key == "gen_store" and dev["type"] == "battery":

                        st.markdown(
                            "<hr style='border:0;border-top:1px dotted #bbf7d0;margin:0.4rem 0;'/>",
                            unsafe_allow_html=True
                        )
                        st.markdown("**Battery settings can be input in next page**")
                        # Hide button
                        if st.button("▲ Hide details", key=f"{settings_id}_batt_hide"):
                            st.session_state[f"settings_open_{settings_id}"]=False
                            st.rerun()
                    
                    # =========================================================
                    # FC (Generation & Storage)
                    # =========================================================
                    elif cat_key == "gen_store" and dev["type"] == "fuel_cell":
                        st.markdown(
                            "<hr style='border:0;border-top:1px dotted #bbf7d0;margin:0.4rem 0;'/>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("**Fuel cell (methanol-reformed HT-PEM)**")

                        # --- Basic sizing ---
                        c1, c2 = st.columns(2)
                        cfg["p_rated_kw"] = c1.number_input(
                            "Rated electrical power (kW)",
                            min_value=0.5,
                            max_value=50.0,
                            step=0.5,
                            value=float(cfg.get("p_rated_kw", 4.8)),   # ~4.8 kW example
                            key=f"{settings_id}_fc_prated",
                        )
                        cfg["p_min_frac"] = c2.slider(
                            "Minimum loading (% of rated)",
                            min_value=0,
                            max_value=100,
                            value=int(cfg.get("p_min_frac", 40)),
                            step=5,
                            key=f"{settings_id}_fc_pminfrac",
                        )

                        p_rated_kw = float(cfg["p_rated_kw"])
                        p_min_kw   = p_rated_kw * (cfg["p_min_frac"] / 100.0)

                        # store in W for the cost model
                        p_rated_W = p_rated_kw * 1000.0
                        p_min_W   = p_min_kw   * 1000.0

                        # --- Fuel cost parameter (equivalent methanol cost) ---
                        st.markdown("**Fuel economics**")
                        cfg["methanol_price"] = st.number_input(
                            "Methanol cost (DKK per kWh of fuel)",
                            min_value=0.0,
                            max_value=50.0,
                            step=0.1,
                            value=float(cfg.get("methanol_price", 3.75)),  # choose whatever default you used before
                            key=f"{settings_id}_fc_methprice",
                        )
                        meth_price = float(cfg["methanol_price"])

                        st.caption(
                            "This value should be consistent with the methanol cost you purchased "
                        )

                        # --- Economic schedule for selected day (15-min) ---
                        st.markdown("**Economic schedule for selected daymarkdown**")

                        if st.button("💡 Suggest FC power schedule (minute)", key=f"{settings_id}_fc_suggest"):
                            if "price_daily" not in st.session_state or st.session_state["price_daily"].empty:
                                st.warning("No price data or selected day found.")
                            else:
                                price_tot = st.session_state["price_daily"]
                                sel_day = st.session_state["day"]
                                day_start = pd.Timestamp(sel_day)
                                day_end   = day_start + pd.Timedelta(days=1)

                                # exact minute-resolution selection
                                price_min = price_tot.loc[(price_tot.index >= day_start) & (price_tot.index < day_end)]
                                if price_min.empty:
                                    st.warning("No price data for the day.")
                                else:
                                    prices = price_min.values  # 1440 points

                                    opti_W, fvals = solve_fc_schedule_minute(
                                        prices,
                                        Price_ch3oh=meth_price,
                                        Pmin_W=p_min_W,
                                        Prated_W=p_rated_W,
                                    )
                                    idx_min = price_min.index
                                    p_fc_raw = pd.Series(opti_W, index=idx_min, name="fc_kw")
                                    p_fc_smooth = smooth_fc_schedule(
                                        p_fc_raw,
                                        dt_min=1,
                                        min_on_min=60,   # e.g. at least 1 h once started
                                        min_off_min=30,  # don't stop for tiny 15-min gaps
                                    )


                                    cfg["fc_schedule_minute_kw"] = p_fc_smooth / 1000.0
                                    cfg["fc_schedule_minute_index"] = price_min.index.astype(str).tolist()

                                    st.success("FC schedule computed (minute-level).")


                        # --- Plot schedule if available ---
                        if "fc_schedule_minute_kw" in cfg and cfg.get("fc_schedule_minute_kw") is not None:
                            try:
                                idx = pd.to_datetime(cfg.get("fc_schedule_minute_index", []))
                                p_kw = np.asarray(cfg["fc_schedule_minute_kw"], dtype=float)
                                if len(idx) == len(p_kw) and len(idx) > 0:
                                    s = pd.Series(p_kw, index=idx, name="P_FC_kW")

                                    fig_fc = go.Figure()
                                    fig_fc.add_scatter(
                                        x=s.index,
                                        y=s.values,
                                        mode="lines",
                                        name="P_FC_kW",
                                    )
                                    st.markdown("**Suggested FC electrical power (15-min)**")
                                    fig_fc.update_layout(
                                        height=200,
                                        margin=dict(l=10, r=10, t=10, b=8),
                                        xaxis_title="Time",
                                        yaxis_title="kW",
                                        showlegend=False,
                                    )
                                    st.plotly_chart(fig_fc, use_container_width=True)
                                else:
                                    st.caption("No valid FC schedule stored yet.")
                            except Exception:
                                st.caption("No valid FC schedule stored yet.")

                        # Hide button
                        if st.button("▲ Hide details", key=f"{settings_id}_hide_fc"):
                            st.session_state[f"settings_open_{settings_id}"] = False
                            st.rerun()

                    elif cat_key == "ev" and dev["type"] == "ev":
                        st.markdown(
                            """
                            <hr style="border: 0; border-top: 1px dotted #bfdbfe; margin: 0.4rem 0;" />
                            """,
                            unsafe_allow_html=True,
                        )
                        st.markdown("**EV charging settings (01:00–06:00 window)**")

                        # --- Charger & battery ---
                        c_p, c_cap = st.columns(2)
                        cfg["power_kw"] = c_p.number_input(
                            "Charger power (kW)",
                            min_value=1.0,
                            max_value=50.0,
                            step=0.5,
                            value=float(cfg.get("power_kw", 11.0)),
                            key=f"{settings_id}_ev_power",
                        )
                        cfg["capacity_kwh"] = c_cap.number_input(
                            "EV battery capacity (kWh)",
                            min_value=10.0,
                            max_value=200.0,
                            step=1.0,
                            value=float(cfg.get("capacity_kwh", 75.0)),
                            key=f"{settings_id}_ev_cap",
                        )

                        c_soc_a, c_soc_t = st.columns(2)
                        cfg["soc_arrive"] = c_soc_a.number_input(
                            "Arrival SOC (%)",
                            min_value=0.0,
                            max_value=100.0,
                            step=5.0,
                            value=float(cfg.get("soc_arrive", 20.0)),
                            key=f"{settings_id}_ev_soc_a",
                        )
                        cfg["soc_target"] = c_soc_t.number_input(
                            "Target SOC at departure (%)",
                            min_value=0.0,
                            max_value=100.0,
                            step=5.0,
                            value=float(cfg.get("soc_target", 80.0)),
                            key=f"{settings_id}_ev_soc_t",
                        )
                        if cfg["soc_target"] < cfg["soc_arrive"]:
                            cfg["soc_target"] = cfg["soc_arrive"]

                        # Needed energy and duration (minutes)
                        delta_soc = max(cfg["soc_target"] - cfg["soc_arrive"], 0.0) / 100.0
                        energy_need = delta_soc * cfg["capacity_kwh"]      # kWh
                        if cfg["power_kw"] > 0 and energy_need > 0:
                            duration_min = int(np.ceil(energy_need * 60.0 / cfg["power_kw"]))
                        else:
                            duration_min = 0
                        cfg["duration_min"] = duration_min

                        st.caption(
                            f"Energy needed ≈ **{energy_need:.1f} kWh**, "
                            f"charging time ≈ **{duration_min} min** at {cfg['power_kw']:.1f} kW."
                        )

                        # Cost/CO2 preference
                        cfg["w_cost"] = float(
                            st.slider(
                                "Preference (0 = CO₂ only, 1 = cost only)",
                                min_value=0.0,
                                max_value=1.0,
                                step=0.05,
                                value=float(cfg.get("w_cost", 0.5)),
                                key=f"{settings_id}_ev_w_cost",
                            )
                        )

                        # Single interval in cfg
                        from datetime import datetime, timedelta, time as _time

                        intervals = cfg.setdefault("intervals", [])
                        if not intervals:
                            intervals.append(
                                {"start": _time(1, 0), "end": _time(6, 0)}
                            )
                        elif len(intervals) > 1:
                            intervals[:] = intervals[:1]

                        current_iv = intervals[0]

                        st.caption("Current scheduled interval (single continuous block):")
                        c_a, c_b = st.columns(2)
                        with c_a:
                            start_time = c_a.time_input(
                                "Start (01:00–06:00 preferred)",
                                value=current_iv.get("start", _time(1, 0)),
                                key=f"{settings_id}_ev_start",
                            )

                        # recompute end from duration
                        if duration_min > 0:
                            dt0 = datetime.combine(date.today(), start_time)
                            dt1 = dt0 + timedelta(minutes=duration_min)
                            end_time = dt1.time()
                        else:
                            end_time = current_iv.get("end", _time(1, 30))

                        with c_b:
                            st.time_input(
                                "End (computed)",
                                value=end_time,
                                key=f"{settings_id}_ev_end_display",
                                disabled=True,
                            )

                        current_iv["start"], current_iv["end"] = start_time, end_time

                        # Suggest button (search only 01:00–06:00 using price/CO₂)
                        if st.button("💡 Suggest cheapest/cleanest in 01:00–06:00",
                                     key=f"{settings_id}_ev_suggest"):
                            if duration_min <= 0:
                                st.warning("No energy needed (arrival SOC ≥ target SOC).")
                            else:
                                interval = suggest_best_interval_for_ev(
                                    duration_min=duration_min,
                                    w_cost=cfg["w_cost"],
                                    window_start_min=60,
                                    window_end_min=360,
                                )
                                if interval is None:
                                    st.warning(
                                        "No price/CO₂ data or no selected day. "
                                        "Please select a day and fetch data on page 1 first."
                                    )
                                else:
                                    intervals[0] = interval
                                    st.success(
                                        f"Suggested interval: "
                                        f"{interval['start'].strftime('%H:%M')}–"
                                        f"{interval['end'].strftime('%H:%M')}"
                                    )
                                    st.rerun()

                        # Preview profile (kW)
                        st.markdown("**Daily EV charging profile (preview)**")
                        prof_ev = build_minute_profile(
                            power_w=cfg["power_kw"] * 1000.0,  # W input
                            intervals=intervals,
                            step_min=1,
                        )
                        cfg["profile_index"] = prof_ev.index.astype(str).tolist()
                        cfg["profile_kw"]    = prof_ev.values.tolist()

                        fig_ev = go.Figure()
                        fig_ev.add_scatter(
                            x=prof_ev.index,
                            y=prof_ev.values,
                            mode="lines",
                            name="P_EV_kW",
                        )
                        fig_ev.update_layout(
                            height=180,
                            margin=dict(l=10, r=10, t=10, b=8),
                            xaxis_title="Time",
                            yaxis_title="kW",
                            showlegend=False,
                        )
                        st.plotly_chart(fig_ev, use_container_width=True)

                        if st.button("▲ Hide details", key=f"{settings_id}_hide_ev"):
                            st.session_state[f"settings_open_{settings_id}"] = False
                            st.rerun()


                    else:
                        # other categories – placeholder for now
                        st.info("Settings for this device type will be added later.")
                        if st.button("▲ Hide details", key=f"{settings_id}_hide_other"):
                            st.session_state[f"settings_open_{settings_id}"] = False
                            st.rerun()

                    # bottom dashed line
                    st.markdown(
                        "<hr style='border-top: 1px dashed #bbb;'/>",
                        unsafe_allow_html=True,
                    )

        else:
            st.caption("No devices in this group yet.")

        st.markdown("")  # small gap

        # Add new device
        c_add1, c_add2, c_add3 = st.columns([0.4, 0.4, 0.2])
        with c_add1:
            label = st.selectbox(
                "Type",
                list(type_choices.keys()),
                key=f"{cat_key}_new_type",
            )
        with c_add2:
            default_name = type_choices[label].replace("_", " ").title()
            name = st.text_input(
                "Name",
                value=default_name,
                key=f"{cat_key}_new_name",
            )
        with c_add3:
            add_clicked = st.button("➕", key=f"{cat_key}_add_btn")

        max_cap = MAX_PER_CATEGORY.get(cat_key, 99)
        if add_clicked:
            if len(dev_list) >= max_cap:
                st.info(
                    f"Cannot add more than {max_cap} devices in this group. "
                    "Please delete one before adding another."
                )
            else:
                new_id = f"{cat_key}_{len(dev_list)+1}"
                dev_list.append(
                    {"id": new_id, "type": type_choices[label], "name": name}
                )
                st.rerun()

        devices[cat_key] = dev_list

    
    # ------------------------------------------------------------------ #
    # LEFT + RIGHT COLUMNS
    # ------------------------------------------------------------------ #
    left_col, right_col = st.columns([1.5, 2])

    # ---------------- LEFT: device lists ----------------
    with left_col:
        st.markdown("### Choose devices")

        render_category_ui(
            "elec_fixed",
            "1. Household electrical – fixed",
            {
                "Lights": "lights",
                "Range hood": "hood",
                "Refrigerator": "fridge",
                "Other fixed load": "other",
            },
        )

        st.markdown("---")

        render_category_ui(
            "elec_flex",
            "1b. Household electrical – flexible",
            {
                "Washing machine": "wm",
                "Dishwasher": "dw",
                "Dryer": "dryer",
                "Other flexible load": "other",
            },
        )

        st.markdown("---")

        render_category_ui(
            "thermal",
            "2. Household thermal",
            {
                "Heat pump": "hp",
                "Electric heater": "e_heater",
                "Hot tub": "hot_tub",
                "DHW tank": "dhw",
            },
        )

        st.markdown("---")

        render_category_ui(
            "gen_store",
            "3. Generation & storage",
            {
                "PV system": "pv",
                "Battery": "battery",
                "Fuel cell": "fuel_cell",
            },
        )

        st.markdown("---")

        render_category_ui(
            "ev",
            "4. Outside",
            {
                "EV charger": "ev",
            },
        )

    # ------------------------------------------------------------------ #
    # Derive a simple devices_enabled dict for compatibility
    # ------------------------------------------------------------------ #
    enabled = {
        "lights": False,
        "hood": False,
        "fridge": False,
        "wm": False,
        "dw": False,
        "dryer": False,
        "hp": False,
        "e_heater": False,
        "hot_tub": False,
        "pv": False,
        "battery": False,
        "fuel_cell": False,
        "diesel": False,
        "ev": False,
    }
    for dev_list in devices.values():
        for d in dev_list:
            t = d["type"]
            if t in enabled:
                enabled[t] = True
    st.session_state["devices_enabled"] = enabled
    st.session_state["device_configs"] = device_configs  # <<< write back configs

    # ---------------- RIGHT: house drawing ----------------
    with right_col:
        st.markdown("### House layout")

        fig = go.Figure()
        shapes = []

        # HOUSE BODY
        shapes.append(
            dict(
                type="rect",
                x0=1, y0=2, x1=9, y1=7,
                line=dict(width=2),
                fillcolor="rgba(245,245,245,0.8)",
            )
        )

        # ROOF = Generation & storage
        roof_path = "M 1 7 L 5 9 L 9 7 Z"
        shapes.append(
            dict(
                type="path",
                path=roof_path,
                line=dict(width=2),
                fillcolor="rgba(180, 255, 180, 0.5)",
            )
        )

        # Electrical zone (upper half)
        shapes.append(
            dict(
                type="rect",
                x0=1.05, y0=4.0, x1=8.95, y1=6.95,
                line=dict(width=1, dash="dot"),
                fillcolor="rgba(210, 225, 255, 0.4)",
            )
        )
        # Left = fixed
        shapes.append(
            dict(
                type="rect",
                x0=1.1, y0=4.1, x1=4.8, y1=6.85,
                line=dict(width=1, dash="dot"),
                fillcolor="rgba(150, 200, 255, 0.4)",
            )
        )
        # Right = flexible
        shapes.append(
            dict(
                type="rect",
                x0=5.2, y0=4.1, x1=8.85, y1=6.85,
                line=dict(width=1, dash="dot"),
                fillcolor="rgba(255, 235, 170, 0.4)",
            )
        )

        # Thermal zone (lower half)
        shapes.append(
            dict(
                type="rect",
                x0=1.05, y0=2.05, x1=8.95, y1=3.95,
                line=dict(width=1, dash="dot"),
                fillcolor="rgba(255, 190, 190, 0.45)",
            )
        )

        # EV outside
        shapes.append(
            dict(
                type="rect",
                x0=9.2, y0=1.95, x1=10.1, y1=3.02,
                line=dict(width=1.5),
                fillcolor="rgba(250,250,250,0.9)",
            )
        )

        fig.update_layout(shapes=shapes)

        # ---- Zone labels ----
        annotations = [
            dict(x=2.0, y=4.3, text="Electrical (fixed)",    showarrow=False, font=dict(size=10)),
            dict(x=6.2, y=4.3, text="Electrical (flexible)", showarrow=False, font=dict(size=10)),
            dict(x=1.8, y=2.2, text="Thermal",               showarrow=False, font=dict(size=10)),
            dict(x=5.0, y=7.2, text="Generation & Storage",  showarrow=False, font=dict(size=11)),
        ]

        # ---- Device positions per zone ----
        def add_zone_devices_grid(
            dev_list,
            base_x,
            base_y,
            max_cols,
            max_rows,
            dx,
            dy,
        ):
            capacity = max_cols * max_rows
            for j, dev in enumerate(dev_list[:capacity]):
                col = j % max_cols
                row = j // max_cols
                x = base_x + col * dx
                y = base_y - row * dy
                icon = TYPE_ICONS.get(dev["type"], "🔌")
                label = f"{icon} {dev['name']}"
                annotations.append(
                    dict(
                        x=x,
                        y=y,
                        text=label,
                        showarrow=False,
                        font=dict(size=11),
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="rgba(0,0,0,0.25)",
                        borderwidth=1,
                        borderpad=2,
                        xanchor="center",
                        yanchor="middle",
                    )
                )

        # 1) Electrical fixed – 5 x 4 grid
        add_zone_devices_grid(
            devices["elec_fixed"],
            base_x=1.6,
            base_y=6.5,
            max_cols=4,
            max_rows=4,
            dx=0.91,
            dy=0.6,
        )

        # 2) Electrical flexible – 5 x 4 grid
        add_zone_devices_grid(
            devices["elec_flex"],
            base_x=5.65,
            base_y=6.5,
            max_cols=4,
            max_rows=4,
            dx=0.91,
            dy=0.6,
        )
        # 3) Thermal – 4 x 4 grid
        add_zone_devices_grid(
            devices["thermal"],
            base_x=3.0,
            base_y=3.7,
            max_cols=4,
            max_rows=4,
            dx=1.3,
            dy=0.45,
        )
        # 4) Generation & storage – triangular layout (2 + 4 + 6 = 12 slots)
        gen_slots = [
            # top row (2)
            (4.6, 8.4),
            (5.4, 8.4),
            # middle row (4)
            (3.9, 8.0),
            (4.6, 8.0),
            (5.4, 8.0),
            (6.2, 8.0),
            # bottom row (6)
            (2.9, 7.5),
            (3.7, 7.5),
            (4.5, 7.5),
            (5.3, 7.5),
            (6.1, 7.5),
            (7.0, 7.5),
        ]

        for j, dev in enumerate(devices["gen_store"][: len(gen_slots)]):
            x, y = gen_slots[j]
            icon = TYPE_ICONS.get(dev["type"], "🔌")
            label = f"{icon} {dev['name']}"
            annotations.append(
                dict(
                    x=x,
                    y=y,
                    text=label,
                    showarrow=False,
                    font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.25)",
                    borderwidth=1,
                    borderpad=2,
                    xanchor="center",
                    yanchor="middle",
                )
            )
        # 5) EV – up to 2 stacked outside
        ev_slots = [
            (9.65, 2.7),
            (9.65, 2.2),
        ]
        for j, dev in enumerate(devices["ev"][: len(ev_slots)]):
            x, y = ev_slots[j]
            icon = TYPE_ICONS.get(dev["type"], "🚗")
            label = f"{icon} {dev['name']}"
            annotations.append(
                dict(
                    x=x,
                    y=y,
                    text=label,
                    showarrow=False,
                    font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.25)",
                    borderwidth=1,
                    borderpad=2,
                    xanchor="center",
                    yanchor="middle",
                )
            )

       

        fig.update_layout(
            annotations=annotations,
            xaxis=dict(visible=False, range=[0, 11]),
            yaxis=dict(visible=False, range=[0, 10]),
            margin=dict(l=10, r=10, t=10, b=10),
            template="plotly_white",
            autosize=True,
            height=None,
        )

        fig.update_yaxes(scaleanchor=None)
        st.plotly_chart(fig, use_container_width=True, config={"responsive": True})
        # ------------------------------------------------------------------ #
        # Combined daily profiles: fixed, flexible, EV, total load & PV
        # ------------------------------------------------------------------ #
        # ---------------------------------------------------------------------------
        # Combined per-minute electric load and PV generation for all devices
        # ---------------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### Daily power profiles – per device")

        device_configs = st.session_state.get("device_configs", {})
        devices        = st.session_state.get("devices", {})

        series_by_device = {}
        type_counter = {}  
        series_meta = {}

        for cat_key, dev_list in devices.items():
            for dev in dev_list:
                settings_id = f"{cat_key}_{dev['id']}"
                cfg = device_configs.get(settings_id)
                if not cfg:
                    continue

                if "profile_index" not in cfg or "profile_kw" not in cfg:
                    # this device hasn’t generated a preview yet
                    continue

                try:
                    idx = pd.to_datetime(cfg["profile_index"])
                    vals = np.asarray(cfg["profile_kw"], dtype=float)
                    if len(idx) != len(vals) or len(idx) == 0:
                        continue
                except Exception:
                    continue

                # ---- build unique label ----
                dev_type = dev["type"]
                type_counter[dev_type] = type_counter.get(dev_type, 0) + 1
                icon = TYPE_ICONS.get(dev_type, "🔌")
                if type_counter[dev_type] == 1:
                    label = f"{icon} {dev['name']}"
                else:
                    label = f"{icon} {dev['name']} #{type_counter[dev_type]}"

                s = pd.Series(vals, index=idx, name=label)
                s = normalize_to_dummy_day(s)   # <<< normalize to 2000-01-01, keep only HH:MM
                series_by_device[settings_id] = s
                series_meta[settings_id] = dev_type


        if not series_by_device:
            st.caption("No device profiles stored yet – open a device’s settings to generate its preview.")
        else:
            fig_all = go.Figure()
            all_load_series = []
            pv_series_list  = []

            # --- Separate PV vs non-PV ---
            for sid, s in series_by_device.items():
                d_type = series_meta.get(sid, "")

                if d_type == "pv":
                    pv_series_list.append(s)
                else:
                    all_load_series.append(s)
                    fig_all.add_scatter(
                        x=s.index,
                        y=s.values,
                        mode="lines",
                        name=s.name,
                    )

            # --- Total load (only non-PV devices) ---
            if all_load_series:
                total_load = all_load_series[0].copy()
                for s in all_load_series[1:]:
                    total_load = total_load.add(s, fill_value=0.0)
                total_load.name = "Total load"
                fig_all.add_scatter(
                    x=total_load.index,
                    y=total_load.values,
                    mode="lines",
                    name="Total load",
                    line=dict(width=3),
                )



            # --- Aggregate PV and plot as area ---
            if pv_series_list:
                pv_total = pv_series_list[0].copy()
                for s in pv_series_list[1:]:
                    pv_total = pv_total.add(s, fill_value=0.0)
                pv_total.name = "PV generation"
                fig_all.add_scatter(
                    x=pv_total.index,
                    y=pv_total.values,
                    mode="lines",
                    name="PV generation",
                    fill="tozeroy",                 # <-- area under the curve
                    fillcolor="rgba(255, 230, 150, 0.4)",  # soft transparent yellow
                    line=dict(width=2),
                )


            fig_all.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=20, b=30),
                xaxis_title="Time of day",
                yaxis_title="kW",
            )
            st.plotly_chart(fig_all, use_container_width=True)

            # save load and PV
            if total_load is not None:
                st.session_state ["load"]= total_load
            if pv_total is not None:
                st.session_state ["pv"]= pv_total

#%% -------------------for page 3------------------------------------




#%% -------- SIDEBAR --------
st.set_page_config(page_title="Daily EMS Sandbox", layout="wide")

with st.sidebar:
    st.header("Daily EMS sandbox")
    st.markdown("### Navigation")
    page = st.radio(
        "Step",
        ["1️⃣ Scenario & data", "2️⃣ Devices & layout", "3️⃣ Analysis"],
        key="page",
    )

#%% # -------- MAIN AREA --------
page = st.session_state.get("page", "1️⃣ Scenario & data")
# page 1
if page.startswith("1️⃣"):
    st.title("1️⃣ Scenario & data")
    
    # 1) Scenario basics
    st.markdown("### Scenario")
    today=date.today()

    selected_day = st.date_input("Day", value=st.session_state["day"], key="day")

    


    min_date= date(2025, 10, 1)
    st.markdown("**Location**")
    preset_locations = {
        "Aalborg (DK1)":    {"lat": 57.0488, "lon": 9.9217,  "area": "DK1"},
        "Aarhus (DK1)":     {"lat": 56.1629, "lon": 10.2039, "area": "DK1"},
        "Odense (DK1)":     {"lat": 55.4038, "lon": 10.4024, "area": "DK1"},
        "Copenhagen (DK2)": {"lat": 55.6761, "lon": 12.5683, "area": "DK2"},
        "Custom":           None,
    }

    choice = st.selectbox("Choose city", list(preset_locations.keys()), key="city_choice")

    # --- Initialize lat/lon + price_area in session_state ---

    if "geo_lat" not in st.session_state:
        st.session_state["geo_lat"] = 57.0488
    if "geo_lon" not in st.session_state:
        st.session_state["geo_lon"] = 9.9217
    if "price_area" not in st.session_state:
        st.session_state["price_area"] = "DK1"  # sensible default for Aalborg



    if choice != "Custom":
        preset = preset_locations[choice]
        st.session_state["geo_lat"] = preset["lat"]
        st.session_state["geo_lon"] = preset["lon"]
        st.session_state["price_area"] = preset["area"]
    else:
        # Custom: we can guess area from longitude, but user can override
        lon_guess = st.session_state["geo_lon"]
        guessed_area = "DK2" if lon_guess > 11.5 else "DK1"
        # Only update default if user hasn't manually changed it before
        if "price_area_manual" not in st.session_state or not st.session_state["price_area_manual"]:
            st.session_state["price_area"] = guessed_area


    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input(
            "Latitude",
            -90.0,  90.0,
            float(st.session_state["geo_lat"]),
            0.0001,
            key="geo_lat",
        )
    with col2:
        lon = st.number_input(
            "Longitude",
            -180.0, 180.0,
            float(st.session_state["geo_lon"]),
            0.0001,
            key="geo_lon",
        )

    # Price area selector (DK1/DK2) – user can always override
    area_options = ["DK1", "DK2"]
    area_index = 0 if st.session_state["price_area"] == "DK1" else 1
    area = st.selectbox(
        "Electricity price area",
        area_options,
        index=area_index,
        key="price_area",
    )
    # Remember that user touched this, so we don't auto-overwrite next time for Custom
    st.session_state["price_area_manual"] = True

    st.markdown("**Map preview**")
    df_map = pd.DataFrame({"lat": [st.session_state["geo_lat"]],
                           "lon": [st.session_state["geo_lon"]]})
    st.map(df_map, zoom=9)
    st.markdown("---")

    st.write(f"Selected day: {st.session_state['day']}")
    st.write(f"Location: lat={st.session_state['geo_lat']:.4f}, lon={st.session_state['geo_lon']:.4f}")
    st.info("Here we will later show selected-period price / CO₂ / temperature overview.")

    st.markdown("### Choose period for overview")

    # Period selection widgets
    col_a, col_b = st.columns(2)
    with col_a:
        period_start = st.date_input(
            "Period start",
            value=today - timedelta(days=15),
            min_value= min_date,
            max_value=today+timedelta(days=1),
            key="period_start",
        )
    with col_b:
        period_end = st.date_input(
            "Period end",
            value=today,
            min_value= min_date,
            max_value=today+timedelta(days=1),
            key="period_end",
        )

    if period_end < period_start:
        st.error("End date must be on or after start date.")
        st.stop()

    if st.button("📥 Fetch CO₂, price and temperature for selected period", key="fetch_period"):
        # Optionally enforce a max length (e.g. 120 days) to avoid huge downloads
        if (period_end - period_start).days > 180:
            st.warning("Please choose a period shorter than 180 days.")
        else:
            # 1) Price CO₂ Temperature (original)
            idx = minute_index(period_start,period_end)
            price_plot, note_price = daily_price_dual(idx, period_start, period_end, area)
            co2,   note_co2   = daily_co2_with_note(idx, period_start, period_end, area)
            weather_hr = fetch_weather_open_meteo(
                lat,
                lon,

                start_date=period_start,
                end_date=period_end,
                tz="Europe/Copenhagen",
            )
            tout_minute, note_temp = daily_temperature_with_note(idx, weather_hr)

            # Save in session_state so we don't lose it when changing pages
            st.session_state["period_range"] = (period_start, period_end)
            st.session_state["price_daily"] = price_plot
            st.session_state["co2_daily"] = co2
            st.session_state["temp_daily"] = tout_minute
            st.session_state["weather_hr"]  = weather_hr
            st.session_state["note_price"]   = note_price
            st.session_state["note_co2"]     = note_co2
            st.session_state["note_temp"]    = note_temp


            st.success("Period data fetched successfully.")


    # --- Always show period overview if data already fetched --- 
    price_series = st.session_state.get("price_daily",  pd.Series(dtype=float))
    co2_series   = st.session_state.get("co2_daily",    pd.Series(dtype=float))
    temp_series  = st.session_state.get("temp_daily",   pd.Series(dtype=float))
    note_price   = st.session_state.get("note_price",   "")
    note_co2     = st.session_state.get("note_co2",     "")
    note_temp    = st.session_state.get("note_temp",    "")

    if not price_series.empty and not co2_series.empty and not temp_series.empty:
        st.markdown("### Period overview")

        # Price (bar version)
        fig_price = plot_period_bar(
            price_series,
            selected_day=selected_day,
            title="Electricity price over selected period",
            ytitle="DKK/kWh",
        )
        st.plotly_chart(fig_price, use_container_width=True)
        if note_price:
            st.caption(f"ℹ️ {note_price}")

        # CO₂ (bar version)
        fig_co2 = plot_period_bar(
            co2_series,
            selected_day=selected_day,
            title="CO₂ intensity over selected period",
            ytitle="gCO₂/kWh",
        )
        st.plotly_chart(fig_co2, use_container_width=True)
        if note_co2:
            st.caption(f"ℹ️ {note_co2}")

        # Temperature plot
        fig_temp = plot_period_minute(
            temp_series,
            selected_day=selected_day,
            title="Outdoor temperature over selected period",
            ytitle="°C",
        )
        st.plotly_chart(fig_temp, use_container_width=True)
        if note_temp:
            st.caption(f"ℹ️ {note_temp}")



#%% page 2
elif page.startswith("2️⃣"):
    st.title("2️⃣ Devices & layout")
    st.info("Here we will later add: baseload, lights, WM/DW/dryer, EV, heat pump, PV, battery, house figure, etc.")
    
    st.markdown("## 2. Devices & House layout")
    render_devices_page_house()

    




#%% page 3
elif page.startswith("3️⃣"):
    st.title("3️⃣ Analysis")
    st.info("Here we will later add: Run simulation, Run EMS, plots, KPIs, and logging.")
    # --- capacity / power / SOC settings ---
    def render_battery_settings_panel():
        """
        Returns control_mode string ("Manual" or "Auto") for convenience.
        Also keeps st.session_state['battery_cfg'] + legacy batt_* keys updated.
        """
        cfg = st.session_state.setdefault("battery_cfg", {
            "capacity_kwh": 70.0,
            "power_kw": 9.0,
            "soc_init_pct": 50.0,
            "soc_min_pct": 15.0,
            "control_mode": "Auto",
            "manual_slots": [],
            "rule_priority": 2,
        })

        st.markdown("### 🔋 Battery settings")

        # --- capacity / power / SOC settings ---
        c1, c2 = st.columns(2)
        cfg["capacity_kwh"] = c1.number_input(
            "Battery capacity (kWh)",
            min_value=0.1, max_value=500.0, step=0.5,
            value=float(cfg.get("capacity_kwh", 75.0)),
            key="batt_cap_kwh_input",
        )
        cfg["power_kw"] = c2.number_input(
            "Battery max charge/discharge power (kW)",
            min_value=0.0, max_value=200.0, step=0.5,
            value=float(cfg.get("power_kw", 10.0)),
            key="batt_power_kw_input",
        )

        c3, c4 = st.columns(2)
        cfg["soc_init_pct"] = c3.number_input(
            "Initial SOC (%)",
            min_value=0.0, max_value=100.0, step=1.0,
            value=float(cfg.get("soc_init_pct", 50.0)),
            key="batt_soc_init_input",
        )
        cfg["soc_min_pct"] = c4.number_input(
            "Minimum SOC (%)",
            min_value=0.0, max_value=100.0, step=1.0,
            value=float(cfg.get("soc_min_pct", 15.0)),
            key="batt_soc_min_input",
        )

        # --- Battery Control Mode ---
        st.markdown("**Control mode**")
        mode = st.radio(
            "Battery scheduling method",
            ["Manual plan (enter 6 slots)", "Auto (Use EMS logic)"],
            index=1 if cfg.get("control_mode", "Auto") == "Auto" else 0,
            key="batt_mode_radio",
        )

        if mode.startswith("Manual"):
            cfg["control_mode"] = "Manual"

            # --------- DEFAULT 6 SLOTS (only if nothing defined yet) ---------
            if not cfg.get("manual_slots"):
                default_starts = [
                    _time(1, 0),   # 01:00
                    _time(6, 0),   # 06:00
                    _time(10, 0),  # 10:00
                    _time(14, 0),  # 14:00
                    _time(18, 0),  # 18:00
                    _time(20, 0),  # 20:00
                ]
                default_ends = [
                    _time(6, 0),    # 06:00
                    _time(10, 0),   # 10:00
                    _time(14, 0),   # 14:00
                    _time(18, 0),   # 18:00
                    _time(20, 0),   # 20:00
                    _time(23, 59),  # 23:59 ≈ 24:00
                ]
                default_socs = [15.0, 20.0, 60.0, 80.0, 100.0, 15.0]

                cfg["manual_slots"] = [
                    dict(start=default_starts[i],
                        end=default_ends[i],
                        soc=default_socs[i],
                        grid=0)
                    for i in range(6)
                ]

            st.markdown("#### Manual 6-slot plan")
            st.caption("Enter start/end (HH:MM), SOC target (0–100%), and whether grid charging is allowed.")

            new_slots = []
            prev = cfg.get("manual_slots", [{}] * 6)

            for i in range(6):
                c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
                st.markdown(f"**Slot {i+1}**")

                default_s = prev[i].get("start", _time(0, 0))
                default_e = prev[i].get("end",   _time(4, 0))
                default_soc = prev[i].get("soc", 50.0)
                default_gc  = prev[i].get("grid", 0)

                st_s = c1.time_input(
                    f"Start {i+1}", value=default_s,
                    key=f"batt_s{i}_start",
                )
                st_e = c2.time_input(
                    f"End {i+1}", value=default_e,
                    key=f"batt_s{i}_end",
                )
                st_soc = c3.number_input(
                    f"SOC% {i+1}",
                    min_value=0.0, max_value=100.0, step=1.0,
                    value=default_soc,
                    key=f"batt_s{i}_soc",
                )
                st_gc = c4.selectbox(
                    f"GridCharge {i+1}",
                    options=[0, 1],
                    index=int(default_gc),
                    key=f"batt_s{i}_gc",
                )

                new_slots.append(dict(start=st_s, end=st_e, soc=st_soc, grid=st_gc))

            cfg["manual_slots"] = new_slots

            # keep the EMS-facing format unchanged
            st.session_state["manual_plan_rows"] = [
                {
                    "start": s["start"],
                    "end": s["end"],
                    "soc_setpoint_pct": s["soc"],
                    "grid_charge_allowed": s["grid"],
                }
                for s in new_slots
            ]

        else:
            cfg["control_mode"] = "Auto"

        # --- Priority setting ---
        cfg["rule_priority"] = st.selectbox(
            "Rule priority",
            options=[2, 1],
            index=0 if cfg.get("rule_priority", 2) == 2 else 1,
            format_func=lambda x: "Load first (2)" if x == 2 else "Battery first (1)",
            key="batt_priority_select",
        )

        # --- Mirror to legacy single variables used by EMS ---
        st.session_state["batt_cap"]      = cfg["capacity_kwh"]
        st.session_state["batt_pow"]      = cfg["power_kw"]
        st.session_state["batt_soc_pct"]  = cfg["soc_init_pct"]
        st.session_state["soc_min_pct"]   = cfg["soc_min_pct"]
        st.session_state["batt_mode"]     = cfg["control_mode"]
        st.session_state["batt_priority"] = cfg["rule_priority"]

        return cfg["control_mode"]
 
    if "load" not in st.session_state:
        st.warning("No load profile found. Please configure devices on Page 2 first.")
        st.stop()
    if "load" not in st.session_state:
        st.warning("No PV installed")
        
    load_day= st.session_state["load"]
    pv_day   = st.session_state.get("pv")  # may be None

    price_all = st.session_state.get("price_daily")
    co2_all   = st.session_state.get("co2_daily")
    price_day=get_selected_day_data(price_all)
    co2_day=get_selected_day_data(co2_all)
    idx = load_day.index          # your minute index






