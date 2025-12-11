#%%
from __future__ import annotations
import streamlit as st
from datetime import datetime, date, time
import pandas as pd  
import numpy as np
from pathlib import Path

from datetime import date, datetime, timedelta
import json
import requests
import plotly.graph_objects as go
import numpy as np
from datetime import time as _time, date as _date
from profiles import minute_index, default_price_profile, default_co2_profile, simple_pv_profile, synthetic_outdoor_temp
from devices import  WeatherHP,WeatherELheater,WeatherHotTub, DHWTank
import pvlib
from pvlib.location import Location
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from scipy.optimize import minimize
from ems import rule_power_share
from Optimization_based import generate_smart_time_slots, assign_data_to_time_slots_single, mpc_opt_single, mpc_opt_multi, format_results_single
#%% front page

LOG_PATH = Path("usage_log.csv")


def log_user_profile_to_csv(profile: dict):
    """Append one row to a local CSV file."""
    is_new = not LOG_PATH.exists()
    df_row = pd.DataFrame(
        [{
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "occupation": profile.get("occupation", ""),
            "location": profile.get("location", ""),
            "session_id": st.session_state.get("session_id", ""),
        }]
    )
    if is_new:
        df_row.to_csv(LOG_PATH, index=False, mode="w")
    else:
        df_row.to_csv(LOG_PATH, index=False, mode="a", header=False)


def ensure_user_profile():
    """
    Hard gate: show a small intro form until user has provided
    occupation + living location.
    Call this ONCE at the start of the app.
    """
    # cheap session id
    st.session_state.setdefault("session_id", datetime.now().strftime("%Y%m%d-%H%M%S"))

    if st.session_state.get("user_profile_confirmed"):
        return  # already done

    st.title("Daily EMS Sandbox")
    st.subheader("Before we start, tell us a bit about yourself 👇")

    occupation = st.radio(
        "Your current role",
        [
            "Bachelor student",
            "Master student",
            "PhD student",
            "Research assistant",
            "Postdoc",
            "Assistant Professor",
            "Associate Professor",
            "Professor",
            "Industry",
            "Others",
        ],
        index=None,
        help="We only use this for anonymous statistics about who is using the tool.",
    )
    

    location = st.text_input(
        "Where do you live?",
        placeholder="City, Country (e.g. Aalborg, Denmark)",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        ready = bool(occupation) and bool(location.strip())
        clicked = st.button("Start using the app ▶️", disabled=not ready)

    if clicked and ready:
        profile = {
            "occupation": occupation,
            "location": location.strip(),
        }
        st.session_state["user_profile"] = profile
        st.session_state["user_profile_confirmed"] = True

        # log to CSV (local; you can later replace this with DB / Google Sheet, etc.)
        log_user_profile_to_csv(profile)

        st.rerun()

    # HARD STOP: do not render rest of the app yet
    st.stop()
ensure_user_profile()
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
from datetime import datetime, date as _date

def get_thermal_building_params():
    """
    Map global house_info (size, insulation) to:
      - ua_base  [kW/°C]
      - q_guess  [kW]
      - Cth_guess [kWh/°C]
    """
    hi = st.session_state.get("house_info", {
        "size": "Medium house",
        "insulation": "Average",
        "residents": 2,
    })
    size = hi.get("size", "Medium house")
    ins  = hi.get("insulation", "Average")

    # base UA & capacity guess by size
    if size == "Small apartment":
        ua_base = 0.10
        q_guess = 4.0
        Cth_guess = 0.50
    elif size == "Large house":
        ua_base = 0.14
        q_guess = 8.0
        Cth_guess = 0.75
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

    return ua, q_guess, Cth_guess, size, ins
def get_outdoor_minute_profile():
    """
    Return (idx_minute, tout_minute Series) for the selected day.
    Uses temp_daily from session if available; otherwise synthetic profile.
    """
    if (
        "temp_daily" in st.session_state
        and isinstance(st.session_state["temp_daily"], pd.Series)
        and not st.session_state["temp_daily"].empty
    ):
        tout_tot = st.session_state["temp_daily"]
        # you already have this helper elsewhere:
        tout_minute = get_selected_day_data(tout_tot)
        idx = tout_minute.index
        st.caption("Using fetched outdoor temperature for this preview.")
        return idx, tout_minute.astype(float)

    # fallback: synthetic daily sinusoid
    idx = pd.date_range("2025-01-10 00:00", periods=24 * 60, freq="min")
    hours = idx.hour + idx.minute / 60.0
    tout_minute = pd.Series(
        5.0 + 5.0 * np.sin(2 * np.pi * (hours - 15) / 24.0),
        index=idx,
        name="Tout_C",
    )
    st.caption(
        "No weather data found, using a synthetic outdoor temperature profile."
    )
    return idx, tout_minute.astype(float)

def suggest_best_interval_for_day(
    duration_min: int,
    w_cost: float = 0.5,
    earliest: time | None = None,
    latest:  time | None = None,
) -> dict | None:
    """
    Returns {"start": time, "end": time} for the chosen day,
    restricted to [earliest, latest] if provided.
    """
    price = st.session_state.get("price_daily")
    co2   = st.session_state.get("co2_daily")

    sel_day = st.session_state.get("day")

    if not isinstance(sel_day, _date):
        pr = st.session_state.get("period_range")
        if pr and len(pr) == 2 and isinstance(pr[1], _date):
            sel_day = pr[1]

    if price is None or co2 is None or len(price) == 0 or not isinstance(sel_day, _date):
        return None

    df = pd.DataFrame(index=price.index.copy())
    df["price"] = np.asarray(price, dtype=float)
    df["co2"]   = co2.reindex(df.index).interpolate().bfill().ffill()

    day_start = pd.Timestamp(sel_day)
    day_end   = day_start + pd.Timedelta(days=1)
    df = df.loc[(df.index >= day_start) & (df.index < day_end)]
    if df.empty:
        return None

    # apply allowed window
    if earliest is not None and latest is not None:
        e_min = earliest.hour * 60 + earliest.minute
        l_min = latest.hour * 60 + latest.minute
        minutes_of_day = df.index.hour * 60 + df.index.minute
        if e_min <= l_min:
            mask = (minutes_of_day >= e_min) & (minutes_of_day <= l_min)
        else:
            # window wraps midnight
            mask = (minutes_of_day >= e_min) | (minutes_of_day <= l_min)
        df = df.loc[mask]
        if df.empty:
            return None

    # normalize
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
    for t0 in range(0, n - dur + 1):
        sc = float(df["score"].iloc[t0:t0 + dur].mean())
        if best_score is None or sc < best_score:
            best_score = sc
            best_t0 = t0

    if best_t0 is None:
        return None

    t_start = df.index[best_t0]
    t_end   = df.index[best_t0 + dur - 1] + pd.Timedelta(minutes=1)

    start_min = t_start.hour * 60 + t_start.minute
    end_min   = t_end.hour * 60 + t_end.minute
    start_min = max(0, min(start_min, 24 * 60 - 1))
    end_min   = max(1, min(end_min,   24 * 60 - 1))

    return {
        "start": _time(start_min // 60, start_min % 60),
        "end":   _time(end_min   // 60, end_min   % 60),
    }


def extract_icon(label: str) -> str:
    """Return only the emoji part of a label."""
    if " " in label:
        return label.split(" ", 1)[0]
    return label  # fallback

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

    

# -------------------------------------------------------------------
# Helper: default config per device type
# -------------------------------------------------------------------
def get_default_config(dev_type: str, category: str) -> dict:
    """Return default config (power, schedule, etc.) for a device."""
    base = dict(
        power_kw=0.5,
        start=_time(18, 0),
        duration_min=60,
    )

    # ====== read house_info and map to small/medium/large ==========
    hi = st.session_state.get("house_info", {
        "size": "Medium house",
        "insulation": "Average",
        "residents": 2,
    })
    size_str = (hi.get("size") or "Medium house").lower()

    if "small" in size_str:
        size = "small"
    elif "large" in size_str:
        size = "large"
    else:
        size = "medium"

    def _by_size(small: int, medium: int, large: int) -> int:
        if size == "small":
            return small
        if size == "large":
            return large
        return medium

    # ======================================================
    # 1) FIXED ELECTRICAL (all 17 + other slots)
    # ======================================================
    if category == "elec_fixed":
        # All powers are per device (W).
        # num_devices scales with house size where it makes sense.
        defaults: dict[str, dict] = {
            "lights": {
                "num_devices": _by_size(8, 12, 18),
                "power_w": 8.0,  # per LED fixture
                "intervals": [
                    {"start": _time(6, 0),  "end": _time(8, 0)},   # morning
                    {"start": _time(17, 0), "end": _time(23, 0)},  # evening
                ],
            },
            "fridge": {
                "num_devices": 1,
                "power_w": 80.0,  # average over compressor cycling
                "intervals": [
                    {"start": _time(0, 0), "end": _time(23, 59)},
                ],
            },
            "freezer": {
                "num_devices": _by_size(1, 1, 2),
                "power_w": 90.0,
                "intervals": [
                    {"start": _time(0, 0), "end": _time(23, 59)},
                ],
            },
            "fridge_freezer": {
                "num_devices": _by_size(1, 1, 2),
                "power_w": 110.0,
                "intervals": [
                    {"start": _time(0, 0), "end": _time(23, 59)},
                ],
            },
            "range_hood": {
                "num_devices": 1,
                "power_w": 120.0,
                "intervals": [
                    {"start": _time(18, 0), "end": _time(19, 0)},  # cooking
                ],
            },
            "oven": {
                "num_devices": _by_size(1, 1, 2),
                "power_w": 2000.0,
                "intervals": [
                    {"start": _time(18, 0), "end": _time(19, 0)},
                ],
            },
            "induction": {
                "num_devices": 1,
                "power_w": 2500.0,
                "intervals": [
                    {"start": _time(17, 30), "end": _time(19, 0)},
                ],
            },
            "microwave": {
                "num_devices": 1,
                "power_w": 1200.0,
                "intervals": [
                    {"start": _time(7,  0), "end": _time(7, 15)},
                    {"start": _time(12, 0), "end": _time(12, 15)},
                    {"start": _time(21, 0), "end": _time(21, 15)},
                ],
            },
            "tv": {
                "num_devices": _by_size(1, 1, 2),
                "power_w": 80.0,
                "intervals": [
                    {"start": _time(19, 0), "end": _time(23, 0)},
                ],
            },
            "router": {
                "num_devices": _by_size(1, 2, 3),
                "power_w": 10.0,
                "intervals": [
                    {"start": _time(0, 0), "end": _time(23, 59)},
                ],
            },
            "pc_desktop": {
                "num_devices": _by_size(1, 1, 2),
                "power_w": 150.0,
                "intervals": [
                    {"start": _time(9, 0), "end": _time(17, 0)},   # work-from-home
                ],
            },
            "laptop": {
                "num_devices": _by_size(1, 2, 3),
                "power_w": 60.0,
                "intervals": [
                    {"start": _time(9, 0),  "end": _time(12, 0)},
                    {"start": _time(19, 0), "end": _time(23, 0)},
                ],
            },
            "game_console": {
                "num_devices": _by_size(1, 1, 2),
                "power_w": 120.0,
                "intervals": [
                    {"start": _time(20, 0), "end": _time(22, 0)},
                ],
            },
            "printer": {
                "num_devices": _by_size(1, 1, 2),
                "power_w": 40.0,
                "intervals": [
                    {"start": _time(10, 0), "end": _time(12, 0)},  # sporadic use
                ],
            },
            "ventilation": {
                "num_devices": 1,
                "power_w": 60.0,   # HRV unit
                "intervals": [
                    {"start": _time(0, 0), "end": _time(23, 59)},
                ],
            },
            "humidifier": {
                "num_devices": _by_size(1, 1, 2),
                "power_w": 40.0,
                "intervals": [
                    {"start": _time(22, 0), "end": _time(7, 0)},  # night
                ],
            },
            "baby_monitor": {
                "num_devices": _by_size(1, 1, 1),
                "power_w": 5.0,
                "intervals": [
                    {"start": _time(19, 0), "end": _time(7, 0)},
                ],
            },
            "smoke_detector": {
                "num_devices": _by_size(2, 3, 4),
                "power_w": 2.0,
                "intervals": [
                    {"start": _time(0, 0), "end": _time(23, 59)},
                ],
            },
            "standby": {
                "num_devices": 1,
                "power_w": _by_size(30.0, 50.0, 80.0),  # sum of small phantom loads
                "intervals": [
                    {"start": _time(0, 0), "end": _time(23, 59)},
                ],
            },

            # You can also initialize the 3 "other" slots
            "other_fixed_1": {
                "num_devices": 1,
                "power_w": 100.0,
                "intervals": [
                    {"start": _time(18, 0), "end": _time(22, 0)},
                ],
            },
            "other_fixed_2": {
                "num_devices": 1,
                "power_w": 100.0,
                "intervals": [
                    {"start": _time(18, 0), "end": _time(22, 0)},
                ],
            },
            "other_fixed_3": {
                "num_devices": 1,
                "power_w": 100.0,
                "intervals": [
                    {"start": _time(18, 0), "end": _time(22, 0)},
                ],
            },
        }

        cfg = defaults.get(dev_type, {}).copy()
        if not cfg:
            # unknown device → just base
            return base

        # derive power_kw / start / duration_min for compatibility
        first_iv = cfg["intervals"][0]
        start_t = first_iv["start"]
        end_t   = first_iv["end"]

        # assume same day; if end < start we treat it as overnight (add 24h)
        start_dt = datetime.combine(date.today(), start_t)
        end_dt   = datetime.combine(date.today(), end_t)
        if end_dt <= start_dt:
            end_dt = end_dt.replace(day=end_dt.day + 1)

        dur_min = int((end_dt - start_dt).total_seconds() / 60.0)
        if dur_min <= 0:
            dur_min = 60

        cfg["power_kw"] = cfg["power_w"] / 1000.0
        cfg["start"] = start_t
        cfg["duration_min"] = dur_min

        # merge with base so we still have generic keys
        return {**base, **cfg}
    
    # ======================================================
    # 2) FLEXIBLE ELECTRICAL (shiftable loads)
    # ======================================================
    if category == "elec_flex":
        # small helper already defined above in your function:
        #   size = "small"/"medium"/"large"
        #   def _by_size(small, medium, large): ...

        defaults: dict[str, dict] = {
            # Washing machine
            "wm": {
                "num_devices": 1,
                "power_w": 1200.0,
                "start": _time(19, 0),      # typical evening wash
                "duration_min": 90,
                "w_cost": 1,
            },
            # Dishwasher
            "dw": {
                "num_devices": 1,
                "power_w": 1400.0,
                "start": _time(21, 0),      # after dinner
                "duration_min": 90,
                "w_cost": 1,
            },
            # Tumble dryer (resistive)
            "dryer": {
                "num_devices": 1,
                "power_w": 2000.0,
                "start": _time(20, 0),
                "duration_min": 60,
                "w_cost": 1,
            },
            # Robot vacuum
            "robot_vac": {
                "num_devices": 1,
                "power_w": 250.0,
                "start": _time(11, 0),      # mid-day cleaning
                "duration_min": 60,
                "w_cost": 1,
            },
            # Workshop / hobby tools
            "workshop": {
                "num_devices": 1,
                "power_w": 700.0,
                "start": _time(17, 0),
                "duration_min": 120,
                "w_cost": 1,
            },

            # you can add more flexible types later, just extend DEVICE_CATEGORIES
            # and add entries here with the same pattern.

            # custom slots – just give a neutral default
            "other_flex_1": {
                "num_devices": 1,
                "power_w": 1000.0,
                "start": _time(18, 0),
                "duration_min": 60,
                "w_cost": 1,
            },
            "other_flex_2": {
                "num_devices": 1,
                "power_w": 1000.0,
                "start": _time(18, 0),
                "duration_min": 60,
                "w_cost": 1,
            },
            "other_flex_3": {
                "num_devices": 1,
                "power_w": 1000.0,
                "start": _time(18, 0),
                "duration_min": 60,
                "w_cost": 1,
            },
        }

        cfg = defaults.get(dev_type, {}).copy()
        if not cfg:
            # unknown flexible type → fall back to generic
            return base

        # derive an initial single interval from start + duration_min
        start_t = cfg.get("start", _time(20, 0))
        dur_min = int(cfg.get("duration_min", 60))
        if dur_min <= 0:
            dur_min = 60

        start_dt = datetime.combine(date.today(), start_t)
        end_dt   = start_dt + timedelta(minutes=dur_min)
        # clamp to time of day (ignore day overflow)
        end_t = (end_dt.time().replace(second=0, microsecond=0))

        cfg["intervals"] = [{"start": start_t, "end": end_t}]
        cfg["power_kw"]  = float(cfg.get("power_w", base["power_kw"] * 1000.0)) / 1000.0
        cfg["start"]     = start_t
        cfg["duration_min"] = dur_min

        # keep w_cost if present, default 0.5
        cfg["w_cost"] = float(cfg.get("w_cost", 1))

        return {**base, **cfg}


    # ======================================================
    # 2) FLEXIBLE / THERMAL / GEN / OUTSIDE  (unchanged)
    # ======================================================

    if category == "thermal":
        defaults = {
            # Space heating: external supply by default → no P_el
            "space_heat": {
                "space_mode": "None (external supply)",
                "t_min_c": 20.0,
                "t_max_c": 22.0,
                # these are only used if user changes away from "None"
                "q_kw": 6.0,
            },
            # DHW: external supply by default → no P_el
            "dhw": {
                "dhw_mode": "None (external supply)",
                "volume_l": 200.0,
                "usage_level": "Medium",
                "t_min_c": 45.0,
                "t_max_c": 55.0,
                "p_el_kw": 2.0,
            },
            # Leisure: all disabled by default → no P_el
            "leisure": {
                "hot_tub_enabled": False,
                "pool_enabled": False,
            },
        }

        cfg = defaults.get(dev_type, {}).copy()
        if not cfg:
            # unknown thermal device: just inherit base but no real load
            cfg = {}

        # keep generic keys so other code that expects them doesn’t crash
        cfg.setdefault("power_kw", base["power_kw"])
        cfg.setdefault("start", base["start"])
        cfg.setdefault("duration_min", base["duration_min"])

        return {**base, **cfg}

    if category == "gen_store":
        # For now we only want PV to matter; others should default to 0 kW.
        # Also override the base so gen_store things don't inherit 0.5 kW.
        base_gen = dict(
            power_kw=0.0,
            start=_time(0, 0),
            duration_min=0,
        )

        defaults = {
            # PV: we don’t actually use power_kw/start/duration for PV,
            # but we can store some sizing-related defaults here if you like.
            "pv": {
                "module_wp": 400.0,
                "n_panels": 16,
                "tilt": 30.0,
                "azimuth": 180.0,
                "loss_frac": 0.14,
                # keep power_kw etc. at 0 so it never shows as a “load”
                "power_kw": 0.0,
                "start": _time(0, 0),
                "duration_min": 0,
            },

            # everything else → no default power / duration
            "battery":      {"power_kw": 0.0, "start": _time(0, 0), "duration_min": 0},
            "fuel_cell":    {"power_kw": 0.0, "start": _time(0, 0), "duration_min": 0},
            "diesel_gen":   {"power_kw": 0.0, "start": _time(0, 0), "duration_min": 0},
            "electrolyzer": {"power_kw": 0.0, "start": _time(0, 0), "duration_min": 0},
        }

        return {**base_gen, **defaults.get(dev_type, {})}

    if category == "outside":
        defaults = {
            "ev11":         dict(power_kw=11.0, start=_time(1, 0), duration_min=240),
            "ev22":         dict(power_kw=22.0, start=_time(1, 0), duration_min=120),
            "ebike":        dict(power_kw=0.5,  start=_time(1, 0), duration_min=180),
            "outdoor_light":dict(power_kw=0.2,  start=_time(17, 0),duration_min=600),
            "patio_heater": dict(power_kw=2.0,  start=_time(18, 0),duration_min=240),
        }
        return {**base, **defaults.get(dev_type, {})}

    return base   

# -------------------------------------------------------------------
# Device catalogue (labels + icons)
# -------------------------------------------------------------------
DEVICE_CATEGORIES = {
    "elec_fixed": {
        "title": "1a. Household electrical – fixed",
        "help": "Devices that are hard to shift in time.",
        "devices": [
            ("lights",        "💡 Lights"),
            ("fridge",        "🧊 Refrigerator"),
            ("range_hood",    "🍳 Range hood"),
            ("oven",          "🔥 Oven"),
            ("induction",     "🍳 Induction stove"),
            ("microwave",     "🎛️ Microwave"),
            ("tv",            "📺 TV"),
            ("router",        "🛜 Router"),
            ("pc_desktop",    "🖥️ Desktop PC"),
            ("laptop",        "💻 Laptop charger"),
            ("game_console",  "🎮 Game console"),
            ("printer",       "🖨️ Printer"),
            ("ventilation",   "𖣘 Ventilation / HRV"),
            ("humidifier",    "💧 Humidifier"),
            ("baby_monitor",  "🚼 Baby monitor"),
            ("smoke_detector","🚨 Smoke detector"),
            ("standby",       "🔌 Standby loads"),

            # --- 3 custom “other fixed” slots ---
            ("other_fixed_1", "🧩 Other #1"),
            ("other_fixed_2", "🧩 Other #2"),
            ("other_fixed_3", "🧩 Other #3"),
        ],
    },
    "elec_flex": {
        "title": "1b. Household electrical – flexible",
        "help": "Shiftable devices that can be move to cheaper/cleaner hours.",
        "devices": [
            ("wm",           "🧺 Washing machine"),
            ("dw",           "🍽 Dishwasher"),
            ("dryer",        "👕 Dryer"),
            ("robot_vac",    "🧹 Robot vacuum"),
            ("workshop",     "🔧 Workshop tools"),

            # --- 3 custom “other flexible” slots ---
            ("other_flex_1", "🧩 Other #1"),
            ("other_flex_2", "🧩 Other #2"),
            ("other_flex_3", "🧩 Other #3"),
        ],
    },
    "thermal": {
        "title": "2. Household thermal",
        "help": "Space heating, domestic hot water and leisure thermal loads.",
        "devices": [
            ("space_heat", "🔥 Space heating"),
            ("dhw",        "💧 DHW system"),
            ("leisure",    "🧖 Leisure thermal loads"),
            ],
        },
    "outside": {
        "title": "3. Electrical Vehicles",
        "help": "Electric vehicles.",
        "devices": [
            ("ev11",        "🚗 EV charger"),
            ("ebike",       "🚲 E-bike charger"),
        ],
    },
    "gen_store": {
        "title": "4. Generation & storage",
        "help": "PV, batteries and other on-site generation/storage units. (Currently only PV is available. Other models are under development)",
        "devices": [
            ("pv",          "☀️ PV system"),
        ],
    },
    
}
from datetime import datetime, date, timedelta

# optional: build a lookup from full_key -> pretty label (for legend)
DEVICE_LABEL_MAP = {
    f"{cat}:{dev}": label
    for cat, info in DEVICE_CATEGORIES.items()
    for dev, label in info["devices"]
}
def resolve_display_label(full_key: str, dev_type: str, cfg_current: dict) -> str:
    """
    Reuse the same logic as device checkboxes:
    - Use catalogue emoji + text
    - For 'other_*' devices, replace text with custom_name if set.
    """
    base_label = DEVICE_LABEL_MAP.get(full_key, dev_type)

    if dev_type.startswith("other"):
        custom_name = cfg_current.get("custom_name")
        if custom_name:
            # keep emoji from base_label if present
            if " " in base_label:
                emoji = base_label.split(" ", 1)[0]
            else:
                emoji = "🧩"
            return f"{emoji} {custom_name}"

    return base_label




def compute_daily_profiles(sel: dict, cfgs: dict):
    """
    Build per-device and total daily profiles [kW] for all selected devices.
    All profiles are returned on a 1-minute grid for a dummy day.
    """

    device_traces: dict[str, pd.Series] = {}

    # Common dummy day + 1-min index for the final plot
    dummy_day = date(2025, 1, 10)
    start_dt  = datetime.combine(dummy_day, time(0, 0))
    idx_common = pd.date_range(start=start_dt, periods=24 * 60, freq="min")

    total = pd.Series(0.0, index=idx_common, name="P_total_kW")

    def _map_to_dummy(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Map any datetime index to dummy_day, keeping only time-of-day."""
        idx = pd.to_datetime(idx)
        return pd.DatetimeIndex(
            [datetime.combine(dummy_day, ts.time()) for ts in idx],
            name="time",
        )

    for full_key, checked in sel.items():
        if not checked:
            continue

        cfg = cfgs.get(full_key, {}) or {}
        cat_key, dev_type = full_key.split(":", 1)

        prof = None

        # ----------------------------------------------------------
        # 1) Try to use stored profile_index / profile_kw (1-min)
        # ----------------------------------------------------------
        idx_list = cfg.get("profile_index")
        kw_list  = cfg.get("profile_kw")

        if (
            isinstance(idx_list, list)
            and isinstance(kw_list, list)
            and len(idx_list) == len(kw_list)
            and len(idx_list) > 0
        ):
            try:
                idx_local = pd.to_datetime(idx_list)
                vals = np.asarray(kw_list, dtype=float)

                idx_norm = _map_to_dummy(idx_local)
                prof = pd.Series(vals, index=idx_norm, name=full_key)

                # make sure we have exactly the full day, fill gaps with 0
                prof = prof.reindex(idx_common, fill_value=0.0)
            except Exception:
                prof = None

        # ----------------------------------------------------------
        # 2) Fallback if we don't have a saved profile
        # ----------------------------------------------------------
        if prof is None:
            if cat_key == "thermal":
                # thermal devices → 0 kW until user opens settings and we save a profile
                prof = pd.Series(0.0, index=idx_common, name=full_key)

            else:
                # generic fallback for electrical (fixed/flex/etc.)
                num = int(cfg.get("num_devices", 1))

                intervals = cfg.get("intervals")
                if not intervals:
                    start_t = cfg.get("start", time(18, 0))
                    dur_min = int(cfg.get("duration_min", 60))
                    start_local = datetime.combine(dummy_day, start_t)
                    end_local   = start_local + timedelta(minutes=dur_min)
                    intervals = [{"start": start_t, "end": end_local.time()}]

                if cat_key in ("elec_fixed", "elec_flex"):
                    power_w_per_dev = float(
                        cfg.get("power_w", cfg.get("power_kw", 0.5) * 1000.0)
                    )
                    power_w = power_w_per_dev * num
                else:
                    # other categories that still use power_kw
                    power_w = float(cfg.get("power_kw", 0.5) * 1000.0) * num

                prof_raw = build_minute_profile(
                    power_w=power_w,
                    intervals=intervals,
                    step_min=1,   # always 1-min internally
                )

                # normalize index to dummy day
                if isinstance(prof_raw.index, pd.DatetimeIndex):
                    prof_raw.index = _map_to_dummy(prof_raw.index)
                else:
                    prof_raw.index = idx_common

                prof = prof_raw.reindex(idx_common, fill_value=0.0)
                prof.name = full_key

        device_traces[full_key] = prof
        if not (cat_key == "gen_store" and dev_type == "pv"):
            total = total.add(prof, fill_value=0.0)

    if not device_traces:
        total = pd.Series(0.0, index=idx_common, name="P_total_kW")

    return idx_common, device_traces, total

def build_series_for_analysis(sel: dict, cfgs: dict):
    """
    Use compute_daily_profiles(...) output and split into:
      - total electrical load (all non-PV devices)
      - total PV generation
      - per-device daily energy (kWh) for loads
    Returns:
      idx, load_tot, pv_tot, energy_per_device
    """
    idx, device_traces, _total = compute_daily_profiles(sel, cfgs)

    load_tot = pd.Series(0.0, index=idx, name="P_load_kW")
    pv_tot   = pd.Series(0.0, index=idx, name="P_pv_kW")
    energy_per_device: dict[str, float] = {}

    for full_key, s in device_traces.items():
        if s is None or s.empty:
            continue

        cat_key, dev_type = full_key.split(":", 1)

        if cat_key == "gen_store" and dev_type == "pv":
            # generation
            pv_tot = pv_tot.add(s, fill_value=0.0)
        else:
            # everything else is electrical consumption
            load_tot = load_tot.add(s, fill_value=0.0)
            # store kWh (1-min resolution → divide by 60)
            energy_per_device[full_key] = float(s.sum() / 60.0)

    return idx, load_tot, pv_tot, energy_per_device

def build_house_layout_figure(sel: dict, cfgs: dict) -> go.Figure:
    """
    Build house layout figure where each selected device appears as an
    icon+name inside its zone.
    """
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

    # EV / outside
    shapes.append(
        dict(
            type="rect",
            x0=9.2, y0=1.95, x1=10.1, y1=3.02,
            line=dict(width=1.5),
            fillcolor="rgba(250,250,250,0.9)",
        )
    )

    fig.update_layout(shapes=shapes)

    # ---- base annotations (zone labels) ----
    annotations = [
        dict(x=2.0, y=4.3, text="Electrical (fixed)",    showarrow=False, font=dict(size=10)),
        dict(x=6.2, y=4.3, text="Electrical (flexible)", showarrow=False, font=dict(size=10)),
        dict(x=1.8, y=2.2, text="Thermal",               showarrow=False, font=dict(size=10)),
        dict(x=5.0, y=7.2, text="Generation & Storage",  showarrow=False, font=dict(size=11)),
    ]

    # ---- collect selected devices per zone ----
    zone_devices = {
        "elec_fixed": [],
        "elec_flex":  [],
        "thermal":    [],
        "gen_store":  [],
        "outside":    [],
    }

    for full_key, checked in sel.items():
        if not checked:
            continue
        if ":" not in full_key:
            continue
        cat_key, dev_type = full_key.split(":", 1)
        if cat_key not in zone_devices:
            continue

        cfg_current = cfgs.get(full_key, {})
        label = resolve_display_label(full_key, dev_type, cfg_current)
        zone_devices[cat_key].append(label)

    # ---- helper: grid layout for zones ----
    def add_zone_devices_grid(
        labels,
        base_x,
        base_y,
        max_cols,
        max_rows,
        dx,
        dy,
    ):
        capacity = max_cols * max_rows
        for j, label in enumerate(labels[:capacity]):
            col = j % max_cols
            row = j // max_cols
            x = base_x + col * dx
            y = base_y - row * dy
            annotations.append(
                dict(
                    x=x,
                    y=y,
                    text=extract_icon(label),
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

    # 1) Electrical fixed – 4 x 4 grid
    add_zone_devices_grid(
        zone_devices["elec_fixed"],
        base_x=1.6,
        base_y=6.5,
        max_cols=5,
        max_rows=4,
        dx=0.68,
        dy=0.6,
    )

    # 2) Electrical flexible – 4 x 4 grid
    add_zone_devices_grid(
        zone_devices["elec_flex"],
        base_x=5.65,
        base_y=6.5,
        max_cols=5,
        max_rows=4,
        dx=0.68,
        dy=0.6,
    )

    # 3) Thermal – 4 x 4 grid
    add_zone_devices_grid(
        zone_devices["thermal"],
        base_x=3.0,
        base_y=3.7,
        max_cols=4,
        max_rows=4,
        dx=1.3,
        dy=0.45,
    )

    # 4) Generation & storage – triangular layout (same as old code)
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

    for j, label in enumerate(zone_devices["gen_store"][: len(gen_slots)]):
        x, y = gen_slots[j]
        annotations.append(
            dict(
                x=x,
                y=y,
                text=extract_icon(label),
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

    # 5) EV / outside – up to 2 stacked
    ev_slots = [
        (9.65, 2.7),
        (9.65, 2.2),
    ]
    for j, label in enumerate(zone_devices["outside"][: len(ev_slots)]):
        x, y = ev_slots[j]
        annotations.append(
            dict(
                x=x,
                y=y,
                text=extract_icon(label),
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
    return fig

def get_house_thermal_params():
    """Derive UA, C_th and default comfort band from house_info."""
    hi = st.session_state.get(
        "house_info",
        {"size": "Medium house", "insulation": "Average", "residents": 2},
    )
    size_str = (hi.get("size") or "Medium house").lower()
    ins_str  = hi.get("insulation", "Average")

    # base UA + Cth by size
    if "small" in size_str:
        ua_base  = 0.10   # kW/°C
        Cth_base = 0.50   # kWh/°C
    elif "large" in size_str:
        ua_base  = 0.14
        Cth_base = 0.75
    else:
        ua_base  = 0.12
        Cth_base = 0.60

    # insulation
    if ins_str == "Poor":
        ua = ua_base * 1.3
    elif ins_str == "Good":
        ua = ua_base * 0.7
    else:
        ua = ua_base

    # default comfort band (can be edited in UI)
    t_min_default = 20.0
    t_max_default = 22.0

    return {
        "ua_kw_per_c": ua,
        "C_th_kwh_per_c": Cth_base,
        "t_min_default": t_min_default,
        "t_max_default": t_max_default,
    }



# -------------------------------------------------------------------
# Main page function
# -------------------------------------------------------------------
def render_devices_page_house():

    st.markdown("### 🏠 House information")

    hi = st.session_state.get("house_info", {
        "size": "Medium house",
        "insulation": "Average",
        "residents": 2,
    })
    prev_hi = st.session_state.get("house_info_prev")
    if prev_hi is None or prev_hi != hi:
        st.session_state["device_configs"] = {}
    st.session_state["house_info_prev"] = hi.copy()


    # ---- init session dicts ----
    if "device_selection" not in st.session_state:
        st.session_state["device_selection"] = {}
    if "device_configs" not in st.session_state:
        st.session_state["device_configs"] = {}

    sel = st.session_state["device_selection"]
    cfgs = st.session_state["device_configs"]

    # ===============================================================
    # 1) TOP: layout (left) + daily power profiles (right)
    # ===============================================================
    top_left, top_right = st.columns([1, 1])

    with top_left:
        st.markdown("### House layout")
        layout_placeholder = st.empty()

    with top_right:
        st.markdown("### Daily power profiles")
        profile_placeholder = st.empty()
        

    st.markdown("---")

    # ===============================================================
    # 2) BOTTOM: device selection – Ninite-style
    # ===============================================================
    st.markdown("### Select the devices in your house")
    # ---------------------------------------------------------------
    # Shared preferences for ALL flexible loads
    # ---------------------------------------------------------------
    flex_prefs = st.session_state.setdefault(
        "flex_prefs",
        {
            "w_cost": 1,
            "window_mode": "Daytime (08–17)",
            "earliest": _time(8, 0),
            "latest":  _time(17, 0),
            "earliest_custom": _time(7, 0),
            "latest_custom":   _time(22, 0),
        },
    )



    def render_category(cat_key: str, n_cols: int = 3):
        info = DEVICE_CATEGORIES[cat_key]
        st.markdown(f"#### {info['title']}")
        st.caption(info["help"])

        # ---- Global preference slider for ALL flexible loads ----
        # ---------- category-level controls for flexible loads ----------
        if cat_key == "elec_flex":
            # make sure we always have a dict in session_state
            default_flex = {
                "w_cost": 1,
                "window_mode": "Daytime (08–17)",
                "earliest": _time(8, 0),
                "latest":  _time(17, 0),
                "earliest_custom": _time(7, 0),
                "latest_custom":   _time(22, 0),
            }
            flex_prefs = st.session_state.get("flex_prefs")
            if not isinstance(flex_prefs, dict):
                flex_prefs = default_flex.copy()
                st.session_state["flex_prefs"] = flex_prefs

            with st.expander("⚙️ Flexible load – global settings", expanded=False):

                # --- allowed window presets ---
                window_options = [
                    "Any time (00–24)",
                    "Daytime (08–17)",
                    "Evening (17–23)",
                    "Night (00–06)",
                    "Custom",
                ]
                flex_prefs["window_mode"] = st.radio(
                    "When is it OK to run flexible devices?",
                    options=window_options,
                    index=window_options.index(flex_prefs.get("window_mode", "Daytime (08–17)")),
                    horizontal=False,
                    key="flex_window_mode",
                )

                mode = flex_prefs["window_mode"]
                if mode == "Any time (00–24)":
                    flex_prefs["earliest"] = _time(0, 0)
                    flex_prefs["latest"]   = _time(23, 59)
                elif mode == "Daytime (08–17)":
                    flex_prefs["earliest"] = _time(8, 0)
                    flex_prefs["latest"]   = _time(17, 0)
                elif mode == "Evening (17–23)":
                    flex_prefs["earliest"] = _time(17, 0)
                    flex_prefs["latest"]   = _time(23, 0)
                elif mode == "Night (00–06)":
                    flex_prefs["earliest"] = _time(0, 0)
                    flex_prefs["latest"]   = _time(6, 0)
                elif mode == "Custom":
                    c1, c2 = st.columns(2)
                    with c1:
                        flex_prefs["earliest_custom"] = c1.time_input(
                            "Earliest allowed start",
                            value=flex_prefs.get("earliest_custom", _time(7, 0)),
                            key="flex_earliest_custom",
                        )
                    with c2:
                        flex_prefs["latest_custom"] = c2.time_input(
                            "Latest allowed finish",
                            value=flex_prefs.get("latest_custom", _time(22, 0)),
                            key="flex_latest_custom",
                        )
                    # copy custom to effective window
                    flex_prefs["earliest"] = flex_prefs["earliest_custom"]
                    flex_prefs["latest"]   = flex_prefs["latest_custom"]

                # --- single preference bar for all flex loads ---
                flex_prefs["w_cost"] = st.slider(
                    "Preference (0 = CO₂ only, 1 = cost only)",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    value=float(flex_prefs.get("w_cost", 1)),
                    key="flex_w_cost",
                )

                cols_pref = st.columns(2)
                with cols_pref[1]:
                    if st.button("↩ Reset flexible loads to defaults", key="flex_reset_all"):
                        new_defaults = default_flex.copy()
                        st.session_state["flex_prefs"] = new_defaults
                        flex_prefs = new_defaults

                        cfgs_local = st.session_state.get("device_configs", {})
                        for full_key in list(cfgs_local.keys()):
                            if full_key.startswith("elec_flex:"):
                                dev_type = full_key.split(":", 1)[1]
                                cfgs_local[full_key] = get_default_config(dev_type, "elec_flex")
                        st.session_state["device_configs"] = cfgs_local

                        st.success("All flexible loads reset to defaults for this house profile.")
                        st.rerun()

                with cols_pref[0]:
                    if st.button("💡 Suggest schedules for all flexible devices",
                                key="flex_suggest_all"):
                        flex_prefs = st.session_state.get("flex_prefs", {})
                        cfgs_local = st.session_state.get("device_configs", {})

                        any_updated = False
                        for full_key, cfg in cfgs_local.items():
                            if not full_key.startswith("elec_flex:"):
                                continue
                            dev_type = full_key.split(":", 1)[1]

                            # ensure we have a config
                            if cfg is None or "duration_min" not in cfg:
                                cfg = get_default_config(dev_type, "elec_flex")

                            duration = int(cfg.get("duration_min", 60))
                            interval = suggest_best_interval_for_day(
                                duration_min=duration,
                                w_cost=flex_prefs.get("w_cost", 0.5),
                                earliest=flex_prefs.get("earliest"),
                                latest=flex_prefs.get("latest"),
                            )
                            if interval is None:
                                continue

                            cfg["intervals"] = [interval]

                            # update preview profile
                            num_devices = int(cfg.get("num_devices", 1))
                            power_w     = float(
                                cfg.get("power_w", cfg.get("power_kw", 0.5) * 1000.0)
                            )
                            prof = build_minute_profile(
                                power_w=power_w * num_devices,
                                intervals=cfg["intervals"],
                                step_min=1,
                            )
                            cfg["profile_index"] = prof.index.astype(str).tolist()
                            cfg["profile_kw"]    = prof.values.tolist()

                            cfgs_local[full_key] = cfg
                            any_updated = True

                        if any_updated:
                            st.session_state["device_configs"] = cfgs_local
                            st.success("All flexible device schedules were updated.")
                            st.rerun()
                        else:
                            st.info("No flexible devices are configured yet, or no price/CO₂ data.")
                
                st.markdown("##### 📋 Flexible devices – schedule overview")
                rows = []
                for dev_type, label in DEVICE_CATEGORIES["elec_flex"]["devices"]:
                    full_key = f"elec_flex:{dev_type}"

                    # only show selected devices
                    if not sel.get(full_key, False):
                        continue

                    cfg = cfgs.get(full_key, {})
                    if not cfg:
                        continue

                    # resolve display name (respect custom name for "other_flex_*")
                    display_label = label
                    if dev_type.startswith("other"):
                        custom_name = cfg.get("custom_name")
                        if custom_name:
                            # keep emoji, change text
                            if " " in label:
                                emoji = label.split(" ", 1)[0]
                            else:
                                emoji = "🧩"
                            display_label = f"{emoji} {custom_name}"

                    intervals = cfg.get("intervals") or []
                    if not intervals:
                        continue

                    # join intervals as "HH:MM–HH:MM, ..."
                    def _fmt_t(t):
                        return t.strftime("%H:%M") if t is not None else "--:--"

                    interval_strs = [
                        f"{_fmt_t(iv.get('start'))}–{_fmt_t(iv.get('end'))}"
                        for iv in intervals
                    ]
                    interval_txt = ", ".join(interval_strs)

                    num_devices = int(cfg.get("num_devices", 1))
                    dur_min = int(cfg.get("duration_min", 0))

                    rows.append(
                        {
                            "Device": display_label,
                            "# units": num_devices,
                            "Duration (min)": dur_min,
                            "Scheduled time(s)": interval_txt,
                        }
                    )

                if rows:
                    df_overview = pd.DataFrame(rows)
                    st.dataframe(
                        df_overview,
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("No flexible devices selected yet, or no schedules defined.")

            st.session_state["flex_prefs"] = flex_prefs

        # ---------- Thermal: shared house model + helpers ----------
        if cat_key == "thermal":
            hpar = get_house_thermal_params()
            ua_base   = hpar["ua_kw_per_c"]
            Cth_base  = hpar["C_th_kwh_per_c"]
            tmin_def  = hpar["t_min_default"]
            tmax_def  = hpar["t_max_default"]

            # store for later use if needed elsewhere
            st.session_state["thermal_house_params"] = hpar


            def _get_outdoor_profile():
                if (
                    "temp_daily" in st.session_state
                    and isinstance(st.session_state["temp_daily"], pd.Series)
                    and not st.session_state["temp_daily"].empty
                ):
                    tout_tot = st.session_state["temp_daily"]
                    tout_minute = get_selected_day_data(tout_tot)
                    idx = tout_minute.index
                else:
                    idx = pd.date_range("2025-01-10 00:00", periods=24 * 60, freq="min")
                    hours = idx.hour + idx.minute / 60.0
                    tout_minute = pd.Series(
                        5.0 + 5.0 * np.sin(2 * np.pi * (hours - 15) / 24.0),
                        index=idx,
                        name="Tout_C",
                    )
                return idx, tout_minute


        devs = info["devices"]
        cols = st.columns(n_cols)

        for i, (dev_type, label) in enumerate(devs):
            col = cols[i % n_cols]
            with col:
                full_key   = f"{cat_key}:{dev_type}"
                open_key   = f"open_cfg_{full_key}"
                settings_id = full_key.replace(":", "_")

                # get current cfg (for custom name)
                cfg_current = cfgs.get(full_key, {})

                # --------- build dynamic label (for "other*" types) ----------
                display_label = resolve_display_label(full_key, dev_type, cfg_current)
        
                c1, c2 = st.columns([0.8, 0.2])

                with c1:
                    default_on = False
                    checked = st.checkbox(
                        display_label,
                        value=sel.get(full_key, default_on),
                        key=f"chk_{full_key}",
                    )
                    sel[full_key] = checked
                if checked and full_key not in cfgs and cat_key in ("elec_fixed", "elec_flex"):
                    cfgs[full_key] = get_default_config(dev_type, cat_key)    

                with c2:
                    if st.button("⚙️", key=f"cfg_{full_key}"):
                        st.session_state[open_key] = not st.session_state.get(open_key, False)
                        st.rerun()

                # --- settings panel (only when opened) ---
                if st.session_state.get(open_key, False):
                    cfg = cfgs.get(full_key)
                    if cfg is None:
                        cfg = get_default_config(dev_type, cat_key)
                        cfgs[full_key] = cfg

                    # --- Custom name for “other*” devices ---
                    if dev_type.startswith("other"):
                        base_label = label
                        if " " in label:
                            base_label = label.split(" ", 1)[1]
                        cfg["custom_name"] = st.text_input(
                            "Name of this device",
                            value=cfg.get("custom_name", base_label),
                            key=f"name_{full_key}",
                            help="E.g. '3D printer', 'Aquarium pump', 'Server rack', etc.",
                        )

                    st.write("")  # small gap

                    # =======================================================
                    # A) Detailed editor for FIXED ELECTRICAL loads
                    # =======================================================
                    if cat_key == "elec_fixed":
                        with st.container():
                            st.markdown(
                                "<hr style='border-top: 1px dashed #bbb;'/>",
                                unsafe_allow_html=True,
                            )

                            # Number of identical devices
                            cfg["num_devices"] = int(
                                st.number_input(
                                    "Number of devices",
                                    min_value=1,
                                    max_value=30,
                                    step=1,
                                    value=int(cfg.get("num_devices", 1)),
                                    key=f"{settings_id}_numdev",
                                )
                            )

                            # Power per device (W)
                            # fall back to power_kw if power_w not set yet
                            default_power_w = float(
                                cfg.get("power_w", cfg.get("power_kw", 0.1) * 1000.0)
                            )
                            cfg["power_w"] = st.number_input(
                                "Power per device (W)",
                                min_value=0.0,
                                max_value=5000.0,
                                step=10.0,
                                value=default_power_w,
                                key=f"{settings_id}_power",
                            )
                            # keep power_kw in sync (for later use if needed)
                            cfg["power_kw"] = cfg["power_w"] / 1000.0

                            # Intervals (multiple allowed)
                            st.caption("On/off intervals (you can add multiple):")
                            intervals = cfg.setdefault("intervals", [])
                            if not intervals:
                                intervals.append({"start": _time(18, 0), "end": _time(23, 0)})

                            to_delete = None
                            for j, iv in enumerate(intervals):
                                c_a, c_b, c_c = st.columns([0.4, 0.4, 0.2])
                                with c_a:
                                    s_t = st.time_input(
                                        "Start",
                                        value=iv.get("start", _time(18, 0)),
                                        key=f"{settings_id}_start_{j}",
                                    )
                                with c_b:
                                    e_t = st.time_input(
                                        "End",
                                        value=iv.get("end", _time(23, 0)),
                                        key=f"{settings_id}_end_{j}",
                                    )
                                with c_c:
                                    if st.button("🗑", key=f"{settings_id}_ivdel_{j}"):
                                        to_delete = j
                                iv["start"], iv["end"] = s_t, e_t

                            if to_delete is not None:
                                intervals.pop(to_delete)
                                st.rerun()

                            if st.button("➕ Add interval", key=f"{settings_id}_add_interval"):
                                intervals.append({"start": _time(18, 0), "end": _time(23, 0)})
                                st.rerun()

                            # small gap
                            st.markdown("<div style='height:0.5rem'></div>",
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
                                name="P_fixed_total_kW",
                            )
                            fig_p.update_layout(
                                height=180,
                                margin=dict(l=10, r=10, t=8, b=8),
                                xaxis_title="Time",
                                yaxis_title="kW",
                                showlegend=False,
                            )
                            st.plotly_chart(fig_p, use_container_width=True)

                            if st.button("▲ Hide", key=f"hide_{full_key}"):
                                st.session_state[open_key] = False
                                st.rerun()

                            st.markdown("</div>", unsafe_allow_html=True)

                    # =======================================================
                    # B) Detailed editor for FLEXIBLE ELECTRICAL loads
                    #    (single shiftable block + preference + suggest button)
                    # =======================================================
                    elif cat_key == "elec_flex":
                        with st.container():
                            st.markdown(
                                "<hr style='border-top: 1px dashed #bbb;'/>",
                                unsafe_allow_html=True,
                            )

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
                            cfg["power_kw"] = cfg["power_w"] / 1000.0

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

                            # Ensure exactly one interval in cfg
                            intervals = cfg.setdefault("intervals", [])
                            if not intervals:
                                intervals.append(
                                    {"start": _time(20, 0), "end": _time(21, 30)}
                                )
                            elif len(intervals) > 1:
                                intervals[:] = intervals[:1]

                            current_iv = intervals[0]

                            st.caption("Current scheduled interval (one continuous block):")
                            c_a, c_b = st.columns(2)
                            with c_a:
                                new_start = st.time_input(
                                    "Start",
                                    value=current_iv.get("start", _time(20, 0)),
                                    key=f"{settings_id}_flex_start",
                                )
                            with c_b:
                                new_end = st.time_input(
                                    "End",
                                    value=current_iv.get("end", _time(21, 30)),
                                    key=f"{settings_id}_flex_end",
                                )
                            current_iv["start"], current_iv["end"] = new_start, new_end

                        
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

                            # per-device reset to defaults
                            if st.button("↩ Reset this device to defaults",
                                         key=f"{settings_id}_reset_flex"):
                                new_cfg = get_default_config(dev_type, "elec_flex")
                                cfgs[full_key] = new_cfg
                                st.success("Device reset to default flexible settings.")
                                st.rerun()

                            if st.button("▲ Hide details", key=f"{settings_id}_hide_flex"):
                                st.session_state[open_key] = False
                                st.rerun()

                    # =======================================================
                    # C) THERMAL DEVICES – behave like normal devices
                    # =======================================================
                    elif cat_key == "thermal":
                        # three dev_types: space_heat, dhw, leisure
                        # panel opens only when st.session_state[open_key] is True
                        cfg = cfgs.get(full_key)
                        if cfg is None:
                            cfg = {}
                            cfgs[full_key] = cfg

                        # read shared params prepared at top
                        hpar = st.session_state.get("thermal_house_params", {})
                        ua_base   = float(hpar.get("ua_kw_per_c", 0.12))
                        Cth_base  = float(hpar.get("C_th_kwh_per_c", 0.60))
                        tmin_def  = float(hpar.get("t_min_default", 20.0))
                        tmax_def  = float(hpar.get("t_max_default", 22.0))

                        def _get_outdoor_profile_local():
                            return _get_outdoor_profile()  # from pre-block above

                        # ----- 1) SPACE HEATING ---------------------------------
                        if dev_type == "space_heat":
                            st.markdown(
                                "<hr style='border-top: 1px dotted #fecaca; margin:0.4rem 0;'/>",
                                unsafe_allow_html=True,
                            )
                            st.markdown("**Space heating system**")

                            # subtype
                            space_options = [
                                "None (external supply)",
                                "Electric panels",
                                "Heat pump – air-to-air",
                                "Heat pump – air-to-water",
                            ]
                            mode_default = cfg.get("space_mode", "Electric panels")
                            if mode_default not in space_options:
                                mode_default = "Electric panels"

                            space_mode = st.radio(
                                "Which space heating system do you use?",
                                options=space_options,
                                index=space_options.index(mode_default),
                                key=f"{settings_id}_space_mode",
                            )
                            cfg["space_mode"] = space_mode

                            # comfort range
                            c_tmin, c_tmax = st.columns(2)
                            cfg["t_min_c"] = c_tmin.number_input(
                                "Min indoor temperature (°C)",
                                min_value=5.0,
                                max_value=30.0,
                                step=0.5,
                                value=float(cfg.get("t_min_c", tmin_def)),
                                key=f"{settings_id}_tmin",
                            )
                            cfg["t_max_c"] = c_tmax.number_input(
                                "Max indoor temperature (°C)",
                                min_value=5.0,
                                max_value=30.0,
                                step=0.5,
                                value=float(cfg.get("t_max_c", tmax_def)),
                                key=f"{settings_id}_tmax",
                            )
                            if cfg["t_max_c"] < cfg["t_min_c"]:
                                cfg["t_max_c"] = cfg["t_min_c"]

                            t_min = float(cfg["t_min_c"])
                            t_max = float(cfg["t_max_c"])
                            t_set = 0.5 * (t_min + t_max)
                            hyst  = max(t_max - t_min, 0.5)
                            Cth_eff = Cth_base

                            P_space = None
                            Ti_space = None

                            if space_mode == "None (external supply)":
                                idx_hp, _tout = _get_outdoor_profile_local()
                                P_space = pd.Series(0.0, index=idx_hp, name="P_space_kW")
                                st.info("No electric space heating – maybe district heating, gas, or external boiler.")

                            elif space_mode == "Electric panels":
                                cfg["q_kw"] = st.number_input(
                                    "Total panel capacity (kW)",
                                    min_value=1.0,
                                    max_value=30.0,
                                    step=0.5,
                                    value=float(cfg.get("q_kw", 6.0)),
                                    key=f"{settings_id}_q_elec",
                                )
                                q_rated = float(cfg["q_kw"])

                                idx_hp, tout = _get_outdoor_profile_local()
                                eh = WeatherELheater(
                                    ua_kw_per_c=ua_base,
                                    t_set_c=t_set,
                                    q_rated_kw=q_rated,
                                    C_th_kwh_per_c=Cth_eff,
                                    hyst_band_c=hyst,
                                    p_off_kw=0.05,
                                    min_on_min=0,
                                    min_off_min=0,
                                    Ti0_c=t_set,
                                )
                                P_space,Ti_space  = eh.series_kw(idx_hp, tout)

                            elif space_mode == "Heat pump – air-to-air":
                                st.caption("Heats air directly, cannot heat water.")

                                hp_type_default = cfg.get("hp_type", "Fixed power")
                                hp_type = st.radio(
                                    "Heat pump type",
                                    options=["Fixed power", "Variable power"],
                                    index=["Fixed power", "Variable power"].index(hp_type_default),
                                    horizontal=True,
                                    key=f"{settings_id}_hp_ataa_type",
                                )
                                cfg["hp_type"] = hp_type
                                hp_mode = "onoff" if hp_type == "Fixed power" else "modulating"

                                cfg["q_kw"] = st.number_input(
                                    "Heat pump capacity (kW, thermal)",
                                    min_value=1.0,
                                    max_value=20.0,
                                    step=0.5,
                                    value=float(cfg.get("q_kw", 6.0)),
                                    key=f"{settings_id}_q_ataa",
                                )
                                q_rated = float(cfg["q_kw"])

                                idx_hp, tout = _get_outdoor_profile_local()
                                hp = WeatherHP(
                                    mode=hp_mode,
                                    ua_kw_per_c=ua_base,
                                    t_set_c=t_set,
                                    q_rated_kw=q_rated,
                                    cop_at_7c=3.2,
                                    cop_min=1.6,
                                    cop_max=4.2,
                                    C_th_kwh_per_c=Cth_eff,
                                    hyst_band_c=hyst,
                                    p_off_kw=0.05,
                                    defrost=True,
                                    min_on_min=0,
                                    min_off_min=0,
                                    Ti0_c=t_set,
                                )
                                P_space,Ti_space  = hp.series_kw(idx_hp, tout)

                            elif space_mode == "Heat pump – air-to-water":
                                st.caption("Heats water to radiators / floor heating. Can also feed a DHW tank.")

                                hp_type_default = cfg.get("hp_type", "Fixed power")
                                hp_type = st.radio(
                                    "Heat pump type",
                                    options=["Fixed power", "Variable power"],
                                    index=["Fixed power", "Variable power"].index(hp_type_default),
                                    horizontal=True,
                                    key=f"{settings_id}_hp_ataw_type",
                                )
                                cfg["hp_type"] = hp_type
                                hp_mode = "onoff" if hp_type == "Fixed power" else "modulating"

                                cfg["q_kw"] = st.number_input(
                                    "Heat pump capacity (kW, thermal)",
                                    min_value=3.0,
                                    max_value=25.0,
                                    step=0.5,
                                    value=float(cfg.get("q_kw", 8.0)),
                                    key=f"{settings_id}_q_ataw",
                                )
                                q_rated = float(cfg["q_kw"])

                                dist_options = ["Radiators", "Floor heating", "Both"]
                                dist_default = cfg.get("distribution", "Radiators")
                                if dist_default not in dist_options:
                                    dist_default = "Radiators"
                                dist_mode = st.selectbox(
                                    "Heat distribution system",
                                    options=dist_options,
                                    index=dist_options.index(dist_default),
                                    key=f"{settings_id}_distribution",
                                )
                                cfg["distribution"] = dist_mode

                                extra_mass = {"Radiators": 0.0, "Floor heating": 0.5, "Both": 0.3}[dist_mode]
                                Cth_eff = Cth_base * (1.0 + extra_mass)

                                idx_hp, tout = _get_outdoor_profile_local()
                                hp = WeatherHP(
                                    mode=hp_mode,
                                    ua_kw_per_c=ua_base,
                                    t_set_c=t_set,
                                    q_rated_kw=q_rated,
                                    cop_at_7c=3.2,
                                    cop_min=1.6,
                                    cop_max=4.2,
                                    C_th_kwh_per_c=Cth_eff,
                                    hyst_band_c=hyst,
                                    p_off_kw=0.05,
                                    defrost=True,
                                    min_on_min=0,
                                    min_off_min=0,
                                    Ti0_c=t_set,
                                )
                                P_space,Ti_space = hp.series_kw(idx_hp, tout)

                            if P_space is not None:
                                st.markdown("**Space-heating electric power (preview)**")
                                fig = go.Figure()
                                fig.add_scatter(x=P_space.index, y=P_space.values, mode="lines")
                                fig.update_layout(
                                    height=180,
                                    margin=dict(l=10, r=10, t=8, b=8),
                                    xaxis_title="Time",
                                    yaxis_title="kW",
                                    showlegend=False,
                                )
                                st.plotly_chart(fig, use_container_width=True,
                                                key=f"{settings_id}_space_power")   # ← add key here
                                cfg["profile_index"] = P_space.index.astype(str).tolist()
                                cfg["profile_kw"]    = P_space.values.tolist()

                                if 'Ti_space' in locals() and Ti_space is not None:
                                    st.markdown("**Indoor temperature (preview)**")
                                    fig_T = go.Figure()
                                    fig_T.add_scatter(x=Ti_space.index, y=Ti_space.values, mode="lines", name="Ti")
                                    # optional: show comfort band lines
                                    fig_T.add_hline(y=t_min, line_dash="dot")
                                    fig_T.add_hline(y=t_max, line_dash="dot")
                                    fig_T.update_layout(
                                        height=180,
                                        margin=dict(l=10, r=10, t=8, b=8),
                                        xaxis_title="Time",
                                        yaxis_title="°C",
                                        showlegend=False,
                                    )
                                    st.plotly_chart(fig_T, use_container_width=True,key=f"{settings_id}_space_temp")

                            if st.button("▲ Hide", key=f"{settings_id}_hide_space"):
                                st.session_state[open_key] = False
                                st.rerun()

                        # ----- 2) DHW SYSTEM -------------------------------------
                        elif dev_type == "dhw":
                            st.markdown(
                                "<hr style='border-top: 1px dotted #bfdbfe; margin:0.4rem 0;'/>",
                                unsafe_allow_html=True,
                            )
                            st.markdown("**Domestic hot water (DHW)**")

                            dhw_options = [
                                "None (external supply)",
                                "Electric DHW tank",
                                "Heat pump DHW tank",
                            ]
                            mode_default = cfg.get("dhw_mode", "None (external supply)")
                            if mode_default not in dhw_options:
                                mode_default = "None (external supply)"

                            dhw_mode = st.radio(
                                "How is your domestic hot water heated?",
                                options=dhw_options,
                                index=dhw_options.index(mode_default),
                                key=f"{settings_id}_dhw_mode",
                            )
                            cfg["dhw_mode"] = dhw_mode

                            P_dhw = None
                            T_tank = None
                            idx_dhw, tout = _get_outdoor_profile_local()

                            if dhw_mode == "None (external supply)":
                                st.info("DHW provided by district heating / gas / shared system – EMS does not heat it.")
                                P_dhw = pd.Series(0.0, index=idx_dhw, name="P_DHW_kW")
                            else:
                                # defaults from house info
                                hi = st.session_state.get(
                                    "house_info",
                                    {"size": "Medium house", "insulation": "Average", "residents": 2},
                                )
                                n_res = int(hi.get("residents", 2))
                                size_str = (hi.get("size") or "Medium house").lower()
                                if "small" in size_str:
                                    vol_default = 160.0
                                elif "large" in size_str:
                                    vol_default = 300.0
                                else:
                                    vol_default = 200.0
                                if n_res <= 2:
                                    usage_default = "Low"
                                elif n_res <= 4:
                                    usage_default = "Medium"
                                else:
                                    usage_default = "High"

                                c_vol, c_use = st.columns(2)
                                cfg["volume_l"] = c_vol.number_input(
                                    "Tank volume (L)",
                                    min_value=50.0,
                                    max_value=500.0,
                                    step=25.0,
                                    value=float(cfg.get("volume_l", vol_default)),
                                    key=f"{settings_id}_dhw_vol",
                                )

                                label_map = {
                                    "Low – 1–2 persons, short showers": "Low",
                                    "Medium – 3–4 persons, normal use": "Medium",
                                    "High – 5+ persons or long showers": "High",
                                }
                                reverse_map = {v: k for k, v in label_map.items()}
                                usage_now = cfg.get("usage_level", usage_default)
                                usage_label_default = reverse_map[usage_now]

                                usage_label = c_use.selectbox(
                                    "Usage level",
                                    label_map.keys(),
                                    index=list(label_map.keys()).index(usage_label_default),
                                    key=f"{settings_id}_dhw_usage",
                                )
                                cfg["usage_level"] = label_map[usage_label]

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

                                t_min_tank = float(cfg["t_min_c"])
                                t_max_tank = float(cfg["t_max_c"])
                                t_set_tank = 0.5 * (t_min_tank + t_max_tank)
                                hyst_tank  = max(t_max_tank - t_min_tank, 1.0)

                                default_p = 2.0 if dhw_mode == "Electric DHW tank" else 1.5
                                cfg["p_el_kw"] = st.number_input(
                                    "Heater power (kW, thermal side)",
                                    min_value=0.5,
                                    max_value=10.0,
                                    step=0.5,
                                    value=float(cfg.get("p_el_kw", default_p)),
                                    key=f"{settings_id}_dhw_pel",
                                )

                                if dhw_mode == "Electric DHW tank":
                                    tank = DHWTank(
                                        volume_l=float(cfg["volume_l"]),
                                        t_set_c=t_set_tank,
                                        hyst_band_c=hyst_tank,
                                        ua_kw_per_c=0.02,
                                        p_el_kw=float(cfg["p_el_kw"]),
                                        p_off_kw=0.01,
                                        T_cold_c=10.0,
                                        T_amb_c=20.0,
                                        Ti0_c=t_set_tank,
                                        usage_level=cfg["usage_level"],
                                    )
                                    P_dhw,T_tank  = tank.series_kw(idx_dhw, tout)
                                else:
                                    # HP DHW tank – approximate electrical power via COP
                                    tank = DHWTank(
                                        volume_l=float(cfg["volume_l"]),
                                        t_set_c=t_set_tank,
                                        hyst_band_c=hyst_tank,
                                        ua_kw_per_c=0.02,
                                        p_el_kw=float(cfg["p_el_kw"]) * 2.5,  # thermal
                                        p_off_kw=0.01,
                                        T_cold_c=10.0,
                                        T_amb_c=20.0,
                                        Ti0_c=t_set_tank,
                                        usage_level=cfg["usage_level"],
                                    )
                                    Q_th,T_tank = tank.series_kw(idx_dhw, tout)
                                    cop = 2.5
                                    P_dhw = Q_th / cop
                                    P_dhw.name = "P_DHW_HP_kW"

                            if P_dhw is not None:
                                st.markdown("**DHW electrical power (preview)**")
                                fig = go.Figure()
                                fig.add_scatter(x=P_dhw.index, y=P_dhw.values, mode="lines")
                                fig.update_layout(
                                    height=180,
                                    margin=dict(l=10, r=10, t=8, b=8),
                                    xaxis_title="Time",
                                    yaxis_title="kW",
                                    showlegend=False,
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                if 'T_tank' in locals() and T_tank is not None:
                                    st.markdown("**DHW tank temperature (preview)**")
                                    fig_T = go.Figure()
                                    fig_T.add_scatter(x=T_tank.index, y=T_tank.values, mode="lines")
                                    fig_T.add_hline(y=t_min_tank, line_dash="dot")
                                    fig_T.add_hline(y=t_max_tank, line_dash="dot")
                                    fig_T.update_layout(
                                        height=180,
                                        margin=dict(l=10, r=10, t=8, b=8),
                                        xaxis_title="Time",
                                        yaxis_title="°C",
                                        showlegend=False,
                                    )
                                    st.plotly_chart(fig_T, use_container_width=True)
                                cfg["profile_index"] = P_dhw.index.astype(str).tolist()
                                cfg["profile_kw"]    = P_dhw.values.tolist()

                            if st.button("▲ Hide", key=f"{settings_id}_hide_dhw"):
                                st.session_state[open_key] = False
                                st.rerun()

                        # ----- 3) LEISURE THERMAL LOADS (hot tub + pool + sauna) -----
                        elif dev_type == "leisure":
                            st.markdown(
                                "<hr style='border-top: 1px dotted #bbf7d0; margin:0.4rem 0;'/>",
                                unsafe_allow_html=True,
                            )
                            st.markdown("**Leisure thermal loads**")

                            # always get a daily index + outdoor T once
                            idx_day, tout_day = _get_outdoor_profile_local()

                            # ensure cfg is a dict
                            if not isinstance(cfg, dict):
                                cfg = {}
                                cfgs[full_key] = cfg

                            # ---- three toggles in one row ----
                            col_ht, col_pool = st.columns(2)
                            with col_ht:
                                cfg["hot_tub_enabled"] = st.checkbox(
                                    "🛁 Hot tub / spa",
                                    value=bool(cfg.get("hot_tub_enabled", False)),
                                    key=f"{settings_id}_ht_enable",
                                )
                            with col_pool:
                                cfg["pool_enabled"] = st.checkbox(
                                    "🏊 Pool heater",
                                    value=bool(cfg.get("pool_enabled", False)),
                                    key=f"{settings_id}_pool_enable",
                                )
                  

                            # storage for sub-profiles
                            P_ht = P_pool  = None

                            # ======================================================
                            # HOT TUB / SPA
                            # ======================================================
                            if cfg.get("hot_tub_enabled", False):
                                st.markdown("### 🛁 Hot tub")
                                st.caption("This model uses a built-in electric heater (COP = 1).")


                                c_tgt, c_idle = st.columns(2)
                                cfg["ht_target_c"] = c_tgt.number_input(
                                    "Target water temperature (°C)",
                                    min_value=25.0,
                                    max_value=45.0,
                                    step=0.5,
                                    value=float(cfg.get("ht_target_c", 40.0)),
                                    key=f"{settings_id}_ht_Ttarget",
                                )
                                cfg["ht_idle_c"] = c_idle.number_input(
                                    "Idle temperature (°C)",
                                    min_value=10.0,
                                    max_value=40.0,
                                    step=0.5,
                                    value=float(cfg.get("ht_idle_c", 30.0)),
                                    key=f"{settings_id}_ht_Tidle",
                                )
                                if cfg["ht_idle_c"] > cfg["ht_target_c"]:
                                    cfg["ht_idle_c"] = cfg["ht_target_c"]

                                c_vol, c_pow = st.columns(2)
                                cfg["ht_water_l"] = c_vol.number_input(
                                    "Water volume (L)",
                                    min_value=400.0,
                                    max_value=3000.0,
                                    step=50.0,
                                    value=float(cfg.get("ht_water_l", 1200.0)),
                                    key=f"{settings_id}_ht_vol",
                                )
                                cfg["ht_heater_kw"] = c_pow.number_input(
                                    "Heater capacity (kW)",
                                    min_value=1.0,
                                    max_value=12.0,
                                    step=0.5,
                                    value=float(cfg.get("ht_heater_kw", 5.0)),
                                    key=f"{settings_id}_ht_kw",
                                )

                                ins_levels = ["Good cover", "Average", "Poor"]
                                ins_default = cfg.get("ht_insulation", "Average")
                                if ins_default not in ins_levels:
                                    ins_default = "Average"
                                cfg["ht_insulation"] = st.selectbox(
                                    "Cover / insulation level",
                                    ins_levels,
                                    index=ins_levels.index(ins_default),
                                    key=f"{settings_id}_ht_ins",
                                )
                                ins = cfg["ht_insulation"]
                                ua_base_ht = 0.07
                                if ins == "Good cover":
                                    ua_ht = ua_base_ht * 0.6
                                elif ins == "Poor":
                                    ua_ht = ua_base_ht * 1.4
                                else:
                                    ua_ht = ua_base_ht

                                # ---- estimated preheat time ----
                                C_kwh_per_c_ht = max(cfg["ht_water_l"] * 1.16 / 1000.0, 1e-6)
                                delta_c_ht = max(cfg["ht_target_c"] - cfg["ht_idle_c"], 0.0)
                                E_ht = C_kwh_per_c_ht * delta_c_ht   # kWh
                                if cfg["ht_heater_kw"] > 0:
                                    t_est_h_ht = E_ht / cfg["ht_heater_kw"]
                                else:
                                    t_est_h_ht = 0.0
                                t_est_min_ht = int(round(t_est_h_ht * 60.0))
                                st.caption(f"Rough time from idle to target ≈ {t_est_h_ht:.1f} h.")

                 
                                st.markdown("**Use sessions**")
                                sessions = cfg.get("ht_sessions") 
                                if sessions is None:
                                    sessions = []   # start empty the very first time
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
                                    cfg["ht_sessions"] = sessions
                                    st.rerun()

                                if st.button("➕ Add use session", key=f"{settings_id}_ht_add"):
                                    sessions.append({"start": _time(19, 0), "duration_min": 60})
                                    cfg["ht_sessions"] = sessions
                                    st.rerun()

                                cfg["ht_sessions"] = sessions

                                # indoor ambient, e.g. 21 °C
                                ht = WeatherHotTub(
                                    target_c=float(cfg["ht_target_c"]),
                                    idle_c=float(cfg["ht_idle_c"]),
                                    heater_kw=float(cfg["ht_heater_kw"]),
                                    water_l=float(cfg["ht_water_l"]),
                                    ua_kw_per_c=float(ua_ht),
                                    sessions=sessions,
                                    use_outdoor_for_ambient=False,     # ✅ indoor
                                    indoor_ambient_c=21.0,
                                )
                                P_ht, T_ht = ht.series_kw(idx_day, tout_day)

                                st.markdown("**Hot tub electrical power (preview)**")
                                fig_ht = go.Figure()
                                fig_ht.add_scatter(x=P_ht.index, y=P_ht.values, mode="lines")
                                fig_ht.update_layout(
                                    height=160,
                                    margin=dict(l=10, r=10, t=8, b=8),
                                    xaxis_title="Time",
                                    yaxis_title="kW",
                                    showlegend=False,
                                )
                                st.plotly_chart(fig_ht, use_container_width=True)

                                st.markdown("**Hot tub water temperature (preview)**")
                                fig_T = go.Figure()
                                fig_T.add_scatter(x=T_ht.index, y=T_ht.values, mode="lines")
                                fig_T.add_hline(y=cfg["ht_idle_c"],   line_dash="dot")
                                fig_T.add_hline(y=cfg["ht_target_c"], line_dash="dot")
                                fig_T.update_layout(
                                    height=160,
                                    margin=dict(l=10, r=10, t=8, b=8),
                                    xaxis_title="Time",
                                    yaxis_title="°C",
                                    showlegend=False,
                                )
                                st.plotly_chart(fig_T, use_container_width=True)



                            # ======================================================
                            # POOL HEATER – modeled as a big, cooler hot tub
                            # ======================================================
                            if cfg.get("pool_enabled", False):
                                st.markdown("### 🏊 Pool heater settings")
                                st.caption("Most pools use an air-to-water heat pump. This model assumes a pool heat pump with a COP 3.5.")


                                c_tgt2, c_idle2 = st.columns(2)
                                cfg["pool_target_c"] = c_tgt2.number_input(
                                    "Target water temperature (°C)",
                                    min_value=20.0,
                                    max_value=35.0,
                                    step=0.5,
                                    value=float(cfg.get("pool_target_c", 28.0)),
                                    key=f"{settings_id}_pool_Ttarget",
                                )
                                cfg["pool_idle_c"] = c_idle2.number_input(
                                    "Idle temperature (°C)",
                                    min_value=5.0,
                                    max_value=35.0,
                                    step=0.5,
                                    value=float(cfg.get("pool_idle_c", 24.0)),
                                    key=f"{settings_id}_pool_Tidle",
                                )
                                if cfg["pool_idle_c"] > cfg["pool_target_c"]:
                                    cfg["pool_idle_c"] = cfg["pool_target_c"]

                                c_vol2, c_pow2 = st.columns(2)
                                cfg["pool_water_l"] = c_vol2.number_input(
                                    "Water volume (L)",
                                    min_value=5000.0,
                                    max_value=80000.0,
                                    step=500.0,
                                    value=float(cfg.get("pool_water_l", 30000.0)),
                                    key=f"{settings_id}_pool_vol",
                                )
                                cfg["pool_heater_kw"] = c_pow2.number_input(
                                    "Heater capacity (kW, thermal)",
                                    min_value=3.0,
                                    max_value=40.0,
                                    step=1.0,
                                    value=float(cfg.get("pool_heater_kw", 15.0)),
                                    key=f"{settings_id}_pool_kw",
                                )

                                ins_levels_pool = ["Good cover", "Average", "Poor"]
                                ins_default_pool = cfg.get("pool_insulation", "Average")
                                if ins_default_pool not in ins_levels_pool:
                                    ins_default_pool = "Average"
                                cfg["pool_insulation"] = st.selectbox(
                                    "Cover / insulation level",
                                    ins_levels_pool,
                                    index=ins_levels_pool.index(ins_default_pool),
                                    key=f"{settings_id}_pool_ins",
                                )
                                ins_p = cfg["pool_insulation"]
                                ua_base_pool = 0.15
                                if ins_p == "Good cover":
                                    ua_pool = ua_base_pool * 0.6
                                elif ins_p == "Poor":
                                    ua_pool = ua_base_pool * 1.4
                                else:
                                    ua_pool = ua_base_pool

                                # ---- estimated preheat time (ignoring losses) ----
                                C_kwh_per_c_pool = max(cfg["pool_water_l"] * 1.16 / 1000.0, 1e-6)
                                delta_c_pool = max(cfg["pool_target_c"] - cfg["pool_idle_c"], 0.0)
                                E_pool = C_kwh_per_c_pool * delta_c_pool
                                if cfg["pool_heater_kw"] > 0:
                                    t_est_h_pool = E_pool / cfg["pool_heater_kw"]
                                else:
                                    t_est_h_pool = 0.0
                                t_est_min_pool = int(round(t_est_h_pool * 60.0))
                                st.caption(f"Rough time from idle to target ≈ {t_est_h_pool:.1f} h (ignoring losses).")



                                st.markdown("**Use sessions**")
                                pool_sessions = cfg.get("pool_sessions")
                                if pool_sessions is None:
                                    pool_sessions = []   # start empty the very first time
                                del_idx_pool = None
                                for j, sess in enumerate(pool_sessions):
                                    c_s, c_d, c_del = st.columns([0.4, 0.4, 0.2])
                                    with c_s:
                                        s_t = c_s.time_input(
                                            "Start",
                                            value=sess.get("start", _time(8, 0)),
                                            key=f"{settings_id}_pool_s_{j}",
                                        )
                                    with c_d:
                                        dur = c_d.number_input(
                                            "Duration (min)",
                                            min_value=30,
                                            max_value=1440,
                                            step=30,
                                            value=int(sess.get("duration_min", 480)),
                                            key=f"{settings_id}_pool_d_{j}",
                                        )
                                    with c_del:
                                        if c_del.button("🗑", key=f"{settings_id}_pool_del_{j}"):
                                            del_idx_pool = j
                                    sess["start"] = s_t
                                    sess["duration_min"] = dur

                                if del_idx_pool is not None:
                                    pool_sessions.pop(del_idx_pool)
                                    cfg["pool_sessions"] = pool_sessions
                                    st.rerun()

                                if st.button("➕ Add Use sessions", key=f"{settings_id}_pool_add"):
                                    pool_sessions.append({"start": _time(13, 0), "duration_min": 60})
                                    cfg["pool_sessions"] = pool_sessions
                                    st.rerun()

                                cfg["pool_sessions"] = pool_sessions

                                pool_model = WeatherHotTub(
                                    target_c=float(cfg["pool_target_c"]),
                                    idle_c=float(cfg["pool_idle_c"]),
                                    heater_kw=float(cfg["pool_heater_kw"]),
                                    water_l=float(cfg["pool_water_l"]),
                                    ua_kw_per_c=float(ua_pool),
                                    sessions=pool_sessions,
                                    use_outdoor_for_ambient=True,   # ✅ now we use Tout
                                    indoor_ambient_c=21.0,          # unused when use_outdoor_for_ambient=True
                                )
                                # pool uses real outdoor temperature
                                Q_pool_th, T_pool = pool_model.series_kw(idx_day, tout_day)
                                cop_pool = 3.5

                                P_pool = Q_pool_th / cop_pool
                                P_pool.name = "P_pool_HP_kW"

                                st.markdown("**Pool heater electrical power (preview)**")
                                fig_pool = go.Figure()
                                fig_pool.add_scatter(x=P_pool.index, y=P_pool.values, mode="lines")
                                fig_pool.update_layout(
                                    height=160,
                                    margin=dict(l=10, r=10, t=8, b=8),
                                    xaxis_title="Time",
                                    yaxis_title="kW",
                                    showlegend=False,
                                )
                                st.plotly_chart(fig_pool, use_container_width=True)

                                st.markdown("**Pool water temperature (preview)**")
                                fig_Tp = go.Figure()
                                fig_Tp.add_scatter(x=T_pool.index, y=T_pool.values, mode="lines")
                                fig_Tp.add_hline(y=cfg["pool_idle_c"],   line_dash="dot")
                                fig_Tp.add_hline(y=cfg["pool_target_c"], line_dash="dot")
                                fig_Tp.update_layout(
                                    height=160,
                                    margin=dict(l=10, r=10, t=8, b=8),
                                    xaxis_title="Time",
                                    yaxis_title="°C",
                                    showlegend=False,
                                )
                                st.plotly_chart(fig_Tp, use_container_width=True)



                           
                            # ======================================================
                            # AGGREGATED LEISURE PROFILE
                            # ======================================================
                            # start from zero series so we can always sum safely
                            P_total = pd.Series(0.0, index=idx_day, name="P_leisure_kW")
                            any_enabled = False

                            if P_ht is not None:
                                P_total = P_total + P_ht
                                any_enabled = True
                                cfg["hot_tub_profile_index"] = P_ht.index.astype(str).tolist()
                                cfg["hot_tub_profile_kw"]    = P_ht.values.tolist()

                            if P_pool is not None:
                                P_total = P_total + P_pool
                                any_enabled = True
                                cfg["pool_profile_index"] = P_pool.index.astype(str).tolist()
                                cfg["pool_profile_kw"]    = P_pool.values.tolist()

                           
                            if any_enabled:
                                st.markdown("**Total leisure electrical power (all loads)**")
                                fig_tot = go.Figure()
                                fig_tot.add_scatter(x=P_total.index, y=P_total.values, mode="lines")
                                fig_tot.update_layout(
                                    height=180,
                                    margin=dict(l=10, r=10, t=8, b=8),
                                    xaxis_title="Time",
                                    yaxis_title="kW",
                                    showlegend=False,
                                )
                                st.plotly_chart(fig_tot, use_container_width=True)
                            else:
                                # no enabled loads → zero profile
                                P_total = pd.Series(0.0, index=idx_day, name="P_leisure_kW")

                            # main profile for this device (used by rest of the app)
                            cfg["profile_index"] = P_total.index.astype(str).tolist()
                            cfg["profile_kw"]    = P_total.values.tolist()

                            if st.button("▲ Hide", key=f"{settings_id}_hide_leisure"):
                                st.session_state[open_key] = False
                                st.rerun()

                    # =======================================================
                    # EV CHARGING (special editor under outside)
                    # =======================================================
                    elif cat_key == "outside" and dev_type == "ev11":
                        with st.container():
                            st.markdown(
                                "<hr style='border-top: 1px dotted #bfdbfe; margin:0.4rem 0;'/>",
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
                            energy_need = delta_soc * cfg["capacity_kwh"]  # kWh
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
                                    value=float(cfg.get("w_cost", 1)),
                                    key=f"{settings_id}_ev_w_cost",
                                )
                            )

                            # Single interval in cfg
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
                            if st.button(
                                "💡 Suggest cheapest/cleanest in 01:00–06:00",
                                key=f"{settings_id}_ev_suggest",
                            ):
                                if duration_min <= 0:
                                    st.warning("No energy needed (arrival SOC ≥ target SOC).")
                                else:
                                    interval = suggest_best_interval_for_ev(
                                        duration_min=duration_min,
                                        w_cost=cfg["w_cost"],
                                        window_start_min=60,   # 01:00
                                        window_end_min=360,    # 06:00
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
                            cfg["profile_kw"] = prof_ev.values.tolist()

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
                                st.session_state[open_key] = False
                                st.rerun()

                    #########################################################
                    # E-BIKE CHARGER (simpler EV)
                    # =======================================================
                    elif cat_key == "outside" and dev_type == "ebike":
                        with st.container():
                            st.markdown(
                                "<hr style='border-top: 1px dotted #bbf7d0; margin:0.4rem 0;'/>",
                                unsafe_allow_html=True,
                            )
                            st.markdown("**E-bike charging settings**")

                            # --- Charger & battery (smaller by default) ---
                            c_p, c_cap = st.columns(2)
                            cfg["power_kw"] = c_p.number_input(
                                "Charger power (kW)",
                                min_value=0.1,
                                max_value=5.0,
                                step=0.1,
                                value=float(cfg.get("power_kw", 0.5)),   # e-bike: 500 W default
                                key=f"{settings_id}_ebike_power",
                            )
                            cfg["capacity_kwh"] = c_cap.number_input(
                                "Battery capacity (kWh)",
                                min_value=0.2,
                                max_value=5.0,
                                step=0.1,
                                value=float(cfg.get("capacity_kwh", 1.0)),  # e-bike: ~1 kWh
                                key=f"{settings_id}_ebike_cap",
                            )

                            c_soc_a, c_soc_t = st.columns(2)
                            cfg["soc_arrive"] = c_soc_a.number_input(
                                "Arrival SOC (%)",
                                min_value=0.0,
                                max_value=100.0,
                                step=5.0,
                                value=float(cfg.get("soc_arrive", 40.0)),
                                key=f"{settings_id}_ebike_soc_a",
                            )
                            cfg["soc_target"] = c_soc_t.number_input(
                                "Target SOC at departure (%)",
                                min_value=0.0,
                                max_value=100.0,
                                step=5.0,
                                value=float(cfg.get("soc_target", 100.0)),
                                key=f"{settings_id}_ebike_soc_t",
                            )
                            if cfg["soc_target"] < cfg["soc_arrive"]:
                                cfg["soc_target"] = cfg["soc_arrive"]

                            # Needed energy and duration (minutes)
                            delta_soc   = max(cfg["soc_target"] - cfg["soc_arrive"], 0.0) / 100.0
                            energy_need = delta_soc * cfg["capacity_kwh"]  # kWh
                            if cfg["power_kw"] > 0 and energy_need > 0:
                                duration_min = int(np.ceil(energy_need * 60.0 / cfg["power_kw"]))
                            else:
                                duration_min = 0
                            cfg["duration_min"] = duration_min

                            st.caption(
                                f"Energy needed ≈ **{energy_need:.1f} kWh**, "
                                f"charging time ≈ **{duration_min} min** at {cfg['power_kw']:.1f} kW."
                            )

                            # Cost/CO2 preference (same as EV)
                            cfg["w_cost"] = float(
                                st.slider(
                                    "Preference (0 = CO₂ only, 1 = cost only)",
                                    min_value=0.0,
                                    max_value=1.0,
                                    step=0.05,
                                    value=float(cfg.get("w_cost", 1)),
                                    key=f"{settings_id}_ebike_w_cost",
                                )
                            )

                            # Single interval in cfg
                            # For e-bike I assume an evening window by default (19:00–23:00).
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
                                    "Start",
                                    value=current_iv.get("start", _time(1, 0)),
                                    key=f"{settings_id}_ebike_start",
                                )

                            # recompute end from duration
                            if duration_min > 0:
                                dt0 = datetime.combine(date.today(), start_time)
                                dt1 = dt0 + timedelta(minutes=duration_min)
                                end_time = dt1.time()
                            else:
                                end_time = current_iv.get("end", _time(6, 0))

                            with c_b:
                                st.time_input(
                                    "End (computed)",
                                    value=end_time,
                                    key=f"{settings_id}_ebike_end_display",
                                    disabled=True,
                                )

                            current_iv["start"], current_iv["end"] = start_time, end_time

                            # Optional: reuse same suggestion function as car EV,
                            # but with a different window (e.g. 17:00–23:00).
                            if st.button(
                                "💡 Suggest cheapest/cleanest this evening (1:00–6:00)",
                                key=f"{settings_id}_ebike_suggest",
                            ):
                                if duration_min <= 0:
                                    st.warning("No energy needed (arrival SOC ≥ target SOC).")
                                else:
                                    interval = suggest_best_interval_for_ev(
                                        duration_min=duration_min,
                                        w_cost=cfg["w_cost"],
                                        window_start_min=1* 60,  # 17:00
                                        window_end_min=6 * 60,    # 23:00
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
                            st.markdown("**Daily E-bike charging profile (preview)**")
                            prof_eb = build_minute_profile(
                                power_w=cfg["power_kw"] * 1000.0,
                                intervals=intervals,
                                step_min=1,
                            )
                            cfg["profile_index"] = prof_eb.index.astype(str).tolist()
                            cfg["profile_kw"] = prof_eb.values.tolist()

                            fig_eb = go.Figure()
                            fig_eb.add_scatter(
                                x=prof_eb.index,
                                y=prof_eb.values,
                                mode="lines",
                                name="P_Ebike_kW",
                            )
                            fig_eb.update_layout(
                                height=180,
                                margin=dict(l=10, r=10, t=10, b=8),
                                xaxis_title="Time",
                                yaxis_title="kW",
                                showlegend=False,
                            )
                            st.plotly_chart(fig_eb, use_container_width=True)

                            if st.button("▲ Hide details", key=f"{settings_id}_hide_ebike"):
                                st.session_state[open_key] = False
                                st.rerun()

                    #########################################################
                    # =======================================================
                    # PV SYSTEM – simple generation preview
                    # =======================================================
                    elif cat_key == "gen_store":
                        if dev_type == "pv":
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
                                max_value=2000,
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

                            # --- Build 1-minute index for the selected day ---
                            from datetime import timedelta as _td

                            sel_day = st.session_state.get("day")
                            if sel_day is not None:
                                day_start = pd.Timestamp(sel_day)
                                day_end   = day_start + _td(days=1)
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

                            # Store profile in cfg (kW, 1-min)
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
                                name="P_PV_kW",
                            )
                            fig_pv.update_layout(
                                height=180,
                                margin=dict(l=10, r=10, t=10, b=8),
                                xaxis_title="Time",
                                yaxis_title="kW",
                                showlegend=False,
                            )
                            st.plotly_chart(fig_pv, use_container_width=True, key=f"{settings_id}_pv_fig")

                            # Hide button
                            if st.button("▲ Hide details", key=f"{settings_id}_hide_pv"):
                                st.session_state[open_key] = False
                                st.rerun()

                        ###
                        else:
                            with st.container():
                                    st.markdown(
                                        "🚧 **Model not implemented yet**  \n"
                                        "This component will be supported in a future "
                                        "version of the app. For now, it does not "
                                        "affect your daily load or PV coverage.",
                                    )
                                    if st.button("▲ Hide", key=f"{settings_id}_hide_future"):
                                        st.session_state[open_key] = False
                                        st.rerun()
                                    st.markdown("</div>", unsafe_allow_html=True)


                    # =======================================================
                    # B) Generic simple editor for all other categories
                    # =======================================================
                    else:
                        with st.container():
                            st.markdown(
                                "<div style='border:1px solid #ddd;border-radius:4px;padding:0.5rem;'>",
                                unsafe_allow_html=True,
                            )

                            c_p, c_s, c_d = st.columns(3)
                            cfg["power_kw"] = c_p.number_input(
                                "Power (kW)",
                                min_value=0.0,
                                max_value=50.0,
                                step=0.1,
                                value=float(cfg.get("power_kw", 0.5)),
                                key=f"p_{full_key}",
                            )
                            cfg["start"] = c_s.time_input(
                                "Start",
                                value=cfg.get("start", _time(18, 0)),
                                key=f"start_{full_key}",
                            )
                            cfg["duration_min"] = int(
                                c_d.number_input(
                                    "Duration (min)",
                                    min_value=0,
                                    max_value=1440,
                                    step=15,
                                    value=int(cfg.get("duration_min", 60)),
                                    key=f"dur_{full_key}",
                                )
                            )

                            st.caption(
                                "These are simplified settings. "
                                "You can later link them to detailed models (HP, DHW, EV, etc.)."
                            )

                            if st.button("▲ Hide", key=f"hide_{full_key}"):
                                st.session_state[open_key] = False
                                st.rerun()

                            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("")  # gap under category




    # Render all categories
    render_category("elec_fixed")
    render_category("elec_flex")
    render_category("thermal")
    render_category("outside")
    render_category("gen_store")

    # ---- build and show house layout with devices ----
    layout_fig = build_house_layout_figure(sel, cfgs)
    layout_placeholder.plotly_chart(layout_fig, use_container_width=True, config={"responsive": True})


    # ---- build daily profiles for all selected devices ----
    idx, device_traces, total = compute_daily_profiles(sel, cfgs)

    figp = go.Figure()

    # ---------------------------------------------------
    # 1) Add all LOAD traces (everything except PV)
    # ---------------------------------------------------
    for full_key, series in device_traces.items():
        if series.max() <= 0:
            continue

        cat_key, dev_type = full_key.split(":", 1)

        # skip PV here – we add it separately as shaded area
        if cat_key == "gen_store" and dev_type == "pv":
            continue

        label = DEVICE_LABEL_MAP.get(full_key, full_key)
        cfg = cfgs.get(full_key, {})
        num = int(cfg.get("num_devices", 1))
        if num > 1 and not full_key.startswith("other"):
            label = f"{label} (x{num})"

        figp.add_scatter(
            x=idx,
            y=series.values,
            mode="lines",
            name=label,
        )

    # ---------------------------------------------------
    # 2) Add PV as a SHADED AREA (generation)
    # ---------------------------------------------------
    for full_key, series in device_traces.items():
        cat_key, dev_type = full_key.split(":", 1)
        if not (cat_key == "gen_store" and dev_type == "pv"):
            continue
        if series.max() <= 0:
            continue

        label = DEVICE_LABEL_MAP.get(full_key, "PV generation")

        figp.add_scatter(
            x=idx,
            y=series.values,
            mode="lines",
            name=label,
            fill="tozeroy",      # ✅ fill area to zero
            line=dict(width=1),  # thin line; area does the visual work
        )

    # ---------------------------------------------------
    # 3) Add TOTAL LOAD (consumption only, PV excluded in compute_daily_profiles)
    # ---------------------------------------------------
    figp.add_scatter(
        x=idx,
        y=total.values,
        mode="lines",
        name="Total load",
        line=dict(width=3),
    )

    # ---------------------------------------------------
    # 4) Layout – legend UNDER the figure
    # ---------------------------------------------------
    figp.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=80),  # extra bottom space for legend
        xaxis_title="Time of day",
        yaxis_title="kW",
        legend=dict(
            orientation="h",         # horizontal legend
            yanchor="top",
            y=-0.8,                 # move below plot area
            xanchor="center",
            x=0.5,
            itemclick="toggleothers",    # keep your behaviour
            itemdoubleclick="toggle",
        ),
    )


    # update the placeholder that we created in top_right
    profile_placeholder.plotly_chart(figp, use_container_width=True)

    # write back to session
    st.session_state["device_selection"] = sel
    st.session_state["device_configs"] = cfgs


                

#%% -------------------for page 3------------------------------------

def _norm01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = float(np.nanmin(s.values)), float(np.nanmax(s.values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        # flat if constant/missing
        return pd.Series(0.5, index=s.index, dtype=float)
    out = (s - lo) / (hi - lo)
    return out.reindex(s.index).interpolate().bfill().ffill().astype(float)


def _ts_layout(title: str, ytitle: str = "kW", height: int = 280):
    return dict(
        title=dict(
            text=title,
            x=0.01, xanchor="left",
            y=0.98, yanchor="top",
            pad=dict(t=2, b=0, l=0, r=0)   # <- extra space under the title
        ),
        hovermode="x unified",
        height=height,
        margin=dict(l=36, r=16, t=64, b=32),     # <- larger top margin
        xaxis=dict(
            title="Time",
            type="date",
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(title=ytitle),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0)
    )

def ems_power_split_plot(idx, load, pv, grid_import, pbat):
    """Stack-like view for EMS split."""
    fig = go.Figure()
    fig.add_scatter(x=idx, y=load, name="Load (kW)", mode="lines")
    fig.add_scatter(x=idx, y=pv,   name="PV (kW)",   mode="lines")
    fig.add_scatter(x=idx, y=grid_import, name="Grid import (kW)", mode="lines")
    fig.add_scatter(x=idx, y=pbat, name="Battery power (kW, +dis/-ch)", mode="lines")
    fig.update_layout(_ts_layout("Power split (EMS)", ytitle="kW", height=320))
    return fig

def ems_soc_plot(idx, soc_kwh, cap_kwh):
    soc_pct = (soc_kwh / max(cap_kwh, 1e-9)) * 100.0
    fig = go.Figure()
    fig.add_scatter(x=idx, y=soc_pct, name="SOC (%)", mode="lines")
    fig.update_layout(_ts_layout("Battery SOC", ytitle="%", height=220))
    return fig
#%% -------- SIDEBAR --------
st.set_page_config(page_title="Daily EMS Sandbox", layout="wide")

with st.sidebar:
    st.header("Daily EMS Sandbox")
    
    st.header("Navigation")
    page = st.radio(
        "Step",
        ["1️⃣ Scenario & data", "2️⃣ Devices & layout", "3️⃣ Analysis"],
        key="page",
    )

    st.header("House Input")
    if "house_info" not in st.session_state:
        st.session_state["house_info"] = {
            "location": "Aalborg",
            "size": "Medium house",
            "insulation": "Average",
            "residents": 2,
        }

    hi = st.session_state["house_info"]

    # --- House size ---
    house_size_options = {
        "Small apartment": "40–80 m²,  1–3 rooms",
        "Medium house": "90–150 m²,  3–5 rooms",
        "Large house": "160–250 m², 5+ rooms",
    }

    hi["size"] = st.radio(
        "House size (choose what feels closest)",
        options=list(house_size_options.keys()),
        index=["Small apartment", "Medium house", "Large house"].index(hi["size"]),
    )
    st.caption(f"**{house_size_options[hi['size']]}**")

    # --- Insulation ---
    insulation_desc = {
        "Poor": "Old house (pre-1980), thin walls",
        "Average": "Typical house (1980–2010), standard",
        "Good": "New or renovated house, low heat loss",
    }

    hi["insulation"] = st.radio(
        "Insulation quality",
        list(insulation_desc.keys()),
        horizontal=True,
        index=["Poor", "Average", "Good"].index(hi["insulation"]),
    )
    st.caption(f"**{insulation_desc[hi['insulation']]}**")

    # --- Residents ---
    hi["residents"] = st.number_input(
        "How many people live here?",
        min_value=1, max_value=8, step=1,
        value=int(hi["residents"]),
        help="Used to estimate hot-water usage and device patterns.",
    )

    st.session_state["house_info"] = hi


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
        st.plotly_chart(fig_price, width='stretch')
        if note_price:
            st.caption(f"ℹ️ {note_price}")

        # CO₂ (bar version)
        fig_co2 = plot_period_bar(
            co2_series,
            selected_day=selected_day,
            title="CO₂ intensity over selected period",
            ytitle="gCO₂/kWh",
        )
        st.plotly_chart(fig_co2, width='stretch')
        if note_co2:
            st.caption(f"ℹ️ {note_co2}")

        # Temperature plot
        fig_temp = plot_period_minute(
            temp_series,
            selected_day=selected_day,
            title="Outdoor temperature over selected period",
            ytitle="°C",
        )
        st.plotly_chart(fig_temp, width='stretch')
        if note_temp:
            st.caption(f"ℹ️ {note_temp}")



#%% page 2
elif page.startswith("2️⃣"):
    st.title("2️⃣ Devices & layout")
    st.info("Here we will add different kinds of devices for household")
    
    st.markdown("## 2. Devices & House layout")
    render_devices_page_house()

    




#%% page 3
elif page.startswith("3️⃣"):
    st.title("3️⃣ Analysis")
    st.info("Here we will evaluate daily household consumption plots, KPIs, and logging.")


    sel  = st.session_state.get("device_selection", {})
    cfgs = st.session_state.get("device_configs", {})

    if not sel or not any(sel.values()):
        st.warning("Please select at least one device on page 2 first.")
        st.stop()

    # ---------- 1) Build load / PV series ----------
    idx, load_tot, pv_tot, energy_per_device = build_series_for_analysis(sel, cfgs)

    # Grid import = load minus self-consumed PV (no battery yet)
    pv_self_kw  = np.minimum(load_tot, pv_tot)
    grid_import = load_tot - pv_self_kw
    pv_export   = pv_tot - pv_self_kw
    pv_export[pv_export < 0] = 0.0

    # kWh integrals (1-min → divide by 60)
    E_load   = float(load_tot.sum()   / 60.0)
    E_pv     = float(pv_tot.sum()     / 60.0)
    E_self   = float(pv_self_kw.sum() / 60.0)
    E_grid   = float(grid_import.sum() / 60.0)
    E_export = float(pv_export.sum()   / 60.0)

    pv_cov = (E_self / E_load * 100.0) if E_load > 0 else 0.0

    # ---------- 2) Optional cost / CO₂ ----------
    price_series = st.session_state.get("price_daily")  # DKK/kWh
    co2_series   = st.session_state.get("co2_daily")    # g/kWh

    total_cost     = None
    avg_price      = None
    cost_no_pv     = None
    cost_saved_pv  = None

    total_co2_grid_kg  = None
    co2_no_pv_kg       = None
    co2_saved_by_pv_kg = None

    # ---- PRICE (DKK/kWh) ----
    if isinstance(price_series, (pd.Series, pd.DataFrame)):
        if isinstance(price_series, pd.DataFrame):
            price_series = price_series.iloc[:, 0]
        p = price_series.reindex(idx, method="nearest").fillna(0.0)  # DKK/kWh

        # with PV: only grid_import comes from grid
        total_cost = float((grid_import * p).sum() / 60.0)  # DKK
        tot_grid_kwh = E_grid if E_grid > 0 else 1e-9
        avg_price = total_cost / tot_grid_kwh

        # baseline: no PV → all load from grid
        cost_no_pv = float((load_tot * p).sum() / 60.0)
        cost_saved_pv = cost_no_pv - total_cost

    # ---- CO2 (g/kWh → kg/kWh) ----
    if isinstance(co2_series, (pd.Series, pd.DataFrame)):
        if isinstance(co2_series, pd.DataFrame):
            co2_series = co2_series.iloc[:, 0]
        c_g = co2_series.reindex(idx, method="nearest").fillna(0.0)  # g/kWh
        c_kg = c_g / 1000.0  # kg/kWh

        # with PV: emissions only for grid_import
        total_co2_grid_kg = float((grid_import * c_kg).sum() / 60.0)

        # baseline: no PV (all load is from grid)
        co2_no_pv_kg = float((load_tot * c_kg).sum() / 60.0)

        co2_saved_by_pv_kg = co2_no_pv_kg - total_co2_grid_kg



    # ---------- 3) KPI header ----------
    col1, col2, col3 = st.columns(3)
    col1.metric("Total consumption", f"{E_load:.1f} kWh")
    col2.metric("PV generation", f"{E_pv:.1f} kWh")
    col3.metric("PV coverage", f"{pv_cov:.0f} %")

    col4, col5, col6 = st.columns(3)
    col4.metric("Grid import", f"{E_grid:.1f} kWh")
    col5.metric("PV export", f"{E_export:.1f} kWh")
    if total_cost is not None and avg_price is not None:
        col6.metric(
            "Cost (grid)",
            f"{total_cost:.2f} DKK",
        )
    else:
        col6.metric("Cost (grid)", "n/a")

    # ---- Second row: CO2 KPIs ----
    if total_co2_grid_kg is not None:
        c7, c8, c9= st.columns(3)
        c7.metric("CO₂ from grid", f"{total_co2_grid_kg:.1f} kg")
        if co2_saved_by_pv_kg is not None:
            c8.metric("CO₂ avoided by PV", f"{co2_saved_by_pv_kg:.1f} kg")


    # ---------- 4) Time-series plot ----------
    st.markdown("#### Power over the day")

    fig_ts = go.Figure()
    fig_ts.add_scatter(
        x=idx, y=load_tot.values,
        mode="lines",
        name="Load (kW)",
    )
    if pv_tot.max() > 0:
        # PV plotted below zero with filled area → clear visual separation
        fig_ts.add_scatter(
            x=idx,
            y=-pv_tot.values,
            mode="lines",
            name="PV generation (kW)",
            fill="tozeroy",
        )
    fig_ts.add_scatter(
        x=idx, y=grid_import.values,
        mode="lines",
        name="Grid import (kW)",
    )

    fig_ts.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=70),
        xaxis_title="Time of day",
        yaxis_title="kW (PV shown below 0)",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5,
            itemclick="toggleothers",    # keep your behaviour
            itemdoubleclick="toggle",
        ),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # ---------- 5) Load breakdown (pie by category) ----------
    st.markdown("#### Consumption breakdown by category")

    cat_energy = {
        "Fixed electrical": 0.0,
        "Flexible electrical": 0.0,
        "Thermal (HP / DHW / leisure)": 0.0,
        "EV charging": 0.0,
        "EVs": 0.0,
    }

    for full_key, E_kwh in energy_per_device.items():
        cat_key, dev_type = full_key.split(":", 1)
        if cat_key == "elec_fixed":
            cat_energy["Fixed electrical"] += E_kwh
        elif cat_key == "elec_flex":
            if dev_type in ("ev11", "ebike"):
                cat_energy["EV charging"] += E_kwh
            else:
                cat_energy["Flexible electrical"] += E_kwh
        elif cat_key == "thermal":
            cat_energy["Thermal (HP / DHW / leisure)"] += E_kwh
        elif cat_key == "outside":
            cat_energy["EVs"] += E_kwh
        # gen_store (PV) is generation → not part of consumption pie

    labels = []
    values = []
    for k, v in cat_energy.items():
        if v > 0.01:
            labels.append(k)
            values.append(v)

    if values:
        fig_pie = go.Figure(
            data=[go.Pie(labels=labels, values=values, hole=0.35)]
        )
        fig_pie.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=40),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5,
            ),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.caption("No non-zero loads to show in the breakdown.")

    st.markdown("---")

    # ---------- 6) Download data ----------
    st.markdown("#### Download results")

    df_out = pd.DataFrame(
        {
            "P_load_kW": load_tot.values,
            "P_pv_kW": pv_tot.values,
            "P_grid_import_kW": grid_import.values,
            "P_pv_export_kW": pv_export.values,
        },
        index=idx,
    )

    if isinstance(price_series, (pd.Series, pd.DataFrame)):
        if isinstance(price_series, pd.DataFrame):
            price_series = price_series.iloc[:, 0]
        df_out["Price_grid"] = price_series.reindex(idx, method="nearest").values

    if isinstance(co2_series, (pd.Series, pd.DataFrame)):
        if isinstance(co2_series, pd.DataFrame):
            co2_series = co2_series.iloc[:, 0]
        df_out["CO2_intensity"] = co2_series.reindex(idx, method="nearest").values

    kpi_summary = {
        "E_load_kWh": E_load,
        "E_pv_kWh": E_pv,
        "E_self_kWh": E_self,
        "E_grid_kWh": E_grid,
        "E_export_kWh": E_export,
        "PV_coverage_%": pv_cov,
        "Cost_DKK_with_PV": total_cost,
        "Cost_DKK_no_PV": cost_no_pv,
        "Cost_saved_by_PV_DKK": cost_saved_pv,
        "Avg_price_DKK_per_kWh": avg_price,
        "CO2_grid_kg": total_co2_grid_kg,
        "CO2_no_PV_kg": co2_no_pv_kg,
        "CO2_saved_by_PV_kg": co2_saved_by_pv_kg,
    }

    kpi_rows = pd.DataFrame([kpi_summary])
    kpi_rows.index = ["SUMMARY"]

    df_to_save = pd.concat(
        [
            kpi_rows,
            pd.DataFrame(index=[""]),  # blank separator row
            df_out,
        ]
    )

    csv_bytes = df_to_save.to_csv().encode("utf-8")
    st.download_button(
        label="📥 Download CSV (profiles + summary)",
        data=csv_bytes,
        file_name="house_profile_analysis.csv",
        mime="text/csv",
    )



def future_ems():
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


    use_batt = st.checkbox("Use battery in EMS", value=True, key="use_batt_analysis")
    plan_df = None
    control_mode = None
    if use_batt: 
        # This is your existing panel, now moved here:
        control_mode = render_battery_settings_panel()
    else:
        st.info("Battery is disabled for this analysis. ")

    with st.expander("Multi-objective weights (Cost–CO₂–Comfort)", expanded=True):
        w_cost= float(
            st.slider(
                "Preference (0 = CO₂ only, 1 = cost only)",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                value=0.5,
            )
        )

    price_min = price_day 
    co2_min   = co2_day
    price_n = _norm01(price_min)
    co2_n   = _norm01(co2_min)
    signal = (w_cost * co2_n + (1-w_cost) * price_n).rename("device_objective")

    run_ems = st.button("Run EMS",  key="run_ems")




    # --- 1) AUTO mode: button to compute suggested time slots ---
    if run_ems:
        if use_batt and control_mode == "Auto":
            df_minute = (
                pd.DataFrame({
                    "DateTime": idx,                      # datetime index created earlier
                    "Load":     load_day.values,          # kW, minute-cadence
                    "PV":       pv_day.values,            # kW
                    "ElectricityPrice": price_day.values, # DKK/kWh
                    "signal":   signal.values,            # 0..1 signal
                    "co2":      co2_day.values,           # g/kWh
                })
                .sort_values("DateTime")
            )
            time_slots = generate_smart_time_slots(df_minute)
            df_slots   = assign_data_to_time_slots_single(df_minute, time_slots)
            SOC0 = float(st.session_state["batt_soc_pct"])
            SOC_min = float(st.session_state.get("soc_min_pct", 15.0))   # <- user input
            SOC_max = 100.0
            SOC_opt, Qgrid,_ = mpc_opt_single(
                df_slots, SOC0=SOC0, SOC_min=SOC_min, SOC_max=SOC_max,
                Pbat_chargemax=st.session_state["batt_pow"],
                Qbat=st.session_state["batt_cap"],
            )
            df_plan = format_results_single(SOC_opt, Qgrid, df_slots)
            df_today = df_plan[df_plan["Datetime"] == df_plan["Datetime"].iloc[0]].copy()
            df_today["start"] = pd.to_datetime(df_today["TimeSlot"].str.split(" - ").str[0], format="%H:%M").dt.time
            df_today["end"]   = pd.to_datetime(df_today["TimeSlot"].str.split(" - ").str[1], format="%H:%M").dt.time
            df_today["soc_setpoint_pct"]    = df_today["SOC"]
            df_today["grid_charge_allowed"] = df_today["Grid_Charge"]
            plan_df = df_today[["start","end","soc_setpoint_pct","grid_charge_allowed"]]    
            
            st.session_state["ems_plan"] = plan_df.copy()
            ems_out = rule_power_share(
                idx=price_day.index, load_kw=load_day, pv_kw=pv_day,
                plan_slots=plan_df,
                cap_kwh=st.session_state["batt_cap"],
                p_max_kw=st.session_state["batt_pow"],
                soc0_kwh=(st.session_state["batt_soc_pct"] / 100.0) * st.session_state["batt_cap"],
                eta_ch=0.95, eta_dis=0.95,
                energy_pattern=2,
            )
            grid = ems_out["grid_import_kw"]
            pbat = ems_out["batt_discharge_kw"] - ems_out["batt_charge_kw"]  # +dis, -ch
            soc  = ems_out["batt_soc_kwh"]
            
            # ---- Selected time slots & settings (table) ----
            plan_df = st.session_state.get("ems_plan")
            if plan_df is not None and not plan_df.empty:
                # clean + enrich for display
                disp = plan_df.copy()
                # ensure proper dtypes
                disp["start"] = pd.to_datetime(disp["start"].astype(str)).dt.time
                disp["end"]   = pd.to_datetime(disp["end"].astype(str)).dt.time
                # sortable helper for ordering
                disp["_start_sort"] = pd.to_datetime(disp["start"].astype(str), format="%H:%M:%S")
                disp = disp.sort_values("_start_sort").drop(columns=["_start_sort"]).reset_index(drop=True)
                disp.index = np.arange(1, len(disp) + 1)  # Slot numbering

                # Compute duration (min) for each slot on the chosen day
                day0 = pd.Timestamp(price_day.index[0]).normalize()

                s_ts = pd.to_datetime(disp["start"].astype(str).radd(str(day0.date()) + " "))
                e_ts = pd.to_datetime(disp["end"].astype(str).radd(str(day0.date()) + " "))

                # handle wrap-around (end earlier than start means it crosses midnight)
                e_ts = e_ts.mask(e_ts <= s_ts, e_ts + pd.Timedelta(days=1))

                # minutes as integers
                disp["Duration (min)"] = ((e_ts - s_ts) / pd.Timedelta(minutes=1)).astype(int)
                # alternatively:
                # disp["Duration (min)"] = ((e_ts - s_ts).dt.total_seconds() / 60).astype(int)


                st.subheader("Selected battery time slots & settings")
                st.dataframe(
                    disp.style.format({"SOC target (%)": "{:.0f}", "Duration (min)": "{:d}"}),
                    width='stretch'
                )

            st.plotly_chart(ems_power_split_plot(price_day.index, load_day, pv_day, grid, pbat), width='stretch')
            st.plotly_chart(ems_soc_plot(price_day.index, soc, st.session_state["batt_cap"]), width='stretch')
    return e_ts




            
