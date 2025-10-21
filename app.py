from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, time
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from profiles import minute_index, default_price_profile, default_co2_profile, simple_pv_profile, synthetic_outdoor_temp
from devices import (
    FixedDevice, CycledWindowDevice, CyclingDevice, BaseloadSpec,
    WashingMachine, Dishwasher, Dryer, RangeHood, EVCharger, WeatherHP
)
from ems import rule_power_share
from Optimization_based import generate_smart_time_slots, assign_data_to_time_slots_single, mpc_opt_single, format_results_single
from report import save_csv, save_plots, save_summary_md
import pvlib
from pvlib.location import Location
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
# export/report helper imports
import json, os, sys, subprocess
from datetime import datetime


st.set_page_config(page_title="Daily EMS Sandbox", layout="wide")
st.title("🏠 Daily EMS Sandbox")
#----------------------------------------------------------------------------------------------------
# --- at the top, after imports ---
if "sim" not in st.session_state:  # stores load/env results
    st.session_state.sim = None
if "ems" not in st.session_state:  # stores EMS split results
    st.session_state.ems = None
    
if "scheduled_notes" not in st.session_state:
    st.session_state["scheduled_notes"] = []

def note(name, start_idx, duration_min, idx):
    if start_idx is None:
        st.session_state["scheduled_notes"].append((name, "not scheduled", int(duration_min)))
    else:
        st.session_state["scheduled_notes"].append(
            (name, idx[start_idx].strftime("%H:%M"), int(duration_min))
        )


# --- helper functions you already have (sketch) ---
def compute_simulation(day, step_min, controls) -> dict:
    # build idx, load_parts, load, pv, price, co2, tout_minute, notes, parts_df, etc.
    # DO NOT plot here—just return data needed to plot later
    return {
        "idx": idx,
        "dt_h": dt_h,
        "load": load,
        "pv": pv,
        "price": price,
        "co2": co2,
        "tout": tout_minute,
        "parts_df": parts_df,
        "notes": {"temp": note_temp, "price": note_price, "co2": note_co2},
    }

def render_sim(sim):
    idx, load, pv = sim["idx"], sim["load"], sim["pv"]
    # Load & PV
    fig_lpg = go.Figure()
    fig_lpg.add_scatter(x=idx, y=load, name="Load (kW)", mode="lines")
    fig_lpg.add_scatter(x=idx, y=pv,   name="PV (kW)",   mode="lines")
    fig_lpg.update_layout(_ts_layout("Load vs PV"))
    st.plotly_chart(fig_lpg, use_container_width=True, key="sim_load_pv")

    # Per-device
    parts_df = sim["parts_df"]
    if parts_df is not None and not parts_df.empty:
        fig_parts = go.Figure()
        for c in parts_df.columns:
            fig_parts.add_scatter(x=parts_df.index, y=parts_df[c], name=c, mode="lines")
        fig_parts.update_layout(_ts_layout("Per-device power"))
        st.plotly_chart(fig_parts, use_container_width=True, key="sim_parts")

    # Environment
    fig_temp = go.Figure()
    fig_temp.add_scatter(x=idx, y=sim["tout"], name="Outdoor Temp (°C)", mode="lines")
    fig_temp.update_layout(_ts_layout("Outdoor Temperature", ytitle="°C"))
    st.plotly_chart(fig_temp, use_container_width=True, key="sim_temp")

    fig_price = go.Figure()
    fig_price.add_scatter(x=idx, y=sim["price"], name="Price (DKK/kWh)", mode="lines")
    fig_price.update_layout(_ts_layout("Electricity Price", ytitle="DKK/kWh"))
    st.plotly_chart(fig_price, use_container_width=True, key="sim_price")

    fig_co2 = go.Figure()
    fig_co2.add_scatter(x=idx, y=sim["co2"], name="CO₂ (g/kWh)", mode="lines")
    fig_co2.update_layout(_ts_layout("CO₂ Intensity", ytitle="gCO₂/kWh"))
    st.plotly_chart(fig_co2, use_container_width=True, key="sim_co2")

def run_ems_on_sim(sim, plan_df_or_mode) -> dict:
    # If manual plan: build per-minute targets → call rule_power_share(...)
    # If auto plan: make hourly df → slots → optimizer → plan_df → rule_power_share(...)
    grid, pbat, pv_used, soc = rule_power_share(
        load_kw=sim["load"], pv_kw=sim["pv"],
        soc_target_pct=soc_series, grid_charge_flag=gc_series,
        cap_kwh=cap_kwh, p_charge_max_kw=p_kw,
        soc0_kwh=soc0_kwh, energy_pattern=energy_pattern,
        eta_ch=0.95, eta_dis=0.95
    )
    return {"grid": grid, "pbat": pbat, "pv_used": pv_used, "soc": soc}

def render_ems(sim, ems):
    idx = sim["idx"]
    load, pv = sim["load"], sim["pv"]
    grid, pbat, soc = ems["grid"], ems["pbat"], ems["soc"]

    # Power split
    fig_split = go.Figure()
    fig_split.add_scatter(x=idx, y=load, name="Load (kW)", mode="lines")
    fig_split.add_scatter(x=idx, y=pv,   name="PV (kW)", mode="lines")
    fig_split.add_scatter(x=idx, y=grid, name="Grid import (kW)", mode="lines")
    fig_split.add_scatter(x=idx, y=pbat, name="Battery (+dis/-ch kW)", mode="lines")
    fig_split.update_layout(_ts_layout("Power split"))
    st.plotly_chart(fig_split, use_container_width=True, key="ems_split")

    # SOC
    fig_soc = go.Figure()
    fig_soc.add_scatter(x=idx, y=soc, name="Battery SOC (kWh)", mode="lines")
    fig_soc.update_layout(_ts_layout("Battery SOC", ytitle="kWh"))
    st.plotly_chart(fig_soc, use_container_width=True, key="ems_soc")

    # Summary metrics (money, CO2, savings vs baselines)
    dt_h = sim["dt_h"]
    price, co2 = sim["price"], sim["co2"]
    cost = float((price * grid * dt_h).sum())
    co2_kg = float((co2 * grid * dt_h).sum() / 1000.0)
    # baselines:
    grid_only = (sim["load"] * dt_h).sum()  # kWh if no PV/battery (rough)
    grid_with_pv = ((sim["load"] - sim["pv"]).clip(lower=0) * dt_h).sum()
    kwh_with_ems = (grid * dt_h).sum()

    self_consumption = (np.minimum(sim["load"], sim["pv"]) * dt_h).sum() / max((sim["pv"]*dt_h).sum(),1e-9) * 100
    self_sufficiency = (np.minimum(sim["load"], sim["pv"]) * dt_h).sum() / max((sim["load"]*dt_h).sum(),1e-9) * 100

    st.markdown("### EMS Summary")
    st.write(f"- Cost: **{cost:.2f} DKK**")
    st.write(f"- CO₂: **{co2_kg:.2f} kg**")
    st.write(f"- Grid energy (no PV/batt): {grid_only:.2f} kWh")
    st.write(f"- Grid energy (PV only): {grid_with_pv:.2f} kWh")
    st.write(f"- Grid energy (EMS): {kwh_with_ems:.2f} kWh")
    st.write(f"- Self-consumption: **{self_consumption:.1f}%**")
    st.write(f"- Self-sufficiency: **{self_sufficiency:.1f}%**")

#----------about PV-------------------------------------------------------------------------------
# 1) Hourly irradiance + met from Open-Meteo (archive+forecast), like earlier
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
        r = requests.get(url, timeout=30); r.raise_for_status()
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


# 2) Run pvlib POA -> ModelChain (pvwatts) and align to minute index
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



# -------- Helpers ---------------------------------------------------------
def compute_load_env(day, step_min, objective, weather_hr, use_baseload, use_lights, use_hood,
                     use_wm, use_dw, use_dryer, use_ev, use_hp, pv_kwp, idx_preview):
    """Build idx, load components, total load, PV, price, CO2, Tout, etc. No EMS here."""
    idx = minute_index(day, step_min=step_min)
    dt_h = step_min / 60.0

    # signals
    price_plot, price_hourly, note_price = daily_price_dual(idx, day, area="DK1")
    co2,   note_co2   = daily_co2_with_note(idx, day, area="DK1")
    signal = price if objective == "cost" else (co2 if objective == "co2" else price*0.0)

    # temperature
    tout_minute, note_temp = daily_temperature_with_note(idx, weather_hr)

    # PV (from your weather_hr + pvlib)
    pv = pv_from_weather_modelchain_from_df(
        idx_min=idx, dfh=weather_hr,
        lat=float(st.session_state["geo_lat"]),
        lon=float(st.session_state["geo_lon"]),
        kwp=pv_kwp,
        tilt_deg=st.session_state.get("pv_tilt", 30.0),
        az_deg=st.session_state.get("pv_az", 180.0),
        sys_loss_frac=st.session_state.get("pv_loss", 0.14),
    )

    # ---- assemble device loads exactly like you already do ----
    load_parts = []

    # Baseload
    if use_baseload:
        base = BaseloadSpec(
            router_w=st.session_state.get("base_router", 12.0),
            ventilation_w=st.session_state.get("base_vent", 40.0),
            standby_w=st.session_state.get("base_standby", 60.0),
            dhw_recirc_w=st.session_state.get("base_dhw", 0.0),
            fridge_avg_w=st.session_state.get("base_fridge", 45.0),
            other1_w=st.session_state.get("base_o1", 0.0),
            other2_w=st.session_state.get("base_o2", 0.0),
        ).series_kw(idx)
        load_parts.append(base.rename("baseload_kw"))

    # Lights
    if use_lights and st.session_state.get("lights_count", 0) > 0:
        lights = FixedDevice(
            "lights",
            st.session_state["lights_power"],
            st.session_state["lights_count"],
            st.session_state["lights_start"],
            st.session_state["lights_end"],
        ).series_kw(idx)
        load_parts.append(lights)

    # Range hood
    if use_hood:
        rh = RangeHood(
            power_w=float(st.session_state["hood_p"]),
            lunch_start=st.session_state["hood_ls"],
            lunch_duration_min=int(st.session_state["hood_ld"]),
            dinner_start=st.session_state["hood_ds"],
            dinner_duration_min=int(st.session_state["hood_dd"]),
            lunch_enabled=bool(st.session_state["hood_lunch_on"]),
            dinner_enabled=bool(st.session_state["hood_dinner_on"])
        ).series_kw(idx)
        load_parts.append(rh)

    # WM / DW / Dryer / EV (same scheduling as you currently do)
        # ---- WM / DW / Dryer / EV (schedule + append) ----
    # Washing machine
    if use_wm and int(st.session_state.get("wm_dur", 0)) > 0:
        wm = WashingMachine(
            power_w=float(st.session_state["wm_p"]),
            duration_min=int(st.session_state["wm_dur"]),
            window_start=st.session_state["wm_ws"],
            window_end=st.session_state["wm_we"],
        )
        wm_start = build_block(signal, wm.feasible_mask(idx), int(st.session_state["wm_dur"]),
                               st.session_state["wm_sched"], st.session_state.get("wm_manual"), idx)
        load_parts.append(block_or_zero(wm, idx, wm_start).rename("washing_machine_kw"))

    # Dishwasher
    if use_dw and int(st.session_state.get("dw_dur", 0)) > 0:
        dw = Dishwasher(
            power_w=float(st.session_state["dw_p"]),
            duration_min=int(st.session_state["dw_dur"]),
            window_start=st.session_state["dw_ws"],
            window_end=st.session_state["dw_we"],
        )
        dw_start = build_block(signal, dw.feasible_mask(idx), int(st.session_state["dw_dur"]),
                               st.session_state["dw_sched"], st.session_state.get("dw_manual"), idx)
        load_parts.append(block_or_zero(dw, idx, dw_start).rename("dishwasher_kw"))

    # Dryer
    if use_dryer and int(st.session_state.get("dr_dur", 0)) > 0:
        dr = Dryer(
            power_w=float(st.session_state["dr_p"]),
            duration_min=int(st.session_state["dr_dur"]),
            window_start=st.session_state["dr_ws"],
            window_end=st.session_state["dr_we"],
        )
        dr_start = build_block(signal, dr.feasible_mask(idx), int(st.session_state["dr_dur"]),
                               st.session_state["dr_sched"], st.session_state.get("dr_manual"), idx)
        load_parts.append(block_or_zero(dr, idx, dr_start).rename("dryer_kw"))

    # EV (needs dt_h)
    if use_ev and float(st.session_state.get("ev_e", 0)) > 0 and float(st.session_state.get("ev_p", 0)) > 0:
        ev = EVCharger(
            power_kw=float(st.session_state["ev_p"]),
            energy_target_kwh=float(st.session_state["ev_e"]),
            window_start=st.session_state["ev_ws"],
            window_end=st.session_state["ev_we"],
        )
        ev_dur = ev.duration_minutes(dt_h)
        ev_start = build_block(signal, ev.feasible_mask(idx), int(ev_dur),
                               st.session_state["ev_sched"], st.session_state.get("ev_manual"), idx)
        load_parts.append(block_or_zero(ev, idx, ev_start, dt_h).rename("ev_kw"))
    # ---- reuse your existing block scheduling code here (omitted for brevity) ----
    # Append each device's series to load_parts when scheduled.

    # Heat pump
    if use_hp:
        if not weather_hr.empty and "temp" in weather_hr:
            hourly_temp = weather_hr["temp"].rename("Temperature")
            tout = to_minute_series_from_hourly(hourly_temp, idx)
        else:
            tout = pd.Series(0.0, index=idx, name="Tout_C")

        hp = WeatherHP(
            ua_kw_per_c=float(st.session_state["hp_ua"]),
            t_set_c=float(st.session_state["hp_tset"]),
            q_rated_kw=float(st.session_state["hp_qr"]),
            cop_at_7c=float(st.session_state["hp_cop7"]),
            cop_a=float(st.session_state["hp_copa"]) if st.session_state.get("hp_adv_on") else None,
            cop_b=float(st.session_state["hp_copb"]) if st.session_state.get("hp_adv_on") else None,
            cop_min=float(st.session_state.get("hp_copmin", 1.6)),
            cop_max=float(st.session_state.get("hp_copmax", 4.2)),
            defrost=bool(st.session_state.get("hp_def", True)),
        )
        load_parts.append(hp.series_kw(idx, tout_minute))
    else:
        pass  

    # Combine load
    if load_parts:
        parts_df = pd.concat(load_parts, axis=1)
        load = parts_df.sum(axis=1).rename("load_kw")
        parts_df["total_load_kw"] = load
    else:
        parts_df = pd.DataFrame(index=idx)
        load = pd.Series(0.0, index=idx, name="load_kw")

    notes = dict(price=note_price, co2=note_co2, temp=note_temp)
    return dict(idx=idx, dt_h=dt_h, load=load, pv=pv, price=price_plot,price_hourly=price_hourly, co2=co2, tout=tout_minute,
                parts_df=parts_df, notes=notes)

def show_load_env(sim):
    """Plots/tables for the first step."""
    idx = sim["idx"]; load = sim["load"]; pv = sim["pv"]
    price = sim["price"]; co2 = sim["co2"]; tout = sim["tout"]
    parts_df = sim["parts_df"]; notes = sim["notes"]

    # Load & PV
    fig_lpg = go.Figure()
    fig_lpg.add_scatter(x=load.index, y=load, name="Load (kW)", mode="lines")
    fig_lpg.add_scatter(x=pv.index,   y=pv,   name="PV (kW)",   mode="lines")
    fig_lpg.update_layout(_ts_layout("Load & PV", ytitle="kW", height=280))
    st.plotly_chart(fig_lpg, use_container_width=True)

    # Per-device power
    if not parts_df.empty:
        fig_parts = go.Figure()
        for col in parts_df.columns:
            fig_parts.add_scatter(x=parts_df.index, y=parts_df[col], name=col, mode="lines")
        fig_parts.update_layout(_ts_layout("Per-device power (includes total)", ytitle="kW", height=280))
        st.plotly_chart(fig_parts, use_container_width=True)

    # Environment charts
    fig_temp = go.Figure()
    fig_temp.add_scatter(x=tout.index, y=tout, name="Outdoor Temp (°C)", mode="lines")
    fig_temp.update_layout(_ts_layout("Outdoor Temperature", ytitle="°C", height=260))
    st.plotly_chart(fig_temp, use_container_width=True)
    if notes.get("temp"): st.caption(f"ℹ️ {notes['temp']}")

    fig_co2 = go.Figure()
    fig_co2.add_scatter(x=co2.index, y=co2.values, name="CO₂ (g/kWh)", mode="lines")
    fig_co2.update_layout(_ts_layout("CO₂ Intensity", ytitle="gCO₂/kWh", height=260))
    st.plotly_chart(fig_co2, use_container_width=True)
    if notes.get("co2"): st.caption(f"ℹ️ {notes['co2']}")

    fig_price = go.Figure()
    fig_price.add_scatter(x=price.index, y=price, name="Price (DKK/kWh)", mode="lines")
    fig_price.update_layout(_ts_layout("Electricity Price", ytitle="DKK/kWh", height=260))
    st.plotly_chart(fig_price, use_container_width=True)
    if notes.get("price"): st.caption(f"ℹ️ {notes['price']}")

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
def kpi_summary(load, pv, grid, price, co2, dt_h):
    E_grid = float((grid * dt_h).sum())
    E_load = float((load * dt_h).sum())
    E_pv   = float((pv   * dt_h).sum())
    cost   = float((price * grid * dt_h).sum())
    co2_kg = float((co2 * grid * dt_h).sum()) / 1000.0

    sc = (np.minimum(load, pv) * dt_h).sum() / max(E_pv, 1e-9) * 100.0 if E_pv > 1e-9 else 0.0
    ss = (np.minimum(load, pv) * dt_h).sum() / max(E_load, 1e-9) * 100.0 if E_load > 1e-9 else 0.0
    return dict(
        grid_kwh=E_grid, load_kwh=E_load, pv_kwh=E_pv,
        cost_dkk=cost, co2_kg=co2_kg,
        self_consumption_pct=float(sc), self_sufficiency_pct=float(ss)
    )

def compare_scenarios(sim, plan_df, cap_kwh, p_max_kw, soc0_kwh, energy_pattern, eta_ch=0.95, eta_dis=0.95):
    """Build baseline comparisons: (1) No PV & No Batt, (2) PV-only, (3) Batt-only, (4) PV+Batt (actual)."""
    idx = sim["idx"]; dt_h = sim["dt_h"]
    load = sim["load"]; pv = sim["pv"]; price = sim["price"]; co2 = sim["co2"]

    # 1) No PV, No Battery
    grid_base = load.copy()
    kpi_base = kpi_summary(load, pv*0, grid_base, price, co2, dt_h)

    # 2) PV only (no export modeled)
    grid_pv_only = np.clip(load - pv, 0.0, None)
    kpi_pv_only = kpi_summary(load, pv, grid_pv_only, price, co2, dt_h)

    # 3) Battery only (use your rule engine with pv=0 and same plan)
    ems_batt_only = rule_power_share(
        idx=idx, load_kw=load, pv_kw=pv*0,
        plan_slots=plan_df, cap_kwh=cap_kwh, p_max_kw=p_max_kw,
        soc0_kwh=soc0_kwh, energy_pattern=energy_pattern, eta_ch=eta_ch, eta_dis=eta_dis
    )
    grid_batt_only = ems_batt_only["grid_import_kw"]
    kpi_batt_only = kpi_summary(load, pv*0, grid_batt_only, price, co2, dt_h)

    # 4) Actual PV+Battery (caller should provide)
    return kpi_base, kpi_pv_only, kpi_batt_only

def _collapse_quarters_to_hourly(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()].sort_index()
    # If quarter-hourly, take mean per hour; if hourly already, just floor
    minutes = s.index.minute
    is_q = set(minutes.unique()).issubset({0, 15, 30, 45}) and len(minutes.unique()) > 1
    s = s.groupby(s.index.floor("H")).mean() if is_q else s.set_axis(s.index.floor("H"))
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    s.name = s.name or "price_dkk_per_kwh"
    return s


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

def _align_to_day(idx: pd.DatetimeIndex, s: pd.Series) -> pd.Series:
    """Slice/align any series to the selected minute index and interpolate."""
    start, end = idx[0], idx[-1]
    step_min = int((idx[1] - idx[0]).total_seconds() // 60) if len(idx) > 1 else 1
    s = s.loc[(s.index >= start) & (s.index <= end + pd.Timedelta(minutes=step_min))]
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s.reindex(idx)

def load_signal(uploaded, default_func, idx, col_name):
    if uploaded is not None:
        df = pd.read_csv(uploaded, parse_dates=[0])
        df = df.set_index(df.columns[0]).sort_index()
        s = df.iloc[:,0].reindex(idx).interpolate().bfill().ffill()
        s.name = col_name
        return s
    else:
        return default_func(idx)

def to_minute_series_from_csv(file, idx, col_name="Temperature"):
    df = pd.read_csv(file, parse_dates=[0])
    df = df.set_index(df.columns[0]).sort_index()
    if col_name not in df.columns:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                col_name = c; break
    s = df[col_name].astype(float)
    s = s.reindex(idx).interpolate().bfill().ffill()
    return s.rename("Tout_C")

def block_or_zero(device, idx, start_idx, dt_h=None):
    import pandas as pd
    if start_idx is None:
        return pd.Series(0.0, index=idx, name=device.name)
    # EV block_kw may need dt_h; others do not
    try:
        return device.block_kw(idx, start_idx) if dt_h is None else device.block_kw(idx, start_idx, dt_h)
    except TypeError:
        return device.block_kw(idx, start_idx)

def manual_start_index(idx: pd.DatetimeIndex, t) -> int:
    mins = idx.hour*60 + idx.minute
    target = t.hour*60 + t.minute
    return int(np.argmin(np.abs(mins - target)))

def choose_block_start(objective: pd.Series, feasible_mask: pd.Series, duration_min: int) -> int | None:
    obj = objective.values.astype(float)
    feas = feasible_mask.values.astype(bool)
    n = len(obj)
    if duration_min <= 0 or duration_min > n:
        return None
    csum = np.cumsum(np.r_[0.0, obj])
    best_val = np.inf
    best_s = None
    for s in range(n - duration_min + 1):
        if feas[s:s+duration_min].all():
            val = csum[s+duration_min] - csum[s]
            if (best_s is None) or (val < best_val):
                best_val = val
                best_s = s
    return None if best_s is None else int(best_s)


def build_block(obj_series, feas_mask_series, duration_min, sched_mode, manual_time, idx):
    duration_min = int(duration_min)
    if duration_min <= 0:
        return None

    # Manual: use exact time, but only if the full block fits inside the feasible window
    if sched_mode == "Manual fixed start" and manual_time is not None:
        s = manual_start_index(idx, manual_time)
        if s is None:
            return None
        if s + duration_min > len(feas_mask_series):
            return None
        if not feas_mask_series.iloc[s:s+duration_min].all():
            return None
        return s

    # Auto (optimize): choose the minimum-sum objective window that fits
    return choose_block_start(obj_series, feas_mask_series, duration_min)  # may return None

def to_minute_series_from_hourly(hourly_temp: pd.Series, idx_minute: pd.DatetimeIndex) -> pd.Series:
    s = hourly_temp.copy()
    s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()]
    # 🔧 normalize timezone: drop tz so it matches the app's tz-naive minute index
    if getattr(s.index, "tz", None) is not None:
        # If you prefer, tz_convert("Europe/Copenhagen").tz_localize(None)
        s.index = s.index.tz_localize(None)
    # slice to the day and reindex
    start, end = idx_minute[0], idx_minute[-1]
    step_min = int((idx_minute[1] - idx_minute[0]).total_seconds() // 60) if len(idx_minute) > 1 else 1
    s = s.loc[(s.index >= start) & (s.index <= end + pd.Timedelta(minutes=step_min))]
    return s.reindex(idx_minute).interpolate().bfill().ffill().rename("Tout_C")


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


def earliest_fit_start(mask: pd.Series, duration_min: int) -> int | None:
    """Earliest index where `mask` has `duration_min` consecutive True values."""
    a = mask.values.astype(bool)
    run = 0
    for i, ok in enumerate(a):
        if ok:
            run += 1
            if run == duration_min:
                return i - duration_min + 1
        else:
            run = 0
    return None

# -------- Energinet EDS helpers (CO₂ & Elspot prices) --------
EDS_CO2_URL = "https://api.energidataservice.dk/dataset/CO2EmisProg"
EDS_PRICE_URL_OLD = "https://api.energidataservice.dk/dataset/Elspotprices"     # valid up to 2025-09-30
EDS_PRICE_URL_NEW = "https://api.energidataservice.dk/dataset/DayAheadPrices"   # valid from 2025-10-01
_CUTOFF = pd.Timestamp("2025-10-01")



@st.cache_data(ttl=300, show_spinner=False)
def fetch_co2_prog(area: str = "DK1", horizon_hours: int = 48) -> pd.DataFrame:
    """5-min CO₂ intensity prognosis (gCO₂/kWh) as a DataFrame with local naive Time."""
    need = int(horizon_hours) * 12
    limit = max(200, need * 3)
    url = f"{EDS_CO2_URL}?limit={limit}"
    r = requests.get(url, timeout=40); r.raise_for_status()
    recs = r.json().get("records", [])
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(recs)
    req = {"Minutes5UTC","PriceArea","CO2Emission"}
    if not req.issubset(df.columns): return pd.DataFrame()
    df = df[df["PriceArea"] == area]
    if df.empty: return pd.DataFrame()
    t = pd.to_datetime(df["Minutes5UTC"], errors="coerce", utc=True)
    df["Time"] = t.dt.tz_convert("Europe/Copenhagen").dt.tz_localize(None)
    return (df.rename(columns={"CO2Emission":"gCO2_per_kWh"})
              [["Time","PriceArea","gCO2_per_kWh"]]
              .dropna(subset=["Time"])
              .sort_values("Time")
              .reset_index(drop=True))

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_dayahead_prices_latest(area: str = "DK1") -> pd.DataFrame:
    r = requests.get(f"{EDS_PRICE_URL_NEW}?limit=100000", timeout=40); r.raise_for_status()
    df = pd.DataFrame.from_records(r.json().get("records", []))
        # --- Normalize names across schemas ---
    # Time column
    if "TimeDK" in df.columns:
        df = df.rename(columns={"TimeDK": "HourDK"})
    # Price columns (DKK/MWh and EUR/MWh)
    if "DayAheadPriceDKK" in df.columns:
        df = df.rename(columns={"DayAheadPriceDKK": "SpotPriceDKK"})
    if "DayAheadPriceEUR" in df.columns:
        df = df.rename(columns={"DayAheadPriceEUR": "SpotPriceEUR"})

    # Must have these now
    if "HourDK" not in df.columns or "PriceArea" not in df.columns:
        return pd.DataFrame()

    # Area filter
    df = df[df["PriceArea"] == area].copy()
    if df.empty:
        return pd.DataFrame()

    # Build DKK/kWh (prefer DKK, fallback to EUR→DKK)
    if "SpotPriceDKK" in df.columns and df["SpotPriceDKK"].notna().any():
        price_dkk_per_kwh = df["SpotPriceDKK"].astype(float) / 1000.0
    elif "SpotPriceEUR" in df.columns and df["SpotPriceEUR"].notna().any():
        eur_to_dkk = 7.45  # simple fixed FX fallback
        price_dkk_per_kwh = df["SpotPriceEUR"].astype(float) * eur_to_dkk / 1000.0
    else:
        return pd.DataFrame()

    # Clean, de-dup, sort
    df["HourDK"] = pd.to_datetime(df["HourDK"], errors="coerce")
    df = df.dropna(subset=["HourDK"]).sort_values("HourDK")
    df = df[~df["HourDK"].duplicated(keep="first")]

    # Return same shape as your original
    df["price_dkk_per_kwh"] = price_dkk_per_kwh.values
    return df.set_index("HourDK")[["price_dkk_per_kwh"]]


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_elspot_prices(area: str = "DK1") -> pd.DataFrame:
    r = requests.get(f"{EDS_PRICE_URL_OLD}?limit=100000", timeout=40); r.raise_for_status()
    df = pd.DataFrame.from_records(r.json().get("records", []))
    if df.empty or "HourDK" not in df or "PriceArea" not in df or "SpotPriceDKK" not in df:
        return pd.DataFrame()
    df = df[df["PriceArea"] == area][["HourDK","SpotPriceDKK"]].copy()
    df["price_dkk_per_kwh"] = df["SpotPriceDKK"].astype(float) / 1000.0
    return (df.assign(HourDK=pd.to_datetime(df["HourDK"], errors="coerce"))
              .dropna(subset=["HourDK"])
              .sort_values("HourDK")
              .set_index("HourDK")[["price_dkk_per_kwh"]])

def daily_price_dual(idx_min: pd.DatetimeIndex, day: date, area="DK1"):
    """
    Returns:
      - price_plot: native-cadence series aligned to idx_min (for charts)
      - price_hourly: clean hourly series (for EMS/optimization)
      - note: optional note for the UI
    """
    tz = "Europe/Copenhagen"
    day_start = pd.Timestamp(day).tz_localize(tz).tz_localize(None)
    day_end   = day_start + pd.Timedelta(days=1)
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
        idx_h = pd.date_range(start=day_start, periods=24, freq="h")
        price_hourly = pd.Series(price_plot.reindex(idx_h).interpolate().bfill().ffill().values,
                                 index=idx_h, name="price_dkk_per_kwh")
        note = "No day-ahead price data available for this day. Showing a smooth placeholder curve."
        return price_plot, price_hourly, note

    # Build plotting series at native resolution → align to minute index for display only
    price_plot = s_native.reindex(idx_min).interpolate().bfill().ffill().rename("price_dkk_per_kwh")

    # Build hourly series for EMS
    s_hourly = _collapse_quarters_to_hourly(s_native)
    idx_h = pd.date_range(start=day_start, periods=24, freq="h")
    price_hourly = s_hourly.reindex(idx_h).interpolate().bfill().ffill().rename("price_dkk_per_kwh")

    # Optional note if we had gaps
    miss = int(s_native.isna().sum()) if hasattr(s_native, "isna") else 0
    if miss > 0:
        note = f"Filled {miss} missing price points by interpolation."

    return price_plot, price_hourly, note


def align_daily_series(idx: pd.DatetimeIndex, s: pd.Series) -> pd.Series:
    start, end = idx[0], idx[-1]
    # derive step minutes robustly
    step_min = int((idx[1] - idx[0]).total_seconds() // 60) if len(idx) > 1 else 1
    ss = s.loc[(s.index >= start) & (s.index <= end + pd.Timedelta(minutes=step_min))]
    if getattr(ss.index, "tz", None) is not None:
        ss.index = ss.index.tz_localize(None)
    return ss.reindex(idx).interpolate().bfill().ffill()
@st.cache_data(ttl=300, show_spinner=False)
def fetch_co2_for_day(day: date, area: str = "DK1") -> pd.Series:
    """Return local-naive 5-min gCO2/kWh series for the given calendar day. May contain NaNs."""
    tz = "Europe/Copenhagen"
    start_local = pd.Timestamp(day).tz_localize(tz)
    idx5 = pd.date_range(start=start_local, periods=288, freq="5min").tz_localize(None)

    url = "https://api.energidataservice.dk/dataset/CO2EmisProg?limit=20000"
    r = requests.get(url, timeout=20); r.raise_for_status()
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

    # Keep only that day and align to exact 5-min grid
    s = s.loc[(s.index >= idx5[0]) & (s.index <= idx5[-1])].reindex(idx5)
    return s.rename("gCO2_per_kWh")

def daily_co2_with_note(idx_min: pd.DatetimeIndex, day: date, area="DK1") -> tuple[pd.Series, str|None]:
    """
    Returns (co2_series_on_idx_min, note).
    - Validates completeness on 5-min grid (the API cadence),
      then upsamples to the app's minute index for plotting.
    """
    # 5-min local-naive series for the calendar day (may contain NaNs at 5-min timestamps)
    s5 = fetch_co2_for_day(day, area=area).rename("gCO2_per_kWh")  # 288 points

    # Build the exact 5-min grid that covers the selected minute range
    start5 = idx_min[0]
    end5   = idx_min[-1]
    idx5   = pd.date_range(start=start5, end=end5, freq="5min")

    # Align to 5-min grid and count only *real* API gaps on that grid
    s5_aligned = s5.reindex(idx5)
    miss5 = int(s5_aligned.isna().sum())
    note = None

    if s5_aligned.isna().all():
        # No API data at all → synthesize (then upsample)
        hrs = (idx_min - idx_min[0]).total_seconds() / 3600.0
        placeholder = 250.0 + 100.0*np.sin(2*np.pi*(hrs - 15.0)/24.0)  # g/kWh placeholder
        s_min = pd.Series(placeholder, index=idx_min, name="gCO2_per_kWh")
        note = "No CO₂ data from EnergiDataService for this day. Showing a smooth placeholder curve."
        return s_min, note

    # Fill only the 5-min gaps
    if miss5 > 0:
        s5_aligned = s5_aligned.interpolate(limit_direction="both").bfill().ffill()
        note = f"Filled {miss5} missing CO₂ points by interpolation."

    # Finally upsample to the app’s minute index for plotting/usage
    s_min = s5_aligned.reindex(idx_min).interpolate().bfill().ffill().astype(float)
    return s_min.rename("gCO2_per_kWh"), note


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


# -------- Sidebar --------
with st.sidebar:
    st.header("General")
    day = st.date_input("Day", value=date.today())
    step_min = 1
    objective = st.selectbox(
        "Objective", ["cost", "co2", "reservation"], index=0,
        format_func=lambda s: {"cost": "Minimize cost", "co2": "Minimize CO₂", "reservation": "Energy reservation"}[s]
    )
    ems_mode = {"cost": "economic", "reservation": "reserve", "co2": "green"}[objective]
    
    # app.py (inside your sidebar, near other controls)
    slot_mode = st.radio(
        "Battery slot source",
        ["Manual (I enter slots & setpoints)", "Auto (Use EMS logic)"],
        index=1, horizontal=False
    )

    energy_pattern = st.selectbox(
        "Rule priority",
        options=[2, 1],
        index=0,
        format_func=lambda x: "Load first (2)" if x==2 else "Battery first (1)"
    )

    # If Manual: collect 6 slots + setpoints
    manual_plan_rows = []
    if slot_mode.startswith("Manual"):
        st.markdown("**Manual battery plan (6 slots)**")
        cols = st.columns([1,1,1,1,1,1])
        st.caption("Enter start/end (HH:MM), SOC setpoint (%), and GridCharge (0/1)")
        for i in range(6):
            c1, c2, c3, c4 = st.columns([1,1,1,1])
            st.markdown(f"**Slot {i+1}**")
            st_time_start = c1.time_input(f"Start {i+1}", value=time(0,0) if i==0 else time(6,0), key=f"s{i}_s")
            st_time_end   = c2.time_input(f"End {i+1}",   value=time(6,0) if i==0 else time(23,59) if i==5 else time(10,0), key=f"s{i}_e")
            st_soc        = c3.number_input(f"SOC% {i+1}", min_value=0.0, max_value=100.0, value=20.0 if i==0 else 50.0, key=f"s{i}_soc")
            st_grid       = c4.selectbox(f"GridCharge {i+1}", options=[0,1], index=0, key=f"s{i}_gc")
            manual_plan_rows.append(dict(start=st_time_start, end=st_time_end,
                                        soc_setpoint_pct=st_soc, grid_charge_allowed=st_grid))




    st.markdown("**Location**")
    st.number_input("Latitude",  -90.0,  90.0, 55.6761, 0.0001, key="geo_lat")
    st.number_input("Longitude", -180.0, 180.0, 12.5683, 0.0001, key="geo_lon")

# --- Preview index & helpers (for the badges) ---
idx_preview = minute_index(day, step_min=step_min)
dt_h = step_min/60.0
# Preview signals (no uploads; use defaults)
price_prev = default_price_profile(idx_preview)
co2_prev   = default_co2_profile(idx_preview)
signal_preview = price_prev if objective == "cost" else (co2_prev if objective == "co2" else price_prev*0.0)



def energy_kwh(series: pd.Series) -> float:
    return float((series * dt_h).sum())

def preview_block_start(feas_mask: pd.Series, duration_min: int, sched_mode: str, manual_time):
    if duration_min <= 0: 
        return 0
    if sched_mode == "Manual fixed start" and manual_time is not None:
        return manual_start_index(idx_preview, manual_time)


# -------- Power Loads --------
st.markdown("## 1. ⚡ Power Loads")

# Baseload
use_baseload = st.checkbox("Baseload", value=True, key="base_on")
if use_baseload:
    with st.expander("Baseload settings", expanded=False):
        colb1, colb2, colb3 = st.columns(3)
        with colb1:
            router_w   = st.number_input("Router & modem (W)", 0.0, 1000.0, 12.0, 1.0, key="base_router")
            ventilation_w = st.number_input("Ventilation fan (W)", 0.0, 1000.0, 40.0, 1.0, key="base_vent")
            standby_w  = st.number_input("Standby & electronics (W)", 0.0, 1000.0, 60.0, 1.0, key="base_standby")
        with colb2:
            dhw_recirc_w = st.number_input("Hot-water recirculation pump (W)", 0.0, 1000.0, 0.0, 1.0, key="base_dhw")
            fridge_avg_w = st.number_input("Refrigerator average (W)", 0.0, 500.0, 45.0, 1.0, key="base_fridge")
        with colb3:
            other1_label = st.text_input("Other always-on #1 label", value="Aquarium/pond pump", key="base_o1_label")
            other1_w     = st.number_input(f"{other1_label} (W)", 0.0, 1000.0, 0.0, 1.0, key="base_o1")
            other2_label = st.text_input("Other always-on #2 label", value="Network switches", key="base_o2_label")
            other2_w     = st.number_input(f"{other2_label} (W)", 0.0, 1000.0, 0.0, 1.0, key="base_o2")

        st.caption("Baseload = 24/7 background draw. Set any field to 0 if not present.")


# Baseload preview badge
if use_baseload:
    base_prev = BaseloadSpec(
        router_w      = st.session_state.get("base_router", 12.0),
        ventilation_w = st.session_state.get("base_vent", 40.0),
        standby_w     = st.session_state.get("base_standby", 60.0),
        dhw_recirc_w  = st.session_state.get("base_dhw", 0.0),
        fridge_avg_w  = st.session_state.get("base_fridge", 45.0),
        other1_w      = st.session_state.get("base_o1", 0.0),
        other2_w      = st.session_state.get("base_o2", 0.0),
    ).series_kw(idx_preview)
    st.caption(f"≈ {energy_kwh(base_prev):.1f} kWh today")

# Lights
use_lights = st.checkbox("Lights", value=True, key="lights_on")
if use_lights:
    with st.expander("Lights settings"):
        lights_count = st.number_input("Count", 0, 100, 5, 1, key="lights_count")
        lights_power = st.number_input("Power per light (W)", 1, 200, 12, 1, key="lights_power")
        lights_start = st.time_input("Start", value=time(18,0), key="lights_start")
        lights_end = st.time_input("End", value=time(23,30), key="lights_end")
if use_lights and st.session_state["lights_count"]>0:
    lights_prev = FixedDevice(
        "lights",
        st.session_state["lights_power"],
        st.session_state["lights_count"],
        st.session_state["lights_start"],
        st.session_state["lights_end"],
    ).series_kw(idx_preview)
    st.caption(f"≈ {energy_kwh(lights_prev):.1f} kWh today")


# Range hood
use_hood = st.checkbox("Range hood", value=True, key="hood_on")
if use_hood:
    with st.expander("Range hood settings"):
        hood_power = st.number_input("Power (W)", 50, 1000, 150, 10, key="hood_p")
        lunch_enabled = st.checkbox("Lunch block", value=True, key="hood_lunch_on")
        dinner_enabled = st.checkbox("Dinner block", value=True, key="hood_dinner_on")
        lunch_start = st.time_input("Lunch start", value=time(12,30), key="hood_ls")
        lunch_dur = st.number_input("Lunch duration (min)", 5, 120, 20, 5, key="hood_ld")
        dinner_start = st.time_input("Dinner start", value=time(18,0), key="hood_ds")
        dinner_dur = st.number_input("Dinner duration (min)", 5, 180, 30, 5, key="hood_dd")
if use_hood:
    rh_prev = RangeHood(
        power_w=float(st.session_state["hood_p"]),
        lunch_start=st.session_state["hood_ls"],
        lunch_duration_min=int(st.session_state["hood_ld"]),
        dinner_start=st.session_state["hood_ds"],
        dinner_duration_min=int(st.session_state["hood_dd"]),
        lunch_enabled=bool(st.session_state["hood_lunch_on"]),
        dinner_enabled=bool(st.session_state["hood_dinner_on"])
    ).series_kw(idx_preview)
    st.caption(f"≈ {energy_kwh(rh_prev):.1f} kWh today")

# Washing machine
use_wm = st.checkbox("Washing machine", value=True, key="wm_on")
if use_wm:
    with st.expander("Washing machine settings"):
        wm_power = st.number_input("Power (W)", 200, 3000, 1200, 50, key="wm_p")
        wm_duration = st.number_input("Duration (min)", 15, 240, 90, 5, key="wm_dur")
        wm_win_start = st.time_input("Allowed start ≥", value=time(8,0), key="wm_ws")
        wm_win_end = st.time_input("Allowed end <", value=time(22,0), key="wm_we")
        wm_sched = st.radio("Scheduling", ["Auto (optimize)", "Manual fixed start"], horizontal=True, key="wm_sched")
        wm_manual_start = st.time_input("Manual start", value=time(9,0), key="wm_manual") if wm_sched=="Manual fixed start" else None
if use_wm and st.session_state["wm_dur"] > 0:
    wm_prev = WashingMachine(
        power_w=float(st.session_state["wm_p"]),
        duration_min=int(st.session_state["wm_dur"]),
        window_start=st.session_state["wm_ws"],
        window_end=st.session_state["wm_we"],
    )
    wm_start_prev = build_block(
        signal_preview,
        wm_prev.feasible_mask(idx_preview),
        int(st.session_state["wm_dur"]),
        st.session_state["wm_sched"],
        st.session_state.get("wm_manual"),
        idx_preview,
    )
    wm_series_prev = block_or_zero(wm_prev, idx_preview, wm_start_prev)
    st.caption(f"≈ {energy_kwh(wm_series_prev):.1f} kWh today")
    note("Washing machine", wm_start_prev, int(st.session_state["wm_dur"]), idx_preview)


    

# Dishwasher
use_dw = st.checkbox("Dishwasher", value=True, key="dw_on")
if use_dw:
    with st.expander("Dishwasher settings"):
        dw_power = st.number_input("Power (W)", 500, 3000, 1500, 50, key="dw_p")
        dw_duration = st.number_input("Duration (min)", 30, 240, 90, 5, key="dw_dur")
        dw_win_start = st.time_input("Allowed start ≥", value=time(19,0), key="dw_ws")
        dw_win_end = st.time_input("Allowed end <", value=time(7,0), key="dw_we")
        dw_sched = st.radio("Scheduling", ["Auto (optimize)", "Manual fixed start"], horizontal=True, key="dw_sched")
        dw_manual_start = st.time_input("Manual start", value=time(20,30), key="dw_manual") if dw_sched=="Manual fixed start" else None
if use_dw and st.session_state["dw_dur"] > 0:
    dw_prev = Dishwasher(
        power_w=float(st.session_state["dw_p"]),
        duration_min=int(st.session_state["dw_dur"]),
        window_start=st.session_state["dw_ws"],
        window_end=st.session_state["dw_we"],
    )
    dw_start_prev = build_block(
        signal_preview,
        dw_prev.feasible_mask(idx_preview),
        int(st.session_state["dw_dur"]),
        st.session_state["dw_sched"],
        st.session_state.get("dw_manual"),
        idx_preview,
    )
    dw_series_prev = block_or_zero(dw_prev, idx_preview, dw_start_prev)
    st.caption(f"≈ {energy_kwh(dw_series_prev):.1f} kWh today")
    note("Dishwasher", dw_start_prev, int(st.session_state["dw_dur"]), idx_preview)



# Dryer
use_dryer = st.checkbox("Dryer", value=False, key="dr_on")
if use_dryer:
    with st.expander("Dryer settings"):
        dr_power = st.number_input("Power (W)", 500, 3000, 1000, 50, key="dr_p")
        dr_duration = st.number_input("Duration (min)", 30, 240, 90, 5, key="dr_dur")
        dr_win_start = st.time_input("Allowed start ≥", value=time(8,0), key="dr_ws")
        dr_win_end = st.time_input("Allowed end <", value=time(22,0), key="dr_we")
        dr_sched = st.radio("Scheduling", ["Auto (optimize)", "Manual fixed start"], horizontal=True, key="dr_sched")
        dr_manual_start = st.time_input("Manual start", value=time(21,0), key="dr_manual") if dr_sched=="Manual fixed start" else None
if use_dryer and st.session_state["dr_dur"] > 0:
    dr_prev = Dryer(
        power_w=float(st.session_state["dr_p"]),
        duration_min=int(st.session_state["dr_dur"]),
        window_start=st.session_state["dr_ws"],
        window_end=st.session_state["dr_we"],
    )
    dr_start_prev = build_block(
        signal_preview,
        dr_prev.feasible_mask(idx_preview),
        int(st.session_state["dr_dur"]),
        st.session_state["dr_sched"],
        st.session_state.get("dr_manual"),
        idx_preview,
    )
    dr_series_prev = block_or_zero(dr_prev, idx_preview, dr_start_prev)
    st.caption(f"≈ {energy_kwh(dr_series_prev):.1f} kWh today")
    note("Dryer", dr_start_prev, int(st.session_state["dr_dur"]), idx_preview)



# EV
use_ev = st.checkbox("EV charging", value=True, key="ev_on")
if use_ev:
    with st.expander("EV settings"):
        ev_power_kw = st.number_input("Charger power (kW)", min_value=1.0, value=11.0, step=0.5, key="ev_p")
        ev_energy_target = st.number_input("Energy target (kWh)", min_value=0.0, value=50.0, step=1.0, key="ev_e")
        ev_win_start = st.time_input("Allowed start ≥", value=time(1,0), key="ev_ws")
        ev_win_end = st.time_input("Allowed end <", value=time(6,0), key="ev_we")
        ev_sched = st.radio("Scheduling", ["Auto (optimize)", "Manual fixed start"], horizontal=True, key="ev_sched")
        ev_manual_start = st.time_input("Manual start", value=time(1,0), key="ev_manual") if ev_sched=="Manual fixed start" else None
if use_ev and st.session_state["ev_e"] > 0 and st.session_state["ev_p"] > 0:
    ev_prev = EVCharger(
        power_kw=float(st.session_state["ev_p"]),
        energy_target_kwh=float(st.session_state["ev_e"]),
        window_start=st.session_state["ev_ws"],
        window_end=st.session_state["ev_we"],
    )
    ev_dur_prev = ev_prev.duration_minutes(dt_h)
    ev_start_prev = build_block(
        signal_preview,
        ev_prev.feasible_mask(idx_preview),
        int(ev_dur_prev),
        st.session_state["ev_sched"],
        st.session_state.get("ev_manual"),
        idx_preview,
    )
    ev_series_prev = block_or_zero(ev_prev, idx_preview, ev_start_prev, dt_h)
    st.caption(f"≈ {energy_kwh(ev_series_prev):.1f} kWh today")
    note("EV Charging", ev_start_prev, ev_dur_prev, idx_preview)


#----------------------get weather-----------------------------
weather_hr = fetch_weather_open_meteo(
    lat = float(st.session_state["geo_lat"]),
    lon = float(st.session_state["geo_lon"]),

    start_date=day,
    end_date=day,
    tz="Europe/Copenhagen",
)
# Minute temperature for the selected day (always compute it)
def to_minute_temp(idx: pd.DatetimeIndex) -> pd.Series:
    if not weather_hr.empty and "temp" in weather_hr.columns:
        s = weather_hr["temp"].astype(float)
        # Align to selected day & resolution
        start, end = idx[0], idx[-1]
        step_min = int((idx[1] - idx[0]).total_seconds() // 60) if len(idx) > 1 else 1
        s = s.loc[(s.index >= start) & (s.index <= end + pd.Timedelta(minutes=step_min))]
        return s.reindex(idx).interpolate().bfill().ffill().rename("Tout_C")
    return pd.Series(index=idx, dtype=float, name="Tout_C")

# -------- Thermal Loads --------
st.markdown("---")
st.markdown("## 2. 🌡️ Thermal Loads")

use_hp = st.checkbox("Heat pump", value=True, key="hp_on")
if use_hp:
    with st.expander("Heat pump settings", expanded=True):
        st.caption("Outdoor temperature is fetched automatically from Open-Meteo for the selected day.")

        ua    = st.number_input("House loss UA (kW/°C)", 0.05, 2.0, 0.25, 0.05, key="hp_ua")
        tset  = st.number_input("Indoor setpoint (°C)",   10.0, 26.0, 21.0, 0.5,  key="hp_tset")
        qrat  = st.number_input("Rated heat capacity (kW)", 1.0, 25.0, 5.0, 0.5,  key="hp_qr")
        cop7  = st.number_input("COP at 7 °C",             1.0, 8.0,  3.2, 0.1,  key="hp_cop7")

        with st.expander("Advanced COP & defrost (optional)"):
            adv_on = st.checkbox("Override COP curve (a,b)", value=False, key="hp_adv_on")
            copa   = st.number_input("COP intercept a",        0.0, 10.0, 3.2, 0.1, key="hp_copa") if adv_on else None
            copb   = st.number_input("COP slope b (per °C)",  -0.2, 0.2, 0.05, 0.01, key="hp_copb") if adv_on else None
            cop_min= st.number_input("COP min",               1.0, 6.0,  1.6, 0.1, key="hp_copmin")
            cop_max= st.number_input("COP max",               2.0, 8.0,  4.2, 0.1, key="hp_copmax")
            defrost= st.checkbox("Defrost penalty below 3 °C", value=True, key="hp_def")



# -------- Generation & Storage --------
st.markdown("---")
st.markdown("## 3. Installed Power Generation & Storage")

g1, g2 = st.columns(2)
# --- PV sizing UI (drop this where you define pv_kwp) ---
with g1:
    use_pv= st.checkbox("Enable PV", value=True, key="pv_on")
    module_wp = st.number_input("Module nameplate (Wp)", min_value=50, value=400, step=10, disabled=not use_pv,key="pv_mod_wp")
    n_panels  = st.number_input("Number of panels", min_value=0, value=16, step=1, disabled=not use_pv,key="pv_n")
    pv_kwp = (module_wp * n_panels) / 1000.0
    st.caption(f"Total DC size: **{pv_kwp:.2f} kWp**")
    if use_pv:
        with st.expander("PV orientation & losses"):
            # angle from horizontal
            st.number_input("Tilt (°)", min_value=0.0, max_value=90.0, value=30.0, step=1.0, disabled=not use_pv, key="pv_tilt")
            # 0=N, 90=E, 180=S, 270=W
            st.number_input("Azimuth (°; 180 = S)", min_value=0.0, max_value=360.0, value=180.0, step=1.0, disabled=not use_pv, key="pv_az")
            # loss fraction for ‘everything else’
            st.number_input("System losses (fraction)", min_value=0.0, max_value=0.5, value=0.14, step=0.01, disabled=not use_pv, key="pv_loss")




with g2:
    use_batt = st.checkbox("Enable battery", value=True, key="batt_on")
    cap_kwh = st.number_input("Battery capacity (kWh)", 0.0, 100.0, 75.0, 0.5, disabled=not use_batt, key="batt_cap")
    p_kw    = st.number_input("Battery power (kW)",      0.0, 50.0,  9.0, 0.5, disabled=not use_batt, key="batt_pow")
    soc_pct = st.number_input("Initial SOC (%)",         0.0, 100.0, 20.0, 1.0, disabled=not use_batt, key="batt_soc_pct")

soc0_kwh = (st.session_state["batt_soc_pct"] / 100.0) * st.session_state["batt_cap"] if use_batt else 0.0


#%%
st.markdown("---")
st.markdown("## 4. Run")

sim_col, ems_col = st.columns([1,1])

# ---------- Phase 1: simulate only ----------
if st.session_state["scheduled_notes"]:
    st.subheader("4.1 Schedules chosen today")
    for name, start_txt, dur in st.session_state["scheduled_notes"]:
        st.write(f"• **{name}** → {start_txt} for {dur} min")

if sim_col.button("▶️ Run simulation"):
    st.session_state["scheduled_notes"] = [] 
    sim = compute_load_env(
        day=day, step_min=step_min, objective=objective, weather_hr=weather_hr,
        use_baseload=use_baseload, use_lights=use_lights, use_hood=use_hood,
        use_wm=use_wm, use_dw=use_dw, use_dryer=use_dryer, use_ev=use_ev, use_hp=use_hp,
        pv_kwp=pv_kwp, idx_preview=idx_preview
    )
    st.session_state["sim"] = sim          # store results
    st.session_state["ems"] = None         # optional: clear previous EMS
    st.success("Simulation complete (no EMS).")



# ---------- Phase 2: EMS on existing simulation ----------
ems_disabled = st.session_state.get("sim") is None
if ems_col.button("⚡ Run EMS on current simulation", disabled=ems_disabled):
    sim = st.session_state.get("sim")
    if sim is None:
        st.warning("Run simulation first.")
    else:
        # Build hourly df → auto/ manual slots → plan_df exactly as you already do:
        df_hourly = (pd.DataFrame({
            "load": sim["load"], "pv": sim["pv"], "price": sim["price_hourly"],
        }).resample("h").mean()
          .reset_index()
          .rename(columns={"index":"DateTime","price":"ElectricityPrice","load":"Load","pv":"PV"}))

        # Choose slot source (manual from sidebar, or auto)
        if slot_mode.startswith("Manual"):
            # Collect your 6 rows from `manual_plan_rows` you already built in the sidebar:
            plan_df = pd.DataFrame(manual_plan_rows)  # columns: start,end,soc_setpoint_pct,grid_charge_allowed
        else:
            # Auto slots from your existing functions
            time_slots = generate_smart_time_slots(df_hourly)
            df_slots   = assign_data_to_time_slots_single(df_hourly, time_slots)
            SOC0 = float(st.session_state["batt_soc_pct"])
            SOC_min = 15.0 if objective != "reservation" else 50.0
            SOC_max = 100.0
            SOC_opt, Qgrid, _ = mpc_opt_single(
                df_slots, SOC0=SOC0, SOC_min=SOC_min, SOC_max=SOC_max,
                Pbat_chargemax=st.session_state["batt_pow"],
                Qbat=st.session_state["batt_cap"]
            )
            df_plan = format_results_single(SOC_opt, Qgrid, df_slots)
            df_today = df_plan[df_plan["Datetime"] == df_plan["Datetime"].iloc[0]].copy()
            df_today["start"] = pd.to_datetime(df_today["TimeSlot"].str.split(" - ").str[0], format="%H:%M").dt.time
            df_today["end"]   = pd.to_datetime(df_today["TimeSlot"].str.split(" - ").str[1], format="%H:%M").dt.time
            df_today["soc_setpoint_pct"]    = df_today["SOC"]
            df_today["grid_charge_allowed"] = df_today["Grid_Charge"]
            plan_df = df_today[["start","end","soc_setpoint_pct","grid_charge_allowed"]]

        # Run EMS (rule engine with your same semantics)
        st.session_state["ems_plan"] = plan_df.copy()
        ems_out = rule_power_share(
            idx=sim["idx"], load_kw=sim["load"], pv_kw=sim["pv"],
            plan_slots=plan_df,
            cap_kwh=st.session_state["batt_cap"],
            p_max_kw=st.session_state["batt_pow"],
            soc0_kwh=(st.session_state["batt_soc_pct"] / 100.0) * st.session_state["batt_cap"],
            eta_ch=0.95, eta_dis=0.95,
            energy_pattern=energy_pattern
        )
        grid = ems_out["grid_import_kw"]
        pbat = ems_out["batt_discharge_kw"] - ems_out["batt_charge_kw"]  # +dis, -ch
        soc  = ems_out["batt_soc_kwh"]

        # Plots
        st.header("B) EMS Result")
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
            day0 = pd.Timestamp(sim["idx"][0]).normalize()

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
                use_container_width=True
            )

        st.plotly_chart(ems_power_split_plot(sim["idx"], sim["load"], sim["pv"], grid, pbat), use_container_width=True)
        st.plotly_chart(ems_soc_plot(sim["idx"], soc, st.session_state["batt_cap"]), use_container_width=True)

        # KPI summary + comparisons
        kpi_actual = kpi_summary(sim["load"], sim["pv"], grid, sim["price"], sim["co2"], sim["dt_h"])
        kpi_base, kpi_pv_only, kpi_batt_only = compare_scenarios(
            sim, plan_df,
            cap_kwh=st.session_state["batt_cap"],
            p_max_kw=st.session_state["batt_pow"],
            soc0_kwh=(st.session_state["batt_soc_pct"] / 100.0) * st.session_state["batt_cap"],
            energy_pattern=energy_pattern,
        )

        # Build a table of results and savings
        def flat(label, k):
            return dict(Scenario=label, **{
                "Cost (DKK)": k["cost_dkk"], "CO₂ (kg)": k["co2_kg"],
                "Grid (kWh)": k["grid_kwh"], "Load (kWh)": k["load_kwh"], "PV (kWh)": k["pv_kwh"],
                "Self-consumption (%)": k["self_consumption_pct"], "Self-sufficiency (%)": k["self_sufficiency_pct"],
            })

        rows = [
            flat("No PV, No Battery", kpi_base),
            flat("PV only",           kpi_pv_only),
            flat("Battery only",      kpi_batt_only),
            flat("PV + Battery (EMS)",kpi_actual),
        ]
        df_kpi = pd.DataFrame(rows)
        # Savings columns
        base_cost, base_co2 = kpi_base["cost_dkk"], kpi_base["co2_kg"]
        df_kpi.loc[df_kpi["Scenario"]=="PV + Battery (EMS)","Saved DKK vs Base"] = base_cost - kpi_actual["cost_dkk"]
        df_kpi.loc[df_kpi["Scenario"]=="PV + Battery (EMS)","Saved CO₂ kg vs Base"] = base_co2 - kpi_actual["co2_kg"]
        df_kpi.loc[df_kpi["Scenario"]=="PV + Battery (EMS)","Saved DKK vs PV-only"] = kpi_pv_only["cost_dkk"] - kpi_actual["cost_dkk"]
        df_kpi.loc[df_kpi["Scenario"]=="PV + Battery (EMS)","Saved CO₂ kg vs PV-only"] = kpi_pv_only["co2_kg"] - kpi_actual["co2_kg"]
        df_kpi.loc[df_kpi["Scenario"]=="PV + Battery (EMS)","Saved DKK vs Batt-only"] = kpi_batt_only["cost_dkk"] - kpi_actual["cost_dkk"]
        df_kpi.loc[df_kpi["Scenario"]=="PV + Battery (EMS)","Saved CO₂ kg vs Batt-only"] = kpi_batt_only["co2_kg"] - kpi_actual["co2_kg"]

        st.subheader("Summary & Comparisons")
        st.dataframe(df_kpi.style.format({
            "Cost (DKK)": "{:.2f}", "CO₂ (kg)": "{:.2f}",
            "Grid (kWh)": "{:.2f}", "Load (kWh)": "{:.2f}", "PV (kWh)": "{:.2f}",
            "Self-consumption (%)": "{:.1f}", "Self-sufficiency (%)": "{:.1f}",
            "Saved DKK vs Base": "{:.2f}", "Saved CO₂ kg vs Base": "{:.2f}",
            "Saved DKK vs PV-only": "{:.2f}", "Saved CO₂ kg vs PV-only": "{:.2f}",
            "Saved DKK vs Batt-only": "{:.2f}", "Saved CO₂ kg vs Batt-only": "{:.2f}",
        }), use_container_width=True)

# ---------- Render what we have (so both sections show after EMS) ----------
if st.session_state.get("sim") is not None:
    st.header("A) Simulation results")
    show_load_env(st.session_state["sim"])   # make sure charts use unique keys, e.g., key="sim_price", ...

if st.session_state.get("ems") is not None:
    st.header("B) EMS results")
    render_ems(st.session_state["sim"], st.session_state["ems"])  # keys like "ems_split", "ems_soc"