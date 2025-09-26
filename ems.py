import numpy as np
import pandas as pd
from datetime import time as _time  # avoid shadowing

# -- public API used by app.py ------------------------------------------------
def rule_power_share(
    idx: pd.DatetimeIndex,
    load_kw: pd.Series,
    pv_kw: pd.Series,
    plan_slots: pd.DataFrame,          # columns: start, end, soc_setpoint_pct, grid_charge_allowed
    cap_kwh: float,
    p_max_kw: float,
    soc0_kwh: float,
    energy_pattern: int = 2,           # 2=Load-first, 1=Battery-first  (keep your semantics)
    eta_ch: float = 0.95,
    eta_dis: float = 0.95,
    p_fc_kw: float | pd.Series = 0.0,  # optional other local source
):
    """
    Wrapper that expands 6 user/EMS slots into per-minute setpoint/flag series,
    then calls the series dispatcher.
    """
    # 1) expand the 6 slots to per-minute series
    soc_target_pct, grid_charge_flag = _expand_plan_to_series(
        idx=idx,
        plan_slots=plan_slots,
        default_soc_pct=0.0,
        default_grid_flag=0,
    )

    # 2) run the series-based engine
    out = _rule_power_share_series(
        load_kw=load_kw,
        pv_kw=pv_kw,
        soc_target_pct=soc_target_pct,
        grid_charge_flag=grid_charge_flag,
        cap_kwh=cap_kwh,
        p_charge_max_kw=p_max_kw,
        soc0_kwh=soc0_kwh,
        energy_pattern=energy_pattern,
        eta_ch=eta_ch,
        eta_dis=eta_dis,
        p_fc_kw=p_fc_kw,
    )

    # 3) return as a dict (what app.py expects)
    grid, pbat, pv_used, soc = out
    return {
        "grid_import_kw": grid.rename("grid_import_kw"),
        "batt_charge_kw": pbat.clip(upper=0).abs().rename("batt_charge_kw"),
        "batt_discharge_kw": pbat.clip(lower=0).rename("batt_discharge_kw"),
        "batt_soc_kwh": soc.rename("batt_soc_kwh"),
        "pv_used_to_load_kw": pv_used.rename("pv_used_to_load_kw"),
    }


# -- helper: expand 6 slots into per-minute series ----------------------------
def _expand_plan_to_series(
    idx: pd.DatetimeIndex,
    plan_slots: pd.DataFrame,  # start (time), end (time), soc_setpoint_pct (float), grid_charge_allowed (0/1)
    default_soc_pct: float = 0.0,
    default_grid_flag: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """
    Creates two per-index series:
      - soc_target_pct (float%)
      - grid_charge_flag (0/1)
    by painting values across the provided [start,end) slots.
    """
    soc_target = pd.Series(default_soc_pct, index=idx, dtype=float)
    grid_flag  = pd.Series(int(default_grid_flag), index=idx, dtype=int)

    # normalize columns
    df = plan_slots.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    # expected: 'start','end','soc_setpoint_pct','grid_charge_allowed'

    # paint each slot on [start, end)
    for _, row in df.iterrows():
        st: _time = row["start"]
        en: _time = row["end"]
        m = (idx.time >= st) & (idx.time < en) if en != _time(23,59) else (idx.time >= st)  # include tail for last slot
        if "soc_setpoint_pct" in row:
            soc_target.loc[m] = float(row["soc_setpoint_pct"])
        if "grid_charge_allowed" in row:
            grid_flag.loc[m]  = int(row["grid_charge_allowed"])

    # ensure full coverage
    soc_target = soc_target.astype(float).ffill().bfill()
    grid_flag  = grid_flag.astype(int).ffill().bfill()
    return soc_target, grid_flag


# -- series-based engine (very close to what you wrote) -----------------------
def _rule_power_share_series(
    load_kw: pd.Series,                # +kW
    pv_kw: pd.Series,                  # +kW
    soc_target_pct: pd.Series | float, # SOC target (%) per step or scalar
    grid_charge_flag: pd.Series | int, # 0/1 per step or scalar
    cap_kwh: float,
    p_charge_max_kw: float,
    soc0_kwh: float,
    energy_pattern: int = 2,
    eta_ch: float = 0.95,
    eta_dis: float = 0.95,
    p_fc_kw: float | pd.Series = 0.0,
):
    idx = load_kw.index
    assert pv_kw.index.equals(idx), "pv_kw index must match load_kw"

    # Broadcast scalars to series
    if np.isscalar(soc_target_pct):
        soc_target_pct = pd.Series(float(soc_target_pct), index=idx)
    if np.isscalar(grid_charge_flag):
        grid_charge_flag = pd.Series(int(grid_charge_flag), index=idx)
    if np.isscalar(p_fc_kw):
        p_fc_kw = pd.Series(float(p_fc_kw), index=idx)

    # time step
    dt_h = (idx[1] - idx[0]).total_seconds() / 3600.0 if len(idx) > 1 else 1/60.0

    grid = np.zeros(len(idx))
    pbat = np.zeros(len(idx))      # +discharge to load, -charge from grid/PV
    pv_used = np.zeros(len(idx))
    soc = np.zeros(len(idx))

    soc_kwh = float(soc0_kwh)

    def clamp_charge_discharge(p_req_kw, soc_kwh_local):
        """Apply power limit and SOC headroom/tailroom with efficiencies."""
        if p_req_kw >= 0:  # discharge
            dis_room_kw = max(0.0, soc_kwh_local) * eta_dis / dt_h
            return min(p_req_kw, p_charge_max_kw, dis_room_kw)
        else:              # charge (negative)
            ch_room_kw = max(0.0, cap_kwh - soc_kwh_local) / (eta_ch * dt_h)
            return -min(abs(p_req_kw), p_charge_max_kw, ch_room_kw)

    for i, t in enumerate(idx):
        P_pv   = float(pv_kw.iat[i])
        P_load = float(load_kw.iat[i])
        P_fc   = float(p_fc_kw.iat[i])
        SOClvl = float(soc_target_pct.iat[i])   # target %
        Gflag  = int(grid_charge_flag.iat[i])   # 0/1
        SOCpct = (soc_kwh / cap_kwh * 100.0) if cap_kwh > 0 else 0.0

        P_load_eff = P_load - P_fc

        Pgrid = 0.0
        Pbatt = 0.0

        # ---- your MATLAB-style rule logic (kept 1:1) ----
        if energy_pattern == 2:  # Load first
            if P_load_eff > 0:
                if P_pv >= P_load_eff:
                    if SOCpct < SOClvl and Gflag == 1:
                        Pgrid = max(p_charge_max_kw - (P_pv - P_load_eff), 0.0)
                        Pbatt = -p_charge_max_kw
                    elif SOCpct < SOClvl and Gflag == 0:
                        Pgrid = 0.0
                        Pbatt = -min(p_charge_max_kw, (P_pv - P_load_eff))
                    elif SOCpct >= SOClvl:
                        Pgrid = 0.0
                        Pbatt = -min(p_charge_max_kw, (P_pv - P_load_eff))
                    if SOCpct >= 100.0:
                        Pgrid = 0.0
                        Pbatt = 0.0
                else:
                    if SOCpct < SOClvl and Gflag == 1:
                        Pbatt = -p_charge_max_kw
                        Pgrid = p_charge_max_kw + P_load_eff - P_pv
                    elif SOCpct < SOClvl and Gflag == 0:
                        Pgrid = P_load_eff - P_pv
                        Pbatt = 0.0
                    elif SOCpct >= SOClvl:
                        deficit = P_load_eff - P_pv
                        if (p_charge_max_kw - deficit) > 0:
                            Pbatt = deficit
                            Pgrid = 0.0
                        else:
                            Pbatt = p_charge_max_kw
                            Pgrid = deficit - p_charge_max_kw
            else:
                if SOClvl > SOCpct and Gflag == 0:
                    Pbatt = -min(p_charge_max_kw, (P_pv - P_load_eff))
                    Pgrid = 0.0
                elif SOClvl > SOCpct and Gflag == 1:
                    Pbatt = -p_charge_max_kw
                    Pgrid = max(p_charge_max_kw - (P_pv - P_load_eff), 0.0)
                else:  # SOCpct >= SOClvl
                    Pgrid = 0.0
                    Pbatt = -min(p_charge_max_kw, (P_pv - P_load_eff))
                if SOCpct >= 100.0:
                    Pgrid = 0.0
                    Pbatt = 0.0

        else:  # Battery first
            if P_load_eff > 0:
                if P_pv >= P_load_eff:
                    if SOClvl > SOCpct and Gflag == 1:
                        Pgrid = max(p_charge_max_kw - (P_pv - P_load_eff), 0.0)
                        Pbatt = -p_charge_max_kw
                    elif SOClvl > SOCpct and Gflag == 0:
                        Pbatt = -min(p_charge_max_kw, P_pv)
                        Pgrid = P_load_eff - P_pv - Pbatt
                    else:  # SOCpct >= SOClvl
                        Pgrid = 0.0
                        Pbatt = -min(p_charge_max_kw, (P_pv - P_load_eff))
                    if SOCpct >= 100.0:
                        Pgrid = 0.0
                        Pbatt = 0.0
                else:
                    if SOClvl > SOCpct and Gflag == 1:
                        Pbatt = -p_charge_max_kw
                        Pgrid = p_charge_max_kw + P_load_eff - P_pv
                    elif SOClvl > SOCpct and Gflag == 0:
                        Pbatt = -min(p_charge_max_kw, P_pv)
                        Pgrid = P_load_eff - P_pv - Pbatt
                    else:  # SOCpct >= SOClvl
                        deficit = P_load_eff - P_pv
                        if (p_charge_max_kw - deficit) > 0:
                            Pbatt = deficit
                            Pgrid = 0.0
                        else:
                            Pbatt = p_charge_max_kw
                            Pgrid = deficit - p_charge_max_kw
            else:
                if SOClvl > SOCpct and Gflag == 0:
                    Pbatt = -min(p_charge_max_kw, (P_pv - P_load_eff))
                    Pgrid = 0.0
                elif SOClvl > SOCpct and Gflag == 1:
                    Pbatt = -p_charge_max_kw
                    Pgrid = max(p_charge_max_kw - (P_pv - P_load_eff), 0.0)
                else:  # SOCpct >= SOClvl
                    Pgrid = 0.0
                    Pbatt = -min(p_charge_max_kw, (P_pv - P_load_eff))
                if SOCpct >= 100.0:
                    Pgrid = 0.0
                    Pbatt = 0.0

        # enforce battery constraints
        Pbatt = clamp_charge_discharge(Pbatt, soc_kwh)

        # SOC update (kWh)
        if Pbatt >= 0:  # discharge
            soc_kwh = max(0.0, soc_kwh - (Pbatt / eta_dis) * dt_h)
        else:           # charge
            soc_kwh = min(cap_kwh, soc_kwh + (-Pbatt * eta_ch) * dt_h)

        Pgrid = max(0.0, Pgrid)  # no export in this model
        P_pv_used = P_load_eff - Pgrid - Pbatt

        grid[i] = Pgrid
        pbat[i] = Pbatt
        pv_used[i] = max(0.0, P_pv_used)
        soc[i] = soc_kwh

    # outputs as series (engine-level)
    grid_s = pd.Series(grid, index=idx, name="grid_import_kw")
    pbat_s = pd.Series(pbat, index=idx, name="batt_power_kw")  # +dis, -ch
    pv_s   = pd.Series(pv_used, index=idx, name="pv_used_to_load_kw")
    soc_s  = pd.Series(soc, index=idx, name="batt_soc_kwh")
    return grid_s, pbat_s, pv_s, soc_s
