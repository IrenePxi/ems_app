"""
API endpoints for Energy Management System (EMS) operations
"""
from fastapi import APIRouter, HTTPException, Body
from typing import Dict, List, Optional
from datetime import date, time
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path (go up to ems_app_v3)
backend_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(backend_path))
from ems import rule_power_share

router = APIRouter()

@router.post("/run")
async def run_ems(
    load_kw: Dict = Body(..., description="Load profile (dict with index and data)"),
    pv_kw: Optional[Dict] = Body(None, description="PV profile (optional)"),
    plan_slots: List[Dict] = Body(..., description="Battery plan slots"),
    battery_config: Dict = Body(..., description="Battery configuration"),
    energy_pattern: int = Body(2, description="Energy pattern: 1=Battery-first, 2=Load-first")
):
    """Run EMS rule-based power sharing"""
    try:
        # Convert input data to pandas Series
        idx = pd.to_datetime(load_kw["index"])
        load_series = pd.Series(load_kw["data"], index=idx)
        
        if pv_kw:
            pv_series = pd.Series(pv_kw["data"], index=pd.to_datetime(pv_kw["index"]))
            pv_series = pv_series.reindex(idx, fill_value=0.0)
        else:
            pv_series = pd.Series(0.0, index=idx)
        
        # Convert plan slots to DataFrame
        plan_df = pd.DataFrame(plan_slots)
        if "start" in plan_df.columns:
            plan_df["start"] = pd.to_datetime(plan_df["start"].astype(str)).dt.time
        if "end" in plan_df.columns:
            plan_df["end"] = pd.to_datetime(plan_df["end"].astype(str)).dt.time
        
        # Run EMS
        result = rule_power_share(
            idx=idx,
            load_kw=load_series,
            pv_kw=pv_series,
            plan_slots=plan_df,
            cap_kwh=battery_config["capacity_kwh"],
            p_max_kw=battery_config["power_kw"],
            soc0_kwh=(battery_config["soc_init_pct"] / 100.0) * battery_config["capacity_kwh"],
            energy_pattern=energy_pattern,
            eta_ch=battery_config.get("eta_ch", 0.95),
            eta_dis=battery_config.get("eta_dis", 0.95),
        )
        
        # Calculate KPIs
        grid_import = result["grid_import_kw"]
        batt_charge = result["batt_charge_kw"]
        batt_discharge = result["batt_discharge_kw"]
        soc = result["batt_soc_kwh"]
        pv_used = result["pv_used_to_load_kw"]
        
        # Calculate metrics
        dt_h = (idx[1] - idx[0]).total_seconds() / 3600.0 if len(idx) > 1 else 1/60.0
        
        total_load_kwh = load_series.sum() * dt_h
        total_pv_kwh = pv_series.sum() * dt_h
        total_grid_kwh = grid_import.sum() * dt_h
        total_pv_used_kwh = pv_used.sum() * dt_h
        
        self_consumption_pct = (total_pv_used_kwh / total_pv_kwh * 100) if total_pv_kwh > 0 else 0
        self_sufficiency_pct = (total_pv_used_kwh / total_load_kwh * 100) if total_load_kwh > 0 else 0
        
        return {
            "grid_import_kw": grid_import.to_dict(),
            "batt_charge_kw": batt_charge.to_dict(),
            "batt_discharge_kw": batt_discharge.to_dict(),
            "batt_soc_kwh": soc.to_dict(),
            "pv_used_to_load_kw": pv_used.to_dict(),
            "index": [str(ts) for ts in idx],
            "kpis": {
                "total_load_kwh": float(total_load_kwh),
                "total_pv_kwh": float(total_pv_kwh),
                "total_grid_kwh": float(total_grid_kwh),
                "self_consumption_pct": float(self_consumption_pct),
                "self_sufficiency_pct": float(self_sufficiency_pct),
                "peak_grid_import_kw": float(grid_import.max())
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running EMS: {str(e)}")
