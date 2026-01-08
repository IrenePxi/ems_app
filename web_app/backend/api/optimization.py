"""
API endpoints for optimization-based battery scheduling
"""
from fastapi import APIRouter, HTTPException, Body
from typing import Dict, List
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path (go up to ems_app_v3)
backend_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(backend_path))
from Optimization_based import (
    generate_smart_time_slots,
    assign_data_to_time_slots_single,
    mpc_opt_single,
    format_results_single
)

router = APIRouter()

@router.post("/generate-slots")
async def generate_optimized_slots(
    load_kw: Dict = Body(..., description="Load profile"),
    pv_kw: Dict = Body(..., description="PV profile"),
    price: Dict = Body(..., description="Price profile"),
    co2: Dict = Body(..., description="CO2 profile"),
    signal: Dict = Body(..., description="Combined signal (weighted price+CO2)")
):
    """Generate optimized time slots using smart detection"""
    try:
        # Convert to DataFrame format expected by optimization
        idx = pd.to_datetime(load_kw["index"])
        df_minute = pd.DataFrame({
            "DateTime": idx,
            "Load": pd.Series(load_kw["data"], index=idx).values,
            "PV": pd.Series(pv_kw["data"], index=idx).values,
            "ElectricityPrice": pd.Series(price["data"], index=idx).values,
            "signal": pd.Series(signal["data"], index=idx).values,
            "co2": pd.Series(co2["data"], index=idx).values,
        }).sort_values("DateTime")
        
        # Generate time slots
        time_slots = generate_smart_time_slots(df_minute)
        df_slots = assign_data_to_time_slots_single(df_minute, time_slots)
        
        # Convert to response format
        slots_list = []
        for day, slots in time_slots.items():
            for start_time, end_time in slots:
                slots_list.append({
                    "date": str(day),
                    "start": start_time,
                    "end": end_time
                })
        
        return {
            "time_slots": slots_list,
            "slot_data": df_slots.to_dict("records")
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating slots: {str(e)}")

@router.post("/optimize-battery")
async def optimize_battery_schedule(
    slot_data: List[Dict] = Body(..., description="Time slot data from generate-slots"),
    battery_config: Dict = Body(..., description="Battery configuration")
):
    """Run MPC optimization to determine optimal SOC targets and grid charging"""
    try:
        df_slots = pd.DataFrame(slot_data)
        
        SOC0 = battery_config["soc_init_pct"]
        SOC_min = battery_config.get("soc_min_pct", 15.0)
        SOC_max = 100.0
        Pbat_chargemax = battery_config["power_kw"]
        Qbat = battery_config["capacity_kwh"]
        
        SOC_opt, Qgrid, _ = mpc_opt_single(
            df_slots,
            SOC0=SOC0,
            SOC_min=SOC_min,
            SOC_max=SOC_max,
            Pbat_chargemax=Pbat_chargemax,
            Qbat=Qbat
        )
        
        df_plan = format_results_single(SOC_opt, Qgrid, df_slots)
        
        # Convert to response format
        plan_slots = []
        for _, row in df_plan.iterrows():
            time_slot = row["TimeSlot"]
            start_str, end_str = time_slot.split(" - ")
            
            plan_slots.append({
                "start": start_str,
                "end": end_str,
                "soc_setpoint_pct": float(row["SOC"]),
                "grid_charge_allowed": int(row["Grid_Charge"])
            })
        
        return {
            "plan_slots": plan_slots,
            "optimization_result": {
                "soc_targets": SOC_opt.tolist(),
                "grid_charge": Qgrid.tolist()
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error optimizing battery: {str(e)}")
