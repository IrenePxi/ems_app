"""
API endpoints for device configuration and load profile generation
"""
from fastapi import APIRouter, HTTPException, Body
from typing import Dict, List, Optional
from datetime import date
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path (go up to ems_app_v3)
backend_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(backend_path))
from devices import WeatherHP, WeatherELheater, WeatherHotTub, DHWTank
from profiles import minute_index

router = APIRouter()

@router.post("/calculate-load")
async def calculate_load_profile(
    devices: List[Dict] = Body(..., description="List of device configurations"),
    start_date: date = Body(..., description="Start date"),
    end_date: date = Body(..., description="End date"),
    outdoor_temp: Optional[Dict] = Body(None, description="Outdoor temperature series (optional)")
):
    """Calculate total load profile from configured devices"""
    try:
        idx = minute_index(start_date, end_date)
        
        # Prepare outdoor temperature
        if outdoor_temp:
            tout_series = pd.Series(outdoor_temp["data"], index=pd.to_datetime(outdoor_temp["index"]))
            tout_series = tout_series.reindex(idx, method="nearest")
        else:
            tout_series = pd.Series(10.0, index=idx)  # Default 10°C
        
        total_load = pd.Series(0.0, index=idx)
        device_outputs = {}
        
        for device_config in devices:
            dev_type = device_config.get("type")
            params = device_config.get("params", {})
            
            if dev_type == "heat_pump":
                hp = WeatherHP(**params)
                p_kw, ti_c = hp.series_kw(idx, tout_series)
                total_load += p_kw
                device_outputs[device_config.get("name", "heat_pump")] = {
                    "power_kw": p_kw.to_dict(),
                    "indoor_temp_c": ti_c.to_dict()
                }
            
            elif dev_type == "electric_heater":
                eh = WeatherELheater(**params)
                p_kw, ti_c = eh.series_kw(idx, tout_series)
                total_load += p_kw
                device_outputs[device_config.get("name", "electric_heater")] = {
                    "power_kw": p_kw.to_dict(),
                    "indoor_temp_c": ti_c.to_dict()
                }
            
            elif dev_type == "hot_tub":
                ht = WeatherHotTub(**params)
                p_kw, tw_c = ht.series_kw(idx, tout_series)
                total_load += p_kw
                device_outputs[device_config.get("name", "hot_tub")] = {
                    "power_kw": p_kw.to_dict(),
                    "water_temp_c": tw_c.to_dict()
                }
            
            elif dev_type == "dhw_tank":
                dhw = DHWTank(**params)
                p_kw, tt_c = dhw.series_kw(idx, tout_series)
                total_load += p_kw
                device_outputs[device_config.get("name", "dhw_tank")] = {
                    "power_kw": p_kw.to_dict(),
                    "tank_temp_c": tt_c.to_dict()
                }
        
        return {
            "total_load_kw": total_load.to_dict(),
            "index": [str(ts) for ts in idx],
            "device_outputs": device_outputs
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating load profile: {str(e)}")

@router.get("/device-defaults/{device_type}")
async def get_device_defaults(device_type: str):
    """Get default parameters for a device type"""
    defaults = {
        "heat_pump": {
            "name": "heat_pump_weather",
            "ua_kw_per_c": 0.25,
            "t_set_c": 21.0,
            "q_rated_kw": 6.0,
            "cop_at_7c": 3.2,
            "cop_min": 1.6,
            "cop_max": 4.2,
            "hyst_band_c": 0.6,
            "C_th_kwh_per_c": 3.0,
            "Ti0_c": 21.0,
            "mode": "onoff"
        },
        "electric_heater": {
            "name": "el_heater_weather",
            "ua_kw_per_c": 0.25,
            "t_set_c": 21.0,
            "q_rated_kw": 6.0,
            "hyst_band_c": 0.6,
            "C_th_kwh_per_c": 3.0,
            "Ti0_c": 21.0
        },
        "hot_tub": {
            "name": "hot_tub_weather",
            "target_c": 38.0,
            "idle_c": 32.0,
            "heater_kw": 3.0,
            "water_l": 800.0,
            "ua_kw_per_c": 0.02,
            "sessions": []
        },
        "dhw_tank": {
            "name": "dhw_tank",
            "volume_l": 200.0,
            "t_set_c": 50.0,
            "hyst_band_c": 5.0,
            "ua_kw_per_c": 0.02,
            "p_el_kw": 2.0,
            "usage_level": "Medium"
        }
    }
    
    if device_type not in defaults:
        raise HTTPException(status_code=404, detail=f"Unknown device type: {device_type}")
    
    return defaults[device_type]
