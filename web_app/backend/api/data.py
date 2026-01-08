"""
API endpoints for data fetching (prices, CO2, weather, PV)
"""
from fastapi import APIRouter, HTTPException, Query
from datetime import date, datetime
from typing import Optional
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path to import modules (go up to ems_app_v3)
backend_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(backend_path))
from profiles import minute_index, default_price_profile, default_co2_profile, simple_pv_profile, synthetic_outdoor_temp
import pvlib
from pvlib.location import Location

router = APIRouter()

# Import data fetching functions from app.py (we'll extract these)
EDS_PRICE_URL_NEW = "https://api.energidataservice.dk/dataset/DayAheadPrices"
EDS_CO2_HIST_URL = "https://api.energidataservice.dk/dataset/CO2Emis"
EDS_CO2_PROG_URL = "https://api.energidataservice.dk/dataset/CO2EmisProg"
TZ_DK = "Europe/Copenhagen"

@router.get("/price")
async def get_price_data(
    start_date: date = Query(..., description="Start date"),
    end_date: date = Query(..., description="End date"),
    area: str = Query("DK1", description="Price area (DK1 or DK2)")
):
    """Fetch electricity price data for a date range"""
    try:
        import requests
        r = requests.get(f"{EDS_PRICE_URL_NEW}?limit=200000", timeout=40)
        r.raise_for_status()
        recs = r.json().get("records", [])
        
        if not recs:
            # Fallback to default profile
            idx = minute_index(start_date, end_date)
            price_series = default_price_profile(idx)
            return {
                "data": price_series.to_dict(),
                "index": [str(ts) for ts in price_series.index],
                "note": "Using synthetic data - API returned no records"
            }
        
        df = pd.DataFrame.from_records(recs)
        if "TimeDK" in df.columns:
            df = df.rename(columns={"TimeDK": "HourDK"})
        if "DayAheadPriceDKK" in df.columns:
            df = df.rename(columns={"DayAheadPriceDKK": "SpotPriceDKK"})
        
        df = df[df["PriceArea"] == area].copy()
        df["HourDK"] = pd.to_datetime(df["HourDK"], errors="coerce")
        df = df.dropna(subset=["HourDK"]).sort_values("HourDK")
        
        if "SpotPriceDKK" in df.columns:
            df["price_dkk_per_kwh"] = df["SpotPriceDKK"].astype(float) / 1000.0
        
        # Create minute-level index and interpolate
        idx = minute_index(start_date, end_date)
        price_hourly = pd.Series(df["price_dkk_per_kwh"].values, index=df["HourDK"])
        price_minute = price_hourly.reindex(idx, method="ffill").fillna(method="bfill")
        
        return {
            "data": price_minute.to_dict(),
            "index": [str(ts) for ts in price_minute.index],
            "note": None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching price data: {str(e)}")

@router.get("/co2")
async def get_co2_data(
    start_date: date = Query(..., description="Start date"),
    end_date: date = Query(..., description="End date"),
    area: str = Query("DK1", description="Price area (DK1 or DK2)")
):
    """Fetch CO2 intensity data for a date range"""
    try:
        import requests
        # Try historical first
        url = f"{EDS_CO2_HIST_URL}?start={start_date}&end={end_date}&limit=10000"
        r = requests.get(url, timeout=40)
        
        if r.status_code == 200:
            recs = r.json().get("records", [])
            if recs:
                df = pd.DataFrame.from_records(recs)
                if "Minutes5DK" in df.columns:
                    df["Minutes5DK"] = pd.to_datetime(df["Minutes5DK"])
                    df = df.sort_values("Minutes5DK")
                    co2_series = pd.Series(df["CO2Emission"].values, index=df["Minutes5DK"])
                    
                    idx = minute_index(start_date, end_date)
                    co2_minute = co2_series.reindex(idx, method="nearest").fillna(method="ffill")
                    
                    return {
                        "data": co2_minute.to_dict(),
                        "index": [str(ts) for ts in co2_minute.index],
                        "note": None
                    }
        
        # Fallback to default
        idx = minute_index(start_date, end_date)
        co2_series = default_co2_profile(idx)
        return {
            "data": co2_series.to_dict(),
            "index": [str(ts) for ts in co2_series.index],
            "note": "Using synthetic data - API returned no records"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching CO2 data: {str(e)}")

@router.get("/weather")
async def get_weather_data(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    start_date: date = Query(..., description="Start date"),
    end_date: date = Query(..., description="End date")
):
    """Fetch weather data from Open-Meteo"""
    try:
        import requests
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "hourly": "temperature_2m",
            "timezone": "Europe/Copenhagen"
        }
        r = requests.get(url, params=params, timeout=40)
        r.raise_for_status()
        data = r.json()
        
        if "hourly" in data:
            times = pd.to_datetime(data["hourly"]["time"])
            temps = data["hourly"]["temperature_2m"]
            temp_hourly = pd.Series(temps, index=times)
            
            idx = minute_index(start_date, end_date)
            temp_minute = temp_hourly.reindex(idx, method="ffill").fillna(method="bfill")
            
            return {
                "data": temp_minute.to_dict(),
                "index": [str(ts) for ts in temp_minute.index],
                "note": None
            }
        
        # Fallback
        idx = minute_index(start_date, end_date)
        temp_series = synthetic_outdoor_temp(idx)
        return {
            "data": temp_series.to_dict(),
            "index": [str(ts) for ts in temp_series.index],
            "note": "Using synthetic data"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching weather data: {str(e)}")

@router.get("/pv")
async def get_pv_data(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    kwp: float = Query(3.0, description="PV capacity in kWp"),
    start_date: date = Query(..., description="Start date"),
    end_date: date = Query(..., description="End date")
):
    """Generate PV production profile"""
    try:
        idx = minute_index(start_date, end_date)
        
        # Try to use pvlib with weather data
        try:
            location = Location(lat, lon, tz=TZ_DK)
            # Simplified PV calculation - you can enhance this with actual weather
            pv_series = simple_pv_profile(idx, kwp)
        except:
            pv_series = simple_pv_profile(idx, kwp)
        
        return {
            "data": pv_series.to_dict(),
            "index": [str(ts) for ts in pv_series.index],
            "note": None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating PV data: {str(e)}")
