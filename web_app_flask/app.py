"""
Flask-based web application (no Node.js required)
"""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from datetime import date, datetime
import pandas as pd
import sys
from pathlib import Path

# Import original modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from profiles import minute_index, default_price_profile, default_co2_profile, simple_pv_profile, synthetic_outdoor_temp
from devices import WeatherHP, WeatherELheater, WeatherHotTub, DHWTank
from ems import rule_power_share
from Optimization_based import generate_smart_time_slots, assign_data_to_time_slots_single, mpc_opt_single, format_results_single

app = Flask(__name__)
CORS(app)

# Store session data in memory (use Redis/database in production)
sessions = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data/price', methods=['GET'])
def get_price():
    """Fetch electricity price data"""
    try:
        start_date = date.fromisoformat(request.args.get('start_date'))
        end_date = date.fromisoformat(request.args.get('end_date'))
        area = request.args.get('area', 'DK1')
        
        import requests
        url = "https://api.energidataservice.dk/dataset/DayAheadPrices?limit=200000"
        r = requests.get(url, timeout=40)
        
        if r.status_code == 200:
            recs = r.json().get("records", [])
            if recs:
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
                
                idx = minute_index(start_date, end_date)
                price_hourly = pd.Series(df["price_dkk_per_kwh"].values, index=df["HourDK"])
                price_minute = price_hourly.reindex(idx, method="ffill").fillna(method="bfill")
                
                return jsonify({
                    "data": price_minute.to_dict(),
                    "index": [str(ts) for ts in price_minute.index],
                    "note": None
                })
        
        # Fallback
        idx = minute_index(start_date, end_date)
        price_series = default_price_profile(idx)
        return jsonify({
            "data": price_series.to_dict(),
            "index": [str(ts) for ts in price_series.index],
            "note": "Using synthetic data"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/co2', methods=['GET'])
def get_co2():
    """Fetch CO2 intensity data"""
    try:
        start_date = date.fromisoformat(request.args.get('start_date'))
        end_date = date.fromisoformat(request.args.get('end_date'))
        area = request.args.get('area', 'DK1')
        
        import requests
        url = f"https://api.energidataservice.dk/dataset/CO2Emis?start={start_date}&end={end_date}&limit=10000"
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
                    
                    return jsonify({
                        "data": co2_minute.to_dict(),
                        "index": [str(ts) for ts in co2_minute.index],
                        "note": None
                    })
        
        # Fallback
        idx = minute_index(start_date, end_date)
        co2_series = default_co2_profile(idx)
        return jsonify({
            "data": co2_series.to_dict(),
            "index": [str(ts) for ts in co2_series.index],
            "note": "Using synthetic data"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/weather', methods=['GET'])
def get_weather():
    """Fetch weather data"""
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        start_date = date.fromisoformat(request.args.get('start_date'))
        end_date = date.fromisoformat(request.args.get('end_date'))
        
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
        
        if r.status_code == 200:
            data = r.json()
            if "hourly" in data:
                times = pd.to_datetime(data["hourly"]["time"])
                temps = data["hourly"]["temperature_2m"]
                temp_hourly = pd.Series(temps, index=times)
                
                idx = minute_index(start_date, end_date)
                temp_minute = temp_hourly.reindex(idx, method="ffill").fillna(method="bfill")
                
                return jsonify({
                    "data": temp_minute.to_dict(),
                    "index": [str(ts) for ts in temp_minute.index],
                    "note": None
                })
        
        # Fallback
        idx = minute_index(start_date, end_date)
        temp_series = synthetic_outdoor_temp(idx)
        return jsonify({
            "data": temp_series.to_dict(),
            "index": [str(ts) for ts in temp_series.index],
            "note": "Using synthetic data"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/devices/calculate-load', methods=['POST'])
def calculate_load():
    """Calculate load profile from devices"""
    try:
        data = request.json
        devices = data.get('devices', [])
        start_date = date.fromisoformat(data['start_date'])
        end_date = date.fromisoformat(data['end_date'])
        outdoor_temp = data.get('outdoor_temp')
        
        idx = minute_index(start_date, end_date)
        
        if outdoor_temp:
            tout_series = pd.Series(outdoor_temp["data"], index=pd.to_datetime(outdoor_temp["index"]))
            tout_series = tout_series.reindex(idx, method="nearest")
        else:
            tout_series = pd.Series(10.0, index=idx)
        
        total_load = pd.Series(0.0, index=idx)
        device_outputs = {}
        
        for device_config in devices:
            dev_type = device_config.get("type")
            params = device_config.get("params", {})
            
            if dev_type == "heat_pump":
                hp = WeatherHP(**params)
                p_kw, ti_c = hp.series_kw(idx, tout_series)
                total_load += p_kw
            elif dev_type == "electric_heater":
                eh = WeatherELheater(**params)
                p_kw, ti_c = eh.series_kw(idx, tout_series)
                total_load += p_kw
            elif dev_type == "hot_tub":
                ht = WeatherHotTub(**params)
                p_kw, tw_c = ht.series_kw(idx, tout_series)
                total_load += p_kw
            elif dev_type == "dhw_tank":
                dhw = DHWTank(**params)
                p_kw, tt_c = dhw.series_kw(idx, tout_series)
                total_load += p_kw
        
        return jsonify({
            "total_load_kw": total_load.to_dict(),
            "index": [str(ts) for ts in idx],
            "device_outputs": device_outputs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ems/run', methods=['POST'])
def run_ems():
    """Run EMS"""
    try:
        data = request.json
        load_kw = data['load_kw']
        pv_kw = data.get('pv_kw')
        plan_slots = data['plan_slots']
        battery_config = data['battery_config']
        energy_pattern = data.get('energy_pattern', 2)
        
        idx = pd.to_datetime(load_kw["index"])
        load_series = pd.Series(load_kw["data"], index=idx)
        
        if pv_kw:
            pv_series = pd.Series(pv_kw["data"], index=pd.to_datetime(pv_kw["index"]))
            pv_series = pv_series.reindex(idx, fill_value=0.0)
        else:
            pv_series = pd.Series(0.0, index=idx)
        
        plan_df = pd.DataFrame(plan_slots)
        if "start" in plan_df.columns:
            plan_df["start"] = pd.to_datetime(plan_df["start"].astype(str)).dt.time
        if "end" in plan_df.columns:
            plan_df["end"] = pd.to_datetime(plan_df["end"].astype(str)).dt.time
        
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
        
        grid_import = result["grid_import_kw"]
        batt_charge = result["batt_charge_kw"]
        batt_discharge = result["batt_discharge_kw"]
        soc = result["batt_soc_kwh"]
        pv_used = result["pv_used_to_load_kw"]
        
        dt_h = (idx[1] - idx[0]).total_seconds() / 3600.0 if len(idx) > 1 else 1/60.0
        
        total_load_kwh = load_series.sum() * dt_h
        total_pv_kwh = pv_series.sum() * dt_h
        total_grid_kwh = grid_import.sum() * dt_h
        total_pv_used_kwh = pv_used.sum() * dt_h
        
        self_consumption_pct = (total_pv_used_kwh / total_pv_kwh * 100) if total_pv_kwh > 0 else 0
        self_sufficiency_pct = (total_pv_used_kwh / total_load_kwh * 100) if total_load_kwh > 0 else 0
        
        return jsonify({
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
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
