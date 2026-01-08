from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from flask_session import Session
from datetime import datetime, date, timedelta, time as _time
import pandas as pd
import numpy as np
from pathlib import Path
import json
import uuid
import os

# Import functions from the original app.py
from app import (
    _fetch_dayahead_prices_latest, _fetch_elspot_prices,
    fetch_co2_for_day, fetch_weather_open_meteo, plot_period_bar, plot_period_minute,
    build_series_for_analysis, rule_power_share, generate_smart_time_slots,
    mpc_opt_single, format_results_single, daily_price_dual, daily_co2_with_note,
    daily_temperature_with_note, minute_index, DEVICE_CATEGORIES, DEVICE_LABEL_MAP,
    get_default_config, build_minute_profile, compute_daily_profiles,
    build_house_layout_figure, resolve_display_label,
    suggest_best_interval_for_day
)

# Import thermal device models
from devices import WeatherHP, WeatherELheater, WeatherHotTub, DHWTank

# Helper function for extracting icons (if not imported)
def extract_icon(label: str) -> str:
    """Return only the emoji part of a label."""
    if " " in label:
        return label.split(" ", 1)[0]
    return label

def get_house_thermal_params(house_info: dict = None):
    """Derive UA, C_th from house_info (Flask-compatible version)."""
    if house_info is None:
        house_info = {"size": "Medium house", "insulation": "Average", "residents": 2}
    
    size_str = (house_info.get("size") or "Medium house").lower()
    ins_str = house_info.get("insulation", "Average")

    # base UA + Cth by size
    if "small" in size_str:
        ua_base = 0.10   # kW/°C
        Cth_base = 0.50  # kWh/°C
    elif "large" in size_str:
        ua_base = 0.14
        Cth_base = 0.75
    else:
        ua_base = 0.12
        Cth_base = 0.60

    # insulation factor
    if ins_str == "Poor":
        ua = ua_base * 1.3
    elif ins_str == "Good":
        ua = ua_base * 0.7
    else:
        ua = ua_base

    # default comfort band
    t_min_default = 20.0
    t_max_default = 22.0

    return {
        "ua_kw_per_c": ua,
        "C_th_kwh_per_c": Cth_base,
        "t_min_default": t_min_default,
        "t_max_default": t_max_default,
    }

def get_default_config_web(dev_type: str, category: str, house_info: dict) -> dict:
    """Get default config for a device based on house_info (Flask version)."""
    base = dict(
        power_kw=0.5,
        start="18:00",  # Use string instead of time object for JSON serialization
        duration_min=60,
    )
    
    # Map house size to small/medium/large
    size_str = (house_info.get("size") or "Medium house").lower()
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
    
    # Fixed electrical devices
    if category == "elec_fixed":
        defaults = {
            "lights": {
                "num_devices": _by_size(8, 12, 18),
                "power_w": 8.0,
                "intervals": [
                    {"start": _time(6, 0), "end": _time(8, 0)},
                    {"start": _time(17, 0), "end": _time(23, 0)},
                ],
            },
            "fridge": {
                "num_devices": 1,
                "power_w": 80.0,
                "intervals": [{"start": _time(0, 0), "end": _time(23, 59)}],
            },
            "range_hood": {
                "num_devices": 1,
                "power_w": 120.0,
                "intervals": [{"start": _time(18, 0), "end": _time(19, 0)}],
            },
            "oven": {
                "num_devices": _by_size(1, 1, 2),
                "power_w": 2000.0,
                "intervals": [{"start": _time(18, 0), "end": _time(19, 0)}],
            },
            "induction": {
                "num_devices": 1,
                "power_w": 2500.0,
                "intervals": [{"start": _time(17, 30), "end": _time(19, 0)}],
            },
            "microwave": {
                "num_devices": 1,
                "power_w": 1200.0,
                "intervals": [
                    {"start": _time(7, 0), "end": _time(7, 15)},
                    {"start": _time(12, 0), "end": _time(12, 15)},
                    {"start": _time(21, 0), "end": _time(21, 15)},
                ],
            },
            "tv": {
                "num_devices": _by_size(1, 1, 2),
                "power_w": 80.0,
                "intervals": [{"start": _time(19, 0), "end": _time(23, 0)}],
            },
            "router": {
                "num_devices": _by_size(1, 2, 3),
                "power_w": 10.0,
                "intervals": [{"start": _time(0, 0), "end": _time(23, 59)}],
            },
            "pc_desktop": {
                "num_devices": _by_size(1, 1, 2),
                "power_w": 150.0,
                "intervals": [{"start": _time(9, 0), "end": _time(17, 0)}],
            },
            "laptop": {
                "num_devices": _by_size(1, 2, 3),
                "power_w": 60.0,
                "intervals": [
                    {"start": _time(9, 0), "end": _time(12, 0)},
                    {"start": _time(19, 0), "end": _time(23, 0)},
                ],
            },
            "game_console": {
                "num_devices": _by_size(1, 1, 2),
                "power_w": 120.0,
                "intervals": [{"start": _time(20, 0), "end": _time(22, 0)}],
            },
            "printer": {
                "num_devices": _by_size(1, 1, 2),
                "power_w": 40.0,
                "intervals": [{"start": _time(10, 0), "end": _time(12, 0)}],
            },
            "ventilation": {
                "num_devices": 1,
                "power_w": 60.0,
                "intervals": [{"start": _time(0, 0), "end": _time(23, 59)}],
            },
            "humidifier": {
                "num_devices": _by_size(1, 1, 2),
                "power_w": 40.0,
                "intervals": [{"start": _time(22, 0), "end": _time(7, 0)}],
            },
            "baby_monitor": {
                "num_devices": _by_size(1, 1, 1),
                "power_w": 5.0,
                "intervals": [{"start": _time(19, 0), "end": _time(7, 0)}],
            },
            "smoke_detector": {
                "num_devices": _by_size(2, 3, 4),
                "power_w": 2.0,
                "intervals": [{"start": _time(0, 0), "end": _time(23, 59)}],
            },
            "standby": {
                "num_devices": 1,
                "power_w": _by_size(30.0, 50.0, 80.0),
                "intervals": [{"start": _time(0, 0), "end": _time(23, 59)}],
            },
            "other_fixed_1": {
                "num_devices": 1,
                "power_w": 100.0,
                "intervals": [{"start": _time(18, 0), "end": _time(22, 0)}],
            },
            "other_fixed_2": {
                "num_devices": 1,
                "power_w": 100.0,
                "intervals": [{"start": _time(18, 0), "end": _time(22, 0)}],
            },
            "other_fixed_3": {
                "num_devices": 1,
                "power_w": 100.0,
                "intervals": [{"start": _time(18, 0), "end": _time(22, 0)}],
            },
        }
        
        cfg = defaults.get(dev_type, {}).copy()
        if not cfg:
            return base
        
        # Convert time objects to strings for JSON serialization
        if "intervals" in cfg:
            try:
                cfg["intervals"] = [
                    {
                        "start": f"{iv['start'].hour:02d}:{iv['start'].minute:02d}",
                        "end": f"{iv['end'].hour:02d}:{iv['end'].minute:02d}"
                    }
                    for iv in cfg["intervals"]
                ]
            except Exception as e:
                print(f"ERROR converting intervals for {dev_type}: {e}")
                print(f"Intervals data: {cfg.get('intervals')}")
                raise
        
        cfg["power_kw"] = cfg["power_w"] / 1000.0
        return {**base, **cfg}
    
    # Flexible electrical devices
    if category == "elec_flex":
        defaults = {
            "wm": {
                "num_devices": 1,
                "power_w": 1200.0,
                "start": "19:00",
                "duration_min": 90,
                "w_cost": 1.0,
            },
            "dw": {
                "num_devices": 1,
                "power_w": 1400.0,
                "start": "21:00",
                "duration_min": 90,
                "w_cost": 1.0,
            },
            "dryer": {
                "num_devices": 1,
                "power_w": 2000.0,
                "start": "20:00",
                "duration_min": 60,
                "w_cost": 1.0,
            },
            "robot_vac": {
                "num_devices": 1,
                "power_w": 250.0,
                "start": "11:00",
                "duration_min": 60,
                "w_cost": 1.0,
            },
            "workshop": {
                "num_devices": 1,
                "power_w": 700.0,
                "start": "17:00",
                "duration_min": 120,
                "w_cost": 1.0,
            },
            "other_flex_1": {
                "num_devices": 1,
                "power_w": 1000.0,
                "start": "18:00",
                "duration_min": 60,
                "w_cost": 1.0,
            },
            "other_flex_2": {
                "num_devices": 1,
                "power_w": 1000.0,
                "start": "18:00",
                "duration_min": 60,
                "w_cost": 1.0,
            },
            "other_flex_3": {
                "num_devices": 1,
                "power_w": 1000.0,
                "start": "18:00",
                "duration_min": 60,
                "w_cost": 1.0,
            },
        }
        
        cfg = defaults.get(dev_type, {}).copy()
        if not cfg:
            return base
        
        # Calculate end time from start + duration
        start_str = cfg.get("start", "18:00")
        start_parts = start_str.split(":")
        start_hour = int(start_parts[0])
        start_min = int(start_parts[1])
        duration_min = cfg.get("duration_min", 60)
        
        # Calculate end time
        total_minutes = start_hour * 60 + start_min + duration_min
        end_hour = (total_minutes // 60) % 24
        end_min = total_minutes % 60
        
        # Create single interval
        cfg["intervals"] = [
            {
                "start": start_str,
                "end": f"{end_hour:02d}:{end_min:02d}"
            }
        ]
        
        cfg["power_kw"] = cfg["power_w"] / 1000.0
        return {**base, **cfg}
    
    # Thermal devices
    if category == "thermal":
        # Get house thermal parameters for building model
        thermal_params = get_house_thermal_params(house_info)
        ua_base = thermal_params.get("ua_kw_per_c", 0.12)
        cth_base = thermal_params.get("C_th_kwh_per_c", 0.60)
        
        # House-size-based heater capacity (from get_thermal_building_params in Streamlit)
        # Small: 4 kW, Medium: 6 kW, Large: 8 kW
        q_guess = _by_size(4.0, 6.0, 8.0)
        
        # DHW volume based on house size
        vol_default = _by_size(160.0, 200.0, 300.0)
        
        # DHW usage based on residents
        n_res = int(house_info.get("residents", 2))
        if n_res <= 2:
            usage_default = "Low"
        elif n_res <= 4:
            usage_default = "Medium"
        else:
            usage_default = "High"
        
        defaults = {
            "space_heat": {
                "space_mode": "None (external supply)",
                "t_min_c": 20.0,
                "t_max_c": 22.0,
                "q_kw": q_guess,  # House-size based
                "hp_type": "Fixed power",
                "distribution": "Radiators",
                "ua_kw_per_c": ua_base,
                "C_th_kwh_per_c": cth_base,
            },
            "dhw": {
                "dhw_mode": "None (external supply)",
                "volume_l": vol_default,  # House-size based
                "usage_level": usage_default,  # Resident-count based
                "t_min_c": 45.0,
                "t_max_c": 55.0,
                "p_el_kw": 2.0,
            },
            "leisure": {
                "hot_tub_enabled": False,
                "pool_enabled": False,
                # Hot tub defaults
                "ht_target_c": 40.0,
                "ht_idle_c": 30.0,
                "ht_water_l": 1200.0,
                "ht_heater_kw": 5.0,
                "ht_insulation": "Average",
                "ht_sessions": [],
                # Pool defaults
                "pool_target_c": 28.0,
                "pool_idle_c": 24.0,
                "pool_water_l": 30000.0,
                "pool_heater_kw": 8.0,
                "pool_insulation": "Average",
                "pool_sessions": [],
            },
        }
        
        cfg = defaults.get(dev_type, {}).copy()
        if not cfg:
            cfg = {}
        
        # Add base fields
        cfg.setdefault("power_kw", base["power_kw"])
        
        return {**base, **cfg}
    
    # For other categories, return base for now
    return base

app = Flask(__name__)
# Use a string secret key instead of bytes to avoid encoding issues
app.secret_key = os.urandom(24).hex()

# Configure server-side sessions to avoid cookie size limits
# This stores session data on the server (filesystem) instead of in cookies
# Only a small session ID is stored in the cookie, not the entire session data
app.config['SESSION_TYPE'] = 'filesystem'
session_dir = Path('flask_session')
session_dir.mkdir(exist_ok=True)
app.config['SESSION_FILE_DIR'] = str(session_dir.absolute())
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = False  # Disable signer to avoid bytes/string encoding issues
app.config['SESSION_KEY_PREFIX'] = 'session:'
app.config['SESSION_COOKIE_NAME'] = 'ems_session'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
# Set session lifetime to 24 hours (sessions will expire after 24 hours of inactivity)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

Session(app)

# Clean up old session files on startup (older than 7 days)
import time
def cleanup_old_sessions():
    """Remove session files older than 7 days."""
    try:
        if session_dir.exists():
            now = time.time()
            for session_file in session_dir.glob('*'):
                if session_file.is_file():
                    # Check if file is older than 7 days
                    if now - session_file.stat().st_mtime > 7 * 24 * 3600:
                        try:
                            session_file.unlink()
                        except:
                            pass
    except:
        pass

cleanup_old_sessions()

ADMIN_PASSWORD = "FCCOGEN"
LOG_PATH = Path("usage_log.csv")

def get_admin_stats():
    """Process usage log and return statistics for admin page."""
    if not LOG_PATH.exists():
        return None
    
    df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
    
    # Convert usage_over_time dates to strings for JSON serialization
    usage_over_time = df.groupby(df["timestamp"].dt.date).size()
    usage_over_time_dict = {str(date): int(count) for date, count in usage_over_time.items()}
    
    # Calculate location distribution
    location_counts = df["location"].value_counts().to_dict()
    
    stats = {
        "total_clicks": len(df),
        "unique_sessions": df["session_id"].nunique(),
        "unique_locations": df["location"].nunique(),
        "occupation_counts": df["occupation"].value_counts().to_dict(),
        "location_counts": location_counts,
        "usage_over_time": usage_over_time_dict,
        "raw_log": df.to_dict("records")
    }
    
    return stats

@app.route('/')
def index():
    """Front page for user profile setup."""
    is_logged_in = 'user_profile' in session
    user_profile = session.get('user_profile', {})
    return render_template('index.html', is_logged_in=is_logged_in, user_profile=user_profile)

@app.route('/login', methods=['POST'])
def login():
    """Handle user profile submission."""
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()
    
    occupation = data.get('occupation', '').strip()
    location = data.get('location', '').strip()
    
    if not occupation or not location:
        return jsonify({'success': False, 'error': 'Please fill in all fields'}), 400
    
    # Generate session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    # Create profile
    profile = {
        'occupation': occupation,
        'location': location
    }
    
    # Log to CSV (adapting the function to work without Streamlit)
    clicked_ts = datetime.now().isoformat(timespec="seconds")
    is_new = not LOG_PATH.exists()
    
    df_row = pd.DataFrame([{
        "timestamp": clicked_ts,
        "occupation": occupation,
        "location": location,
        "session_id": session.get("session_id", ""),
    }])
    
    df_row.to_csv(
        LOG_PATH,
        index=False,
        mode="w" if is_new else "a",
        header=is_new,
    )
    
    # Store in session
    session['user_profile'] = profile
    session['user_profile_confirmed'] = True
    
    return jsonify({'success': True, 'redirect': '/scenario'})

@app.route('/logout')
def logout():
    """Clear user session."""
    session.clear()
    return redirect(url_for('index'))

@app.route('/scenario')
def scenario():
    """Page 1: Scenario & Data."""
    if 'user_profile' not in session:
        return redirect(url_for('index'))
    
    today = date.today()
    period_start = today - timedelta(days=14)
    period_end = today
    selected_day = today
    
    # Initialize day in session if not exists
    if 'day' not in session:
        session['day'] = selected_day
    
    return render_template('scenario.html', 
                         period_start=period_start.strftime('%Y-%m-%d'),
                         period_end=period_end.strftime('%Y-%m-%d'),
                         selected_day=selected_day.strftime('%Y-%m-%d'),
                         current_day=today.strftime('%Y-%m-%d'))

@app.route('/devices')
def devices():
    """Page 2: Devices & Layout."""
    if 'user_profile' not in session:
        return redirect(url_for('index'))
    
    # Initialize house_info if not exists
    if 'house_info' not in session:
        session['house_info'] = {
            'size': 'Medium house',
            'insulation': 'Average',
            'residents': 2
        }
    
    # Initialize device selection and configs if not exists
    if 'device_selection' not in session:
        session['device_selection'] = {}
    if 'device_configs' not in session:
        session['device_configs'] = {}
    
    return render_template('devices.html', 
                         house_info=session.get('house_info', {}),
                         device_categories=DEVICE_CATEGORIES,
                         device_selection=session.get('device_selection', {}),
                         device_configs=session.get('device_configs', {}))

@app.route('/analysis')
def analysis():
    """Page 3: Analysis."""
    if 'user_profile' not in session:
        return redirect(url_for('index'))
    
    return render_template('analysis.html')

@app.route('/admin')
def admin():
    """Admin statistics page."""
    verified = request.args.get('verified') == '1'
    if not verified:
        return redirect(url_for('index'))
    
    stats = get_admin_stats()
    return render_template('admin.html', stats=stats)

@app.route('/api/admin-stats', methods=['POST'])
def api_admin_stats():
    """API endpoint for admin password verification."""
    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Invalid request format'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        password = data.get('password', '')
        
        if not password:
            return jsonify({'success': False, 'error': 'Password is required'}), 400
        
        if password != ADMIN_PASSWORD:
            return jsonify({'success': False, 'error': 'Invalid password'}), 401
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/download-log')
def download_log():
    """Download usage log CSV."""
    if not LOG_PATH.exists():
        return jsonify({'error': 'Log file not found'}), 404
    
    return send_file(str(LOG_PATH), as_attachment=True, download_name='usage_log.csv')

@app.route('/api/save-house-info', methods=['POST'])
def save_house_info():
    """Save house information to session."""
    data = request.get_json()
    session['house_info'] = {
        'size': data.get('size', 'Medium house'),
        'insulation': data.get('insulation', 'Average'),
        'residents': int(data.get('residents', 2))
    }
    return jsonify({'success': True})

@app.route('/api/fetch-price-data', methods=['POST'])
def fetch_price_data():
    """Fetch electricity price data for period."""
    data = request.get_json()
    period_start = datetime.strptime(data['period_start'], '%Y-%m-%d').date()
    period_end = datetime.strptime(data['period_end'], '%Y-%m-%d').date()
    selected_day = datetime.strptime(data['selected_day'], '%Y-%m-%d').date()
    area = data.get('area', 'DK1')
    
    try:
        idx = minute_index(period_start, period_end)
        price_plot, note = daily_price_dual(idx, period_start, period_end, area)
        
        # Store in session for later use (e.g., suggest interval)
        session['price_daily'] = {
            'index': [d.isoformat() for d in price_plot.index],
            'values': price_plot.values.tolist()
        }
        session['day'] = selected_day
        session.modified = True  # Mark session as modified for Flask-Session
        
        # Convert to JSON-serializable format
        period_data = {
            'index': [d.isoformat() for d in price_plot.index],
            'values': price_plot.values.tolist()
        }
        
        # Extract selected day data
        selected_date = datetime.combine(selected_day, datetime.min.time())
        selected_mask = pd.to_datetime(price_plot.index).date == selected_day
        selected_day_data = {
            'index': [d.isoformat() for d in price_plot.index[selected_mask]],
            'values': price_plot.values[selected_mask].tolist()
        }
        
        return jsonify({
            'success': True,
            'period_data': period_data,
            'selected_day_data': selected_day_data,
            'note': note
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/fetch-co2-data', methods=['POST'])
def fetch_co2_data():
    """Fetch CO2 intensity data for period."""
    data = request.get_json()
    period_start = datetime.strptime(data['period_start'], '%Y-%m-%d').date()
    period_end = datetime.strptime(data['period_end'], '%Y-%m-%d').date()
    selected_day = datetime.strptime(data['selected_day'], '%Y-%m-%d').date()
    area = data.get('area', 'DK1')
    
    try:
        idx = minute_index(period_start, period_end)
        co2_series, note = daily_co2_with_note(idx, period_start, period_end, area)
        
        # Store in session for later use (e.g., suggest interval)
        session['co2_daily'] = {
            'index': [d.isoformat() for d in co2_series.index],
            'values': co2_series.values.tolist()
        }
        session.modified = True  # Mark session as modified for Flask-Session
        
        # Convert to JSON-serializable format
        period_data = {
            'index': [d.isoformat() for d in co2_series.index],
            'values': co2_series.values.tolist()
        }
        
        # Extract selected day data
        selected_mask = pd.to_datetime(co2_series.index).date == selected_day
        selected_day_data = {
            'index': [d.isoformat() for d in co2_series.index[selected_mask]],
            'values': co2_series.values[selected_mask].tolist()
        }
        
        return jsonify({
            'success': True,
            'period_data': period_data,
            'selected_day_data': selected_day_data,
            'note': note
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/fetch-weather', methods=['POST'])
def fetch_weather():
    """Fetch weather data for period."""
    data = request.get_json()
    lat = float(data['lat'])
    lon = float(data['lon'])
    period_start = datetime.strptime(data['period_start'], '%Y-%m-%d').date()
    period_end = datetime.strptime(data['period_end'], '%Y-%m-%d').date()
    selected_day = datetime.strptime(data['selected_day'], '%Y-%m-%d').date()
    
    try:
        idx = minute_index(period_start, period_end)
        weather_hr = fetch_weather_open_meteo(
            lat, lon,
            start_date=period_start,
            end_date=period_end,
            tz="Europe/Copenhagen"
        )
        temp_series, note = daily_temperature_with_note(idx, weather_hr)
        
        # Convert to JSON-serializable format
        period_data = {
            'index': [d.isoformat() for d in temp_series.index],
            'values': temp_series.values.tolist()
        }
        
        # Extract selected day data
        selected_mask = pd.to_datetime(temp_series.index).date == selected_day
        selected_day_data = {
            'index': [d.isoformat() for d in temp_series.index[selected_mask]],
            'values': temp_series.values[selected_mask].tolist()
        }
        
        # IMPORTANT: Store temperature data in session for thermal device calculations
        session['temp_daily'] = period_data
        session.modified = True
        print(f"DEBUG fetch_weather: Stored temp_daily with {len(period_data['values'])} data points")
        
        return jsonify({
            'success': True,
            'period_data': period_data,
            'selected_day_data': selected_day_data,
            'note': note
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/save-device-selection', methods=['POST'])
def save_device_selection():
    """Save device selection to session."""
    data = request.get_json()
    if 'device_selection' not in session:
        session['device_selection'] = {}
    session['device_selection'].update(data.get('selection', {}))
    session.modified = True
    return jsonify({'success': True})

@app.route('/api/save-device-config', methods=['POST'])
def save_device_config():
    """Save device configuration to session."""
    data = request.get_json()
    if 'device_configs' not in session:
        session['device_configs'] = {}
    full_key = data.get('full_key')
    config = data.get('config', {})
    if full_key:
        # If intervals are provided, regenerate the profile
        # This ensures the profile matches the interval settings
        if 'intervals' in config and config['intervals']:
            try:
                cat_key = full_key.split(':')[0] if ':' in full_key else ''
                if cat_key in ('elec_fixed', 'elec_flex'):
                    # Convert string times to time objects for profile generation
                    intervals_for_profile = []
                    for iv in config['intervals']:
                        if isinstance(iv, dict):
                            start_val = iv.get('start')
                            end_val = iv.get('end')
                            
                            # Convert string to time object
                            def str_to_time(val):
                                if isinstance(val, _time):
                                    return val
                                if isinstance(val, str) and ':' in val:
                                    parts = val.split(':')
                                    return _time(int(parts[0]), int(parts[1]))
                                return _time(18, 0)
                            
                            intervals_for_profile.append({
                                'start': str_to_time(start_val),
                                'end': str_to_time(end_val)
                            })
                    
                    if intervals_for_profile:
                        # Get power and num_devices
                        num_devices = int(config.get('num_devices', 1))
                        power_w = float(config.get('power_w', config.get('power_kw', 0.5) * 1000.0))
                        total_power_w = power_w * num_devices
                        
                        # Generate profile
                        prof = build_minute_profile(
                            power_w=total_power_w,
                            intervals=intervals_for_profile,
                            step_min=1
                        )
                        
                        # Store profile in config
                        config['profile_index'] = prof.index.astype(str).tolist()
                        config['profile_kw'] = prof.values.tolist()
                        print(f"DEBUG save-device-config: Regenerated profile for {full_key}, intervals={intervals_for_profile}")
            except Exception as e:
                print(f"ERROR regenerating profile for {full_key}: {e}")
                import traceback
                traceback.print_exc()
        
        session['device_configs'][full_key] = config
        session.modified = True
        # Force session to be saved
        try:
            session.permanent = True
        except:
            pass
    return jsonify({'success': True, 'saved': full_key is not None})

@app.route('/api/get-device-config', methods=['POST'])
def get_device_config():
    """Get current device configuration (saved or default)."""
    data = request.get_json()
    full_key = data.get('full_key')
    dev_type = data.get('dev_type')
    category = data.get('category')
    
    # First check if we have a saved config
    if 'device_configs' in session and full_key in session['device_configs']:
        saved_config = session['device_configs'][full_key]
        if saved_config and len(saved_config) > 0:
            return jsonify({'success': True, 'config': saved_config})
    
    # No saved config, return empty so frontend can fetch defaults
    return jsonify({'success': True, 'config': {}})

@app.route('/api/get-device-default', methods=['POST'])
def get_device_default():
    """Get default configuration for a device based on house_info."""
    data = request.get_json()
    dev_type = data.get('dev_type')
    category = data.get('category')
    house_info = session.get('house_info', {
        'size': 'Medium house',
        'insulation': 'Average',
        'residents': 2
    })
    
    def time_to_str(val):
        """Convert time object to string."""
        if isinstance(val, _time):
            return f"{val.hour:02d}:{val.minute:02d}"
        return val
    
    try:
        default_config = get_default_config_web(dev_type, category, house_info)
        print(f"DEBUG get-device-default: dev_type={dev_type}, category={category}, config keys={list(default_config.keys())}")
        if 'intervals' in default_config:
            print(f"DEBUG intervals in default: {default_config['intervals']}")
        
        # Ensure all time objects are converted to strings for JSON serialization
        config_serializable = {}
        for key, value in default_config.items():
            if isinstance(value, _time):
                config_serializable[key] = time_to_str(value)
            elif key == 'intervals' and isinstance(value, list):
                # Convert interval start/end times to strings
                serialized_intervals = []
                for iv in value:
                    if isinstance(iv, dict):
                        serialized_iv = {}
                        for iv_key, iv_val in iv.items():
                            serialized_iv[iv_key] = time_to_str(iv_val)
                        serialized_intervals.append(serialized_iv)
                    else:
                        serialized_intervals.append(iv)
                config_serializable[key] = serialized_intervals
            else:
                config_serializable[key] = value
        
        # Generate profile for the default config (for elec_fixed and elec_flex)
        if category in ('elec_fixed', 'elec_flex') and 'intervals' in default_config:
            try:
                intervals_for_profile = default_config['intervals']
                num_devices = int(default_config.get('num_devices', 1))
                power_w = float(default_config.get('power_w', default_config.get('power_kw', 0.5) * 1000.0))
                total_power_w = power_w * num_devices
                
                prof = build_minute_profile(
                    power_w=total_power_w,
                    intervals=intervals_for_profile,
                    step_min=1
                )
                
                config_serializable['profile_index'] = prof.index.astype(str).tolist()
                config_serializable['profile_kw'] = prof.values.tolist()
                print(f"DEBUG get-device-default: Generated profile for {dev_type}, power={total_power_w}W, intervals={len(intervals_for_profile)}")
            except Exception as e:
                print(f"ERROR generating profile in get-device-default: {e}")
                import traceback
                traceback.print_exc()
        
        return jsonify({'success': True, 'config': config_serializable})
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        print(f"ERROR in get-device-default: {error_msg}")
        return jsonify({'success': False, 'error': error_msg}), 500

@app.route('/api/suggest-interval', methods=['POST'])
def suggest_interval():
    """Suggest best interval for a flexible device based on price/CO2."""
    try:
        data = request.get_json()
        print(f"\n{'='*60}")
        print(f"DEBUG suggest-interval: NEW REQUEST")
        print(f"DEBUG suggest-interval: received data: {data}")
        duration_min = int(data.get('duration_min', 90))
        w_cost = float(data.get('w_cost', 1.0))
        earliest_str = data.get('earliest')  # "HH:MM" or None
        latest_str = data.get('latest')      # "HH:MM" or None
        
        # Get price and CO2 data from session
        # Note: Flask sessions can't store pandas Series directly, so we need to reconstruct them
        # The data should be stored as dict with 'index' and 'values' keys
        # Force session to be loaded from filesystem
        session.modified = True
        price_data = session.get('price_daily')
        co2_data = session.get('co2_daily')
        selected_day = session.get('day', date.today())
        
        print(f"DEBUG suggest-interval: price_data type: {type(price_data)}, co2_data type: {type(co2_data)}")
        print(f"DEBUG suggest-interval: selected_day: {selected_day}")
        print(f"DEBUG suggest-interval: session keys: {list(session.keys())}")
        if price_data:
            print(f"DEBUG suggest-interval: price_data keys: {list(price_data.keys()) if isinstance(price_data, dict) else 'N/A'}")
        if co2_data:
            print(f"DEBUG suggest-interval: co2_data keys: {list(co2_data.keys()) if isinstance(co2_data, dict) else 'N/A'}")
        
        if price_data is None or co2_data is None:
            missing = []
            if price_data is None:
                missing.append('price')
            if co2_data is None:
                missing.append('CO2')
            error_msg = f'{", ".join(missing).capitalize()} data not available. Please go to page 1 (Scenario & Data), click "Fetch Data" button, then try again.'
            print(f"ERROR suggest-interval: {error_msg}")
            return jsonify({'success': False, 'error': error_msg}), 400
        
        # Convert to pandas Series
        import pandas as pd
        try:
            if isinstance(price_data, dict) and 'index' in price_data:
                price_series = pd.Series(
                    price_data['values'],
                    index=pd.to_datetime(price_data['index']),
                    name='price_dkk_per_kwh'
                )
                print(f"DEBUG suggest-interval: price_series created, length={len(price_series)}")
            elif isinstance(price_data, pd.Series):
                price_series = price_data
            else:
                error_msg = f'Invalid price data format: {type(price_data)}, keys={list(price_data.keys()) if isinstance(price_data, dict) else "N/A"}'
                print(f"ERROR suggest-interval: {error_msg}")
                return jsonify({'success': False, 'error': error_msg}), 400
            
            if isinstance(co2_data, dict) and 'index' in co2_data:
                co2_series = pd.Series(
                    co2_data['values'],
                    index=pd.to_datetime(co2_data['index']),
                    name='gCO2_per_kWh'
                )
                print(f"DEBUG suggest-interval: co2_series created, length={len(co2_series)}")
            elif isinstance(co2_data, pd.Series):
                co2_series = co2_data
            else:
                error_msg = f'Invalid CO2 data format: {type(co2_data)}, keys={list(co2_data.keys()) if isinstance(co2_data, dict) else "N/A"}'
                print(f"ERROR suggest-interval: {error_msg}")
                return jsonify({'success': False, 'error': error_msg}), 400
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f'Error converting data to Series: {str(e)}'
            print(f"ERROR suggest-interval: {error_msg}")
            return jsonify({'success': False, 'error': error_msg}), 400
        
        # Convert earliest/latest strings to time objects
        earliest = None
        latest = None
        if earliest_str:
            parts = earliest_str.split(':')
            earliest = _time(int(parts[0]), int(parts[1]))
        if latest_str:
            parts = latest_str.split(':')
            latest = _time(int(parts[0]), int(parts[1]))
        
        # Call the suggest function (we need to adapt it for Flask)
        # Since suggest_best_interval_for_day uses st.session_state, we need to create a Flask-compatible version
        interval = suggest_best_interval_for_day_flask(
            price_series, co2_series, selected_day, duration_min, w_cost, earliest, latest
        )
        
        if interval is None:
            error_msg = 'Could not suggest interval. Check that price/CO2 data is available for the selected day.'
            print(f"ERROR suggest-interval: {error_msg}")
            return jsonify({'success': False, 'error': error_msg}), 400
        
        # Convert time objects to strings
        result = {
            'start': f"{interval['start'].hour:02d}:{interval['start'].minute:02d}",
            'end': f"{interval['end'].hour:02d}:{interval['end'].minute:02d}"
        }
        
        print(f"DEBUG suggest-interval: suggested interval: {result}")
        print(f"{'='*60}\n")
        return jsonify({'success': True, 'interval': result})
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        print(f"ERROR in suggest-interval: {error_msg}")
        return jsonify({'success': False, 'error': error_msg}), 500

def suggest_best_interval_for_day_flask(price, co2, sel_day, duration_min, w_cost=0.5, earliest=None, latest=None):
    """Flask-compatible version of suggest_best_interval_for_day."""
    import pandas as pd
    import numpy as np
    
    print(f"DEBUG suggest_flask: earliest={earliest}, latest={latest}, duration={duration_min}, w_cost={w_cost}")
    
    if price is None or co2 is None or len(price) == 0:
        print("DEBUG suggest_flask: No price/co2 data")
        return None
    
    df = pd.DataFrame(index=price.index.copy())
    df["price"] = np.asarray(price, dtype=float)
    df["co2"] = co2.reindex(df.index).interpolate().bfill().ffill()
    
    day_start = pd.Timestamp(sel_day)
    day_end = day_start + pd.Timedelta(days=1)
    df = df.loc[(df.index >= day_start) & (df.index < day_end)]
    print(f"DEBUG suggest_flask: day_start={day_start}, day_end={day_end}, df length after day filter={len(df)}")
    if df.empty:
        print("ERROR suggest_flask: No data for selected day")
        return None
    
    # Apply allowed window
    if earliest is not None and latest is not None:
        e_min = earliest.hour * 60 + earliest.minute
        l_min = latest.hour * 60 + latest.minute
        minutes_of_day = df.index.hour * 60 + df.index.minute
        print(f"DEBUG suggest_flask: window filter e_min={e_min}, l_min={l_min}")
        print(f"DEBUG suggest_flask: minutes_of_day range: {minutes_of_day.min()} to {minutes_of_day.max()}")
        
        if e_min <= l_min:
            mask = (minutes_of_day >= e_min) & (minutes_of_day <= l_min)
        else:
            # window wraps midnight
            mask = (minutes_of_day >= e_min) | (minutes_of_day <= l_min)
        
        n_in_window = mask.sum()
        print(f"DEBUG suggest_flask: {n_in_window} data points in window [{earliest} - {latest}]")
        
        df = df.loc[mask]
        if df.empty:
            print(f"ERROR suggest_flask: No data in allowed window [{earliest} - {latest}]")
            return None
        
        print(f"DEBUG suggest_flask: After window filter, df length={len(df)}, index range: {df.index.min()} to {df.index.max()}")
    
    # Normalize
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
    
    print(f"DEBUG suggest_flask: Searching for best {dur}-minute window in {n} data points")
    
    # Check if duration is longer than available window
    if dur > n:
        print(f"WARNING suggest_flask: duration ({dur}) > available data ({n}), using full window")
        dur = n
    
    search_range = n - dur + 1
    print(f"DEBUG suggest_flask: Will check {search_range} possible starting positions")
    
    for t0 in range(0, search_range):
        sc = float(df["score"].iloc[t0:t0 + dur].mean())
        if best_score is None or sc < best_score:
            best_score = sc
            best_t0 = t0
    
    if best_t0 is None:
        print("ERROR suggest_flask: No valid window found")
        return None
    
    t_start = df.index[best_t0]
    t_end = df.index[best_t0 + dur - 1] + pd.Timedelta(minutes=1)
    
    print(f"DEBUG suggest_flask: Best window found at t0={best_t0}, score={best_score:.4f}")
    print(f"DEBUG suggest_flask: t_start={t_start}, t_end={t_end}")
    
    start_min = t_start.hour * 60 + t_start.minute
    end_min = t_end.hour * 60 + t_end.minute
    start_min = max(0, min(start_min, 24 * 60 - 1))
    end_min = max(1, min(end_min, 24 * 60 - 1))
    
    result = {
        "start": _time(start_min // 60, start_min % 60),
        "end": _time(end_min // 60, end_min % 60),
    }
    print(f"DEBUG suggest_flask: Final result: {result['start']} - {result['end']}")
    
    return result

@app.route('/api/compute-thermal-profiles', methods=['POST'])
def compute_thermal_profiles():
    """Compute thermal device profiles based on weather data and settings."""
    try:
        data = request.get_json() or {}
        preview_config = data.get('preview_config')  # Optional: for modal preview
        preview_dev_type = data.get('dev_type')  # Optional: which device to preview
        
        cfgs = session.get('device_configs', {})
        
        # If preview_config is provided, use it to override the session config for the specific device
        if preview_config and preview_dev_type:
            cfgs = dict(cfgs)  # Make a copy to avoid modifying session
            cfgs[f'thermal:{preview_dev_type}'] = preview_config
            print(f"DEBUG compute_thermal_profiles: Using preview_config for {preview_dev_type}: {preview_config}")
        
        # Get temperature data from session
        temp_data = session.get('temp_daily')
        selected_day = session.get('day', date.today())
        
        print(f"DEBUG compute_thermal_profiles: temp_data available = {temp_data is not None}")
        if temp_data:
            print(f"DEBUG compute_thermal_profiles: temp_data has {len(temp_data.get('values', []))} points")
        print(f"DEBUG compute_thermal_profiles: selected_day = {selected_day}")
        
        # If no temperature data, use synthetic (same formula as Streamlit)
        if temp_data is None:
            print("DEBUG compute_thermal_profiles: Using SYNTHETIC temperature profile")
            # Generate a synthetic temperature profile matching Streamlit
            period_start = selected_day
            period_end = selected_day + timedelta(days=1)
            idx = minute_index(period_start, period_end)
            # Streamlit formula: 5 + 5*sin(2π*(hours-15)/24)
            # Peak at hour 15 (3pm), min at hour 3 (3am)
            hours = np.array(idx.hour) + np.array(idx.minute) / 60.0
            tout_values = 5.0 + 5.0 * np.sin(2 * np.pi * (hours - 15) / 24.0)
            tout_series = pd.Series(tout_values, index=idx)
        else:
            # Reconstruct from stored data (real temperature from API)
            print("DEBUG compute_thermal_profiles: Using REAL temperature data from API")
            tout_series = pd.Series(
                temp_data['values'],
                index=pd.to_datetime(temp_data['index']),
                name='temp_c'
            )
            # Filter to selected day
            day_start = pd.Timestamp(selected_day)
            day_end = day_start + pd.Timedelta(days=1)
            tout_series = tout_series.loc[(tout_series.index >= day_start) & (tout_series.index < day_end)]
            print(f"DEBUG compute_thermal_profiles: Filtered to {len(tout_series)} points for {selected_day}")
        
        if len(tout_series) == 0:
            return jsonify({'success': False, 'error': 'No temperature data available'}), 400
        
        idx = tout_series.index
        profiles = {}
        
        # Get house thermal parameters
        house_info = session.get('house_info', {'size': 'Medium house', 'insulation': 'Average', 'residents': 2})
        thermal_params = get_house_thermal_params(house_info)
        ua_base = thermal_params.get('ua_kw_per_c', 0.25)
        cth_base = thermal_params.get('C_th_kwh_per_c', 3.0)
        
        # ========== SPACE HEATING ==========
        space_cfg = cfgs.get('thermal:space_heat', {})
        space_mode = space_cfg.get('space_mode', 'None (external supply)')
        
        if space_mode != 'None (external supply)':
            t_min = float(space_cfg.get('t_min_c', 20))
            t_max = float(space_cfg.get('t_max_c', 22))
            t_set = 0.5 * (t_min + t_max)
            hyst = max(t_max - t_min, 0.5)
            q_kw = float(space_cfg.get('q_kw', 6))
            
            # Base thermal capacitance (only adjust for air-to-water HP with floor heating)
            cth_eff = cth_base
            
            if space_mode == 'Electric panels':
                heater = WeatherELheater(
                    ua_kw_per_c=ua_base,
                    t_set_c=t_set,
                    q_rated_kw=q_kw,
                    C_th_kwh_per_c=cth_eff,
                    hyst_band_c=hyst,
                    p_off_kw=0.05,
                    min_on_min=0,
                    min_off_min=0,
                    Ti0_c=t_set
                )
                P_space, Ti_space = heater.series_kw(idx, tout_series)
            elif space_mode == 'Heat pump – air-to-air':
                # Air-to-air HP - no distribution adjustment
                hp_type = space_cfg.get('hp_type', 'Fixed power')
                hp_mode = 'onoff' if hp_type == 'Fixed power' else 'modulating'
                
                hp = WeatherHP(
                    mode=hp_mode,
                    ua_kw_per_c=ua_base,
                    t_set_c=t_set,
                    q_rated_kw=q_kw,
                    cop_at_7c=3.2,
                    cop_min=1.6,
                    cop_max=4.2,
                    C_th_kwh_per_c=cth_eff,
                    hyst_band_c=hyst,
                    p_off_kw=0.05,
                    defrost=True,
                    min_on_min=0,
                    min_off_min=0,
                    Ti0_c=t_set
                )
                P_space, Ti_space = hp.series_kw(idx, tout_series)
            else:
                # Heat pump – air-to-water: apply distribution thermal mass adjustment
                hp_type = space_cfg.get('hp_type', 'Fixed power')
                hp_mode = 'onoff' if hp_type == 'Fixed power' else 'modulating'
                
                dist = space_cfg.get('distribution', 'Radiators')
                extra_mass = {'Radiators': 0.0, 'Floor heating': 0.5, 'Both': 0.3}.get(dist, 0.0)
                cth_eff = cth_base * (1.0 + extra_mass)
                
                hp = WeatherHP(
                    mode=hp_mode,
                    ua_kw_per_c=ua_base,
                    t_set_c=t_set,
                    q_rated_kw=q_kw,
                    cop_at_7c=3.2,
                    cop_min=1.6,
                    cop_max=4.2,
                    C_th_kwh_per_c=cth_eff,
                    hyst_band_c=hyst,
                    p_off_kw=0.05,
                    defrost=True,
                    min_on_min=0,
                    min_off_min=0,
                    Ti0_c=t_set
                )
                P_space, Ti_space = hp.series_kw(idx, tout_series)
            
            # Save profile in config
            space_cfg['profile_index'] = P_space.index.astype(str).tolist()
            space_cfg['profile_kw'] = P_space.values.tolist()
            cfgs['thermal:space_heat'] = space_cfg
            
            profiles['space_heat'] = {
                'index': P_space.index.astype(str).tolist(),
                'values': P_space.values.tolist()
            }
        
        # ========== DHW ==========
        dhw_cfg = cfgs.get('thermal:dhw', {})
        dhw_mode = dhw_cfg.get('dhw_mode', 'None (external supply)')
        
        print(f"DEBUG DHW: dhw_mode = '{dhw_mode}'")
        print(f"DEBUG DHW: full dhw_cfg = {dhw_cfg}")
        
        if dhw_mode != 'None (external supply)':
            volume_l = float(dhw_cfg.get('volume_l', 200))
            t_min = float(dhw_cfg.get('t_min_c', 45))
            t_max = float(dhw_cfg.get('t_max_c', 55))
            t_set = 0.5 * (t_min + t_max)
            hyst = max(t_max - t_min, 1.0)
            p_el = float(dhw_cfg.get('p_el_kw', 2))
            usage = dhw_cfg.get('usage_level', 'Medium')
            
            if dhw_mode == 'Electric DHW tank':
                print(f"DEBUG DHW: Creating ELECTRIC tank with p_el_kw={p_el}")
                tank = DHWTank(
                    volume_l=volume_l,
                    t_set_c=t_set,
                    hyst_band_c=hyst,
                    ua_kw_per_c=0.02,
                    p_el_kw=p_el,
                    p_off_kw=0.01,
                    T_cold_c=10.0,
                    T_amb_c=20.0,
                    Ti0_c=t_set,
                    usage_level=usage
                )
                P_dhw, T_tank = tank.series_kw(idx, tout_series)
                print(f"DEBUG DHW Electric: P_dhw max={P_dhw.max():.2f}, mean={P_dhw.mean():.3f}")
            else:
                # Heat pump DHW
                thermal_power = p_el * 2.5
                print(f"DEBUG DHW: Creating HEAT PUMP tank with p_el_kw={thermal_power} (thermal)")
                tank = DHWTank(
                    volume_l=volume_l,
                    t_set_c=t_set,
                    hyst_band_c=hyst,
                    ua_kw_per_c=0.02,
                    p_el_kw=thermal_power,  # thermal output
                    p_off_kw=0.01,
                    T_cold_c=10.0,
                    T_amb_c=20.0,
                    Ti0_c=t_set,
                    usage_level=usage
                )
                Q_th, T_tank = tank.series_kw(idx, tout_series)
                P_dhw = Q_th / 2.5  # divide by COP
                print(f"DEBUG DHW HP: Q_th max={Q_th.max():.2f}, P_dhw max={P_dhw.max():.2f}, mean={P_dhw.mean():.3f}")
            
            dhw_cfg['profile_index'] = P_dhw.index.astype(str).tolist()
            dhw_cfg['profile_kw'] = P_dhw.values.tolist()
            cfgs['thermal:dhw'] = dhw_cfg
            
            profiles['dhw'] = {
                'index': P_dhw.index.astype(str).tolist(),
                'values': P_dhw.values.tolist()
            }
        
        # ========== LEISURE (Hot tub + Pool) ==========
        leisure_cfg = cfgs.get('thermal:leisure', {})
        P_leisure_total = pd.Series(0.0, index=idx)
        
        print(f"DEBUG LEISURE: leisure_cfg = {leisure_cfg}")
        print(f"DEBUG LEISURE: hot_tub_enabled = {leisure_cfg.get('hot_tub_enabled', False)}")
        print(f"DEBUG LEISURE: pool_enabled = {leisure_cfg.get('pool_enabled', False)}")
        
        if leisure_cfg.get('hot_tub_enabled', False):
            # Parse sessions
            sessions = []
            for sess in leisure_cfg.get('ht_sessions', []):
                start_str = sess.get('start', '20:00')
                if isinstance(start_str, str):
                    parts = start_str.split(':')
                    start_time = _time(int(parts[0]), int(parts[1]))
                else:
                    start_time = start_str
                sessions.append({
                    'start': start_time,
                    'duration_min': int(sess.get('duration_min', 60))
                })
            
            # UA based on insulation
            ins = leisure_cfg.get('ht_insulation', 'Average')
            ua_ht = 0.07 * {'Good cover': 0.6, 'Average': 1.0, 'Poor': 1.4}.get(ins, 1.0)
            
            print(f"DEBUG HOT TUB: target={leisure_cfg.get('ht_target_c', 40)}, idle={leisure_cfg.get('ht_idle_c', 30)}, heater_kw={leisure_cfg.get('ht_heater_kw', 5)}, water_l={leisure_cfg.get('ht_water_l', 1200)}, ua={ua_ht}")
            print(f"DEBUG HOT TUB: sessions = {sessions}")
            
            ht = WeatherHotTub(
                target_c=float(leisure_cfg.get('ht_target_c', 40)),
                idle_c=float(leisure_cfg.get('ht_idle_c', 30)),
                heater_kw=float(leisure_cfg.get('ht_heater_kw', 5)),
                water_l=float(leisure_cfg.get('ht_water_l', 1200)),
                ua_kw_per_c=ua_ht,
                sessions=sessions,
                use_outdoor_for_ambient=False,
                indoor_ambient_c=21.0
            )
            P_ht, T_ht = ht.series_kw(idx, tout_series)
            print(f"DEBUG HOT TUB: P_ht max={P_ht.max():.2f}, mean={P_ht.mean():.3f}")
            P_leisure_total = P_leisure_total + P_ht
        
        if leisure_cfg.get('pool_enabled', False):
            # Pool uses outdoor ambient and has heat pump (COP ~3.5)
            ins = leisure_cfg.get('pool_insulation', 'Average')
            ua_base_pool = 0.15  # Base UA for pool (larger surface area)
            ua_pool = ua_base_pool * {'Good cover': 0.6, 'Average': 1.0, 'Poor': 1.4}.get(ins, 1.0)
            
            # Parse pool sessions if any
            pool_sessions = []
            for sess in leisure_cfg.get('pool_sessions', []):
                start_str = sess.get('start', '08:00')
                if isinstance(start_str, str):
                    parts = start_str.split(':')
                    start_time = _time(int(parts[0]), int(parts[1]))
                else:
                    start_time = start_str
                pool_sessions.append({
                    'start': start_time,
                    'duration_min': int(sess.get('duration_min', 480))
                })
            
            print(f"DEBUG POOL: target={leisure_cfg.get('pool_target_c', 28)}, idle={leisure_cfg.get('pool_idle_c', 24)}, heater_kw={leisure_cfg.get('pool_heater_kw', 15)} (thermal), water_l={leisure_cfg.get('pool_water_l', 30000)}, ua={ua_pool}")
            print(f"DEBUG POOL: sessions = {pool_sessions}")
            
            pool = WeatherHotTub(
                target_c=float(leisure_cfg.get('pool_target_c', 28)),
                idle_c=float(leisure_cfg.get('pool_idle_c', 24)),
                heater_kw=float(leisure_cfg.get('pool_heater_kw', 15)),  # thermal output
                water_l=float(leisure_cfg.get('pool_water_l', 30000)),
                ua_kw_per_c=ua_pool,
                sessions=pool_sessions,
                use_outdoor_for_ambient=True,
                indoor_ambient_c=21.0  # unused when use_outdoor_for_ambient=True
            )
            Q_pool_th, T_pool = pool.series_kw(idx, tout_series)
            
            # Pool uses heat pump with COP ~3.5, so electrical power = thermal / COP
            cop_pool = 3.5
            P_pool = Q_pool_th / cop_pool
            print(f"DEBUG POOL: Q_th max={Q_pool_th.max():.2f}, P_pool max={P_pool.max():.2f}, mean={P_pool.mean():.3f}")
            P_leisure_total = P_leisure_total + P_pool
        
        if leisure_cfg.get('hot_tub_enabled', False) or leisure_cfg.get('pool_enabled', False):
            leisure_cfg['profile_index'] = P_leisure_total.index.astype(str).tolist()
            leisure_cfg['profile_kw'] = P_leisure_total.values.tolist()
            cfgs['thermal:leisure'] = leisure_cfg
            
            profiles['leisure'] = {
                'index': P_leisure_total.index.astype(str).tolist(),
                'values': P_leisure_total.values.tolist()
            }
        
        # Save updated configs
        session['device_configs'] = cfgs
        session.modified = True
        
        return jsonify({'success': True, 'profiles': profiles})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/compute-profiles', methods=['GET'])
def compute_profiles():
    """Compute daily profiles for selected devices."""
    try:
        sel = session.get('device_selection', {})
        cfgs = session.get('device_configs', {})
        
        # Debug: log what we're working with
        print(f"DEBUG compute-profiles: selection keys={list(sel.keys())}, config keys={list(cfgs.keys())}")
        for key, cfg in cfgs.items():
            if 'intervals' in cfg:
                print(f"DEBUG {key} intervals: {cfg.get('intervals')}")
        
        # Convert string intervals to time objects for compute_daily_profiles
        # This is needed because frontend saves intervals as strings like "06:00"
        def convert_time_str_to_time(time_val):
            """Convert time string or dict to datetime.time object."""
            if isinstance(time_val, _time):
                return time_val
            elif isinstance(time_val, str):
                parts = time_val.split(':')
                if len(parts) >= 2:
                    return _time(int(parts[0]), int(parts[1]))
            elif isinstance(time_val, dict) and 'hour' in time_val:
                return _time(int(time_val.get('hour', 18)), int(time_val.get('minute', 0)))
            return _time(18, 0)  # default fallback
        
        cfgs_normalized = {}
        for full_key, cfg in cfgs.items():
            cfg_copy = cfg.copy()
            
            # Convert 'start' field if it's a string
            if 'start' in cfg_copy:
                cfg_copy['start'] = convert_time_str_to_time(cfg_copy['start'])
            
            # Convert intervals
            if 'intervals' in cfg_copy and isinstance(cfg_copy['intervals'], list) and len(cfg_copy['intervals']) > 0:
                normalized_intervals = []
                for iv in cfg_copy['intervals']:
                    if isinstance(iv, dict):
                        iv_copy = iv.copy()
                        # Convert start/end from string to time if needed
                        if 'start' in iv_copy:
                            iv_copy['start'] = convert_time_str_to_time(iv_copy['start'])
                        if 'end' in iv_copy:
                            iv_copy['end'] = convert_time_str_to_time(iv_copy['end'])
                        normalized_intervals.append(iv_copy)
                    else:
                        normalized_intervals.append(iv)
                cfg_copy['intervals'] = normalized_intervals
            elif 'intervals' not in cfg_copy or not cfg_copy.get('intervals'):
                # If no intervals, make sure we don't have an empty list
                # This will let compute_daily_profiles use the fallback logic
                if 'intervals' in cfg_copy:
                    del cfg_copy['intervals']
            cfgs_normalized[full_key] = cfg_copy
        
        idx, device_traces, total = compute_daily_profiles(sel, cfgs_normalized)
        
        # Convert to JSON-serializable format
        profiles = {}
        for full_key, series in device_traces.items():
            profiles[full_key] = {
                'index': [str(ts) for ts in series.index],
                'values': series.values.tolist()
            }
        
        total_profile = {
            'index': [str(ts) for ts in total.index],
            'values': total.values.tolist()
        }
        
        return jsonify({
            'success': True,
            'profiles': profiles,
            'total': total_profile
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/house-layout', methods=['GET'])
def get_house_layout():
    """Get house layout visualization data."""
    try:
        sel = session.get('device_selection', {})
        cfgs = session.get('device_configs', {})
        
        fig = build_house_layout_figure(sel, cfgs)
        
        # Convert Plotly figure to JSON
        return jsonify({
            'success': True,
            'figure': fig.to_dict()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
