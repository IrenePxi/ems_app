
#%% 1.1 MQTT read data

import numpy as np
import json
from scipy.optimize import minimize
import time
from datetime import datetime
import pandas as pd
from scipy.signal import find_peaks
from datetime import datetime, timedelta, time, date
import numpy as np
import pandas as pd
from datetime import datetime, time
from sklearn.cluster import KMeans
def detect_price_change_points(day_data):
    price_values = day_data[['signal']].values
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(price_values)
    centers = kmeans.cluster_centers_.flatten()
    order = np.argsort(centers)
    mapping = {order[0]: 1, order[1]: 2, order[2]: 3}
    price_levels = pd.Series([mapping[l] for l in labels], index=day_data.index)

    change_mask = price_levels.astype(int).diff().fillna(0) != 0
    change_points = day_data.loc[change_mask & (day_data['DateTime'].dt.hour > 6), 'DateTime'].tolist()

    if len(change_points) >= 4:
        return [t.time() for t in [change_points[0], change_points[1], change_points[2], change_points[-1]]]
    elif len(change_points) >= 3:
        return [t.time() for t in [change_points[0], change_points[1], change_points[-1]]]
    elif len(change_points) >= 2:
        return [t.time() for t in [change_points[0], change_points[-1]]]
    else:
        return [t.time() for t in [change_points[0]]]

def detect_pv_change_points(day_data):
    gradient_pv = np.abs(np.gradient(day_data['PV'].values))
    change_indices_pv = np.argsort(gradient_pv)[-4:]
    time_changes_pv = sorted(day_data.iloc[change_indices_pv]['DateTime'].dt.time.tolist())
    time_changes_pv = [t for t in time_changes_pv if t > time(6,0)]
    total_day_pv = day_data['PV'].sum()/60
    return time_changes_pv, total_day_pv

def generate_smart_time_slots(df_minute):
    """Returns {day: [(HH:MM, HH:MM) * 6]} exactly as your original code."""
    time_slots_per_day = {}
    df = df_minute.copy().sort_values("DateTime")
    for day, day_data in df.groupby(df['DateTime'].dt.date):
        valid_times = [time(0,0), time(6,0)]

        time_changes_price = detect_price_change_points(day_data)
        time_changes_pv, total_pv = detect_pv_change_points(day_data)

        if total_pv < 8:
            all_time_changes = time_changes_price
        else:
            if len(time_changes_pv) > 2:
                time_changes_pv = [min(time_changes_pv), max(time_changes_pv)]
            all_time_changes = sorted(set(time_changes_price[:2] + time_changes_pv))
            if len(all_time_changes) < 4 and len(time_changes_price) > 2:
                all_time_changes = sorted(set(time_changes_price[:3] + time_changes_pv))
            if len(all_time_changes) < 4 and len(time_changes_price) > 3:
                all_time_changes = sorted(set(time_changes_price[:4] + time_changes_pv))

        while len(all_time_changes) > 4:
            all_time_changes.pop(-1)
        while len(all_time_changes) < 4:
            sorted_times = sorted(valid_times + all_time_changes)
            max_gap = 0; best = None
            for i in range(len(sorted_times)-2):
                start = sorted_times[i+1]; end = sorted_times[i+2]
                gap = (datetime.combine(date.today(), end) - datetime.combine(date.today(), start)).seconds
                if gap > max_gap:
                    max_gap = gap
                    best = time((start.hour + (end.hour - start.hour)//2) % 24, 0)
            if best in all_time_changes:
                best = time((best.hour + 1) % 24, 0)
            all_time_changes.append(best)

        all_time_changes = sorted(set(all_time_changes))
        all_time_changes.append(time(23,59))
        valid_times.extend(all_time_changes)
        valid_times = sorted(set(valid_times))
        if len(valid_times) != 7:
            raise ValueError(f"{day} has {len(valid_times)-1} slots, expected 6.")
        time_slots_per_day[day] = [(valid_times[i].strftime("%H:%M"), valid_times[i+1].strftime("%H:%M")) for i in range(6)]
    return time_slots_per_day
#%% 1.4b assign in the value inside each slot, caculate the mean value

def assign_data_to_time_slots_single(df_anyfreq: pd.DataFrame, time_slots_per_day: dict) -> pd.DataFrame:
    """
    df_anyfreq: columns include ['DateTime','ElectricityPrice','Load','PV'] at minute cadence (or any).
    Returns a DataFrame with per-slot averages for the three columns.
    """
    need = {"DateTime","ElectricityPrice","Load","PV","signal","co2"}
    if not need.issubset(df_anyfreq.columns):
        raise ValueError(f"df_anyfreq must contain {need}")

    df = df_anyfreq.copy()
    df = df.sort_values("DateTime")
    out = []
    for day, slots in time_slots_per_day.items():
        day_mask = df["DateTime"].dt.date == day
        day_data = df.loc[day_mask]
        if day_data.empty:
            continue
        for start_time, end_time in slots:
            st = datetime.combine(day, datetime.strptime(start_time, "%H:%M").time())
            et = datetime.combine(day, datetime.strptime(end_time, "%H:%M").time())
            # half-open interval [st, et) so adjacent slots don’t double count the boundary minute
            slot = day_data[(day_data["DateTime"] >= st) & (day_data["DateTime"] < et)]
            if not slot.empty:
                out.append({
                    "Date": day,
                    "TimeSlot": f"{start_time} - {end_time}",
                    "ElectricityPrice": slot["ElectricityPrice"].mean(),
                    "Load": slot["Load"].mean(),
                    "PV": slot["PV"].mean(),
                    "co2": slot["co2"].mean(),
                    "signal": slot["signal"].mean(),
                })
    return pd.DataFrame(out)


# ---------- OPTIMIZATION (single Load) ----------
from scipy.optimize import linprog

def mpc_opt_single(df_slots, SOC0, SOC_min, SOC_max, Pbat_chargemax, Qbat):
    df = df_slots.copy()
    t0 = pd.to_datetime(df['TimeSlot'].str.split(' - ').str[0], format="%H:%M")
    t1 = pd.to_datetime(df['TimeSlot'].str.split(' - ').str[1], format="%H:%M")
    df['Slot_Duration'] = (t1 - t0).dt.total_seconds()/3600.0

    # energies over the slot (if your intent was energy, keep * Slot_Duration)
    df['LoadTot'] = df['Load'] * df['Slot_Duration']
    df['PVTot']   = df['PV']   * df['Slot_Duration']

    N = df.shape[0]
    lb_SOC = SOC_min * np.ones(N); ub_SOC = SOC_max * np.ones(N)
    lb_Qg  = 0 * np.ones(N);       ub_Qg  = 5e5 * np.ones(N)
    lb = np.hstack([lb_SOC, lb_Qg])
    ub = np.hstack([ub_SOC, ub_Qg])

    A_step = np.zeros((N,N))
    for i in range(N):
        A_step[i,i] = 1
        if i>0: A_step[i,i-1] = -1
    A_soc_pos = np.hstack([ A_step, np.zeros((N,N))])
    A_soc_neg = -A_soc_pos.copy()

    A_grid_1 = A_step.copy()
    A_grid_2 = -np.eye(N)
    A_grid   = np.hstack([A_grid_1/ 100.0 * Qbat, A_grid_2]) 

    A = np.vstack([A_soc_pos, A_soc_neg, A_grid])

    b1 = Pbat_chargemax * df['Slot_Duration'].values * 100.0 / Qbat
    b1[0] += SOC0
    b2 = Pbat_chargemax * df['Slot_Duration'].values * 100.0 / Qbat
    b2[0] -= SOC0
    b3 = df['PVTot'].values - df['LoadTot'].values
    b3[0] += SOC0/100.0*Qbat
    b = np.hstack([b1,b2,b3])

    f = np.hstack([np.zeros(N), df["signal"].values])

    res = linprog(f, A_ub=A, b_ub=b, bounds=list(zip(lb,ub)), method="highs-ds")
    if not res.success:
        raise ValueError("Optimization failed: " + res.message)
    SOC_opt = res.x[:N]
    Qgrid   = res.x[N:]
    return SOC_opt, Qgrid, res.fun


    
def mpc_opt_multi(df_slots, weight,  SOC0, SOC_min, SOC_max, Pbat_chargemax, Qbat, Pev, ev_s,ev_e, Qtesla,SOCev0,SOCevmin):
    df = df_slots.copy()
    t0 = pd.to_datetime(df['TimeSlot'].str.split(' - ').str[0], format="%H:%M")
    t1 = pd.to_datetime(df['TimeSlot'].str.split(' - ').str[1], format="%H:%M")
    df['Slot_Duration'] = (t1 - t0).dt.total_seconds()/3600.0

    ub_Qev = float( Pev * (ev_e.hour-ev_s.hour))


    # energies over the slot (if your intent was energy, keep * Slot_Duration)
    df['LoadTot'] = df['Load'] * df['Slot_Duration']
    df['PVTot']   = df['PV']   * df['Slot_Duration']

    N = df.shape[0]
    lb_SOC = SOC_min * np.ones(N); ub_SOC = SOC_max * np.ones(N)
    lb_Qg  = 0 * np.ones(N);       ub_Qg  = 5e5 * np.ones(N)
    lb = np.hstack([lb_SOC, lb_Qg, 0])
    ub = np.hstack([ub_SOC, ub_Qg, ub_Qev])

    A_step = np.zeros((N,N))
    for i in range(N):
        A_step[i,i] = 1
        if i>0: A_step[i,i-1] = -1
    A_soc_pos = np.hstack([ A_step, np.zeros((N,N+1))])
    A_soc_neg = -A_soc_pos.copy()

    A_grid_1 = A_step.copy()
    A_grid_2 = -np.eye(N)
    A_grid_3 = np.zeros((N,1))
    A_grid_3[0]=1
    A_grid   = np.hstack([A_grid_1/ 100.0 * Qbat, A_grid_2, A_grid_3]) 
    A_ev_pos=np.zeros((1,(N*2+1)))
    A_ev_pos[(0,-1)]=1
    A_ev_neg=-A_ev_pos.copy()

    A = np.vstack([A_soc_pos, A_soc_neg, A_grid,A_ev_pos,A_ev_neg])

    b1 = Pbat_chargemax * df['Slot_Duration'].values * 100.0 / Qbat
    b1[0] += SOC0
    b2 = Pbat_chargemax * df['Slot_Duration'].values * 100.0 / Qbat
    b2[0] -= SOC0
    b3 = df['PVTot'].values - df['LoadTot'].values
    b3[0] += SOC0/100.0*Qbat
    bevpos=(100-SOCev0)/100.0*Qtesla
    bevneg=(SOCev0-SOCevmin)/100.0*Qtesla
    b = np.hstack([b1,b2,b3,bevpos,bevneg])

    w_cost, w_co2, w_cmf = map(float, W)
    
    # scale so EV term is comparable to the grid term magnitude
    # typical scale: average signal * typical EV window energy (kWh)



    f = np.hstack([np.zeros(N), df["signal"].values, 0])

    res = linprog(f, A_ub=A, b_ub=b, bounds=list(zip(lb,ub)), method="highs-ds")
    if not res.success:
        raise ValueError("Optimization failed: " + res.message)
    SOC_opt = res.x[:N]
    Qgrid   = res.x[N:2*N]
    Qev = res.x[-1]
    return SOC_opt, Qgrid, Qev, res.fun

def format_results_single(SOC_opt, Qgrid, df_slots):
    return pd.DataFrame({
        "Datetime": df_slots["Date"],
        "TimeSlot": df_slots["TimeSlot"],
        "SOC": SOC_opt,
        "Grid_Charge": (Qgrid > 0).astype(int)
    })


##########################################################################################################################

# %%
