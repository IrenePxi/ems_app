import React, { useState, useEffect } from 'react'
import axios from 'axios'
import Plot from 'react-plotly.js'
import '../App.css'

const API_BASE = '/api'

function AnalysisPage() {
  const [batteryConfig, setBatteryConfig] = useState({
    capacity_kwh: 70.0,
    power_kw: 9.0,
    soc_init_pct: 50.0,
    soc_min_pct: 15.0,
    control_mode: 'Auto',
    energy_pattern: 2
  })
  const [planSlots, setPlanSlots] = useState([])
  const [emsResult, setEmsResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const runEMS = async () => {
    setLoading(true)
    try {
      const storedData = JSON.parse(localStorage.getItem('ems_data'))
      const loadProfile = JSON.parse(localStorage.getItem('ems_load_profile'))
      
      if (!storedData || !loadProfile) {
        alert('Please complete Scenario and Devices pages first!')
        return
      }

      // Generate optimized slots if Auto mode
      let slots = planSlots
      if (batteryConfig.control_mode === 'Auto' && slots.length === 0) {
        // Generate signal (weighted price + CO2)
        const priceData = storedData.price
        const co2Data = storedData.co2
        
        // Normalize and combine
        const priceValues = Object.values(priceData.data)
        const co2Values = Object.values(co2Data.data)
        const priceMin = Math.min(...priceValues)
        const priceMax = Math.max(...priceValues)
        const co2Min = Math.min(...co2Values)
        const co2Max = Math.max(...co2Values)
        
        const priceNorm = priceValues.map(v => (v - priceMin) / (priceMax - priceMin))
        const co2Norm = co2Values.map(v => (v - co2Min) / (co2Max - co2Min))
        const signal = priceNorm.map((p, i) => 0.5 * p + 0.5 * co2Norm[i])
        
        const optRes = await axios.post(`${API_BASE}/optimization/generate-slots`, {
          load_kw: loadProfile.total_load_kw,
          pv_kw: { data: new Array(Object.keys(loadProfile.total_load_kw).length).fill(0), index: loadProfile.index },
          price: priceData,
          co2: co2Data,
          signal: { data: signal, index: priceData.index }
        })
        
        const optBattery = await axios.post(`${API_BASE}/optimization/optimize-battery`, {
          slot_data: optRes.data.slot_data,
          battery_config: batteryConfig
        })
        
        slots = optBattery.data.plan_slots
        setPlanSlots(slots)
      }

      // Run EMS
      const response = await axios.post(`${API_BASE}/ems/run`, {
        load_kw: loadProfile.total_load_kw,
        pv_kw: { data: new Array(Object.keys(loadProfile.total_load_kw).length).fill(0), index: loadProfile.index },
        plan_slots: slots,
        battery_config: batteryConfig,
        energy_pattern: batteryConfig.energy_pattern
      })

      setEmsResult(response.data)
    } catch (err) {
      alert(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="main-content">
      <div className="card">
        <h2>3️⃣ Analysis</h2>

        <div className="grid grid-2">
          <div className="form-group">
            <label>Battery Capacity (kWh)</label>
            <input
              type="number"
              step="0.5"
              value={batteryConfig.capacity_kwh}
              onChange={(e) => setBatteryConfig({ ...batteryConfig, capacity_kwh: parseFloat(e.target.value) })}
            />
          </div>
          <div className="form-group">
            <label>Battery Power (kW)</label>
            <input
              type="number"
              step="0.5"
              value={batteryConfig.power_kw}
              onChange={(e) => setBatteryConfig({ ...batteryConfig, power_kw: parseFloat(e.target.value) })}
            />
          </div>
        </div>

        <div className="grid grid-2">
          <div className="form-group">
            <label>Initial SOC (%)</label>
            <input
              type="number"
              step="1"
              min="0"
              max="100"
              value={batteryConfig.soc_init_pct}
              onChange={(e) => setBatteryConfig({ ...batteryConfig, soc_init_pct: parseFloat(e.target.value) })}
            />
          </div>
          <div className="form-group">
            <label>Minimum SOC (%)</label>
            <input
              type="number"
              step="1"
              min="0"
              max="100"
              value={batteryConfig.soc_min_pct}
              onChange={(e) => setBatteryConfig({ ...batteryConfig, soc_min_pct: parseFloat(e.target.value) })}
            />
          </div>
        </div>

        <div className="form-group">
          <label>Control Mode</label>
          <select
            value={batteryConfig.control_mode}
            onChange={(e) => setBatteryConfig({ ...batteryConfig, control_mode: e.target.value })}
          >
            <option value="Auto">Auto (Optimization-based)</option>
            <option value="Manual">Manual (6-slot plan)</option>
          </select>
        </div>

        <div className="form-group">
          <label>Energy Pattern</label>
          <select
            value={batteryConfig.energy_pattern}
            onChange={(e) => setBatteryConfig({ ...batteryConfig, energy_pattern: parseInt(e.target.value) })}
          >
            <option value="2">Load First</option>
            <option value="1">Battery First</option>
          </select>
        </div>

        <button className="btn btn-primary" onClick={runEMS} disabled={loading}>
          {loading ? 'Running EMS...' : 'Run EMS'}
        </button>

        {emsResult && (
          <div style={{ marginTop: '2rem' }}>
            <h3>Results</h3>
            <div className="grid grid-3" style={{ marginBottom: '2rem' }}>
              <div className="card">
                <h4>Self-Consumption</h4>
                <p style={{ fontSize: '2rem', fontWeight: 'bold' }}>
                  {emsResult.kpis.self_consumption_pct.toFixed(1)}%
                </p>
              </div>
              <div className="card">
                <h4>Self-Sufficiency</h4>
                <p style={{ fontSize: '2rem', fontWeight: 'bold' }}>
                  {emsResult.kpis.self_sufficiency_pct.toFixed(1)}%
                </p>
              </div>
              <div className="card">
                <h4>Peak Grid Import</h4>
                <p style={{ fontSize: '2rem', fontWeight: 'bold' }}>
                  {emsResult.kpis.peak_grid_import_kw.toFixed(2)} kW
                </p>
              </div>
            </div>

            <div className="card">
              <h4>Power Flow</h4>
              <Plot
                data={[
                  {
                    x: emsResult.index,
                    y: Object.values(emsResult.grid_import_kw),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Grid Import',
                    line: { color: 'red' }
                  },
                  {
                    x: emsResult.index,
                    y: Object.values(emsResult.batt_discharge_kw),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Battery Discharge',
                    line: { color: 'blue' }
                  }
                ]}
                layout={{ width: '100%', height: 400, title: 'Power Flow (kW)' }}
              />
            </div>

            <div className="card">
              <h4>Battery SOC</h4>
              <Plot
                data={[{
                  x: emsResult.index,
                  y: Object.values(emsResult.batt_soc_kwh).map(v => (v / batteryConfig.capacity_kwh) * 100),
                  type: 'scatter',
                  mode: 'lines',
                  name: 'SOC (%)',
                  line: { color: 'green' }
                }]}
                layout={{ width: '100%', height: 400, title: 'Battery State of Charge (%)' }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default AnalysisPage
