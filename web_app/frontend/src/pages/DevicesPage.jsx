import React, { useState, useEffect } from 'react'
import axios from 'axios'
import '../App.css'

const API_BASE = '/api'

function DevicesPage() {
  const [devices, setDevices] = useState([])
  const [houseInfo, setHouseInfo] = useState({
    size: 'Medium house',
    insulation: 'Average',
    residents: 2
  })
  const [loading, setLoading] = useState(false)
  const [loadProfile, setLoadProfile] = useState(null)

  const deviceTypes = [
    { value: 'heat_pump', label: 'Heat Pump' },
    { value: 'electric_heater', label: 'Electric Heater' },
    { value: 'hot_tub', label: 'Hot Tub' },
    { value: 'dhw_tank', label: 'DHW Tank' }
  ]

  const addDevice = () => {
    setDevices([...devices, {
      id: Date.now(),
      type: 'heat_pump',
      name: `Device ${devices.length + 1}`,
      params: {}
    }])
  }

  const removeDevice = (id) => {
    setDevices(devices.filter(d => d.id !== id))
  }

  const updateDevice = (id, field, value) => {
    setDevices(devices.map(d => 
      d.id === id ? { ...d, [field]: value } : d
    ))
  }

  const calculateLoad = async () => {
    setLoading(true)
    try {
      const storedData = JSON.parse(localStorage.getItem('ems_data'))
      if (!storedData) {
        alert('Please fetch data on Scenario page first!')
        return
      }

      const response = await axios.post(`${API_BASE}/devices/calculate-load`, {
        devices: devices.map(d => ({
          type: d.type,
          name: d.name,
          params: d.params
        })),
        start_date: storedData.periodStart,
        end_date: storedData.periodEnd,
        outdoor_temp: storedData.weather
      })

      setLoadProfile(response.data)
      localStorage.setItem('ems_load_profile', JSON.stringify(response.data))
    } catch (err) {
      alert(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="main-content">
      <div className="card">
        <h2>2️⃣ Devices & Layout</h2>

        <div className="form-group">
          <label>House Size</label>
          <select
            value={houseInfo.size}
            onChange={(e) => setHouseInfo({ ...houseInfo, size: e.target.value })}
          >
            <option value="Small apartment">Small apartment (40–80 m²)</option>
            <option value="Medium house">Medium house (90–150 m²)</option>
            <option value="Large house">Large house (160–250 m²)</option>
          </select>
        </div>

        <div className="form-group">
          <label>Insulation Quality</label>
          <select
            value={houseInfo.insulation}
            onChange={(e) => setHouseInfo({ ...houseInfo, insulation: e.target.value })}
          >
            <option value="Poor">Poor (pre-1980)</option>
            <option value="Average">Average (1980–2010)</option>
            <option value="Good">Good (new/renovated)</option>
          </select>
        </div>

        <div className="form-group">
          <label>Number of Residents</label>
          <input
            type="number"
            min="1"
            max="8"
            value={houseInfo.residents}
            onChange={(e) => setHouseInfo({ ...houseInfo, residents: parseInt(e.target.value) })}
          />
        </div>

        <hr style={{ margin: '2rem 0' }} />

        <h3>Devices</h3>
        <button className="btn btn-primary" onClick={addDevice}>
          + Add Device
        </button>

        {devices.map(device => (
          <div key={device.id} className="card" style={{ marginTop: '1rem' }}>
            <div className="grid grid-2">
              <div className="form-group">
                <label>Device Type</label>
                <select
                  value={device.type}
                  onChange={(e) => updateDevice(device.id, 'type', e.target.value)}
                >
                  {deviceTypes.map(dt => (
                    <option key={dt.value} value={dt.value}>{dt.label}</option>
                  ))}
                </select>
              </div>
              <div className="form-group">
                <label>Device Name</label>
                <input
                  type="text"
                  value={device.name}
                  onChange={(e) => updateDevice(device.id, 'name', e.target.value)}
                />
              </div>
            </div>
            <button className="btn btn-secondary" onClick={() => removeDevice(device.id)}>
              Remove
            </button>
          </div>
        ))}

        <button 
          className="btn btn-primary" 
          onClick={calculateLoad}
          disabled={loading || devices.length === 0}
          style={{ marginTop: '2rem' }}
        >
          {loading ? 'Calculating...' : 'Calculate Load Profile'}
        </button>

        {loadProfile && (
          <div className="success" style={{ marginTop: '1rem' }}>
            Load profile calculated! Total load: {Object.values(loadProfile.total_load_kw).reduce((a, b) => a + b, 0).toFixed(2)} kW
          </div>
        )}
      </div>
    </div>
  )
}

export default DevicesPage
