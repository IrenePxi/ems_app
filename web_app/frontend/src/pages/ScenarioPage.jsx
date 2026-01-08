import React, { useState, useEffect } from 'react'
import axios from 'axios'
import '../App.css'

const API_BASE = '/api'

function ScenarioPage() {
  const [selectedDay, setSelectedDay] = useState(new Date().toISOString().split('T')[0])
  const [location, setLocation] = useState({ lat: 57.0488, lon: 9.9217, area: 'DK1' })
  const [periodStart, setPeriodStart] = useState(
    new Date(Date.now() - 15 * 24 * 60 * 60 * 1000).toISOString().split('T')[0]
  )
  const [periodEnd, setPeriodEnd] = useState(new Date().toISOString().split('T')[0])
  const [loading, setLoading] = useState(false)
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  const presetLocations = {
    'Aalborg (DK1)': { lat: 57.0488, lon: 9.9217, area: 'DK1' },
    'Aarhus (DK1)': { lat: 56.1629, lon: 10.2039, area: 'DK1' },
    'Odense (DK1)': { lat: 55.4038, lon: 10.4024, area: 'DK1' },
    'Copenhagen (DK2)': { lat: 55.6761, lon: 12.5683, area: 'DK2' },
  }

  const handleLocationChange = (preset) => {
    if (presetLocations[preset]) {
      setLocation(presetLocations[preset])
    }
  }

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const [priceRes, co2Res, weatherRes] = await Promise.all([
        axios.get(`${API_BASE}/data/price`, {
          params: {
            start_date: periodStart,
            end_date: periodEnd,
            area: location.area
          }
        }),
        axios.get(`${API_BASE}/data/co2`, {
          params: {
            start_date: periodStart,
            end_date: periodEnd,
            area: location.area
          }
        }),
        axios.get(`${API_BASE}/data/weather`, {
          params: {
            lat: location.lat,
            lon: location.lon,
            start_date: periodStart,
            end_date: periodEnd
          }
        })
      ])

      setData({
        price: priceRes.data,
        co2: co2Res.data,
        weather: weatherRes.data
      })

      // Store in localStorage for other pages
      localStorage.setItem('ems_data', JSON.stringify({
        price: priceRes.data,
        co2: co2Res.data,
        weather: weatherRes.data,
        location,
        selectedDay,
        periodStart,
        periodEnd
      }))
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="main-content">
      <div className="card">
        <h2>1️⃣ Scenario & Data</h2>
        
        <div className="form-group">
          <label>Selected Day</label>
          <input
            type="date"
            value={selectedDay}
            onChange={(e) => setSelectedDay(e.target.value)}
          />
        </div>

        <div className="form-group">
          <label>Location</label>
          <select onChange={(e) => handleLocationChange(e.target.value)}>
            {Object.keys(presetLocations).map(key => (
              <option key={key} value={key}>{key}</option>
            ))}
          </select>
        </div>

        <div className="grid grid-2">
          <div className="form-group">
            <label>Latitude</label>
            <input
              type="number"
              step="0.0001"
              value={location.lat}
              onChange={(e) => setLocation({ ...location, lat: parseFloat(e.target.value) })}
            />
          </div>
          <div className="form-group">
            <label>Longitude</label>
            <input
              type="number"
              step="0.0001"
              value={location.lon}
              onChange={(e) => setLocation({ ...location, lon: parseFloat(e.target.value) })}
            />
          </div>
        </div>

        <div className="form-group">
          <label>Price Area</label>
          <select
            value={location.area}
            onChange={(e) => setLocation({ ...location, area: e.target.value })}
          >
            <option value="DK1">DK1</option>
            <option value="DK2">DK2</option>
          </select>
        </div>

        <div className="grid grid-2">
          <div className="form-group">
            <label>Period Start</label>
            <input
              type="date"
              value={periodStart}
              onChange={(e) => setPeriodStart(e.target.value)}
            />
          </div>
          <div className="form-group">
            <label>Period End</label>
            <input
              type="date"
              value={periodEnd}
              onChange={(e) => setPeriodEnd(e.target.value)}
            />
          </div>
        </div>

        <button className="btn btn-primary" onClick={fetchData} disabled={loading}>
          {loading ? 'Fetching...' : '📥 Fetch CO₂, Price and Temperature'}
        </button>

        {error && <div className="error">Error: {error}</div>}
        
        {data && (
          <div className="success">
            Data fetched successfully! You can now proceed to configure devices.
          </div>
        )}
      </div>
    </div>
  )
}

export default ScenarioPage
