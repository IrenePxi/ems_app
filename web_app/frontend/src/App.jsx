import React, { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import ScenarioPage from './pages/ScenarioPage'
import DevicesPage from './pages/DevicesPage'
import AnalysisPage from './pages/AnalysisPage'
import './App.css'

function App() {
  return (
    <Router>
      <div className="app">
        <NavBar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<ScenarioPage />} />
            <Route path="/devices" element={<DevicesPage />} />
            <Route path="/analysis" element={<AnalysisPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

function NavBar() {
  const location = useLocation()
  
  const navItems = [
    { path: '/', label: '1️⃣ Scenario & Data', icon: '📊' },
    { path: '/devices', label: '2️⃣ Devices & Layout', icon: '🏠' },
    { path: '/analysis', label: '3️⃣ Analysis', icon: '⚡' },
  ]

  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <h1>Daily EMS Sandbox</h1>
      </div>
      <div className="navbar-links">
        {navItems.map(item => (
          <Link
            key={item.path}
            to={item.path}
            className={`nav-link ${location.pathname === item.path ? 'active' : ''}`}
          >
            <span className="nav-icon">{item.icon}</span>
            <span className="nav-label">{item.label}</span>
          </Link>
        ))}
      </div>
    </nav>
  )
}

export default App
