# EMS Web Application

A modern web application for Energy Management System simulation, converted from Streamlit to a FastAPI + React architecture.

## Architecture

- **Backend**: FastAPI (Python) - REST API
- **Frontend**: React + Vite - Modern single-page application
- **Communication**: REST API with JSON

## Project Structure

```
web_app/
├── backend/
│   ├── main.py              # FastAPI application entry point
│   ├── api/
│   │   ├── data.py          # Data fetching endpoints (price, CO2, weather, PV)
│   │   ├── devices.py       # Device configuration and load calculation
│   │   ├── ems.py           # EMS execution endpoints
│   │   └── optimization.py  # Optimization-based scheduling
│   └── requirements.txt     # Python dependencies
└── frontend/
    ├── src/
    │   ├── App.jsx          # Main React app with routing
    │   ├── pages/
    │   │   ├── ScenarioPage.jsx    # Page 1: Scenario & Data
    │   │   ├── DevicesPage.jsx     # Page 2: Devices & Layout
    │   │   └── AnalysisPage.jsx    # Page 3: Analysis
    │   └── main.jsx
    ├── package.json
    └── vite.config.js
```

## Setup Instructions

### Backend Setup

1. Navigate to backend directory:
```bash
cd web_app/backend
```

2. Create virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the backend server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd web_app/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## API Documentation

Once the backend is running, visit:
- API docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

## Features

- ✅ Modern web interface (no Streamlit dependency)
- ✅ RESTful API architecture
- ✅ Real-time data fetching from EnergiDataService
- ✅ Device modeling and load calculation
- ✅ EMS optimization and execution
- ✅ Interactive visualizations with Plotly
- ✅ Responsive design

## Deployment

### Backend Deployment

The FastAPI backend can be deployed to:
- Heroku
- AWS Lambda (with Mangum)
- DigitalOcean
- Any Python hosting service

### Frontend Deployment

The React frontend can be deployed to:
- Vercel
- Netlify
- AWS S3 + CloudFront
- Any static hosting service

Build the frontend:
```bash
cd frontend
npm run build
```

The `dist/` folder contains the production build.

## Differences from Streamlit Version

1. **State Management**: Uses localStorage and React state instead of Streamlit session_state
2. **API Layer**: All business logic exposed via REST API
3. **Frontend**: Modern React components instead of Streamlit widgets
4. **Routing**: React Router instead of Streamlit page navigation
5. **Deployment**: Can be deployed as separate services

## Notes

- The backend imports modules from the parent directory (original app location)
- Make sure the original `devices.py`, `ems.py`, `profiles.py`, and `Optimization_based.py` are accessible
- Session management uses localStorage (consider adding proper backend sessions for production)
