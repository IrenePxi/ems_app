# Quick Start Guide

## Prerequisites

- Python 3.8+ with pip
- Node.js 16+ with npm
- All original modules from `ems_app_v3` (devices.py, ems.py, profiles.py, Optimization_based.py)

## Quick Setup (Windows)

### 1. Backend Setup

```bash
cd web_app\backend
pip install -r requirements.txt
python main.py
```

The backend will start on `http://localhost:8000`

### 2. Frontend Setup (in a new terminal)

```bash
cd web_app\frontend
npm install
npm run dev
```

The frontend will start on `http://localhost:3000`

## Using the Application

1. **Open** `http://localhost:3000` in your browser
2. **Page 1 - Scenario & Data**: 
   - Select date and location
   - Click "Fetch CO₂, Price and Temperature"
3. **Page 2 - Devices & Layout**:
   - Configure house parameters
   - Add devices (heat pump, hot tub, etc.)
   - Click "Calculate Load Profile"
4. **Page 3 - Analysis**:
   - Configure battery settings
   - Choose Auto or Manual mode
   - Click "Run EMS"
   - View results and visualizations

## API Testing

Visit `http://localhost:8000/docs` for interactive API documentation where you can test endpoints directly.

## Troubleshooting

### Import Errors
If you see import errors, make sure you're running from the correct directory and that all original modules (devices.py, ems.py, etc.) are in the parent `ems_app_v3` directory.

### Port Already in Use
- Backend: Change port in `backend/main.py` (line with `uvicorn.run`)
- Frontend: Change port in `frontend/vite.config.js`

### CORS Errors
Make sure the frontend proxy is configured correctly in `vite.config.js` and both servers are running.

## Next Steps

- Add authentication if needed
- Implement proper session management (currently uses localStorage)
- Add database for persistent storage
- Deploy to production hosting
