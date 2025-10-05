# Exoplanet Detection System

Machine learning-powered web app that classifies potential exoplanets as **Confirmed**, **Candidate**, or **False Positive** using NASA Kepler/TESS data.

## Features

- Real-time ML predictions with 87% accuracy
- Physics-based validation rules
- Interactive React UI with confidence scores
- Ensemble model (XGBoost, Random Forest, LightGBM)

## Tech Stack

**Frontend:** React, Tailwind CSS  
**Backend:** Flask, Python, scikit-learn, XGBoost

## Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```
Runs on `http://localhost:5000`

### Frontend
```bash
cd frontend
npm install
npm start
```
Runs on `http://localhost:3000`

## Input Parameters

| Parameter | Example |
|-----------|---------|
| Orbital Period (days) | 35.0 |
| Transit Duration (hours) | 4.8 |
| Transit Depth (ppm) | 420 |
| Planet Radius (R⊕) | 1.8 |
| Equilibrium Temp (K) | 285 |
| Stellar Temp (K) | 5200 |
| Stellar Radius (R☉) | 0.92 |
| Stellar Mass (M☉) | 0.88 |

## API Endpoints

- `POST /predict` - Classify exoplanet candidate
- `GET /api/model_info` - Model information
- `GET /health` - Health check

## Deployment (Render)

**Backend:**
- Root: `backend`
- Build: `pip install -r requirements.txt`
- Start: `gunicorn app:app`

**Frontend:**
- Root: `frontend`
- Build: `npm install && npm run build`
- Publish: `build`

## License

MIT
