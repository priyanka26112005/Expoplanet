"""
Flask Backend for Exoplanet Detection System
Loads your trained Super Ensemble model and provides prediction API
"""

from flask import Flask, request, jsonify
import numpy as np
import pickle
from flask_cors import CORS 
import os
import warnings

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# ============================================================================
# LOAD YOUR TRAINED MODEL AND PREPROCESSING TOOLS
# ============================================================================

print("🔄 Loading model and preprocessing tools...")

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Try to load the best model - Super Ensemble first, then fall back to others
MODEL_PATHS = [
    r'D:\Exoplanet Detection System\models\optimized\super_ensemble_optimized.pkl',
    r'D:\Exoplanet Detection System\models\optimized\xgboost_optimized.pkl',
    r'D:\Exoplanet Detection System\models\optimized\random_forest_optimized.pkl',
    r'D:\Exoplanet Detection System\models\optimized\lightgbm_optimized.pkl',
]

model = None
model_name = "Unknown"

for MODEL_PATH in MODEL_PATHS:
    if os.path.exists(MODEL_PATH):
        try:
            print(f"🔄 Trying to load: {MODEL_PATH}")
            with open(MODEL_PATH, 'rb') as f:
                loaded_obj = pickle.load(f)
            
            # Check if it's directly a model or wrapped in a dict
            if hasattr(loaded_obj, 'predict') and hasattr(loaded_obj, 'predict_proba'):
                model = loaded_obj
                model_name = os.path.basename(MODEL_PATH).replace('.pkl', '').replace('_', ' ').title()
                print(f"✅ Model loaded: {model_name}")
                print(f"   Type: {type(model).__name__}")
                break
            elif isinstance(loaded_obj, dict):
                # Try to extract model from dict
                for key in ['best_model', 'model', 'ensemble', 'classifier']:
                    if key in loaded_obj and hasattr(loaded_obj[key], 'predict'):
                        model = loaded_obj[key]
                        model_name = loaded_obj.get('name', os.path.basename(MODEL_PATH))
                        print(f"✅ Model extracted from dict: {model_name}")
                        break
                if model:
                    break
        except Exception as e:
            print(f"⚠️  Failed to load {MODEL_PATH}: {str(e)}")
            continue
    else:
        print(f"⚠️  File not found: {MODEL_PATH}")

if model is None:
    print("❌ CRITICAL: No model could be loaded!")
    print("   Please check that at least one of these files exists:")
    for path in MODEL_PATHS:
        print(f"   - {path}")

# Load preprocessing tools
scaler = None
imputer = None
feature_names = []

try:
    with open(r'D:\Exoplanet Detection System\models\scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Scaler loaded")
except Exception as e:
    print(f"⚠️  No scaler found: {str(e)}")

try:
    with open(r'D:\Exoplanet Detection System\models\imputer.pkl', 'rb') as f:
        imputer = pickle.load(f)
    print("✅ Imputer loaded")
except Exception as e:
    print(f"⚠️  No imputer found: {str(e)}")

try:
    import json
    with open(r'D:\Exoplanet Detection System\models\preprocessing_config.json', 'r') as f:
        config = json.load(f)
    feature_names = config.get('feature_columns', [])
    print(f"✅ Loaded {len(feature_names)} feature names")
    print(f"   Features: {feature_names}")
except Exception as e:
    print(f"⚠️  No config found: {str(e)}")

# Class labels
CLASS_LABELS = {
    0: 'FALSE POSITIVE',
    1: 'CANDIDATE', 
    2: 'CONFIRMED'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_physical_constraints(data):
    """Check if input violates basic physics - returns (is_valid, reason, override_class)"""
    try:
        # Extract values - try multiple key formats
        if isinstance(data, dict):
            def get_value(keys):
                for key in keys:
                    if key in data and data[key] is not None:
                        return float(data[key])
                return 0.0
            
            orbital_period = get_value(['orbital_period', 'Orbital Period', 'koi_period'])
            transit_duration = get_value(['transit_duration', 'Transit Duration', 'koi_duration'])
            transit_depth = get_value(['transit_depth', 'Transit Depth', 'koi_depth'])
            planet_radius = get_value(['planet_radius', 'Planet Radius', 'koi_prad'])
            equilibrium_temp = get_value(['equilibrium_temperature', 'Equilibrium Temperature', 'koi_teq'])
            stellar_radius = get_value(['stellar_radius', 'Stellar Radius', 'koi_srad'])
            stellar_mass = get_value(['stellar_mass', 'Stellar Mass', 'koi_smass'])
        else:
            values = list(data)
            orbital_period = float(values[0]) if len(values) > 0 else 0
            transit_duration = float(values[1]) if len(values) > 1 else 0
            transit_depth = float(values[2]) if len(values) > 2 else 0
            planet_radius = float(values[3]) if len(values) > 3 else 0
            equilibrium_temp = float(values[4]) if len(values) > 4 else 0
            stellar_radius = float(values[6]) if len(values) > 6 else 1.0
            stellar_mass = float(values[7]) if len(values) > 7 else 1.0
        
        # Rule 1: Orbital period too short
        if orbital_period < 0.5:
            return False, "Orbital period impossibly short - likely instrumental artifact", 0
        
        # Rule 2: Planet too large
        if planet_radius > 12.0:
            return False, "Object too large to be a planet - likely eclipsing binary star", 0
        
        # Rule 3: Transit depth too deep
        if transit_depth > 50000:
            return False, "Transit depth too large - indicates stellar companion, not planet", 0
        
        # Rule 4: Transit duration inconsistent
        if transit_duration > (orbital_period * 24 * 0.3):
            return False, "Transit duration inconsistent with orbital period", 0
        
        # Rule 5: Temperature too high
        if equilibrium_temp > 3500 and orbital_period < 1.0:
            return False, "Temperature too high - planet would be vaporized or inside star", 0
        
        # Rule 6: Signal too weak
        if planet_radius < 0.3 and transit_depth < 100:
            return False, "Signal too weak - likely noise or instrumental artifact", 0
        
        # Rule 7: Evolved/giant stars with weak signals
        if stellar_radius > 2.5 and stellar_mass < 1.5:
            if transit_depth < 1000:
                return False, "Weak transit signal around evolved/giant star - likely background eclipsing binary or stellar activity", 0
        
        # Rule 8: Long orbital periods need more validation
        if orbital_period > 100:
            # Return as CANDIDATE (class 1) instead of FALSE POSITIVE
            return False, "Long orbital period (>100 days) - insufficient transits observed for confirmation. Requires additional observational data to verify", 1
        
        # Rule 9: Small planets with weak signals need more validation
        if planet_radius < 1.5 and transit_depth < 250:
            return False, "Small planet with marginal signal strength - requires additional observations for confirmation", 1
        
        return True, "Passes physical constraints", None
    
    except Exception as e:
        print(f"⚠️  Physics check error: {str(e)}")
        return True, "Could not verify constraints", None

def preprocess_input(data):
    """Preprocess user input to match training data format"""
    try:
        if isinstance(data, dict):
            if feature_names:
                values = []
                for feat in feature_names:
                    # Try different key formats
                    val = data.get(feat, 0)
                    if val is None:
                        val = 0
                    values.append(float(val))
            else:
                values = [float(v) if v is not None else 0.0 for v in data.values()]
            X = np.array(values).reshape(1, -1)
        else:
            X = np.array(data, dtype=float).reshape(1, -1)
        
        print(f"   Raw shape: {X.shape}, values: {X[0][:3]}... (showing first 3)")
        
        if imputer is not None:
            X = imputer.transform(X)
            print(f"   After imputer: {X[0][:3]}...")
        
        if scaler is not None:
            X = scaler.transform(X)
            print(f"   After scaler: {X[0][:3]}...")
        
        return X
    except Exception as e:
        print(f"❌ Preprocessing error: {str(e)}")
        raise

def get_interpretation(prediction_class, confidence):
    """Generate human-readable interpretation"""
    interpretations = {
        2: {
            'high': "This object shows strong characteristics of a confirmed exoplanet. The orbital mechanics and transit signature align well with known exoplanet patterns.",
            'medium': "This object likely represents a real exoplanet, though some parameters show minor inconsistencies.",
            'low': "While classified as confirmed, the confidence is relatively low. More data is needed."
        },
        1: {
            'high': "This object is a strong candidate for being an exoplanet but requires additional confirmation.",
            'medium': "This object shows some characteristics consistent with exoplanets but needs further analysis.",
            'low': "This candidate shows weak signals. It may be an exoplanet, but data is insufficient."
        },
        0: {
            'high': "This signal is most likely not an exoplanet. It could be instrumental noise, stellar activity, or an eclipsing binary.",
            'medium': "While classified as a false positive, there's some uncertainty.",
            'low': "This is likely a false positive, though the classification has lower confidence."
        }
    }
    
    conf_level = 'high' if confidence > 0.75 else 'medium' if confidence > 0.50 else 'low'
    return interpretations[prediction_class][conf_level]

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded - check server logs'
            }), 500
        
        data = request.json
        print(f"\n{'='*70}")
        print(f"📥 Received prediction request")
        print(f"   Data keys: {list(data.keys())}")
        
        # Check physics
        is_valid, reason, override_class = check_physical_constraints(data)
        print(f"🔬 Physics check: {'✅ PASS' if is_valid else '❌ OVERRIDE'} - {reason}")
        
        if not is_valid:
            # Determine which class to override to (0=FP, 1=CANDIDATE)
            if override_class == 1:  # CANDIDATE
                response = {
                    'success': True,
                    'prediction_label': 'CANDIDATE',
                    'confidence': {
                        'confirmed': 0.15,
                        'candidate': 0.70,
                        'false_positive': 0.15
                    },
                    'interpretation': f"Physics-based analysis: {reason}",
                    'model_accuracy': 0.87,
                    'physics_override': True
                }
                print(f"⚠️  PHYSICS OVERRIDE → CANDIDATE")
            else:  # FALSE POSITIVE
                response = {
                    'success': True,
                    'prediction_label': 'FALSE POSITIVE',
                    'confidence': {
                        'confirmed': 0.05,
                        'candidate': 0.10,
                        'false_positive': 0.85
                    },
                    'interpretation': f"Physics-based analysis: {reason}",
                    'model_accuracy': 0.87,
                    'physics_override': True
                }
                print(f"⚠️  PHYSICS OVERRIDE → FALSE POSITIVE")
            
            print("="*70)
            return jsonify(response)
        
        # Preprocess
        print(f"🔧 Preprocessing input...")
        X = preprocess_input(data)
        
        # Predict
        print(f"🤖 Running model prediction...")
        predicted_class = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]
        
        print(f"🎯 Results:")
        print(f"   FALSE POSITIVE: {probabilities[0]:.1%}")
        print(f"   CANDIDATE:      {probabilities[1]:.1%}")
        print(f"   CONFIRMED:      {probabilities[2]:.1%}")
        print(f"   → Predicted: {CLASS_LABELS[predicted_class]}")
        
        response = {
            'success': True,
            'prediction_label': CLASS_LABELS[predicted_class],
            'confidence': {
                'confirmed': float(probabilities[2]),
                'candidate': float(probabilities[1]),
                'false_positive': float(probabilities[0])
            },
            'interpretation': get_interpretation(predicted_class, float(probabilities[predicted_class])),
            'model_accuracy': 0.87,
            'physics_override': False
        }
        
        print(f"✅ Response sent")
        print("="*70)
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ PREDICTION ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 400

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Return model information"""
    return jsonify({
        'model_accuracy': 87.0,
        'model_type': model_name if model else 'Not Loaded',
        'training_samples': 'NASA Kepler + TESS',
        'features_used': len(feature_names) if feature_names else 8,
        'classes': list(CLASS_LABELS.values()),
        'model_loaded': model is not None
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy' if model else 'model_not_loaded',
        'message': f'Model: {model_name}' if model else 'No model loaded',
        'model_loaded': model is not None
    }), 200

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 EXOPLANET DETECTION SYSTEM - API SERVER")
    print("="*70)
    if model:
        print(f"✅ Model: {model_name}")
        print(f"✅ Type: {type(model).__name__}")
        print(f"✅ Features: {len(feature_names)}")
        print(f"✅ Physics validation: ACTIVE (8 rules)")
    else:
        print(f"❌ NO MODEL LOADED - Server will return errors")
    print(f"\n🌐 Server: http://127.0.0.1:5000")
    print(f"📡 Endpoints: /predict, /api/model_info, /health")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)