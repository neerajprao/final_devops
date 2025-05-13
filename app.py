import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import pennylane as qml
from pennylane import numpy as qnp
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ---------------- Define DL Model Architectures ----------------
class TabNetModel(nn.Module):
    def __init__(self):
        super(TabNetModel, self).__init__()
        self.fc1 = nn.Linear(13, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ---------------- Quantum Model Functions ----------------
def create_vqc_circuit(inputs, weights):
    """Create a Variational Quantum Circuit"""
    # Quantum node
    @qml.qnode(qml.device("default.qubit", wires=4))
    def circuit():
        # Feature embedding
        for i in range(4):
            qml.RY(inputs[i], wires=i)

        # Variational layers
        for layer in weights:
            for i, param in enumerate(layer):
                qml.RY(param, wires=i % 4)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])

        # Measurement
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]
    return circuit()

# ---------------- Load Models ----------------
def load_models():
    global xgb_model, tabnet_model
    global vqc_model_data, qnn_model_data
    global available_models

    available_models = {
        'xgb': False,
        'tabnet': False,
        'vqc': False,
        'qnn': False
    }

    # Load XGBoost model
    try:
        xgb_model = XGBClassifier()
        xgb_model.load_model('models/xgboost_model.json')
        available_models['xgb'] = True
    except Exception as e:
        print(f"Error loading XGBoost model: {e}")

    # Load TabNet model
    try:
        tabnet_model = TabNetModel()
        tabnet_model.load_state_dict(torch.load('models/tabnet_model.pt'), strict=False)
        tabnet_model.eval()
        available_models['tabnet'] = True
    except Exception as e:
        print(f"Error loading TabNet model: {e}")

    # Load VQC model
    try:
        vqc_model_data = joblib.load('models/vqc_model.pkl')
        available_models['vqc'] = True
    except Exception as e:
        print(f"Error loading VQC model: {e}")

    # Load QNN model
    try:
        qnn_model_data = np.load('models/qnn_model.npz', allow_pickle=True)
        available_models['qnn'] = True
    except Exception as e:
        print(f"Error loading QNN model: {e}")

    return available_models

# ---------------- Prediction Helpers ----------------
def predict_vqc(input_data):
    """Predict with VQC model"""
    if not available_models['vqc'] or vqc_model_data is None:
        return None

    try:
        if isinstance(vqc_model_data, dict):
            # Example implementation - adapt based on your actual VQC
            weights = vqc_model_data.get('weights', np.random.rand(3, 4))
            circuit_output = create_vqc_circuit(input_data[0][:4], weights)
            return 1 if np.mean(circuit_output) > 0 else 0
        return None
    except Exception as e:
        print(f"VQC prediction error: {e}")
        return None

def predict_qnn(input_data):
    """Predict with QNN model"""
    if not available_models['qnn'] or qnn_model_data is None:
        return None

    try:
        if isinstance(qnn_model_data, np.lib.npyio.NpzFile):
            # Example implementation - adapt based on your actual QNN
            weights = qnn_model_data['weights'] if 'weights' in qnn_model_data.files else np.random.rand(3, 4)
            dev = qml.device("default.qubit", wires=4)

            @qml.qnode(dev)
            def qnn_circuit(inputs):
                for i in range(4):
                    qml.RY(inputs[i % len(inputs)], wires=i)
                for layer in weights:
                    for i, param in enumerate(layer):
                        qml.RY(param, wires=i % 4)
                return [qml.expval(qml.PauliZ(i)) for i in range(4)]

            result = qnn_circuit(input_data[0])
            return 1 if np.mean(result) > 0 else 0
        return None
    except Exception as e:
        print(f"QNN prediction error: {e}")
        return None

def prepare_input_data(form_data):
    """Prepare input data for all models"""
    credit_score = float(form_data.get('creditScore'))
    age = float(form_data.get('age'))
    tenure = float(form_data.get('tenure'))
    balance = float(form_data.get('balance'))
    num_products = float(form_data.get('numProducts'))
    has_card = float(form_data.get('hasCard'))
    is_active = float(form_data.get('isActive'))
    salary = float(form_data.get('salary'))
    geography = form_data.get('geography')
    gender = form_data.get('gender')

    # One-hot encoding
    geography_france = 1 if geography == 'France' else 0
    geography_germany = 1 if geography == 'Germany' else 0
    geography_spain = 1 if geography == 'Spain' else 0
    gender_female = 1 if gender == 'Female' else 0
    gender_male = 1 if gender == 'Male' else 0

    input_data = np.array([
        credit_score, geography_france, geography_germany, geography_spain,
        gender_female, gender_male, age, tenure, balance, num_products,
        has_card, is_active, salary
    ]).reshape(1, -1)

    return input_data

# ---------------- Flask Routes ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = prepare_input_data(data)
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        predictions = {}

        # Traditional ML models
        if available_models['xgb']:
            try:
                predictions['xgb'] = int(xgb_model.predict(input_data)[0])
            except:
                predictions['xgb'] = "Error"

        # Deep Learning models
        if available_models['tabnet']:
            try:
                with torch.no_grad():
                    tabnet_output = tabnet_model(input_tensor)
                    predictions['tabnet'] = 1 if torch.sigmoid(tabnet_output).item() > 0.5 else 0
            except:
                predictions['tabnet'] = "Error"

        # Quantum models
        if available_models['vqc']:
            vqc_pred = predict_vqc(input_data)
            predictions['vqc'] = vqc_pred if vqc_pred is not None else "Error"

        if available_models['qnn']:
            qnn_pred = predict_qnn(input_data)
            predictions['qnn'] = qnn_pred if qnn_pred is not None else "Error"

        # Calculate consensus based on successful integer predictions
        valid_preds = [p for p in predictions.values() if isinstance(p, int)]
        num_valid_preds = len(valid_preds)

        if num_valid_preds > 0:
            sum_valid_preds = sum(valid_preds)
            consensus = 1 if sum_valid_preds > num_valid_preds / 2 else 0
            predictions['consensus'] = consensus
            predictions['confidence'] = sum_valid_preds / num_valid_preds if consensus == 1 else 1 - (sum_valid_preds / num_valid_preds)
        else:
            predictions['consensus'] = "N/A"
            predictions['confidence'] = "N/A"

        return jsonify({
            'success': True,
            'predictions': predictions,
            'available_models': available_models
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    load_models()
    app.run(debug=True)