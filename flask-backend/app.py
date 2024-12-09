# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import json
import logging
import ezkl
import asyncio
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS

# Set logging level to ERROR to reduce ezkl warnings
logging.getLogger('ezkl').setLevel(logging.ERROR)

# Define the base directory where all files are located
BASE_DIR = '/home/q/Downloads/BM_hack/fnirs_ezkl/fNIRSNET/save/MA/KFold/1/1/'

# Ensure BASE_DIR exists
if not os.path.isdir(BASE_DIR):
    os.makedirs(BASE_DIR)
    print(f"Created base directory at: {BASE_DIR}")

# Define full paths based on BASE_DIR
MODEL_PATH = os.path.join(BASE_DIR, 'model.onnx')  # The uploaded ONNX model will be saved here
COMPILED_MODEL_PATH = os.path.join(BASE_DIR, 'model.compiled')
PK_PATH = os.path.join(BASE_DIR, 'test.pk')
VK_PATH = os.path.join(BASE_DIR, 'test.vk')
SETTINGS_PATH = os.path.join(BASE_DIR, 'settings.json')

WITNESS_PATH = os.path.join(BASE_DIR, 'witness.json')
DATA_PATH = os.path.join(BASE_DIR, 'input.json')
CAL_PATH = os.path.join(BASE_DIR, 'calibration.json')

PROOF_PATH = os.path.join(BASE_DIR, 'test.pf')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Received a file upload request.")
    
    if 'file' not in request.files:
        print("No file part in the request.")
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        print("No file selected.")
        return jsonify({'message': 'No selected file'}), 400

    # Save the uploaded .onnx file to BASE_DIR/model.onnx
    print(f"Saving uploaded file to {MODEL_PATH}")
    file.save(MODEL_PATH)

    # Run the processing script
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        print("Starting model processing...")
        outputs = loop.run_until_complete(process_model(MODEL_PATH))
        print("Model processed successfully.")
    except Exception as e:
        print(f"Error during model processing: {e}")
        return jsonify({'message': 'Error processing the model', 'error': str(e)}), 500
    finally:
        loop.close()

    # After processing, attempt to load the generated JSON files
    input_content = {}
    witness_content = {}

    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'r') as f:
            input_content = json.load(f)

    if os.path.exists(WITNESS_PATH):
        with open(WITNESS_PATH, 'r') as f:
            witness_content = json.load(f)

    return jsonify({
        'message': 'Model processed successfully',
        'outputs': outputs,
        'input_data': input_content,
        'witness_data': witness_content
    }), 200

async def process_model(model_path):
    # Paths are already defined globally based on BASE_DIR
    global COMPILED_MODEL_PATH, PK_PATH, VK_PATH, SETTINGS_PATH
    global WITNESS_PATH, DATA_PATH, CAL_PATH, PROOF_PATH

    outputs = []

    # ==============================
    # Step 1: Prepare input.json
    # ==============================

    print("Step 1: Preparing input data for the model.")
    outputs.append("Step 1: Preparing input data for the model.")
    
    # Load test data from 'test_data.npz'
    test_data_npz_path = os.path.join(BASE_DIR, 'test_data.npz')
    if not os.path.isfile(test_data_npz_path):
        raise FileNotFoundError(f"Test data file not found at {test_data_npz_path}")
    
    test_data = np.load(test_data_npz_path)
    X_test = test_data['X']
    y_test = test_data['y']  # Not used for EZKL, but available
    
    print(f"Loaded test data from {test_data_npz_path}")
    outputs.append(f"Loaded test data from {test_data_npz_path}")
    print(f"Test data shape: {X_test.shape}")  # Expected shape: (N, 1, 72, 30)
    outputs.append(f"Test data shape: {X_test.shape}")

    # Select a single test sample to create input.json
    # Here, we take the first sample; you can choose any
    sample_index = 0  # Change this index to select a different sample
    sample_input = X_test[sample_index]  # Shape: (1, 72, 30)
    
    print(f"Selected sample index: {sample_index}")
    outputs.append(f"Selected sample index: {sample_index}")
    
    # Normalize or preprocess sample_input if required
    # Ensure it matches the preprocessing done during training
    # Example: normalization (assuming it was done during training)
    sample_input_normalized = (sample_input - sample_input.mean()) / (sample_input.std() + 1e-5)
    
    print("Normalized the sample input.")
    outputs.append("Normalized the sample input.")
    
    # Convert to list and reshape as needed
    data_array = sample_input_normalized.reshape([-1]).tolist()
    
    data = {"input_data": [data_array]}
    
    # Serialize data into file:
    with open(DATA_PATH, 'w') as f:
        json.dump(data, f)
    
    print(f"Input data saved to: {DATA_PATH}")
    outputs.append(f"Input data saved to: {DATA_PATH}")
    print(f"Input data shape for EZKL: {np.array(data['input_data']).shape}")  # Should be (1, 72*30)
    outputs.append(f"Input data shape for EZKL: {np.array(data['input_data']).shape}")

    # ==============================
    # Step 2: Generate EZKL Settings
    # ==============================
    
    print("Step 2: Generating EZKL settings for the ONNX model.")
    outputs.append("Step 2: Generating EZKL settings for the ONNX model.")
    
    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "private"
    py_run_args.output_visibility = "public"
    py_run_args.param_visibility = "fixed"  # private by default
    
    res = ezkl.gen_settings(model_path, SETTINGS_PATH, py_run_args=py_run_args)
    if res:
        print(f"EZKL settings successfully generated and saved to: {SETTINGS_PATH}")
        outputs.append(f"EZKL settings successfully generated and saved to: {SETTINGS_PATH}")
    else:
        print("Error in generating EZKL settings.")
        outputs.append("Error in generating EZKL settings.")
        raise Exception("Error in generating EZKL settings.")
    
    # ==============================
    # Step 3: Prepare Calibration Data
    # ==============================
    
    print("Step 3: Preparing calibration data using existing test samples.")
    outputs.append("Step 3: Preparing calibration data using existing test samples.")
    
    num_cal_samples = 20  # Number of calibration samples
    if X_test.shape[0] < num_cal_samples:
        raise ValueError(f"Not enough test samples ({X_test.shape[0]}) for calibration.")
    
    calibration_samples = X_test[:num_cal_samples]
    calibration_samples_normalized = (calibration_samples - calibration_samples.mean(axis=(1,2), keepdims=True)) / (calibration_samples.std(axis=(1,2), keepdims=True) + 1e-5)
    
    cal_data_array = calibration_samples_normalized.reshape(num_cal_samples, -1).tolist()
    
    cal_data = {"input_data": cal_data_array}
    
    # Serialize calibration data into file:
    with open(CAL_PATH, 'w') as f:
        json.dump(cal_data, f)
    
    print(f"Calibration data saved to: {CAL_PATH}")
    outputs.append(f"Calibration data saved to: {CAL_PATH}")
    print(f"Calibration data shape for EZKL: {np.array(cal_data['input_data']).shape}")  # Should be (20, 72*30)
    outputs.append(f"Calibration data shape for EZKL: {np.array(cal_data['input_data']).shape}")

    # ==============================
    # Step 4: Calibrate Settings
    # ==============================
    
    print("Step 4: Calibrating model settings...")
    outputs.append("Step 4: Calibrating model settings...")
    
    await ezkl.calibrate_settings(CAL_PATH, model_path, SETTINGS_PATH, "resources")
    
    print("Model calibration completed.")
    outputs.append("Model calibration completed.")

    # ==============================
    # Step 5: Compile Circuit
    # ==============================
    
    print("Step 5: Compiling the model circuit for ZK proof generation.")
    outputs.append("Step 5: Compiling the model circuit for ZK proof generation.")
    
    res = ezkl.compile_circuit(model_path, COMPILED_MODEL_PATH, SETTINGS_PATH)
    if res:
        print(f"Model circuit compiled successfully and saved to: {COMPILED_MODEL_PATH}")
        outputs.append(f"Model circuit compiled successfully and saved to: {COMPILED_MODEL_PATH}")
    else:
        print("Error in compiling the model circuit.")
        outputs.append("Error in compiling the model circuit.")
        raise Exception("Error in compiling the model circuit.")

    # ==============================
    # Step 6: Obtain SRS (Structured Reference String)
    # ==============================
    
    print("Step 6: Obtaining Structured Reference String (SRS) for ZK proof.")
    outputs.append("Step 6: Obtaining Structured Reference String (SRS) for ZK proof.")
    
    res = await ezkl.get_srs(SETTINGS_PATH)
    print("SRS obtained successfully.")
    outputs.append("SRS obtained successfully.")

    # ==============================
    # Step 7: Generate Witness
    # ==============================
    
    print("Step 7: Generating the witness file.")
    outputs.append("Step 7: Generating the witness file.")
    
    res = await ezkl.gen_witness(DATA_PATH, COMPILED_MODEL_PATH, WITNESS_PATH)
    if os.path.isfile(WITNESS_PATH):
        print(f"Witness file generated and saved to: {WITNESS_PATH}")
        outputs.append(f"Witness file generated and saved to: {WITNESS_PATH}")
    else:
        print("Error in generating the witness file.")
        outputs.append("Error in generating the witness file.")
        raise Exception("Error in generating the witness file.")

    # ==============================
    # Step 8: Setup (Key Generation)
    # ==============================
    
    print("Step 8: Setting up circuit parameters and generating keys.")
    outputs.append("Step 8: Setting up circuit parameters and generating keys.")
    
    res = ezkl.setup(COMPILED_MODEL_PATH, VK_PATH, PK_PATH)
    if res:
        print(f"Verification key saved to: {VK_PATH}")
        print(f"Proving key saved to: {PK_PATH}")
        outputs.append(f"Verification key saved to: {VK_PATH}")
        outputs.append(f"Proving key saved to: {PK_PATH}")
    else:
        print("Error in setting up circuit and generating keys.")
        outputs.append("Error in setting up circuit and generating keys.")
        raise Exception("Error in setting up circuit and generating keys.")

    # ==============================
    # Step 9: Generate a Proof
    # ==============================
    
    print("Step 9: Generating the Zero-Knowledge proof.")
    outputs.append("Step 9: Generating the Zero-Knowledge proof.")
    
    res = ezkl.prove(WITNESS_PATH, COMPILED_MODEL_PATH, PK_PATH, PROOF_PATH, "single")
    if os.path.isfile(PROOF_PATH):
        print(f"Proof successfully generated and saved to: {PROOF_PATH}")
        outputs.append(f"Proof successfully generated and saved to: {PROOF_PATH}")
    else:
        print("Error in generating the proof.")
        outputs.append("Error in generating the proof.")
        raise Exception("Error in generating the proof.")

    # ==============================
    # Step 10: Verify the Proof
    # ==============================
    
    print("Step 10: Verifying the proof.")
    outputs.append("Step 10: Verifying the proof.")
    
    res = ezkl.verify(PROOF_PATH, SETTINGS_PATH, VK_PATH)
    if res:
        print("Proof verified successfully!")
        outputs.append("Proof verified successfully!")
    else:
        print("Proof verification failed.")
        outputs.append("Proof verification failed.")
        raise Exception("Proof verification failed.")

    return outputs

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)

