# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import json
import logging
import ezkl
import asyncio

app = Flask(__name__)
CORS(app)  # Enable CORS

# Set logging level to ERROR to disable warnings from EZKL
logging.getLogger('ezkl').setLevel(logging.ERROR)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    # Save the uploaded .onnx file
    model_path = os.path.join('uploaded_model.onnx')
    file.save(model_path)

    # Run the processing script
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        outputs = loop.run_until_complete(process_model(model_path))
    except Exception as e:
        print(e)
        return jsonify({'message': 'Error processing the model', 'error': str(e)}), 500
    finally:
        loop.close()

    # Optionally delete the uploaded file after processing
    # os.remove(model_path)

    return jsonify({'message': 'Model processed successfully', 'outputs': outputs}), 200

async def process_model(model_path):
    # Paths for model and output files
    compiled_model_path = os.path.join('network.compiled')
    pk_path = os.path.join('test.pk')
    vk_path = os.path.join('test.vk')
    settings_path = os.path.join('settings.json')
    witness_path = os.path.join('witness.json')
    data_path = os.path.join('input.json')
    cal_path = os.path.join('calibration.json')
    proof_path = os.path.join('test.pf')

    outputs = []

    # 1. Prepare input data
    outputs.append("Step 1: Preparing input data for the model.")
    input_data = torch.rand(1, 1, 30, 72).detach().numpy()  
    data_array = input_data.reshape([-1]).tolist()

    # Save input data to a JSON file
    outputs.append("Saving input data to JSON format.")
    data = dict(input_data=[data_array])
    json.dump(data, open(data_path, 'w'))
    outputs.append(f"Input data saved to: {data_path}")

    # 2. Generate EZKL settings
    outputs.append("Step 2: Generating EZKL settings for the ONNX model.")
    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "public"
    py_run_args.output_visibility = "public"
    py_run_args.param_visibility = "private"

    res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)
    if res:
        outputs.append(f"EZKL settings successfully generated and saved to: {settings_path}")
    else:
        outputs.append("Error in generating EZKL settings.")
        raise Exception("Error in generating EZKL settings.")

    # 3. Calibrate the model settings using random data
    outputs.append("Step 3: Preparing random data for model calibration.")
    calibration_data = (torch.rand(20, 1, 30, 72).detach().numpy()).reshape([-1]).tolist()
    calibration_dict = dict(input_data=[calibration_data])
    json.dump(calibration_dict, open(cal_path, 'w'))
    outputs.append(f"Calibration data saved to: {cal_path}")

    # Calibrate the settings
    outputs.append("Calibrating model settings. This may take a moment...")
    await ezkl.calibrate_settings(cal_path, model_path, settings_path, "resources")
    outputs.append("Model calibration completed.")

    # 4. Compile the circuit
    outputs.append("Step 4: Compiling the model circuit for ZK proof generation.")
    res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
    if res:
        outputs.append(f"Model circuit compiled successfully and saved to: {compiled_model_path}")
    else:
        outputs.append("Error in compiling the model circuit.")
        raise Exception("Error in compiling the model circuit.")

    # 5. Fetch the Structured Reference String (SRS)
    outputs.append("Step 5: Fetching Structured Reference String (SRS) for ZK proof.")
    res = await ezkl.get_srs(settings_path)
    outputs.append("SRS fetched successfully.")

    # 6. Generate the witness file
    outputs.append("Step 6: Generating the witness file based on the input data and compiled circuit.")
    res = await ezkl.gen_witness(data_path, compiled_model_path, witness_path)
    if os.path.isfile(witness_path):
        outputs.append(f"Witness file generated and saved to: {witness_path}")
    else:
        outputs.append("Error in generating the witness file.")
        raise Exception("Error in generating the witness file.")

    # 7. Setup the circuit parameters and generate keys
    outputs.append("Step 7: Setting up the circuit parameters and generating keys (verification and proving keys).")
    res = ezkl.setup(compiled_model_path, vk_path, pk_path)
    if res:
        outputs.append(f"Verification key saved to: {vk_path}")
        outputs.append(f"Proving key saved to: {pk_path}")
    else:
        outputs.append("Error in setting up the circuit and generating keys.")
        raise Exception("Error in setting up the circuit and generating keys.")

    # 8. Generate a proof
    outputs.append("Step 8: Generating the Zero-Knowledge proof.")
    res = ezkl.prove(witness_path, compiled_model_path, pk_path, proof_path, "single")
    if os.path.isfile(proof_path):
        outputs.append(f"Proof successfully generated and saved to: {proof_path}")
    else:
        outputs.append("Error in generating the proof.")
        raise Exception("Error in generating the proof.")

    # 9. Verify the proof
    outputs.append("Step 9: Verifying the proof to ensure correctness.")
    res = ezkl.verify(proof_path, settings_path, vk_path)
    if res:
        outputs.append("Proof verified successfully! Everything is correct.")
    else:
        outputs.append("Proof verification failed.")
        raise Exception("Proof verification failed.")

    return outputs

if __name__ == '__main__':
    app.run(debug=True)

