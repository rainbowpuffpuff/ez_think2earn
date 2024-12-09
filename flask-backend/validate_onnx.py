import onnx

# Replace with the actual path to your ONNX model
model_path = '/home/q/Downloads/BM_hack/fnirs_ezkl/fNIRSNET/save/MA/KFold/1/1/model.onnx'

try:
    # Load the ONNX model
    model = onnx.load(model_path)
    
    # Check the model for consistency and correctness
    onnx.checker.check_model(model)
    
    print("The ONNX model is valid.")
except onnx.checker.ValidationError as e:
    print(f"The ONNX model is invalid: {e}")
except Exception as e:
    print(f"An unexpected error occurred while validating the model: {e}")

import onnx

model_path = '/home/q/Downloads/BM_hack/fnirs_ezkl/fNIRSNET/save/MA/KFold/1/1/model.onnx'

try:
    model = onnx.load(model_path)
    opset_version = model.opset_import[0].version
    print(f"Opset Version: {opset_version}")
except Exception as e:
    print(f"Failed to determine opset version: {e}")

import onnx

model_path = '/home/q/Downloads/BM_hack/fnirs_ezkl/fNIRSNET/save/MA/KFold/1/1/model.onnx'

try:
    model = onnx.load(model_path)
    opset_version = model.opset_import[0].version
    operators = set([node.op_type for node in model.graph.node])
    
    print(f"Opset Version: {opset_version}")
    print(f"Operators Used: {operators}")
except Exception as e:
    print(f"Failed to inspect operators: {e}")
