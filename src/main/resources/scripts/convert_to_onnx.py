"""
convert_to_onnx.py
Converts a pre-trained PyTorch or Keras/TensorFlow model to ONNX and verifies the exported model with ONNX Runtime.

Usage examples:
# PyTorch:
python convert_to_onnx.py --framework pytorch --model-path model.pth --output slr_model.onnx \
    --seq-len 30 --feature-dim 49152 --opset 14 --apply-softmax

# Keras (.h5) or SavedModel:
python convert_to_onnx.py --framework keras --model-path model.h5 --output slr_model.onnx \
    --seq-len 30 --feature-dim 49152 --opset 14 --apply-softmax

Dependencies:
pip install torch torchvision onnx onnxruntime numpy tensorflow tf2onnx
(If you only use PyTorch -> ONNX, tensorflow/tf2onnx is not required.)
"""
import argparse
import os
import sys
import numpy as np

# Try optional imports (import inside functions for clarity)
def export_pytorch(model_path, output_path, seq_len, feature_dim, opset, apply_softmax, dynamic):
    import torch
    import torch.nn as nn

    # Load model: assume the user saved full state_dict or scripted model.
    # Try loading state_dict into a user-provided class is not possible generically,
    # so if saving only state_dict, user must provide code to re-create the model and load state_dict.
    # We attempt torch.jit.load first (scripted), then torch.load (may be state_dict or whole model).
    print("Loading PyTorch model from:", model_path)
    model = None
    try:
        # Try JIT script/module
        model = torch.jit.load(model_path, map_location='cpu')
        print("Loaded model as TorchScript.")
    except Exception:
        # Try loading a pickled model (may be whole model)
        obj = torch.load(model_path, map_location='cpu')
        if isinstance(obj, nn.Module):
            model = obj
            print("Loaded model object from torch.load.")
        else:
            raise RuntimeError("torch.load returned non-Module. If you saved state_dict, re-create your model in code and load state_dict, then provide the model object or save a scripted model.")

    model.eval()
    # Optionally wrap with Softmax so exported model outputs probabilities
    if apply_softmax:
        class Wrapped(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.sm = nn.Softmax(dim=-1)
            def forward(self, x):
                out = self.m(x)
                return self.sm(out)
        model = Wrapped(model)

    # Dummy input: shape [batch, seq_len, feature_dim]
    dummy = torch.randn(1, seq_len, feature_dim, dtype=torch.float32)
    input_names = ["input"]
    output_names = ["output"]

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {'input': {0: 'batch_size', 1: 'sequence_length'}, 'output': {0: 'batch_size'}}

    print(f"Exporting to ONNX (opset={opset}) -> {output_path}")
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    print("Export complete.")

def export_keras(model_path, output_path, seq_len, feature_dim, opset, apply_softmax, dynamic):
    import tensorflow as tf
    import tf2onnx

    print("Loading Keras/TensorFlow model from:", model_path)
    # model_path can be .h5 or a directory for SavedModel
    if os.path.isdir(model_path):
        model = tf.keras.models.load_model(model_path)
        print("Loaded SavedModel directory.")
    else:
        model = tf.keras.models.load_model(model_path)
        print("Loaded Keras model file.")

    # Optionally wrap with softmax
    if apply_softmax:
        # create a new model that applies softmax to the outputs
        inputs = model.inputs
        outputs = model.outputs
        import tensorflow as tf
        soft_outputs = [tf.keras.layers.Softmax(axis=-1)(out) for out in outputs]
        model = tf.keras.Model(inputs=inputs, outputs=soft_outputs)
        print("Wrapped model with Softmax layer for outputs.")

    # Prepare input signature: allow dynamic batch and sequence dims if requested
    batch_dim = None if dynamic else 1
    seq_dim = None if dynamic else seq_len
    input_signature = [tf.TensorSpec([batch_dim, seq_dim, feature_dim], tf.float32, name="input")]

    print(f"Converting Keras model to ONNX (opset={opset}) -> {output_path}")
    # tf2onnx.convert.from_keras returns (onnx_model_proto, external_tensor_storage)
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
        output_path=output_path
    )
    print("Conversion complete.")

def verify_onnx(onnx_path, framework, model_path, seq_len, feature_dim, apply_softmax, dynamic, atol=1e-5):
    import onnxruntime as ort
    import numpy as np
    print("Verifying exported ONNX model:", onnx_path)
    # create random test input
    test_batch = 2 if dynamic else 1
    seq = seq_len if not dynamic else max(1, seq_len)  # use provided seq_len for shape if possible
    x = np.random.randn(test_batch, seq, feature_dim).astype(np.float32)

    # Run ONNX Runtime
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: x})[0]

    print("ONNX runtime output shape:", onnx_out.shape)

    # If framework available, run original model and compare (best-effort)
    if framework == 'pytorch':
        try:
            import torch
            import torch.nn as nn
            print("Loading PyTorch model for numeric comparison...")
            # Attempt JIT load or torch.load
            try:
                model = torch.jit.load(model_path, map_location='cpu')
            except Exception:
                obj = torch.load(model_path, map_location='cpu')
                if isinstance(obj, nn.Module):
                    model = obj
                else:
                    print("Could not obtain a runnable PyTorch model for numeric comparison (state_dict or unknown object). Skipping numeric compare.")
                    return
            model.eval()
            if apply_softmax:
                import torch.nn.functional as F
                # ensure wrapped in softmax; if you didn't wrap at export time, result differs
                def run_model_np(inp):
                    t = torch.from_numpy(inp).float()
                    with torch.no_grad():
                        out = model(t)
                        out = F.softmax(out, dim=-1)
                    return out.cpu().numpy()
            else:
                def run_model_np(inp):
                    t = torch.from_numpy(inp).float()
                    with torch.no_grad():
                        out = model(t)
                    return out.cpu().numpy()
            fw_out = run_model_np(x)
            print("Framework output shape:", fw_out.shape)
            # Compare shapes
            if fw_out.shape != onnx_out.shape:
                print("Shape mismatch between framework and ONNX outputs. Skipping numeric compare.")
                return
            maxdiff = float(np.max(np.abs(fw_out - onnx_out)))
            print(f"Max abs diff between PyTorch and ONNX outputs: {maxdiff:.6f}")
            if maxdiff <= atol:
                print("Verification PASSED (within tolerance).")
            else:
                print("Verification WARNING: difference exceeds tolerance.")
        except Exception as e:
            print("Skipping framework numeric verification for PyTorch due to error:", e)
    elif framework == 'keras':
        try:
            import tensorflow as tf
            print("Loading Keras model for numeric comparison...")
            model = tf.keras.models.load_model(model_path)
            if apply_softmax:
                import tensorflow as tf
                soft_outs = tf.nn.softmax(model(tf.constant(x)), axis=-1).numpy()
                fw_out = soft_outs
            else:
                fw_out = model.predict(x)
            if fw_out.shape != onnx_out.shape:
                print("Shape mismatch between framework and ONNX outputs. Skipping numeric compare.")
                return
            maxdiff = float(np.max(np.abs(fw_out - onnx_out)))
            print(f"Max abs diff between Keras and ONNX outputs: {maxdiff:.6f}")
            if maxdiff <= atol:
                print("Verification PASSED (within tolerance).")
            else:
                print("Verification WARNING: difference exceeds tolerance.")
        except Exception as e:
            print("Skipping framework numeric verification for Keras due to error:", e)
    else:
        print("Framework not recognized for numeric verification. Only ONNX runtime check completed.")

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch or Keras/TensorFlow sequence model to ONNX and verify it.")
    parser.add_argument("--framework", required=True, choices=["pytorch", "keras"], help="Framework of the source model")
    parser.add_argument("--model-path", required=True, help="Path to the saved model (PyTorch .pt/.pth or Keras .h5 or SavedModel dir)")
    parser.add_argument("--output", required=True, help="Output ONNX file path")
    parser.add_argument("--seq-len", type=int, default=30, help="Sequence length the model expects (frames)")
    parser.add_argument("--feature-dim", type=int, default=128*128*3, help="Feature dimension per frame")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version to export with")
    parser.add_argument("--apply-softmax", action="store_true", help="Apply Softmax in exported model (if original model output logits)")
    parser.add_argument("--dynamic", action="store_true", help="Allow dynamic batch & sequence dimensions in exported ONNX")
    parser.add_argument("--verify", action="store_true", help="Run ONNX Runtime verification (and best-effort numeric compare)")
    args = parser.parse_args()

    if args.framework == "pytorch":
        export_pytorch(args.model_path, args.output, args.seq_len, args.feature_dim, args.opset, args.apply_softmax, args.dynamic)
    else:
        export_keras(args.model_path, args.output, args.seq_len, args.feature_dim, args.opset, args.apply_softmax, args.dynamic)

    if args.verify:
        verify_onnx(args.output, args.framework, args.model_path, args.seq_len, args.feature_dim, args.apply_softmax, args.dynamic)

if __name__ == "__main__":
    main()