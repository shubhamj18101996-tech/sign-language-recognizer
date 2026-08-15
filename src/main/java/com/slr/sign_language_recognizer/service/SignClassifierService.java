package com.slr.sign_language_recognizer.service;

import ai.onnxruntime.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.nio.FloatBuffer;
import java.util.*;

@Service
public class SignClassifierService {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final List<String> labels;
    private final int sequenceLength;
    private final int featureDim;

    public SignClassifierService(OrtEnvironment env,
                                 OrtSession session,
                                 @Value("${model.labels}") String labelsCsv,
                                 @Value("${model.sequence.length}") int sequenceLength,
                                 @Value("${model.feature.dim}") int featureDim) {
        this.env = env;
        this.session = session;
        this.labels = parseLabels(labelsCsv);
        this.sequenceLength = sequenceLength;
        this.featureDim = featureDim;
    }

    private List<String> parseLabels(String csv) {
        String[] arr = csv.split(",");
        List<String> list = new ArrayList<>();
        for (String s : arr) list.add(s.trim());
        return list;
    }

    /**
     * Accepts a sequence of frame-level feature vectors (length must be >= sequenceLength).
     * Uses the last sequenceLength frames for prediction.
     */
    public String predictFromFrames(List<float[]> bufferedFrames) throws OrtException {
        if (bufferedFrames.size() < sequenceLength) {
            return "Buffering frames (" + bufferedFrames.size() + "/" + sequenceLength + ")";
        }

        // Use last sequenceLength frames
        int start = bufferedFrames.size() - sequenceLength;
        long[] shape = new long[] {1, sequenceLength, featureDim};
        float[] flat = new float[sequenceLength * featureDim];

        for (int i = 0; i < sequenceLength; i++) {
            float[] frame = bufferedFrames.get(start + i);
            if (frame.length != featureDim) {
                throw new IllegalArgumentException("Frame feature length mismatch: expected " + featureDim + " got " + frame.length);
            }
            System.arraycopy(frame, 0, flat, i * featureDim, featureDim);
        }

        FloatBuffer fb = FloatBuffer.wrap(flat);
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, fb, shape)) {
            String inputName = session.getInputNames().iterator().next();
            try (OrtSession.Result results = session.run(Collections.singletonMap(inputName, tensor))) {
                // assume model outputs [1, num_classes] probabilities or logits
                Object raw = results.get(0).getValue();
                float[] scores;
                if (raw instanceof float[][]) {
                    scores = ((float[][]) raw)[0];
                } else if (raw instanceof float[]) {
                    scores = (float[]) raw;
                } else {
                    throw new RuntimeException("Unsupported ONNX output type: " + raw.getClass());
                }

                int best = argmax(scores);
                float confidence = scores[best];
                String label = best < labels.size() ? labels.get(best) : "class_" + best;
                return String.format("%s (%.0f%%)", label, confidence * 100f);
            }
        }
    }

    private int argmax(float[] arr) {
        int idx = 0;
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > arr[idx]) idx = i;
        }
        return idx;
    }
}
