package com.slr.sign_language_recognizer.controller;

import com.slr.sign_language_recognizer.service.FeatureExtractor;
import com.slr.sign_language_recognizer.service.SignClassifierService;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.LinkedList;
import java.util.List;

@RestController
@RequestMapping("/api")
public class VideoController {

    private final FeatureExtractor extractor;
    private final SignClassifierService classifier;
    private final int sequenceLength;
    private final int featureDim;

    public VideoController(FeatureExtractor extractor,
                           SignClassifierService classifier,
                           @Value("${model.sequence.length}") int sequenceLength,
                           @Value("${model.feature.dim}") int featureDim) {
        this.extractor = extractor;
        this.classifier = classifier;
        this.sequenceLength = sequenceLength;
        this.featureDim = featureDim;
    }

    /**
     * POST /api/predict
     * multipart/form-data: file=<video file>
     *
     * Returns JSON text result like "Hello (95%)"
     */
    @PostMapping("/predict")
    public ResponseEntity<?> predict(@RequestPart("file") MultipartFile file) {
        try {
            // Extract up to (sequenceLength * k) frames; here we take sequenceLength frames (can be adjusted)
            List<float[]> frames = extractor.extractFeaturesFromVideo(file, sequenceLength);

            // If fewer frames than sequenceLength, you may pad or reject; this example buffers until enough
            if (frames.size() < sequenceLength) {
                return ResponseEntity.badRequest().body("Video too short: need at least " + sequenceLength + " frames.");
            }

            // Call classifier
            String prediction = classifier.predictFromFrames(new LinkedList<>(frames));
            return ResponseEntity.ok().body(prediction);

        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.internalServerError().body("Error processing video: " + e.getMessage());
        }
    }
}
