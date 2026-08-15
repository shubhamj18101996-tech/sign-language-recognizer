package com.slr.sign_language_recognizer.service;

import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Very small feature extractor that:
 *  - reads video file (MultipartFile) with FFmpegFrameGrabber
 *  - for each grabbed frame, resizes to 128x128, converts to RGB, and returns a normalized float[] per frame
 *
 * NOTE: For production or better accuracy, replace pixel-based features with MediaPipe keypoints (hands/pose/face).
 */
@Component
public class FeatureExtractor {

    private static final int WIDTH = 128;
    private static final int HEIGHT = 128;
    private static final int CHANNELS = 3;

    /**
     * Reads up to maxFrames frames from the uploaded video and returns a list of normalized feature arrays.
     * Each float[] length = WIDTH * HEIGHT * CHANNELS
     */
    public List<float[]> extractFeaturesFromVideo(MultipartFile file, int maxFrames) throws Exception {
        File tmp = File.createTempFile("upload-", ".mp4");
        try (FileOutputStream fos = new FileOutputStream(tmp)) {
            fos.write(file.getBytes());
        }

        List<float[]> frames = new ArrayList<>();
        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(tmp);
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        try {
            grabber.start();
            Frame frame;
            int count = 0;
            while ((frame = grabber.grabImage()) != null && count < maxFrames) {
                Mat mat = converter.convert(frame);
                if (mat == null) continue;

                Mat resized = new Mat();
                opencv_imgproc.resize(mat, resized, new org.bytedeco.opencv.opencv_core.Size(WIDTH, HEIGHT));
                Mat rgb = new Mat();
                opencv_imgproc.cvtColor(resized, rgb, opencv_imgproc.COLOR_BGR2RGB);

                int total = WIDTH * HEIGHT * CHANNELS;
                byte[] buffer = new byte[total];
                rgb.data().get(buffer);

                float[] features = new float[total];
                for (int i = 0; i < total; i++) {
                    features[i] = (buffer[i] & 0xFF) / 255.0f;
                }

                frames.add(features);

                // release mats
                resized.close();
                rgb.close();
                mat.close();

                count++;
            }
        } finally {
            grabber.stop();
            // delete temp file
            tmp.delete();
        }

        return frames;
    }
}
