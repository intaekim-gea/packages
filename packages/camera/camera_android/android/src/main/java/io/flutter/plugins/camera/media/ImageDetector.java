package io.flutter.plugins.camera.media;

import android.graphics.Bitmap;
import android.media.Image;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.BackgroundSubtractorMOG2;
import org.opencv.video.Video;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

public class ImageDetector {
    static {
        System.loadLibrary("opencv_java4");
        System.loadLibrary("c++_shared");
    }

    private static final String TAG = "ImageDetector";

    private static final List<Mat> validFrames = new ArrayList<>();
    private static final int maxFrameCount = 10;
    private static final int captureInterval = 5;
    private static int frameCounter = 0;
    private static final long inactivityPeriod = 2000L;
    private static long lastActivityTime = System.currentTimeMillis();
    private static final BackgroundSubtractorMOG2 backgroundSubtractor = Video.createBackgroundSubtractorMOG2(500, 35.0, false);

    public static class ImageDetectResult {
        static final long clearUiList = 0;
        static final long imageForUi = 1;
        static final long imageToAnalyze = 2;

        public byte[] jpegBytes;
        public long activityType;

        public ImageDetectResult(byte[] jpegBytes, long activityType) {
            this.jpegBytes = jpegBytes;
            this.activityType = activityType;
        }
    }

    static public void resetAll() {
        clearValidFrames();
        frameCounter = 0;
        lastActivityTime = System.currentTimeMillis();
    }

    static private void clearValidFrames() {
        for (Mat frame : validFrames) {
            frame.release();
        }
        validFrames.clear();
    }

    static public ImageDetectResult processImage(Image image) {
        Mat matCurrent = getMatFromImage(image);

        // Apply background subtraction
        Mat foregroundMask = new Mat();
        backgroundSubtractor.apply(matCurrent, foregroundMask);

        // Apply noise reduction
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5.0, 5.0));
        Imgproc.morphologyEx(foregroundMask, foregroundMask, Imgproc.MORPH_OPEN, kernel);
        Imgproc.medianBlur(foregroundMask, foregroundMask, 7);

        // Find contours to detect movement
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(foregroundMask, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        boolean significantMovementDetected = false;
        for (MatOfPoint contour : contours) {
            if (Imgproc.contourArea(contour) > 1200) {
                significantMovementDetected = true;
                break;
            }
        }

        if (significantMovementDetected && frameCounter % captureInterval == 0) {
            if (validFrames.size() >= maxFrameCount) {
                validFrames.remove(0).release();
                Log.d(TAG, "Removed oldest frame to maintain max size of " + maxFrameCount + ".");
            }
            validFrames.add(matCurrent.clone());
            Log.d(TAG, "Added new frame. Total frames: " + validFrames.size());
            lastActivityTime = System.currentTimeMillis();
            byte[] jpegBytes = updateConcatenatedImage();
            if (jpegBytes != null) {
                matCurrent.release();
                // For Displying the image in the Flutter app
                return new ImageDetectResult(jpegBytes, ImageDetectResult.imageForUi);
            }
        }

        if (!significantMovementDetected && System.currentTimeMillis() - lastActivityTime > inactivityPeriod) {
            if (validFrames.size() > 3) {
                Log.i(TAG, "Preparing concatenated image for API. with "+validFrames.size()+ " frames.");
                Mat concatenatedMat = concatenateFramesHorizontally(validFrames);
                byte[] jpegBytes = convertMatToJpeg(concatenatedMat);

                concatenatedMat.release();
                clearValidFrames();
                lastActivityTime = System.currentTimeMillis();

                matCurrent.release();
                // To send to the API
                return new ImageDetectResult(jpegBytes, ImageDetectResult.imageToAnalyze);
            }

            byte[] jpegBytes = convertMatToJpeg(matCurrent);
            matCurrent.release();
            clearValidFrames();
            lastActivityTime = System.currentTimeMillis();
            // to Clear UI Images
            return new ImageDetectResult(jpegBytes, ImageDetectResult.clearUiList);
        }

        matCurrent.release();
        kernel.release();
        foregroundMask.release();
        frameCounter++;
        return null;
    }

    public static Mat getMatFromImage(Image image) {
        Image.Plane[] planes = image.getPlanes();
        int width = image.getWidth();
        int height = image.getHeight();

        // Get the Y, U, and V planes
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        // Calculate the size of the Y, U, and V planes
        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        // Create a byte array to hold the YUV data
        byte[] yuvBytes = new byte[ySize + uSize + vSize];

        // Copy the Y, U, and V planes into the byte array
        yBuffer.get(yuvBytes, 0, ySize);
        uBuffer.get(yuvBytes, ySize, uSize);
        vBuffer.get(yuvBytes, ySize + uSize, vSize);

        // Create a Mat to hold the YUV data
        Mat yuvMat = new Mat(height + height / 2, width, CvType.CV_8UC1);
        yuvMat.put(0, 0, yuvBytes);

        // Convert the YUV Mat to RGB
        Mat rgbMat = new Mat();
        Imgproc.cvtColor(yuvMat, rgbMat, Imgproc.COLOR_YUV2RGB_NV21);

        yuvMat.release();
        return rgbMat;
    }

    private static byte[] updateConcatenatedImage() {
        if (validFrames.isEmpty()) return null;

        Log.i(TAG, "Updating concatenated image.");
        Mat concatenatedMat = concatenateFramesHorizontally(validFrames);
        Bitmap bitmap = Bitmap.createBitmap(concatenatedMat.cols(), concatenatedMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(concatenatedMat, bitmap);

        byte[] jpegBytes = convertMatToJpeg(concatenatedMat);
        concatenatedMat.release();
        return jpegBytes;
    }

    private static Mat concatenateFramesHorizontally(List<Mat> frames) {
        if (frames.isEmpty()) {
            throw new IllegalArgumentException("No frames to concatenate.");
        }
        Mat concatenatedMat = frames.get(0).clone();
        for (int i = 1; i < frames.size(); i++) {
            Mat tempMat = new Mat();
            List<Mat> mats = new ArrayList<>();
            mats.add(concatenatedMat);
            mats.add(frames.get(i));
            Core.hconcat(mats, tempMat);
            concatenatedMat.release();
            concatenatedMat = tempMat;
        }
        return concatenatedMat;
    }

    public static byte[] convertMatToJpeg(Mat mat) {
        MatOfByte matOfByte = new MatOfByte();
        Imgcodecs.imencode(".jpg", mat, matOfByte);
        return matOfByte.toArray();
    }
}
