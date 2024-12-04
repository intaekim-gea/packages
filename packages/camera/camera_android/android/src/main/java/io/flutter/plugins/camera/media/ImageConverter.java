package io.flutter.plugins.camera.media;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.Log;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

public class ImageConverter {

    public static byte[] convertToJpegBytes(Image image, int quality) throws IOException {
        int format = image.getFormat();
        if (format == ImageFormat.JPEG) {
            ByteBuffer buffer = image.getPlanes()[0].getBuffer();
            byte[] bytes = new byte[buffer.remaining()];
            buffer.get(bytes);
            return bytes;
        } else if (format == ImageFormat.YUV_420_888) {
            YuvImage yuvImage = toYuvImage(image);
            int width = image.getWidth();
            int height = image.getHeight();

            // Convert to jpeg
            try (ByteArrayOutputStream out = new ByteArrayOutputStream()) {
                yuvImage.compressToJpeg(new Rect(0, 0, width, height), quality, out);
                return out.toByteArray();
            }
        } else {
            throw new IllegalArgumentException("Unsupported image format: " + format);
        }
    }

    static YuvImage toYuvImage(Image image) {
        if (image.getFormat() != ImageFormat.YUV_420_888) {
            throw new IllegalArgumentException("Invalid image format");
        }

        int width = image.getWidth();
        int height = image.getHeight();

        // Order of U/V channel guaranteed, read more:
        // https://developer.android.com/reference/android/graphics/ImageFormat#YUV_420_888
        Image.Plane yPlane = image.getPlanes()[0];
        Image.Plane uPlane = image.getPlanes()[1];
        Image.Plane vPlane = image.getPlanes()[2];

        ByteBuffer yBuffer = yPlane.getBuffer();
        ByteBuffer uBuffer = uPlane.getBuffer();
        ByteBuffer vBuffer = vPlane.getBuffer();

        // Full size Y channel and quarter size U+V channels.
        int numPixels = (int) (width * height * 1.5f);
        byte[] nv21 = new byte[numPixels];
        int index = 0;

        // Copy Y channel.
        int yRowStride = yPlane.getRowStride();
        int yPixelStride = yPlane.getPixelStride();
        for(int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                nv21[index++] = yBuffer.get(y * yRowStride + x * yPixelStride);
            }
        }

        // Copy VU data; NV21 format is expected to have YYYYVU packaging.
        // The U/V planes are guaranteed to have the same row stride and pixel stride.
        int uvRowStride = uPlane.getRowStride();
        int uvPixelStride = uPlane.getPixelStride();
        int uvWidth = width / 2;
        int uvHeight = height / 2;

        for(int y = 0; y < uvHeight; ++y) {
            for (int x = 0; x < uvWidth; ++x) {
                int bufferIndex = (y * uvRowStride) + (x * uvPixelStride);
                // V channel.
                nv21[index++] = vBuffer.get(bufferIndex);
                // U channel.
                nv21[index++] = uBuffer.get(bufferIndex);
            }
        }
        return new YuvImage(
                nv21, ImageFormat.NV21, width, height, /* strides= */ null);
    }

    public static boolean isMostlyDarkImage(byte[] jpegBytes, double threshold) throws IOException {
        long startTime = System.currentTimeMillis();

        Bitmap bitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.length);
        if (bitmap == null) {
            throw new IllegalArgumentException("Invalid JPEG image");
        }

        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int totalPixels = width * height;
        double totalBrightness = 0;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pixel = bitmap.getPixel(x, y);
                int red = (pixel >> 16) & 0xFF;
                int green = (pixel >> 8) & 0xFF;
                int blue = pixel & 0xFF;
                double brightness = (0.299 * red + 0.587 * green + 0.114 * blue) / 255;
                totalBrightness += brightness;
            }
        }

        double averageBrightness = totalBrightness / totalPixels;
        Log.d("ImageConverter", "isMostlyBlackImage(" + averageBrightness + ") isDark: " + (averageBrightness >= threshold) + ", Execution time: " + (System.currentTimeMillis() - startTime) + " ms");
        return averageBrightness < threshold;
    }
}
