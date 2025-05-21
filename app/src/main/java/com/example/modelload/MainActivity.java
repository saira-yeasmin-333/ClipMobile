package com.example.modelload;

import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;
import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";
    private Module imageEncoder;
    private Module textEncoder;
    private EditText queryEditText;
    private Button submitButton;
    private TextView resultTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        Log.d(TAG,"successed in launch");
        
        // Initialize UI elements
        queryEditText = findViewById(R.id.queryEditText);
        submitButton = findViewById(R.id.submitButton);
        resultTextView = findViewById(R.id.resultTextView);

        // Load models
        try {
            imageEncoder = Module.load(assetFilePath(this, "clip_image_encoder.pt"));
            textEncoder = Module.load(assetFilePath(this,"clip_text_encoder_android_v2.pt"));
            Log.d(TAG,"model loaded successfully");

            // Set up button click listener
            submitButton.setOnClickListener(v -> processQuery());

        } catch (Exception e) {
            Log.e(TAG, "Error loading models", e);
            String errorMessage = "Failed to load model: " + e.getMessage();
            if (e.getMessage().contains("scaled_dot_product_attention")) {
                errorMessage = "This model version is not compatible with the current PyTorch Mobile version. Please use a model that doesn't use scaled_dot_product_attention operation.";
            }
            resultTextView.setText(errorMessage);
        }

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    private void processQuery() {
        String selectedQuery = queryEditText.getText().toString().trim().toLowerCase();
        
        // Validate input
        if (!selectedQuery.matches("cat|dog|flower|fruit|horse")) {
            resultTextView.setText("Please enter a valid query: cat, dog, flower, fruit, or horse");
            return;
        }

        try {
            // Load and process images
            android.graphics.Bitmap catBitmap = loadBitmapFromAssets("cat.jpg");
            android.graphics.Bitmap dogBitmap = loadBitmapFromAssets("dog.jpg");
            android.graphics.Bitmap flowerBitmap = loadBitmapFromAssets("flower.jpg");
            android.graphics.Bitmap fruitBitmap = loadBitmapFromAssets("fruit.jpg");
            android.graphics.Bitmap horseBitmap = loadBitmapFromAssets("horse.jpg");

            float[] catEmbedding = encodeImage(catBitmap);
            float[] dogEmbedding = encodeImage(dogBitmap);
            float[] flowerEmbedding = encodeImage(flowerBitmap);
            float[] fruitEmbedding = encodeImage(fruitBitmap);
            float[] horseEmbedding = encodeImage(horseBitmap);
            Log.d(TAG,"encoded all images");

            float[] textEmbedding = encodeText(selectedQuery);
            Log.d(TAG,"text embedding created for query: " + selectedQuery);

            float catScore = cosineSimilarity(catEmbedding, textEmbedding);
            float dogScore = cosineSimilarity(dogEmbedding, textEmbedding);
            float flowerScore = cosineSimilarity(flowerEmbedding, textEmbedding);
            float fruitScore = cosineSimilarity(fruitEmbedding, textEmbedding);
            float horseScore = cosineSimilarity(horseEmbedding, textEmbedding);

            String result = "Query: " + selectedQuery + "\n\n" +
                    "Cat similarity: " + String.format("%.4f", catScore) + "\n" +
                    "Dog similarity: " + String.format("%.4f", dogScore) + "\n" +
                    "Flower similarity: " + String.format("%.4f", flowerScore) + "\n" +
                    "Fruit similarity: " + String.format("%.4f", fruitScore) + "\n" +
                    "Horse similarity: " + String.format("%.4f", horseScore);
            
            resultTextView.setText(result);
            Log.d(TAG, result);
        } catch (Exception e) {
            Log.e(TAG, "Error processing query", e);
            resultTextView.setText("Error processing query: " + e.getMessage());
        }
    }

    private String assetFilePath(android.content.Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    // Method to encode image
    private float[] encodeImage(android.graphics.Bitmap bitmap) {
        // Resize to 224x224 as required by CLIP
        android.graphics.Bitmap resizedBitmap = android.graphics.Bitmap.createScaledBitmap(bitmap, 224, 224, true);

        org.pytorch.Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            resizedBitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        );
        
        IValue output = imageEncoder.forward(IValue.from(inputTensor));
        float[] features = output.toTensor().getDataAsFloatArray();
        
        // Normalize features
        float norm = 0;
        for (float f : features) {
            norm += f * f;
        }
        norm = (float) Math.sqrt(norm);
        for (int i = 0; i < features.length; i++) {
            features[i] /= norm;
        }
        
        return features;
    }

    // Method to encode text
    private float[] encodeText(String query) {
        // Hardcoded token values for each query
        long[] tokenValues;
        switch (query.toLowerCase()) {
            case "cat":
                tokenValues = new long[]{
                    49406, 2368, 49407, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                };
                break;
            case "dog":
                tokenValues = new long[]{
                    49406, 1929, 49407, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0
                };
                break;
            case "flower":
                tokenValues = new long[]{
                    49406, 4055, 49407, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0
                };
                break;
            case "fruit":
                tokenValues = new long[]{
                    49406, 5190, 49407, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0
                };
                break;
            case "horse":
                tokenValues = new long[]{
                        49406, 4558, 49407, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0
                };
                break;
            default:
                throw new IllegalArgumentException("Unknown query: " + query);
        }
        // Create tensor from the hardcoded tokens
        org.pytorch.Tensor inputTensor = org.pytorch.Tensor.fromBlob(
            tokenValues,
            new long[]{1, 77}
        );
        
        IValue output = textEncoder.forward(IValue.from(inputTensor));
        float[] features = output.toTensor().getDataAsFloatArray();
        
        // Normalize features
        float norm = 0;
        for (float f : features) {
            norm += f * f;
        }
        norm = (float) Math.sqrt(norm);
        for (int i = 0; i < features.length; i++) {
            features[i] /= norm;
        }
        
        return features;
    }

    // Method to compute cosine similarity
    private float cosineSimilarity(float[] a, float[] b) {
        float dotProduct = 0;
        float normA = 0;
        float normB = 0;
        
        for (int i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        
        return dotProduct / ((float) Math.sqrt(normA) * (float) Math.sqrt(normB));
    }

    // Helper to load Bitmap from assets
    private android.graphics.Bitmap loadBitmapFromAssets(String fileName) throws IOException {
        try (InputStream is = getAssets().open(fileName)) {
            return android.graphics.BitmapFactory.decodeStream(is);
        }
    }
}