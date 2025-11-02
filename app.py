import os
import sqlite3
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
from datetime import datetime

# --- Configuration ---
IMG_WIDTH, IMG_HEIGHT = 48, 48
TFLITE_MODEL_FILENAME = 'face_emotionModel.tflite' # Changed to TFLite model
DATABASE_FILENAME = 'database.db'
EMOTION_LABELS = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
    4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

# --- Initialization ---
app = Flask(__name__, template_folder='templates') # Explicitly set template folder for clarity

# Load the TFLite model using the Interpreter
try:
    # Initialize the TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_FILENAME)
    interpreter.allocate_tensors()
    print(f"Successfully loaded TFLite model: {TFLITE_MODEL_FILENAME}")
    
    # Get tensor details for later use
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

except Exception as e:
    print(f"Error loading TFLite model: {TFLITE_MODEL_FILENAME}. Ensure the file exists. Error: {e}")
    interpreter = None
    input_details = None
    output_details = None


# --- Database Functions ---

def init_db():
    """Initializes the SQLite database and the results table, including the new 'full_name' field."""
    conn = sqlite3.connect(DATABASE_FILENAME)
    c = conn.cursor()
    # Updated table structure to include full_name
    c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            full_name TEXT NOT NULL,
            emotion TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_result(user_id, full_name, emotion):
    """Saves a prediction result with user details to the database."""
    conn = sqlite3.connect(DATABASE_FILENAME)
    c = conn.cursor()
    c.execute("INSERT INTO results (user_id, full_name, emotion) VALUES (?, ?, ?)", 
              (user_id, full_name, emotion))
    conn.commit()
    conn.close()

# The get_all_results function is no longer strictly necessary since the HTML history sidebar was removed, 
# but we'll keep a stub for completeness if you ever want to add history back.
def get_all_results():
    """Retrieves all stored results from the database."""
    conn = sqlite3.connect(DATABASE_FILENAME)
    c = conn.cursor()
    c.execute("SELECT user_id, full_name, emotion, timestamp FROM results ORDER BY timestamp DESC")
    results = c.fetchall()
    conn.close()
    return results

# --- Prediction Function (UPDATED FOR TFLite) ---

def predict_emotion(image_path):
    """Preprocesses an image and uses the TFLite interpreter to predict the emotion."""
    global interpreter, input_details, output_details
    
    if interpreter is None:
        return "Model not loaded"

    try:
        # Open and convert to grayscale
        img = Image.open(image_path).convert('L')
        # Resize to 48x48
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        # Convert to numpy array and normalize
        face = np.array(img, 'float32') / 255.0
        
        # Reshape for model input: (1, 48, 48, 1). TFLite expects a float32 tensor.
        processed_img = face.reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)

        # 1. Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], processed_img)
        
        # 2. Run inference
        interpreter.invoke()
        
        # 3. Get the results (predictions is a numpy array)
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        # Determine the predicted class
        predicted_class = np.argmax(predictions[0])
        predicted_emotion = EMOTION_LABELS.get(predicted_class, "Unknown")
        
        return predicted_emotion

    except Exception as e:
        return f"Prediction Error: {e}"

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    # Initialize variables for template rendering
    full_name = "" 
    user_id = ""

    if request.method == 'POST':
        # Retrieve data from the POST form submission
        full_name = request.form.get('full_name', '')
        user_id = request.form.get('user_id', '')

        # Check if the post request has the file part
        if 'image' not in request.files:
            return redirect(url_for('index'))
        file = request.files['image']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return redirect(url_for('index'))

        if file:
            # Save the uploaded file temporarily
            temp_path = os.path.join('temp_image.png')
            file.save(temp_path)
            
            # Predict emotion
            prediction_result = predict_emotion(temp_path)
            
            # Save result to database (UPDATED to include full_name)
            save_result(user_id, full_name, prediction_result)
            
            # Clean up the temporary file
            os.remove(temp_path)

    # Render template, passing the user details and prediction
    return render_template('index.html', 
                            prediction=prediction_result, 
                            user_id=user_id,
                            full_name=full_name) # Passing full_name for template use

# --- Main Execution ---

if __name__ == '__main__':
    # Ensure the database is initialized before starting the app
    init_db()
    
    # Run the application
    app.run(debug=True, threaded=False)
