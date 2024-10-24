import joblib
import numpy as np
import cv2
from flask import Flask, request, jsonify
from deepface import DeepFace

app = Flask(__name__)

# Load the SVM model and label encoder
svm_model = joblib.load("svm_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load the FaceNet model from DeepFace
facenet_model = DeepFace.build_model("Facenet")
print("Facenet model loaded successfully.")

# Function to generate embeddings from an image
def generate_embedding(image):
    embedding = DeepFace.represent(img_path=image, model_name="Facenet", detector_backend='mtcnn', enforce_detection=False)[0]['embedding']
    return np.array(embedding)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded frame from the request
        file = request.files['frame'].read()
        nparr = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        predictions = []
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]  # Crop the detected face
            embedding = generate_embedding(face)  # Generate embedding

            # Reshape the embedding and make a prediction
            embedding = embedding.reshape(1, -1)
            prediction = svm_model.predict(embedding)
            label = label_encoder.inverse_transform(prediction)
            predictions.append(label[0])

        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
