
# from flask import Flask, request, jsonify
# from PIL import Image
# import cv2
# import numpy as np
# import joblib
# from skimage.feature import hog
# from skimage import feature
# import logging
# import google.generativeai as genai
# import os

# genai.configure(api_key=os.environ["GENAI_API_KEY"])

# app = Flask(__name__)

# # Setup logging
# logging.basicConfig(level=logging.INFO)

# # Load the trained RandomForestClassifier model   
# model = joblib.load('plant.joblib')

# # Function to extract features (should be identical to training)
# def extract_features(image):
#     try:
#         image = cv2.resize(image, (128, 128))

#         # Extract color histogram features
#         hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#         hist = cv2.normalize(hist, hist).flatten()

#         # Extract HOG features
#         hog_features, _ = hog(image, orientations=8, pixels_per_cell=(16, 16),
#                               cells_per_block=(1, 1), visualize=True, channel_axis=-1)
        
#         # Extract LBP features
#         lbp_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         lbp = feature.local_binary_pattern(lbp_image, P=8, R=1, method='uniform')
#         (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
#         lbp_hist = lbp_hist.astype("float")
#         lbp_hist /= lbp_hist.sum()

#         # Combine all features into a single feature vector
#         combined_features = np.hstack([hist, hog_features, lbp_hist])

#         return combined_features
#     except Exception as e:
#         app.logger.error(f"Error extracting features: {e}")
#         raise

# @app.route('/predict', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         app.logger.error("No file part in the request")
#         return jsonify({"error": "No file found"}), 400

#     image_file = request.files['file']
#     app.logger.info(f"Received file: {image_file.filename}")

#     try:
#         # Load and process the image
#         image = Image.open(image_file)
#         image = np.array(image)

#         # Extract features from the image
#         processed_features = extract_features(image).reshape(1, -1)

#         # Predict the class using the trained model
#         prediction = model.predict(processed_features)

#         # Map the prediction to the class label
#         class_labels = {
#             0: 'apple scab',
#             1: 'apple black rot',
#             2: 'apple cedar apple rust',
#             3: 'apple healthy',
#             4: 'blueberry healthy',
#             5: 'cherry powdery mildew',
#             6: 'cherry healthy',
#             7: 'corn cercospora leaf spot gray leaf spot',
#             8: 'corn common rust',
#             9: 'corn northern leaf blight',
#             10: 'corn healthy',
#             11: 'grape black rot',
#             12: 'grape esca (black measles)',
#             13: 'grape leaf blight (isariopsis leaf spot)',
#             14: 'grape healthy',
#             15: 'orange haunglongbing (citrus greening)',
#             16: 'peach bacterial spot',
#             17: 'peach healthy'
#         }

#         predicted_class = class_labels.get(prediction[0], "Unknown")
#         prompt = f"{predicted_class} overcome measures "
#         model = genai.GenerativeModel("gemini-1.5-flash")
#         response = model.generate_content(prompt)
#         print(response.text)
#         return jsonify({"prediction":response }), 200

#     except Exception as e:
#         app.logger.error(f"Prediction error: {e}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)




# from flask import Flask, request, jsonify
# from PIL import Image
# import cv2
# import numpy as np
# import joblib
# from skimage.feature import hog
# from skimage import feature
# import logging
# import google.generativeai as genai
# import os

# # Configure the Generative AI API
# genai.configure(api_key=os.environ["GENAI_API_KEY"])

# app = Flask(__name__)

# # Setup logging
# logging.basicConfig(level=logging.INFO)

# # Load the trained RandomForestClassifier model   
# model = joblib.load('plant.joblib')

# # Initialize the Generative Model
# generative_model = genai.GenerativeModel("gemini-1.5-flash")

# # Function to extract features (should be identical to training)
# def extract_features(image):
#     try:
#         image = cv2.resize(image, (128, 128))

#         # Extract color histogram features
#         hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#         hist = cv2.normalize(hist, hist).flatten()

#         # Extract HOG features
#         hog_features, _ = hog(image, orientations=8, pixels_per_cell=(16, 16),
#                               cells_per_block=(1, 1), visualize=True, channel_axis=-1)
        
#         # Extract LBP features
#         lbp_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         lbp = feature.local_binary_pattern(lbp_image, P=8, R=1, method='uniform')
#         (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
#         lbp_hist = lbp_hist.astype("float")
#         lbp_hist /= lbp_hist.sum()

#         # Combine all features into a single feature vector AIzaSyCg4uPLTTi1yKUjnIhLvwtwXmZvw2kky50
#         combined_features = np.hstack([hist, hog_features, lbp_hist])

#         return combined_features
#     except Exception as e:
#         app.logger.error(f"Error extracting features: {e}")
#         raise

# @app.route('/predict', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         app.logger.error("No file part in the request")
#         return jsonify({"error": "No file found"}), 400

#     image_file = request.files['file']
#     app.logger.info(f"Received file: {image_file.filename}")

#     try:
#         # Load and process the image
#         image = Image.open(image_file)
#         image = np.array(image)

#         # Extract features from the image
#         processed_features = extract_features(image).reshape(1, -1)

#         # Predict the class using the trained model
#         prediction = model.predict(processed_features)

#         # Map the prediction to the class label
#         class_labels = {
#             0: 'apple scab',
#             1: 'apple black rot',
#             2: 'apple cedar apple rust',
#             3: 'apple healthy',
#             4: 'blueberry healthy',
#             5: 'cherry powdery mildew',
#             6: 'cherry healthy',
#             7: 'corn cercospora leaf spot gray leaf spot',
#             8: 'corn common rust',
#             9: 'corn northern leaf blight',
#             10: 'corn healthy',
#             11: 'grape black rot',
#             12: 'grape esca (black measles)',
#             13: 'grape leaf blight (isariopsis leaf spot)',
#             14: 'grape healthy',
#             15: 'orange haunglongbing (citrus greening)',
#             16: 'peach bacterial spot',
#             17: 'peach healthy'
#         }

#         predicted_class = class_labels.get(prediction[0], "Unknown")
#         prompt = f"{predicted_class} overcome measures "
#         response = generative_model.generate_content(prompt)
#         response_text = response.text if hasattr(response, 'text') else str(response)
#         print(response_text)
#         return jsonify({"prediction": response_text}), 200


#     except Exception as e:
#         app.logger.error(f"Prediction error: {e}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)




# import tensorflow as tf
# from flask import Flask, request, jsonify
# from PIL import Image
# import cv2
# import numpy as np
# import joblib
# from skimage.feature import hog
# from skimage import feature
# import logging
# import google.generativeai as genai
# import os

# # Configure the Generative AI API
# genai.configure(api_key=os.environ["GENAI_API_KEY"])

# app = Flask(__name__)

# # Setup logging
# logging.basicConfig(level=logging.INFO)

# # Load the trained RandomForestClassifier model   
# model =  tf.keras.models.load_model('cnn_model.h5')

# # Initialize the Generative Model
# generative_model = genai.GenerativeModel("gemini-1.5-flash")

# # Function to extract features (should be identical to training)
# def extract_features(image):
#     try:
#         image = cv2.resize(image, (128, 128))

#         # Extract color histogram features
#         hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#         hist = cv2.normalize(hist, hist).flatten()

#         # Extract HOG features
#         hog_features, _ = hog(image, orientations=8, pixels_per_cell=(16, 16),
#                               cells_per_block=(1, 1), visualize=True, channel_axis=-1)
        
#         # Extract LBP features
#         lbp_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         lbp = feature.local_binary_pattern(lbp_image, P=8, R=1, method='uniform')
#         (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
#         lbp_hist = lbp_hist.astype("float")
#         lbp_hist /= lbp_hist.sum()

#         # Combine all features into a single feature vector AIzaSyCg4uPLTTi1yKUjnIhLvwtwXmZvw2kky50
#         combined_features = np.hstack([hist, hog_features, lbp_hist])

#         return combined_features
#     except Exception as e:
#         app.logger.error(f"Error extracting features: {e}")
#         raise
# @app.route('/predict', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         app.logger.error("No file part in the request")
#         return jsonify({"error": "No file found"}), 400

#     image_file = request.files['file']
#     app.logger.info(f"Received file: {image_file.filename}")

#     try:
#         # Load and process the image
#         image = Image.open(image_file).convert('RGB')  # Ensure image is in RGB mode
#         image = image.resize((224, 224))  # Resize to the expected input size
#         image_array = np.array(image) / 255.0  # Normalize the image to [0, 1]
#         image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

#         # Predict the class using the trained model
#         prediction = model.predict(image_array)

#         # Map the prediction to the class label
#         predicted_class = np.argmax(prediction, axis=1)[0]
#         class_labels = {
#     0: 'Pepper bell Bacterial spot',
#     1: 'Pepper bell healthy',
#     2: 'Potato Early blight',
#     3: 'Potato healthy',
#     4: 'Potato Late blight',
#     5: 'Tomato Bacterial spot',
#     6: 'Tomato Early blight',
#     7: 'Tomato healthy',
#     8: 'Tomato Late blight',
#     9: 'Tomato Leaf Mold',
#     10: 'Tomato Septoria leaf spot',
#     11: 'Tomato Spider mites Two spotted spider mite',
#     12: 'Tomato Target Spot',
#     13: 'Tomato Tomato mosaic virus',
#     14: 'Tomato Tomato YellowLeaf Curl Virus'
# }


#         predicted_label = class_labels.get(predicted_class, "Unknown")
#         prompt = f"{predicted_label} overcome measures "
#         response = generative_model.generate_content(prompt)
#         response_text = response.text if hasattr(response, 'text') else str(response)
#         print(response_text)
#         return jsonify({"prediction": response_text}), 200

#     except Exception as e:
#         app.logger.error(f"Prediction error: {e}")
#         return jsonify({"error": str(e)}), 500


# if __name__ == '__main__':
#     app.run(debug=True, port=5000)


import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import cv2
import numpy as np
import logging
import google.generativeai as genai
import os


genai.configure(api_key=os.environ["GENAI_API_KEY"])

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

model = tf.keras.models.load_model('C:/jaff/model/cnn_model.keras')

# Initialize the Generative Model
generative_model = genai.GenerativeModel("gemini-1.5-flash")

# Class labels based on your model's training
class_labels = {
    0: 'Apple Apple scab',
    1: 'Apple Black rot',
    2: 'Apple Cedar apple rust',
    3: 'Apple healthy',
    4: 'Blueberry healthy',
    5: 'Cherry (including sour) healthy',
    6: 'Cherry (including sour) Powdery mildew',
    7: 'Corn (maize) Cercospora leaf spot Gray leaf spot',
    8: 'Corn (maize) Common rust',
    9: 'Corn (maize) healthy',
    10: 'Corn (maize) Northern Leaf Blight',
    11: 'Grape Black rot',
    12: 'Grape Esca (Black Measles)',
    13: 'Grape healthy',
    14: 'Grape Leaf blight (Isariopsis Leaf Spot)',
    15: 'Orange Haunglongbing (Citrus greening)',
    16: 'Peach Bacterial spot',
    17: 'Peach healthy',
    18: 'Pepper bell Bacterial spot',
    19: 'Pepper bell healthy',
    20: 'Potato Early blight',
    21: 'Potato healthy',
    22: 'Potato Late blight',
    23: 'Raspberry healthy',
    24: 'Soybean healthy',
    25: 'Squash Powdery mildew',
    26: 'Strawberry healthy',
    27: 'Strawberry Leaf scorch',
    28: 'Tomato Bacterial spot',
    29: 'Tomato Early blight',
    30: 'Tomato healthy',
    31: 'Tomato Late blight',
    32: 'Tomato Leaf Mold',
    33: 'Tomato Septoria leaf spot',
    34: 'Tomato Spider mites Two-spotted spider mite',
    35: 'Tomato Target Spot',
    36: 'Tomato Tomato mosaic virus',
    37: 'Tomato Tomato Yellow Leaf Curl Virus'
}


@app.route('/predict', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        app.logger.error("No file part in the request")
        return jsonify({"error": "No file found"}), 400

    image_file = request.files['file']
    app.logger.info(f"Received file: {image_file.filename}")

    try:
        # Load and process the image
        image = Image.open(image_file).convert('RGB')  # Ensure image is in RGB mode
        image = image.resize((256, 256))  # Resize to the expected input size (256x256)
        image_array = np.array(image) / 255.0  # Normalize the image to [0, 1]
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict the class using the trained model
        prediction = model.predict(image_array)

        # Map the prediction to the class label
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_labels.get(predicted_class, "Unknown")

        # Generate a response using the generative model
        prompt = f"{predicted_label} overcome measures and prevention measures. give only the content."
        print(prompt)
        response = generative_model.generate_content(prompt)
        response_text = response.text if hasattr(response, 'text') else str(response)

        app.logger.info(f"Prediction: {predicted_label}, Response: {response_text}")
        return jsonify({"prediction": response_text}), 200

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
