import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load pre-trained leaf classification model
leaf_classifier = tf.keras.models.load_model("trained_plant_disease_model.h5")

# Function to check if the uploaded image is a leaf
def is_leaf(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0  # Normalize image
    image = np.expand_dims(image, axis=0)
    prediction = leaf_classifier.predict(image)
    if prediction[0][0] > 0.5:  # Assuming 0.5 as the threshold for leaf detection
        return True
    else:
        return False

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

#Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE PREDICTION AND PESTICIDE RECOMMENDATION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition and pesticide recommendation System! 🌿🔍
    
    This application helps in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases and also recommends pesticides

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    """)

#About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (64368 images)
                2. test (33 images)
                3. validation (16098 images)

                """)

#Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image is not None:
        img = Image.open(test_image)
        st.image(img, width=400, use_column_width=True)
        
        # Check if the uploaded image is a leaf
        if not is_leaf(img):
            st.warning("Please upload only leaf images.")
        
        if st.button("Predict"):
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            
            # Reading Labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                          'Tomato___healthy']
            
            pesticide_recommendations = {
                'Apple___Apple_scab': 'Copper-based fungicides, Sulfur-based fungicides',
                'Apple___Black_rot': 'Captan, Fenarimol',
                'Apple___Cedar_apple_rust': 'Chlorothalonil, Myclobutanil',
                'Apple___healthy': 'No specific pesticide recommendation',
                'Blueberry___healthy': 'No specific pesticide recommendation',
                'Cherry_(including_sour)___Powdery_mildew': 'Sulfur-based fungicides, Potassium bicarbonate',
                'Cherry_(including_sour)___healthy': 'No specific pesticide recommendation',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Chlorothalonil, Azoxystrobin',
                'Corn_(maize)___Common_rust_': 'Fungicides containing azoxystrobin, Fungicides containing propiconazole',
                'Corn_(maize)___Northern_Leaf_Blight': 'Chlorothalonil, Azoxystrobin',
                'Corn_(maize)___healthy': 'No specific pesticide recommendation',
                'Grape___Black_rot': 'Captan, Mancozeb',
                'Grape___Esca_(Black_Measles)': 'Chlorothalonil, Thiophanate-methyl',
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Mancozeb, Myclobutanil',
                'Grape___healthy': 'No specific pesticide recommendation',
                'Orange___Haunglongbing_(Citrus_greening)': 'No cure available, Manage through cultural practices',
                'Peach___Bacterial_spot': 'Copper-based bactericides, Streptomycin',
                'Peach___healthy': 'No specific pesticide recommendation',
                'Pepper,_bell___Bacterial_spot': 'Copper-based bactericides, Streptomycin',
                'Pepper,_bell___healthy': 'No specific pesticide recommendation',
                'Potato___Early_blight': 'Chlorothalonil, Mancozeb',
                'Potato___Late_blight': 'Chlorothalonil, Mancozeb',
                'Potato___healthy': 'No specific pesticide recommendation',
                'Raspberry___healthy': 'No specific pesticide recommendation',
                'Soybean___healthy': 'No specific pesticide recommendation',
                'Squash___Powdery_mildew': 'Sulfur-based fungicides, Potassium bicarbonate',
                'Strawberry___Leaf_scorch': 'Copper-based bactericides, Streptomycin',
                'Strawberry___healthy': 'No specific pesticide recommendation',
                'Tomato___Bacterial_spot': 'Copper-based bactericides, Streptomycin',
                'Tomato___Early_blight': 'Chlorothalonil, Mancozeb',
                'Tomato___Late_blight': 'Chlorothalonil, Mancozeb',
                'Tomato___Leaf_Mold': 'Chlorothalonil, Mancozeb',
                'Tomato___Septoria_leaf_spot': 'Chlorothalonil, Mancozeb',
                'Tomato___Spider_mites Two-spotted_spider_mite': 'Insecticidal soap, Neem oil',
                'Tomato___Target_Spot': 'Chlorothalonil, Mancozeb',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'No cure available, Manage through cultural practices',
                'Tomato___Tomato_mosaic_virus': 'No cure available, Manage through cultural practices',
                'Tomato___healthy': 'No specific pesticide recommendation'
            }
            
            prediction_result = class_name[result_index]
            if prediction_result in pesticide_recommendations:
                recommendation = pesticide_recommendations[prediction_result]
                st.success(f"Model predicts {prediction_result} and recommends {recommendation}.")
            else:
                st.success(f"Model predicts {prediction_result} but no specific pesticide recommendation available.")
