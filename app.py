import streamlit as st
from transformers import pipeline
from PIL import Image
import base64

# Function to add a background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        .dark-box {{
            background-color: rgba(0, 0, 0, 0.7);
            padding: 20px;
            border-radius: 10px;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# Set the path to your background image
background_image_path = 'download.jpg'  # Update this path

# Load the model from the Hugging Face hub
pipe = pipeline("image-classification", model="Abhiram4/PlantDiseaseDetectorSwinv2")

# Advice messages based on predictions
advice_messages = {
    "Tomato___Late_blight": "Use fungicides containing copper or mancozeb; practice crop rotation and remove infected plants promptly.",
    "Tomato___healthy": "Ensure adequate sunlight, water, and nutrients; use disease-resistant varieties and practice good garden hygiene.",
    "Grape___healthy": "Prune vines regularly, manage vineyard canopy for good airflow, and apply appropriate fungicides as needed.",
    "Orange___Haunglongbing_(Citrus_greening)": "Remove infected trees promptly, control psyllid vectors with insecticides, and plant disease-free nursery stock.",
    "Soybean___healthy": "Plant disease-resistant varieties, practice crop rotation, and use appropriate fungicides and insecticides.",
    "Squash___Powdery_mildew": "Apply sulfur or potassium bicarbonate fungicides, maintain good airflow around plants, and avoid overhead watering.",
    "Potato___healthy": "Rotate crops, plant certified disease-free seed potatoes, and apply fungicides as a preventive measure.",
    "Corn_(maize)___Northern_Leaf_Blight": "Rotate crops, plant resistant varieties, and use fungicides if necessary.",
    "Tomato___Early_blight": "Mulch around plants to reduce soil splash, water at the base of plants, and apply fungicides preventatively.",
    "Tomato___Septoria_leaf_spot": "Rotate crops, water at the base of plants, and apply fungicides early in the season.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Rotate crops, plant resistant varieties, and apply fungicides preventatively.",
    "Strawberry___Leaf_scorch": "Plant disease-resistant varieties, ensure good drainage, and remove infected plants promptly.",
    "Peach___healthy": "Prune trees to improve airflow, apply fungicides preventatively, and remove infected branches.",
    "Apple___Apple_scab": "Plant resistant varieties, prune to improve air circulation, and apply fungicides preventatively.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies with insecticides, plant virus-resistant varieties, and remove infected plants.",
    "Tomato___Bacterial_spot": "Rotate crops, avoid overhead watering, and apply copper-based fungicides.",
    "Apple___Black_rot": "Prune to improve air circulation, remove mummified fruit, and apply fungicides preventatively.",
    "Blueberry___healthy": "Plant disease-resistant varieties, maintain soil pH, and prune to improve air circulation.",
    "Cherry_(including_sour)___Powdery_mildew": "Apply sulfur or potassium bicarbonate fungicides, maintain good airflow around plants, and avoid overhead watering.",
    "Peach___Bacterial_spot": "Prune to improve airflow, apply copper-based fungicides, and remove infected branches.",
    "Apple___Cedar_apple_rust": "Plant resistant varieties, prune to improve air circulation, and apply fungicides preventatively.",
    "Tomato___Target_Spot": "Rotate crops, practice good garden hygiene, and apply fungicides preventatively.",
    "Pepper,_bell___healthy": "Ensure adequate spacing between plants, avoid overhead watering, and apply fungicides preventatively.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Prune to improve air circulation, remove infected leaves, and apply fungicides preventatively.",
    "Potato___Late_blight": "Rotate crops, plant certified disease-free seed potatoes, and apply fungicides preventatively.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants promptly, control aphid vectors, and plant resistant varieties.",
    "Strawberry___healthy": "Plant disease-resistant varieties, maintain good soil moisture, and remove infected plants.",
    "Apple___healthy": "Prune to improve air circulation, apply fungicides preventatively, and remove infected branches.",
    "Grape___Black_rot": "Prune infected parts promptly, apply fungicides before flowering, and ensure good air circulation around plants.",
    "Potato___Early_blight": "Rotate crops, plant resistant varieties, and apply fungicides preventatively.",
    "Cherry_(including_sour)___healthy": "Prune to improve air circulation, apply fungicides preventatively, and remove infected branches.",
    "Corn_(maize)___Common_rust_": "Plant resistant varieties, practice crop rotation, and apply fungicides preventatively.",
    "Grape___Esca_(Black_Measles)": "Prune vines to improve air circulation, remove infected wood, and apply appropriate fungicides.",
    "Raspberry___healthy": "Prune to improve air circulation, remove and destroy infected canes, and apply fungicides preventatively.",
    "Tomato___Leaf_Mold": "Provide good airflow around plants, avoid overhead watering, and apply fungicides preventatively.",
    "Tomato___Spider_mites_Two-spotted_spider_mite": "Use insecticidal soap or neem oil, maintain humidity levels, and remove heavily infested plants.",
    "Pepper,_bell___Bacterial_spot": "Rotate crops, avoid overhead watering, and apply copper-based fungicides preventatively.",
    "Corn_(maize)___healthy": "Plant disease-resistant varieties, practice crop rotation, and use appropriate cultural practices and treatments as needed.",
    "red_spot": "Apply fungicides such as mancozeb or chlorothalonil and ensure proper air circulation around plants.",
    "helopeltis": "Use insecticides like imidacloprid and practice regular pruning to remove infested parts.",
    "gray_blight": "Spray with fungicides containing copper or mancozeb, and remove and destroy infected plant debris.",
    "brown_blight": "Apply fungicides such as chlorothalonil and remove affected plant parts to prevent spread.",
    "algal_spot": "Use copper-based fungicides and improve air circulation and drainage around the plants."
}


# Streamlit app
def main():
    add_bg_from_local(background_image_path)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["Instructions", "About Dataset and Training", "Plant Disease Detector"])

    if page == "Instructions":
        instructions()
    elif page == "About Dataset and Training":
        about_dataset_and_training()
    else:
        plant_disease_detector()

def instructions():
    st.title("Instructions")
    st.write("""
    Welcome to the Plant Disease Detector app! Here are the steps to use the interface:
    
    1. **Upload an Image**: Click on the "Choose an image..." button to upload an image of a plant leaf.
    2. **View the Image**: The uploaded image will be displayed on the screen.
    3. **Classify the Image**: The app will automatically classify the uploaded image and provide top prediction and confidence score. Higher the score, shows that model is more confident. (Range is from 0 to 1)
    4. **Get Advice**: Based on the prediction, you will receive advice on how to handle the detected plant disease.
    """)

def about_dataset_and_training():
    st.title("About Dataset and Training")
    st.write("""
    This Plant Disease Detector app uses "microsoft/swinv2-tiny-patch4-window8-256" which is a pre-trained image classification model from Hugging Face, that was then finetuned on combined dataset of "New Plant Diseases" and "Tea_Leaf_Disease" from kaggle.

    ### Dataset
    The dataset used for training includes 92 thousand images of plant leaves labeled with different diseases categorized into 44 classes. Each image is annotated with the type of disease or if the plant is healthy.

    ### Training Details
    - **Model**: The model used is "microsoft/swinv2-tiny-patch4-window8-256k", a pre-trained image classification model from Hugging Face.
    - **Training**: The model was fine-tuned on the plant disease dataset to improve its accuracy in detecting specific plant diseases.
    - **Data Augmentation**: For the training dataset, a series of transformations were applied including resized, horizontal flipping, color jitter, and normalization. For validation and test datasets, resizing and normalization were applied to maintain consistency.
    - **Preprocessing**: Functions were defined to preprocess batches of training and validation images by applying the respective transformations. The dataset was split into training, validation, and test sets with an 80-10-10 split. Each subset was then transformed using the defined preprocessing functions to prepare the data for model training and evaluation.
    - **Training Process**: The model was trained for 3 epochs with a batch size of 64.
    - **Performance**: The highest training accuracy achieved was 99.7546% and the test accuracy was 99.72%.
    - **Evaluation**: The model's performance was evaluated using standard metrics such as accuracy, precision, and recall to ensure reliable predictions.
    """)

def plant_disease_detector():
    st.title("Plant Disease Detector")
    st.write("Upload an image of a plant leaf to detect its disease.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")

        # Convert the image to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Get predictions
        predictions = pipe(image)

        # Extract the top prediction
        if predictions:
            top_prediction = predictions[0]  # Top prediction is the first one in the list
            label = top_prediction['label']
            confidence = top_prediction['score']
            st.write(f"Top Prediction: {label}, Confidence: {confidence:.4f}")

            # Display advice message for top prediction
            formatted_label = label.replace("___", ": ").replace("_", " ")
            if label in advice_messages:
                st.write("Advice:")
                st.write(advice_messages[label])

if __name__ == '__main__':
    main()
