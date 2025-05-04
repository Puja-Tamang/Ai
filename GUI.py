import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import matplotlib.pyplot as plt
from torchvision import models
import timm
import torch.nn.functional as F

st.set_page_config(
    page_title="Image Classifier",
    page_icon="🇳🇵",
    layout="wide",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

# Load trained model
@st.cache_resource
def load_model():
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=10)
    # If you have a custom-trained model, load the weights here
    # Replace this path with the actual path to your model weights file
    MODEL_PATH = 'best_model.pth'

    # Load the model weights (if you have them saved locally)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    return model

model = load_model()


# Move the model to the appropriate device (GPU/CPU)
model.to(device)
model.eval()

# Class labels
idx_to_class = {
    0: 'tench',
    1: 'English springer',
    2: 'cassette player',
    3: 'chain saw',
    4: 'church',
    5: 'French horn',
    6: 'garbage truck',
    7: 'gas pump',
    8: 'golf ball',
    9: 'parachute'
}


# Image transform to match the CIFAR-10 input requirements
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to preprocess the image and make predictions
def predict_image(image):
    # Preprocess the image
    image = transform(image).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)  # Get softmax probabilities
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item(), probabilities.squeeze()

# Streamlit GUI
st.title("Imagenette Image Classifier using EfficientNet-B0")
st.write(
    "Upload an image from Imagenette dataset, and the model will predict the class."
)

# Create 3 columns
col1, col2, col3 = st.columns([1, 1, 1])  # Equal width for each column

# **Column 1: Upload Image**
with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# **Column 2: Display Image**
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Ensure RGB format
    if image.mode != "RGB":
        image = image.convert("RGB")

    predict_clicked = False  # Flag to track button click

    with col2:
        st.header("Preview")
        new_width = 400
        new_height = 300
        # Resize the image using PIL
        resized_image = image.resize((new_width, new_height))
        
        # Display the resized image
        st.image(resized_image, caption="Uploaded Image", use_column_width=False)
        #st.image(image, caption="Uploaded Image", use_column_width=True)
        # Predict Button
        if st.button("Predict"):
            predict_clicked = True  # Update flag when button is clicked

    # **Column 3: Prediction and Confidence Chart**
    with col3:
        if uploaded_file is not None and predict_clicked:
            preds, probs = predict_image(image)
            # Display predicted class
            predicted_class_name = idx_to_class[preds]
            st.markdown(f"<h3 style='color: #4CAF50;'>Predicted: {predicted_class_name}</h3>", unsafe_allow_html=True)

            # Create confidence bar chart
            fig, ax = plt.subplots()
            ax.bar(list(idx_to_class.values()), probs.tolist(), color="skyblue")
            ax.set_xlabel("Classes")
            ax.set_ylabel("Confidence")
            ax.set_xticklabels(labels=list(idx_to_class.values()), rotation=45)
            st.pyplot(fig)



# Run the Streamlit app with the command:
# streamlit run app.py
