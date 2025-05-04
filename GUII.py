import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image

# -------------------- Setup --------------------
st.set_page_config(page_title="ImageNette Classifier", layout="centered")
st.title("🧠 ImageNette Image Classifier")
st.caption("Upload an image to classify it using a model based on EfficientNet_b0.")

# -------------------- Class Labels --------------------
class_names = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute"
]

# -------------------- Image Transform --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Standard for ImageNet
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------- Model Loader --------------------
@st.cache_resource
def load_model():
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=len(class_names))
    state_dict = torch.load('best_model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# -------------------- Prediction Function --------------------
def predict(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_class = torch.max(probs, dim=0)
    return class_names[top_class], top_prob.item()

# -------------------- UI: Upload + Predict --------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Classifying..."):
            label, confidence = predict(image)
        st.success(f"✅ Predicted: *{label}* ({confidence*100:.2f}% confidence)")