import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


# 1. Define the model architecture
# This must match the class in your notebook exactly for the weights to load
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()

        # Layer 1: 3 input channels (RGB) -> 32 output channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        # Layer 2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Layer 3: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fully Connected Layers
        # Image reduces from 96 -> 48 -> 24 -> 12
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 12 * 12, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# 2. Page Config
st.title("Model: Cancer Detection Tool")
st.write("Upload a tissue scan to check for metastasis.")


# 3. Load the model
@st.cache_resource
def load_model():
    model = MyCNN()
    try:
        # Load the file we saved in the notebook
        model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        return None


model = load_model()

if model is None:
    st.error("Error: 'model.pth' not found. Please run your notebook to train the model first.")

# 4. File Uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "tif"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Scan', width=300)

    # Analyze Button
    if st.button("Analyze Image"):
        # Preprocess the image (resize to 96x96 and normalize)
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            score = output.item()

        # Display Result
        st.write("---")
        st.subheader("Result:")

        if score > 0.5:
            st.error("⚠️ TUMOR DETECTED")
            st.write(f"Confidence: {score * 100:.2f}%")
        else:
            st.success("✅ HEALTHY TISSUE")
            st.write(f"Confidence: {(1 - score) * 100:.2f}%")