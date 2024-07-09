# referenced from lightning.ai
import streamlit as st
import torch
# import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import numpy as np
from typing import Union, List
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model and preprocessing
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
model_name = "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)


def preprocess(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.Tensor:
    return processor(texts, max_length=context_length, padding=True, truncation=truncate, return_tensors="pt").input_ids.to(device)


# Function to predict descriptions and probabilities
def predict(image, descriptions):
    image = preprocess(image).unsqueeze(0).to(device)
    text = tokenize(descriptions).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return descriptions[np.argmax(probs)], np.max(probs)

# Streamlit app
def main():
    st.title("Two Lies and One Truth: Image Understanding Game")

    # Instructions for the user
    st.markdown("---")
    st.markdown("### Upload an image and let the model guess the truth!")

    # Upload image through Streamlit with a unique key
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"], key="uploaded_image")

    if uploaded_image is not None:
        # Convert the uploaded image to PIL Image
        pil_image = Image.open(uploaded_image)

        # Limit the height of the displayed image to 400px
        st.image(pil_image, caption="Uploaded Image.", use_column_width=True, width=200)
        
        # Instructions for the user
        st.markdown("### Two lies and one truth")
        st.markdown("Write 3 descriptions about the image, one must be true.")

        # Get user input for descriptions
        description1 = st.text_input("Description 1:", placeholder='A red apple')
        description2 = st.text_input("Description 2:", placeholder='A car parked in a garage')
        description3 = st.text_input("Description 3:", placeholder='An orange fruit on a tree')

        descriptions = [description1, description2, description3]

        # Button to trigger prediction
        if st.button("Predict"):
            if all(descriptions):
                # Make predictions
                best_description, best_prob = predict(pil_image, descriptions)

                # Display the highest probability description and its probability
                st.write(f"**Best Description:** {best_description}")
                st.write(f"**Prediction Probability:** {best_prob:.2%}")

                # Display progress bar for the highest probability
                st.progress(float(best_prob))

if __name__ == "__main__":
    main()
