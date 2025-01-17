import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

# Load CLIP model and processor outside the main function
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def download_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# Function to predict descriptions and probabilities
def predict(image, descriptions):
    model, processor = download_clip_model()
    inputs = processor(text=descriptions, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    return descriptions[np.argmax(probs)], np.max(probs)

# Streamlit app
def main():
    st.title("Two Lies, One Truth - Can you fool the model?")

    # Instructions for the user
    st.markdown("---")
    st.markdown("### Upload an image to trick the model!")

    # Upload image through Streamlit with a unique key
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"], key="uploaded_image")

    if uploaded_image is not None:
        try:
            # Convert the uploaded image to PIL Image
            pil_image = Image.open(uploaded_image).convert("RGB")

            # Limit the height of the displayed image to 400px
            st.image(pil_image, caption="Uploaded Image.", use_column_width=True, width=200)

            # Instructions for the user
            st.markdown("### 2 Lies and 1 Truth")
            st.markdown("Write 3 descriptions about the image, 1 must be true.")

            # Get user input for descriptions
            description1 = st.text_input("Description 1:", placeholder='A red apple')
            description2 = st.text_input("Description 2:", placeholder='A car parked in a garage')
            description3 = st.text_input("Description 3:", placeholder='An orange fruit on a tree')

            descriptions = [description1, description2, description3]

            # Button to trigger prediction
            if st.button("Guess"):
                if all(descriptions):
                    # Make predictions
                    best_description, best_prob = predict(pil_image, descriptions)

                    # Display the highest probability description and its probability
                    st.write(f"**Best Description:** {best_description}")
                    st.write(f"**Prediction Probability:** {best_prob:.2%}")

                    # Display progress bar for the highest probability
                    st.progress(float(best_prob))

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
