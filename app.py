import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Import necessary libraries for the model
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

# Set up model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model.to(device)

feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set generation parameters
gen_kwargs = {
    "max_length": 16,
    "num_beams": 5,
    "temperature": 1.0,
}

# CSS to style the app
st.markdown(
    """
    <style>
    /* Remove white space above title */
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 1rem;
    }
    .main-container {
        background-color: #1e1e2f;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4e79a7;
        text-align: center;
        margin-bottom: 1rem;
    }
    .caption-result {
        font-size: 1.25rem;
        color: #dcdcdc;
        font-weight: bold;
        margin-top: 1rem;
    }
    img {
        border-radius: 10px;
        max-width: 300px;
        margin: 0 auto;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to generate captions
def generate_captions(image, num_captions=3):
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
    
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Ensure num_return_sequences ≤ num_beams
    output_ids = model.generate(
        pixel_values,
        num_beams=max(num_captions, gen_kwargs["num_beams"]),
        num_return_sequences=num_captions,
        max_length=gen_kwargs["max_length"],
    )

    # Decode captions
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [pred.strip() for pred in preds]

# App Layout
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("<h1 class='title'>Image Caption Generator</h1>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image, and the app will generate multiple descriptive captions for you.", 
    type=["jpg", "jpeg", "png"]
)

# If file uploaded, process it
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Render the image (smaller size)
    st.image(image, caption="Uploaded Image",use_container_width = False, width=300)

    # Get the number of captions from the user
    num_captions = st.slider("How many captions would you like to generate?", min_value=1, max_value=5, value=3)

    # Generate captions when button clicked
    if st.button("Generate Captions"):
        with st.spinner(""):
            captions = generate_captions(image, num_captions=num_captions)
        
        # Display the captions
        st.markdown("<div class='caption-result'>Generated Captions:</div>", unsafe_allow_html=True)
        for i, caption in enumerate(captions, 1):
            st.write(f"{i}. {caption}")

st.markdown("</div>", unsafe_allow_html=True)
