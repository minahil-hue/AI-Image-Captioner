import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
import torch

# Page configuration
st.set_page_config(
    page_title="Real-Time Image Captioning",
    page_icon="📸",
    layout="wide"
)

# Title and description
st.title("📸 Real-Time Image Captioning")
st.markdown("""
Generate descriptive captions for your images using AI! 
Choose from three input methods below.
""")

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_model():
    """Load the BLIP model and processor"""
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_caption(image, processor, model):
    """Generate caption for the given image"""
    try:
        # Convert image to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Process image and generate caption
        inputs = processor(image, return_tensors="pt")
        
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50)
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error generating caption: {str(e)}"

def load_image_from_url(url):
    """Load image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image, None
    except requests.exceptions.RequestException as e:
        return None, f"Error fetching image: {str(e)}"
    except Exception as e:
        return None, f"Error opening image: {str(e)}"

# Load model
with st.spinner("Loading AI model... This may take a moment on first run."):
    processor, model = load_model()

if processor is None or model is None:
    st.error("Failed to load the model. Please refresh the page or check your internet connection.")
    st.stop()

st.success("✅ Model loaded successfully!")

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📁 Upload Image", "🔗 Image URL", "📷 Camera Capture"])

# Tab 1: File Upload
with tab1:
    st.subheader("Upload an Image File")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="upload"
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                with st.spinner("Generating caption..."):
                    caption = generate_caption(image, processor, model)
                st.success("Caption Generated!")
                st.markdown(f"### 💬 Caption:")
                st.info(caption)
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")

# Tab 2: URL Input
with tab2:
    st.subheader("Enter Image URL")
    url_input = st.text_input(
        "Paste image URL here:",
        placeholder="https://example.com/image.jpg",
        key="url"
    )
    
    if st.button("Generate Caption from URL", key="url_button"):
        if url_input:
            with st.spinner("Fetching and processing image..."):
                image, error = load_image_from_url(url_input)
            
            if error:
                st.error(error)
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Image from URL", use_container_width=True)
                
                with col2:
                    with st.spinner("Generating caption..."):
                        caption = generate_caption(image, processor, model)
                    st.success("Caption Generated!")
                    st.markdown(f"### 💬 Caption:")
                    st.info(caption)
        else:
            st.warning("Please enter a valid URL.")

# Tab 3: Camera Capture
with tab3:
    st.subheader("Capture Photo with Camera")
    st.markdown("Click the button below to open your camera and take a photo.")
    
    camera_photo = st.camera_input("Take a picture", key="camera")
    
    if camera_photo is not None:
        try:
            image = Image.open(camera_photo)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Captured Photo", use_container_width=True)
            
            with col2:
                with st.spinner("Generating caption..."):
                    caption = generate_caption(image, processor, model)
                st.success("Caption Generated!")
                st.markdown(f"### 💬 Caption:")
                st.info(caption)
        except Exception as e:
            st.error(f"Error processing camera photo: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
**About this app:** This application uses the BLIP (Bootstrapping Language-Image Pre-training) model 
from Salesforce to generate descriptive captions for images. The model runs locally without requiring 
any API keys or external services.
""")
