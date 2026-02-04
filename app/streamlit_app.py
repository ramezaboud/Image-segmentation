"""
🧠 Brain Tumor Segmentation - Streamlit Web Application
========================================================
Interactive web app for brain tumor segmentation using deep learning.
Upload an MRI image and get instant segmentation results.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# =====================
# Page Configuration
# =====================
st.set_page_config(
    page_title="Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================
# Custom CSS Styling
# =====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-size: 1.1rem;
    }
    .stButton > button:hover {
        background-color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)

# =====================
# Constants
# =====================
IMG_SIZE = (256, 256)
MODEL_PATH = Path(__file__).parent.parent / "brain_segmentation_efficientnet_final.keras"

# Color map for segmentation visualization
COLORS = [
    [0, 0, 0],        # Background - Black
    [255, 0, 0],      # Class 1 - Red
    [0, 255, 0],      # Class 2 - Green
    [0, 0, 255],      # Class 3 - Blue
    [255, 255, 0],    # Class 4 - Yellow
    [255, 0, 255],    # Class 5 - Magenta
    [0, 255, 255],    # Class 6 - Cyan
]

CLASS_NAMES = [
    "Background",
    "Tumor Region 1",
    "Tumor Region 2", 
    "Tumor Region 3",
    "Tumor Region 4",
    "Tumor Region 5",
    "Tumor Region 6",
]

# =====================
# Model Loading
# =====================
@st.cache_resource
def load_model():
    """Load the trained segmentation model"""
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False  # We don't need to compile for inference
        )
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info(f"📁 Expected model path: {MODEL_PATH}")
        return None

# =====================
# Image Processing
# =====================
def preprocess_image(image):
    """Preprocess image for model input"""
    # Resize
    img_resized = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)
    
    # Apply EfficientNet preprocessing
    img_processed = tf.keras.applications.efficientnet.preprocess_input(
        img_resized.astype(np.float32)
    )
    
    return img_resized, img_processed

def create_colored_mask(mask, num_classes):
    """Convert segmentation mask to colored image"""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for c in range(min(num_classes, len(COLORS))):
        colored[mask == c] = COLORS[c]
    
    return colored

def create_overlay(image, mask, alpha=0.5):
    """Create overlay of segmentation on original image"""
    colored_mask = create_colored_mask(mask, len(COLORS))
    
    # Resize mask to match image if needed
    if colored_mask.shape[:2] != image.shape[:2]:
        colored_mask = cv2.resize(colored_mask, (image.shape[1], image.shape[0]))
    
    # Create overlay
    overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    
    return overlay

def calculate_statistics(mask, num_classes):
    """Calculate segmentation statistics"""
    total_pixels = mask.size
    stats = {}
    
    for c in range(num_classes):
        count = np.sum(mask == c)
        percentage = (count / total_pixels) * 100
        stats[c] = {
            'count': count,
            'percentage': percentage
        }
    
    return stats

# =====================
# Main Application
# =====================
def main():
    # Header
    st.markdown('<p class="main-header">🧠 Brain Tumor Segmentation</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an MRI image to detect and segment brain tumors using AI</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Overlay opacity
        overlay_alpha = st.slider(
            "Overlay Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Adjust the transparency of the segmentation overlay"
        )
        
        # Display options
        st.subheader("📊 Display Options")
        show_statistics = st.checkbox("Show Statistics", value=True)
        show_color_legend = st.checkbox("Show Color Legend", value=True)
        
        st.markdown("---")
        
        # Model info
        st.subheader("ℹ️ Model Info")
        st.info("""
        **Architecture:** U-Net  
        **Backbone:** EfficientNetB3  
        **Input Size:** 256×256  
        **Pretrained:** ImageNet
        """)
        
        st.markdown("---")
        
        # About
        st.subheader("📖 About")
        st.write("""
        This application uses deep learning to segment 
        brain tumors from MRI images. The model was 
        trained using transfer learning with 
        EfficientNetB3 backbone.
        """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.warning("⚠️ Model not loaded. Please ensure the model file exists.")
        st.stop()
    
    # Get number of classes from model
    num_classes = model.output_shape[-1]
    
    # File uploader
    st.markdown("### 📤 Upload MRI Image")
    
    uploaded_file = st.file_uploader(
        "Choose an MRI image...",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a brain MRI image in JPG, JPEG, PNG, or BMP format"
    )
    
    # Demo mode
    use_demo = st.checkbox("🎮 Use Demo Image", value=False)
    
    if uploaded_file is not None or use_demo:
        
        # Load image
        if use_demo:
            # Create a demo image (placeholder)
            st.info("📝 Demo mode: Using a placeholder image. Upload a real MRI for actual results.")
            image = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
        else:
            # Read uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        with st.spinner("🔄 Processing image..."):
            img_resized, img_processed = preprocess_image(image)
            
            # Predict
            prediction = model.predict(img_processed[np.newaxis, ...], verbose=0)
            mask = np.argmax(prediction[0], axis=-1)
            
            # Create overlay
            overlay = create_overlay(img_resized, mask, overlay_alpha)
            
            # Calculate statistics
            stats = calculate_statistics(mask, num_classes)
        
        # Display results
        st.markdown("### 🖼️ Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Original Image**")
            st.image(img_resized, use_container_width=True)
        
        with col2:
            st.markdown("**Segmentation Mask**")
            colored_mask = create_colored_mask(mask, num_classes)
            st.image(colored_mask, use_container_width=True)
        
        with col3:
            st.markdown("**Overlay**")
            st.image(overlay, use_container_width=True)
        
        # Color Legend
        if show_color_legend:
            st.markdown("### 🎨 Color Legend")
            legend_cols = st.columns(min(num_classes, 7))
            
            for i, col in enumerate(legend_cols):
                if i < num_classes:
                    color_hex = '#{:02x}{:02x}{:02x}'.format(*COLORS[i])
                    col.markdown(
                        f'<div style="background-color:{color_hex}; '
                        f'padding:10px; border-radius:5px; text-align:center; '
                        f'color:{"white" if i > 0 else "gray"};">'
                        f'{CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}"}</div>',
                        unsafe_allow_html=True
                    )
        
        # Statistics
        if show_statistics:
            st.markdown("### 📊 Segmentation Statistics")
            
            stat_cols = st.columns(min(num_classes, 4))
            
            for i, col in enumerate(stat_cols):
                if i < num_classes and i in stats:
                    with col:
                        st.metric(
                            label=CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}",
                            value=f"{stats[i]['percentage']:.1f}%",
                            delta=f"{stats[i]['count']:,} pixels"
                        )
            
            # Tumor detection status
            tumor_pixels = sum(stats[c]['count'] for c in range(1, num_classes) if c in stats)
            tumor_percentage = (tumor_pixels / mask.size) * 100
            
            st.markdown("---")
            
            if tumor_percentage > 0.5:
                st.error(f"⚠️ **Tumor Detected!** - {tumor_percentage:.2f}% of the image contains tumor regions")
            else:
                st.success("✅ **No significant tumor detected** - Image appears normal")
        
        # Download options
        st.markdown("### 💾 Download Results")
        
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        
        with dl_col1:
            # Convert mask to downloadable format
            mask_pil = Image.fromarray(colored_mask)
            from io import BytesIO
            buf = BytesIO()
            mask_pil.save(buf, format='PNG')
            st.download_button(
                label="📥 Download Mask",
                data=buf.getvalue(),
                file_name="segmentation_mask.png",
                mime="image/png"
            )
        
        with dl_col2:
            # Convert overlay to downloadable format
            overlay_pil = Image.fromarray(overlay)
            buf2 = BytesIO()
            overlay_pil.save(buf2, format='PNG')
            st.download_button(
                label="📥 Download Overlay",
                data=buf2.getvalue(),
                file_name="segmentation_overlay.png",
                mime="image/png"
            )
        
        with dl_col3:
            # Download raw mask as numpy
            buf3 = BytesIO()
            np.save(buf3, mask)
            st.download_button(
                label="📥 Download Raw Mask (.npy)",
                data=buf3.getvalue(),
                file_name="segmentation_mask.npy",
                mime="application/octet-stream"
            )
    
    else:
        # Instructions when no image is uploaded
        st.info("👆 Upload an MRI image to get started, or check 'Use Demo Image' for a quick test.")
        
        # Example usage
        with st.expander("📖 How to use this application"):
            st.markdown("""
            1. **Upload an MRI image** using the file uploader above
            2. **Wait for processing** - the AI model will analyze the image
            3. **View results** - see the original image, segmentation mask, and overlay
            4. **Adjust settings** - use the sidebar to customize the display
            5. **Download results** - save the segmentation mask or overlay
            
            **Supported formats:** JPG, JPEG, PNG, BMP
            
            **Note:** This tool is for research/educational purposes only and should not 
            be used for clinical diagnosis.
            """)

# =====================
# Footer
# =====================
def footer():
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            Made with ❤️ using Streamlit & TensorFlow<br>
            <small>⚠️ This tool is for research purposes only. Not for clinical use.</small>
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================
# Run Application
# =====================
if __name__ == "__main__":
    main()
    footer()
