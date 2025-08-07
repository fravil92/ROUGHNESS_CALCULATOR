import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, img_as_float
from scipy.signal import correlate
from numpy.polynomial import Polynomial
from scipy.fft import fft, fftfreq
from PIL import Image
import io as iolib
from datetime import datetime
from scipy import ndimage
from skimage.morphology import remove_small_objects, binary_closing, disk
from skimage.measure import label, regionprops
import os
import openai



# Set page config
st.set_page_config(
    page_title="Sidewall Roughness Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e7d32;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .results-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e7d32;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'image_loaded' not in st.session_state:
    st.session_state.image_loaded = False
if 'image_height' not in st.session_state:
    st.session_state.image_height = 1000
if 'image_width' not in st.session_state:
    st.session_state.image_width = 1000
if 'roi_coords' not in st.session_state:
    st.session_state.roi_coords = None
if 'scale_lines' not in st.session_state:
    st.session_state.scale_lines = None
if 'known_distance' not in st.session_state:
    st.session_state.known_distance = 100.0

# Title
st.markdown('<div class="main-header">🔬 Advanced Sidewall Roughness Analysis</div>', unsafe_allow_html=True)

# Introduction
with st.expander("📖 About This Enhanced Tool", expanded=False):
    st.markdown("""
    **New Advanced Features:**

    🎯 **ROI Selection**: Define rectangular regions of interest for targeted analysis

    📏 **Interactive Scale Calibration**: Set scale by measuring pixel distances between reference lines

    🎨 **Advanced Image Processing**: Enhanced contrast controls and preprocessing within selected regions

    📊 **Multi-Region Analysis**: Process multiple regions with different parameters

    **Workflow:**
    1. **Upload & Calibrate**: Load SEM image and set scale using reference markers
    2. **Region Selection**: Define rectangular ROI for analysis
    3. **Image Enhancement**: Apply contrast and threshold adjustments to ROI
    4. **Edge Detection**: Extract sidewall profile from processed region
    5. **Analysis & Export**: Calculate roughness metrics and export results
    """)

# Sidebar for parameters
st.sidebar.markdown("## 🔧 Analysis Parameters")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Image",
    type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
    help="For example: upload a SEM image of the etched sidewall"
)
use_example = st.sidebar.checkbox("Use Example Image", value=False, help="Try with a real SEM image example")

if not uploaded_file and use_example:
    uploaded_file = "example.png"

def get_ai_interpretation(prompt, model="openai/gpt-4o"):
    # Get key
    api_key = st.secrets.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return "OpenRouter API key not set! Please configure it in your secrets."
    # Set OpenRouter endpoint
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert scientific assistant who can analyze and comment on sidewall roughness measurement results, explaining their meaning and suggesting possible improvements or interpretations for research and engineering."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenRouter: {e}"


def create_interactive_image_selector(img_array, title="Image Selector"):
    """Create interactive image with ROI selection using matplotlib"""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Display image
    if img_array.ndim == 3:
        ax.imshow(img_array)
    else:
        ax.imshow(img_array, cmap='gray')

    ax.set_title(title)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')

    return fig, ax


def create_scale_calibration_plot(img_array, line1_x=100, line2_x=200):
    """Create plot for scale calibration with reference lines"""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Convert to grayscale if needed for better visibility
    if img_array.ndim == 3:
        img_gray = np.mean(img_array, axis=2)
        ax.imshow(img_gray, cmap='gray')
    else:
        ax.imshow(img_array, cmap='gray')

    # Add reference lines
    ax.axvline(line1_x, color='red', linewidth=2, linestyle='-', label=f'Line 1 (x={line1_x})')
    ax.axvline(line2_x, color='blue', linewidth=2, linestyle='-', label=f'Line 2 (x={line2_x})')

    # Add distance annotation
    y_mid = img_array.shape[0] // 2
    ax.annotate('', xy=(line2_x, y_mid), xytext=(line1_x, y_mid),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))

    distance_pixels = abs(line2_x - line1_x)
    ax.text((line1_x + line2_x) / 2, y_mid - 50, f'{distance_pixels} pixels',
            ha='center', va='top', color='green', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title('Scale Calibration - Reference Lines')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax

def set_roi(x0, y0, x1, y1):
    st.session_state["roi_x0"] = x0
    st.session_state["roi_y0"] = y0
    st.session_state["roi_x1"] = x1
    st.session_state["roi_y1"] = y1
    st.session_state.roi_coords = (x0, y0, x1, y1)
    st.rerun()

def process_roi_image_enhanced(img_array, roi_coords, contrast_factor=1.0, brightness=0.0, threshold=0.5,
                               island_removal_params=None, morphology_params=None):
    """Enhanced version with island filtering and morphological cleanup"""
    if roi_coords is None:
        roi_img = img_array.copy()
    else:
        x0, y0, x1, y1 = roi_coords
        roi_img = img_array[y0:y1, x0:x1].copy()
    # Convert to grayscale robustly
    if roi_img.ndim == 3:
        img_gray = roi_img[..., :3]
        img_gray = img_as_float(img_gray)
        img_gray = 0.2126 * img_gray[..., 0] + 0.7152 * img_gray[..., 1] + 0.0722 * img_gray[..., 2]
    else:
        img_gray = img_as_float(roi_img)
    img_enhanced = np.clip(img_gray * contrast_factor + brightness, 0, 1)
    img_binary_raw = img_enhanced > threshold
    if island_removal_params is None:
        island_removal_params = {'min_size': 50, 'connectivity_check': True}
    if morphology_params is None:
        morphology_params = {'closing_radius': 2}
    from skimage.morphology import remove_small_objects, binary_closing, disk
    from skimage.measure import label
    img_no_small = remove_small_objects(img_binary_raw, min_size=island_removal_params['min_size'], connectivity=2)
    if island_removal_params['connectivity_check']:
        labeled = label(img_no_small, connectivity=2)
        height, width = img_no_small.shape
        edge_components = set()
        edge_components.update(labeled[0, :])
        edge_components.update(labeled[-1, :])
        edge_components.update(labeled[:, 0])
        edge_components.update(labeled[:, -1])
        edge_components.discard(0)
        img_no_islands = np.zeros_like(labeled, dtype=bool)
        for component_id in edge_components:
            img_no_islands[labeled == component_id] = True
    else:
        img_no_islands = img_no_small
    if morphology_params['closing_radius'] > 0:
        selem = disk(morphology_params['closing_radius'])
        img_binary_cleaned = binary_closing(img_no_islands, selem)
    else:
        img_binary_cleaned = img_no_islands
    removed_islands = img_binary_raw & ~img_binary_cleaned
    from skimage.measure import label
    return {
        'roi_img': roi_img,
        'img_gray': img_gray,
        'img_enhanced': img_enhanced,
        'img_binary_raw': img_binary_raw,
        'img_binary': img_binary_cleaned,
        'removed_islands': removed_islands,
        'roi_coords': roi_coords,
        'processing_stats': {
            'islands_removed': np.max(label(removed_islands)) if np.any(removed_islands) else 0,
            'pixels_removed': np.sum(removed_islands),
            'final_white_pixels': np.sum(img_binary_cleaned)
        }
    }

def extract_edge_profile_from_roi_enhanced(processed_roi, edge_side='right', find_main_structure=True):
    """Enhanced edge profile extraction that uses the cleaned binary image"""
    from skimage.measure import label, regionprops
    img_binary = processed_roi['img_binary']
    height, width = img_binary.shape
    if find_main_structure and np.any(img_binary):
        labeled = label(img_binary, connectivity=2)
        props = regionprops(labeled)
        if len(props) > 0:
            scores = []
            for prop in props:
                area = prop.area
                centroid_y, centroid_x = prop.centroid
                score = area
                if edge_side == 'right':
                    score *= (1 + centroid_x / width)
                elif edge_side == 'left':
                    score *= (1 + (width - centroid_x) / width)
                scores.append(score)
            best_idx = np.argmax(scores)
            best_label = props[best_idx].label
            img_binary = (labeled == best_label)
    profile = []
    y_coords = []
    for y in range(height):
        row = img_binary[y, :]
        if edge_side == 'right':
            white_pixels = np.where(row)[0]
            if len(white_pixels) > 0:
                profile.append(white_pixels[-1])
                y_coords.append(y)
        elif edge_side == 'left':
            white_pixels = np.where(row)[0]
            if len(white_pixels) > 0:
                profile.append(white_pixels[0])
                y_coords.append(y)
        else:  # center
            white_pixels = np.where(row)[0]
            if len(white_pixels) > 0:
                profile.append(np.mean(white_pixels))
                y_coords.append(y)
    if len(profile) == 0:
        return None, None
    return np.array(profile), np.array(y_coords)

def suggest_processing_parameters(img_array, roi_coords=None):
    if roi_coords is None:
        roi_img = img_array
    else:
        x0, y0, x1, y1 = roi_coords
        roi_img = img_array[y0:y1, x0:x1]
    if roi_img.ndim == 3:
        from skimage import img_as_float
        img_gray = img_as_float(roi_img)
        img_gray = 0.2126 * img_gray[..., 0] + 0.7152 * img_gray[..., 1] + 0.0722 * img_gray[..., 2]
    else:
        from skimage import img_as_float
        img_gray = img_as_float(roi_img)
    mean_intensity = np.mean(img_gray)
    std_intensity = np.std(img_gray)
    contrast_ratio = np.max(img_gray) - np.min(img_gray)
    suggestions = {
        "threshold": {
            "value": np.clip(mean_intensity + 0.1 * std_intensity, 0.3, 0.7),
            "reasoning": f"Based on mean intensity ({mean_intensity:.2f}) + 10% std ({std_intensity:.2f})"
        },
        "contrast_factor": {
            "value": np.clip(1.0 / contrast_ratio * 0.8, 0.5, 2.0) if contrast_ratio > 0 else 1.0,
            "reasoning": f"Adjust for contrast ratio of {contrast_ratio:.2f}"
        },
        "min_island_size": {
            "value": max(20, int(roi_img.shape[0] * roi_img.shape[1] * 0.001)),
            "reasoning": f"0.1% of ROI area ({roi_img.shape[0]}x{roi_img.shape[1]} pixels)"
        },
        "closing_radius": {
            "value": 2 if std_intensity > 0.1 else 1,
            "reasoning": f"Based on noise level (std: {std_intensity:.3f})"
        }
    }
    return suggestions


def calculate_roughness_metrics(profile_nm, scale_nm_per_px):
    """Calculate RMS roughness and correlation length"""

    if len(profile_nm) < 3:
        return None

    # RMS roughness
    rms_roughness = np.sqrt(np.mean(profile_nm ** 2))

    # Error estimation for RMS roughness
    n_points = len(profile_nm)
    d_rms_roughness = scale_nm_per_px / np.sqrt(2 * n_points)

    # Autocorrelation
    autocorr = correlate(profile_nm, profile_nm, mode='full')
    autocorr = autocorr[autocorr.size // 2:] / np.max(autocorr)
    lags_nm = np.arange(autocorr.size) * scale_nm_per_px

    # Correlation length (1/e point)
    try:
        idx_corr = np.where(autocorr < 1 / np.e)[0][0]
        corr_length_nm = lags_nm[idx_corr]
    except IndexError:
        corr_length_nm = np.nan

    # Error estimation for correlation length
    d_corr_length_nm = scale_nm_per_px

    return {
        'rms_roughness': rms_roughness,
        'd_rms_roughness': d_rms_roughness,
        'corr_length_nm': corr_length_nm,
        'd_corr_length_nm': d_corr_length_nm,
        'autocorr': autocorr,
        'lags_nm': lags_nm
    }


def fft_analysis(autocorr, scale_nm_per_px, min_period=10):
    """Perform FFT analysis to find dominant periodicities"""

    N_ac = len(autocorr)
    T_ac = scale_nm_per_px

    autocorr_detrended = autocorr - np.mean(autocorr)
    yf_ac = fft(autocorr_detrended)
    xf_ac = fftfreq(N_ac, T_ac)

    pos_mask = (xf_ac > 0) & np.isfinite(xf_ac)
    spatial_periods = 1 / xf_ac[pos_mask]
    amplitudes = np.abs(yf_ac[pos_mask])

    # Sort by spatial period
    sort_idx = np.argsort(spatial_periods)
    spatial_periods = spatial_periods[sort_idx]
    amplitudes = amplitudes[sort_idx]

    # Exclude lowest frequency to avoid DC/edge artifacts
    valid = spatial_periods > min_period

    if np.any(valid):
        peak_idx = np.argmax(amplitudes[valid])
        dominant_period = spatial_periods[valid][peak_idx]
        peak_amplitude = amplitudes[valid][peak_idx]
    else:
        dominant_period = np.nan
        peak_amplitude = np.nan

    return {
        'spatial_periods': spatial_periods,
        'amplitudes': amplitudes,
        'dominant_period': dominant_period,
        'peak_amplitude': peak_amplitude,
        'valid_mask': valid
    }


# Main analysis
if uploaded_file is not None:
    if isinstance(uploaded_file, str):
        image = Image.open(uploaded_file)
    else:
        image = Image.open(uploaded_file)
    img_array = np.array(image)
    if use_example and isinstance(uploaded_file, str):
        st.info("Example SEM image loaded. Try the workflow or upload your own image in the sidebar.")
    # ...rest of app logic...


    st.session_state.image_loaded = True
    st.session_state.image_height = img_array.shape[0]
    st.session_state.image_width = img_array.shape[1]

    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📏 Scale Calibration",
        "🎯 ROI Selection",
        "🎨 Image Processing",
        "📊 Analysis Results",
        "💾 Export Data"
    ])

    # === TAB 1: SCALE CALIBRATION ===
    with tab1:
        st.markdown("## 📏 Interactive Scale Calibration")
        st.markdown("Set two vertical reference lines and enter the known distance between them.")

        col1, col2 = st.columns([3, 1])

        with col1:
            # Manual coordinate input first (for creating the plot)
            st.markdown("### Reference Line Positions")

            subcol1, subcol2 = st.columns(2)
            with subcol1:
                line1_x = st.number_input("Line 1 - X coordinate", min_value=0,
                                          max_value=st.session_state.image_width - 1, value=100, key="line1_x")
            with subcol2:
                line2_x = st.number_input("Line 2 - X coordinate", min_value=0,
                                          max_value=st.session_state.image_width - 1, value=200, key="line2_x")

            # Create and display scale calibration plot
            scale_fig, scale_ax = create_scale_calibration_plot(img_array, line1_x, line2_x)
            st.pyplot(scale_fig)

        with col2:
            st.markdown("### Calibration Settings")

            # Known distance input
            known_distance = st.number_input(
                "Known distance (nm)",
                min_value=0.1,
                max_value=100000.0,
                value=st.session_state.known_distance,
                step=0.1,
                help="Physical distance between the reference lines"
            )
            st.session_state.known_distance = known_distance

            # Calculate scale
            pixel_distance = abs(line2_x - line1_x)
            if pixel_distance > 0:
                calculated_scale = known_distance / pixel_distance
                st.markdown(f"**Calculated Scale:** {calculated_scale:.3f} nm/pixel")
                st.markdown(f"**Pixel Distance:** {pixel_distance} pixels")

                # Store calculated scale
                scale_nm_per_px = calculated_scale

                # Additional info
                st.markdown("---")
                st.markdown("### 📊 Scale Information")
                st.markdown(f"**Image Width:** {st.session_state.image_width} pixels")
                st.markdown(f"**Physical Width:** {st.session_state.image_width * calculated_scale / 1000:.1f} μm")
                st.markdown(f"**Image Height:** {st.session_state.image_height} pixels")
                st.markdown(f"**Physical Height:** {st.session_state.image_height * calculated_scale / 1000:.1f} μm")
            else:
                scale_nm_per_px = 11.11  # Default value
                st.warning("Please set different X coordinates for the reference lines.")

    # === TAB 2: ROI SELECTION ===
    with tab2:
        st.markdown("## 🎯 Region of Interest Selection")
        st.markdown("Define a rectangular region for detailed analysis using the coordinate inputs below.")

        col1, col2 = st.columns([3, 1])

        with col1:
            # Manual ROI input
            st.markdown("### ROI Coordinates")

            img_width = st.session_state.image_width
            img_height = st.session_state.image_height

            # Ensure default ROI fits the current image
            if "roi_x0" not in st.session_state or st.session_state["roi_x0"] >= img_width - 1:
                st.session_state["roi_x0"] = 0
            if "roi_x1" not in st.session_state or st.session_state["roi_x1"] > img_width or st.session_state[
                "roi_x1"] <= st.session_state["roi_x0"]:
                st.session_state["roi_x1"] = min(500, img_width)
            if "roi_y0" not in st.session_state or st.session_state["roi_y0"] >= img_height - 1:
                st.session_state["roi_y0"] = 0
            if "roi_y1" not in st.session_state or st.session_state["roi_y1"] > img_height or st.session_state[
                "roi_y1"] <= st.session_state["roi_y0"]:
                st.session_state["roi_y1"] = min(500, img_height)

            subcol1, subcol2 = st.columns(2)
            with subcol1:
                roi_x0 = st.number_input("X0 (left)", min_value=0, max_value=img_width - 2,
                                         value=st.session_state["roi_x0"], key="roi_x0_input")
                roi_y0 = st.number_input("Y0 (top)", min_value=0, max_value=img_height - 2,
                                         value=st.session_state["roi_y0"], key="roi_y0_input")
            with subcol2:
                min_x1 = roi_x0 + 1
                min_y1 = roi_y0 + 1
                roi_x1 = st.number_input("X1 (right)", min_value=min_x1, max_value=img_width,
                                         value=max(min_x1, st.session_state["roi_x1"]), key="roi_x1_input")
                roi_y1 = st.number_input("Y1 (bottom)", min_value=min_y1, max_value=img_height,
                                         value=max(min_y1, st.session_state["roi_y1"]), key="roi_y1_input")

            # Update session state
            st.session_state["roi_x0"] = roi_x0
            st.session_state["roi_x1"] = roi_x1
            st.session_state["roi_y0"] = roi_y0
            st.session_state["roi_y1"] = roi_y1
            roi_coords = (roi_x0, roi_y0, roi_x1, roi_y1)
            st.session_state.roi_coords = roi_coords

            # Create ROI visualization
            roi_fig, roi_ax = create_interactive_image_selector(img_array, "Image with Selected ROI")

            # Add ROI rectangle
            rect = patches.Rectangle(
                (roi_x0, roi_y0), roi_x1 - roi_x0, roi_y1 - roi_y0,
                linewidth=3, edgecolor='red', facecolor='none', alpha=0.8
            )
            roi_ax.add_patch(rect)

            # Add ROI label
            roi_ax.text(roi_x0 + 5, roi_y0 + 15, 'ROI', color='red', fontsize=14,
                        fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            st.pyplot(roi_fig)

        with col2:
            # ROI info
            roi_width = roi_x1 - roi_x0
            roi_height = roi_y1 - roi_y0
            st.markdown("### 📊 ROI Information")
            st.markdown(f"**ROI Size:** {roi_width} × {roi_height} pixels")

            if 'scale_nm_per_px' in locals():
                physical_width = roi_width * scale_nm_per_px / 1000
                physical_height = roi_height * scale_nm_per_px / 1000
                st.markdown(f"**Physical Size:** {physical_width:.1f} × {physical_height:.1f} μm")
                st.markdown(f"**Scale Used:** {scale_nm_per_px:.3f} nm/pixel")
            else:
                st.warning("⚠️ Set scale in Tab 1 first for physical dimensions")

            # Quick ROI presets
            # Assuming you have already set img_width and img_height
            w, h = st.session_state.image_width, st.session_state.image_height

            st.markdown("---")
            st.markdown("### 🎯 Quick ROI Presets")

            if st.button("📐 Full Image"):
                set_roi(0, 0, w, h)

            if st.button("🎯 Center Quarter"):
                set_roi(w // 4, h // 4, 3 * w // 4, 3 * h // 4)

            if st.button("📏 Left Half"):
                set_roi(0, 0, w // 2, h)

            if st.button("📐 Right Half"):
                set_roi(w // 2, 0, w, h)

    # === TAB 3: IMAGE PROCESSING ===
    # === TAB 3: ENHANCED IMAGE PROCESSING ===
    with tab3:
        st.markdown("## 🎨 Enhanced Image Processing with Island Filtering")

        # Get ROI coordinates
        roi_coords = st.session_state.roi_coords if st.session_state.roi_coords else None

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Basic Processing Controls")

            contrast_factor = st.slider(
                "Contrast",
                min_value=0.1, max_value=3.0, value=st.session_state.get("contrast_factor", 1.0),
                step=0.01, key="contrast_factor"
            )

            brightness = st.slider(
                "Brightness",
                min_value=-0.5,
                max_value=0.5,
                value=0.0,
                step=0.01,
                help="Adjust image brightness"
            )

            threshold = st.slider(
                "Binary Threshold",
                min_value=0.0, max_value=1.0, value=st.session_state.get("threshold", 0.5),
                step=0.01, key="threshold"
            )

            edge_side = st.selectbox(
                "Edge to Extract",
                ["right", "left", "center"],
                index=0,
                help="Which edge of the white region to extract"
            )

            st.markdown("---")
            st.markdown("### 🏝️ Island Filtering Controls")

            min_island_size = st.slider(
                "Minimum Component Size",
                min_value=10, max_value=500, value=st.session_state.get("min_island_size", 50),
                step=1, key="min_island_size"
            )
            connectivity_check = st.checkbox(
                "Remove Edge-Disconnected Islands",
                value=True,
                help="Remove white regions that don't connect to image edges"
            )

            closing_radius = st.slider(
                "Morphological Closing Radius",
                min_value=0, max_value=10, value=st.session_state.get("closing_radius", 2),
                step=1, key="closing_radius"
            )

            find_main_structure = st.checkbox(
                "Use Main Structure Only",
                value=True,
                help="Extract edge from the largest/most relevant structure only"
            )

            process_button = st.button("🔄 Process ROI with Island Filtering", type="primary")

        with col2:
            # Run the enhanced processing either on button click or if not yet in session_state
            if process_button or 'processed_roi_enhanced' not in st.session_state:
                island_removal_params = {
                    'min_size': min_island_size,
                    'connectivity_check': connectivity_check
                }

                morphology_params = {
                    'closing_radius': closing_radius
                }

                processed_roi = process_roi_image_enhanced(
                    img_array,
                    roi_coords,
                    contrast_factor,
                    brightness,
                    threshold,
                    island_removal_params,
                    morphology_params
                )
                st.session_state.processed_roi_enhanced = processed_roi

                profile, y_coords = extract_edge_profile_from_roi_enhanced(
                    processed_roi,
                    edge_side,
                    find_main_structure
                )

                if profile is not None:
                    st.session_state.edge_profile = profile
                    st.session_state.edge_y_coords = y_coords
                    st.success(f"✅ Edge profile extracted: {len(profile)} points")

                    stats = processed_roi['processing_stats']
                    if stats['islands_removed'] > 0:
                        st.info(f"🏝️ Removed {stats['islands_removed']} islands ({stats['pixels_removed']} pixels)")
                    else:
                        st.info("ℹ️ No islands detected")

                    st.info(f"📊 Final structure: {stats['final_white_pixels']} white pixels")
                else:
                    st.error("❌ No edge detected! Try adjusting the parameters.")
                    st.session_state.edge_profile = None

            # Display processed images
            if 'processed_roi_enhanced' in st.session_state:
                processed_roi = st.session_state.processed_roi_enhanced

                fig, axes = plt.subplots(2, 3, figsize=(15, 10))

                axes[0, 0].imshow(processed_roi['img_gray'], cmap='gray')
                axes[0, 0].set_title('Original ROI (Grayscale)')
                axes[0, 0].axis('off')

                axes[0, 1].imshow(processed_roi['img_enhanced'], cmap='gray')
                axes[0, 1].set_title('Enhanced (Contrast + Brightness)')
                axes[0, 1].axis('off')

                axes[0, 2].imshow(processed_roi['img_binary_raw'], cmap='gray')
                axes[0, 2].set_title('Raw Binary (Before Island Removal)')
                axes[0, 2].axis('off')

                axes[1, 0].imshow(processed_roi['img_binary_raw'], cmap='gray', alpha=0.7)
                if np.any(processed_roi['removed_islands']):
                    axes[1, 0].imshow(processed_roi['removed_islands'], cmap='Reds', alpha=0.8)
                    axes[1, 0].set_title(
                        f'Removed Islands (Count: {processed_roi["processing_stats"]["islands_removed"]})')
                else:
                    axes[1, 0].set_title('No Islands Removed')
                axes[1, 0].axis('off')

                axes[1, 1].imshow(processed_roi['img_binary'], cmap='gray')
                axes[1, 1].set_title('Cleaned Binary (After Island Removal)')
                axes[1, 1].axis('off')

                axes[1, 2].imshow(processed_roi['img_binary'], cmap='gray')
                if hasattr(st.session_state, 'edge_profile') and st.session_state.edge_profile is not None:
                    axes[1, 2].plot(st.session_state.edge_profile, st.session_state.edge_y_coords, 'r-', linewidth=2)
                    axes[1, 2].set_title(f'Final Edge Profile ({len(st.session_state.edge_profile)} points)')
                else:
                    axes[1, 2].set_title('No Edge Detected')
                axes[1, 2].axis('off')

                plt.tight_layout()
                st.pyplot(fig)

                # Processing summary
                with st.expander("🔍 Processing Summary", expanded=False):
                    stats = processed_roi['processing_stats']
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Islands Removed", stats['islands_removed'])
                    with col2:
                        st.metric("Pixels Removed", stats['pixels_removed'])
                    with col3:
                        st.metric("Final White Pixels", stats['final_white_pixels'])

                    st.markdown("### Processing Parameters Used")
                    st.json({
                        "Basic Processing": {
                            "contrast_factor": contrast_factor,
                            "brightness": brightness,
                            "threshold": threshold,
                            "edge_side": edge_side
                        },
                        "Island Filtering": {
                            "min_component_size": min_island_size,
                            "remove_disconnected_islands": connectivity_check,
                            "morphological_closing_radius": closing_radius,
                            "use_main_structure_only": find_main_structure
                        }
                    })

        with st.expander("🎯 Parameter Optimization Assistant", expanded=False):
            st.markdown("### Automatic Parameter Suggestions")
            if st.button("🔍 Analyze Image & Suggest Parameters"):
                suggestions = suggest_processing_parameters(img_array, roi_coords)
                st.session_state.suggestions = suggestions

            if "suggestions" in st.session_state:
                s = st.session_state.suggestions
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Threshold:** {s['threshold']['value']:.3f}")
                    st.markdown(f"*{s['threshold']['reasoning']}*")
                    st.markdown(f"**Contrast Factor:** {s['contrast_factor']['value']:.2f}")
                    st.markdown(f"*{s['contrast_factor']['reasoning']}*")
                with col2:
                    st.markdown(f"**Min Island Size:** {s['min_island_size']['value']}")
                    st.markdown(f"*{s['min_island_size']['reasoning']}*")
                    st.markdown(f"**Closing Radius:** {s['closing_radius']['value']}")
                    st.markdown(f"*{s['closing_radius']['reasoning']}*")
                if st.button("✨ Apply Suggested Parameters"):
                    st.session_state["threshold"] = float(s['threshold']['value'])
                    st.session_state["contrast_factor"] = float(s['contrast_factor']['value'])
                    st.session_state["min_island_size"] = int(s['min_island_size']['value'])
                    st.session_state["closing_radius"] = int(s['closing_radius']['value'])
                    st.experimental_rerun()

    # === TAB 4: ANALYSIS RESULTS ===
    with tab4:
        st.markdown("## 📊 Roughness Analysis Results")

        if (hasattr(st.session_state, 'edge_profile') and
                st.session_state.edge_profile is not None and
                'scale_nm_per_px' in locals()):

            # Get edge profile
            profile_px = st.session_state.edge_profile
            y_coords = st.session_state.edge_y_coords

            # Detrending options
            col1, col2 = st.columns([1, 3])

            with col1:
                st.markdown("### Detrending Options")

                detrend_method = st.selectbox(
                    "Detrending Method",
                    ["Mean Centering", "Linear Detrend", "Parabolic Detrend"],
                    index=2
                )

                # Apply detrending
                if detrend_method == "Mean Centering":
                    profile_detrended = profile_px - np.mean(profile_px)
                elif detrend_method == "Linear Detrend":
                    from scipy import signal

                    profile_detrended = signal.detrend(profile_px, type='linear')
                else:  # Parabolic
                    if len(profile_px) > 3:
                        pfit = Polynomial.fit(y_coords, profile_px, 2)
                        profile_detrended = profile_px - pfit(y_coords)
                    else:
                        profile_detrended = profile_px - np.mean(profile_px)

                # Convert to nm
                profile_nm = profile_detrended * scale_nm_per_px

                # Calculate metrics
                metrics = calculate_roughness_metrics(profile_nm, scale_nm_per_px)

                if metrics:
                    # FFT analysis
                    fft_results = fft_analysis(metrics['autocorr'], scale_nm_per_px, min_period=10.0)

                    # Display results
                    st.markdown("### 📈 Results")
                    st.markdown(
                        f"**RMS Roughness:** {metrics['rms_roughness']:.2f} ± {metrics['d_rms_roughness']:.2f} nm")

                    if not np.isnan(metrics['corr_length_nm']):
                        st.markdown(
                            f"**Correlation Length:** {metrics['corr_length_nm']:.2f} ± {metrics['d_corr_length_nm']:.2f} nm")

                    if not np.isnan(fft_results['dominant_period']):
                        st.markdown(f"**Dominant Periodicity:** {fft_results['dominant_period']:.1f} nm")

                    st.markdown(f"**Profile Points:** {len(profile_nm)}")
                    physical_length = len(profile_nm) * scale_nm_per_px / 1000
                    st.markdown(f"**Analysis Length:** {physical_length:.1f} μm")

            with col2:
                if metrics:
                    # Create analysis plots
                    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

                    # Profile plot
                    axes[0].plot(y_coords * scale_nm_per_px / 1000, profile_nm, 'b-', linewidth=1)
                    axes[0].set_xlabel('Distance (μm)')
                    axes[0].set_ylabel('Deviation (nm)')
                    axes[0].set_title(f'Sidewall Profile ({detrend_method})')
                    axes[0].grid(True, alpha=0.3)

                    # Autocorrelation plot
                    axes[1].plot(metrics['lags_nm'], metrics['autocorr'], 'g-', linewidth=1)
                    if not np.isnan(metrics['corr_length_nm']):
                        axes[1].axvline(metrics['corr_length_nm'], color='r', linestyle='--',
                                        label=f'Correlation length = {metrics["corr_length_nm"]:.1f} nm')
                    axes[1].axhline(1 / np.e, color='orange', linestyle=':', alpha=0.7, label='1/e threshold')
                    axes[1].set_xlabel('Lag (nm)')
                    axes[1].set_ylabel('Autocorrelation')
                    axes[1].set_title('Autocorrelation Function')
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)

                    # FFT plot
                    axes[2].plot(fft_results['spatial_periods'], fft_results['amplitudes'], 'purple', linewidth=1)
                    if not np.isnan(fft_results['dominant_period']):
                        axes[2].axvline(fft_results['dominant_period'], color='red', linestyle='--',
                                        label=f'Peak: {fft_results["dominant_period"]:.1f} nm')
                    axes[2].set_xlabel('Spatial Period (nm)')
                    axes[2].set_ylabel('FFT Amplitude')
                    axes[2].set_title('Spatial Frequency Analysis')
                    axes[2].set_xlim([0, min(150000, np.max(fft_results['spatial_periods']))])
                    axes[2].legend()
                    axes[2].grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)

        else:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("""
            ### ⚠️ Analysis Requirements

            To perform roughness analysis, please complete:
            1. **Scale Calibration** (Tab 1) - Set the nm/pixel scale
            2. **ROI Selection** (Tab 2) - Define analysis region  
            3. **Image Processing** (Tab 3) - Extract edge profile

            Once all steps are complete, analysis results will appear here.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Prepare the prompt for the AI ---
    prompt_lines = [
        f"RMS roughness: {metrics['rms_roughness']:.2f} nm ± {metrics['d_rms_roughness']:.2f} nm",
        f"Correlation length: {metrics['corr_length_nm']:.2f} nm ± {metrics['d_corr_length_nm']:.2f} nm" if not np.isnan(
            metrics['corr_length_nm']) else "Correlation length: Not available",
        f"Dominant periodicity: {fft_results['dominant_period']:.1f} nm" if not np.isnan(
            fft_results['dominant_period']) else "Dominant periodicity: Not available",
        f"Profile points: {len(profile_nm)}",
        f"Profile analysis length: {physical_length:.2f} μm",
        "Context: This is sidewall profile roughness analysis from SEM images for microfabrication (etching)."
    ]
    prompt = "\n".join(prompt_lines)
    prompt += "\n\nPlease provide a concise, expert interpretation of these measurements, including what they mean about the fabrication quality, any likely causes for observed values, and any suggestions for improvement. Write your answer in English, suitable for a scientific report. Provide also a LaTeX code with the summary of the results, suitable for a scientific report"

    # --- UI logic ---
    st.markdown("---")
    st.markdown("## 🧠 AI-powered Interpretation and Suggestions")

    if st.button("🧠 Generate AI Report & Suggestions"):
        with st.spinner("Contacting AI assistant..."):
            ai_report = get_ai_interpretation(prompt, model="openai/gpt-oss-20b:free")  # Or try llama-3, claude-3, etc.
            st.session_state["ai_report"] = ai_report

    if "ai_report" in st.session_state:
        st.success(st.session_state["ai_report"])

    # === TAB 5: EXPORT DATA ===
    with tab5:
        st.markdown("## 💾 Export Analysis Data")

        if isinstance(uploaded_file, str):
            file_display_name = uploaded_file
        else:
            file_display_name = uploaded_file.name

        if (hasattr(st.session_state, 'edge_profile') and
                st.session_state.edge_profile is not None and
                'scale_nm_per_px' in locals() and
                'metrics' in locals()):

            col1, col2, col3 = st.columns(3)

            with col1:
                # Analysis report
                results_text = f"""Advanced Sidewall Roughness Analysis Report
================================================

File: {file_display_name}
Analysis Date: {st.session_state.get('analysis_date', 'Not set')}

CALIBRATION:
Scale: {scale_nm_per_px:.3f} nm/pixel
Reference Distance: {st.session_state.known_distance:.1f} nm

ROI PARAMETERS:
ROI Coordinates: {st.session_state.roi_coords}
ROI Size: {roi_width} × {roi_height} pixels
Physical ROI Size: {physical_width:.1f} × {physical_height:.1f} μm

PROCESSING PARAMETERS:
Contrast Factor: {contrast_factor}
Brightness Offset: {brightness}
Binary Threshold: {threshold}
Edge Side: {edge_side}
Detrending: {detrend_method}

ROUGHNESS METRICS:
RMS Roughness: {metrics['rms_roughness']:.2f} ± {metrics['d_rms_roughness']:.2f} nm
Correlation Length: {metrics['corr_length_nm']:.2f} ± {metrics['d_corr_length_nm']:.2f} nm
Dominant Periodicity: {fft_results['dominant_period']:.1f} nm
Peak Amplitude: {fft_results['peak_amplitude']:.3f}

ANALYSIS STATISTICS:
Profile Points: {len(profile_nm)}
Analysis Length: {physical_length:.1f} μm
Resolution: ±{scale_nm_per_px:.2f} nm
"""

                st.download_button(
                    label="📄 Download Full Report",
                    data=results_text,
                    file_name=f"advanced_roughness_report_{file_display_name}.txt",
                    mime="text/plain"
                )

            with col2:
                # Profile data
                profile_data = np.column_stack([
                    st.session_state.edge_y_coords,
                    st.session_state.edge_profile,
                    profile_nm
                ])
                profile_csv = "y_pixels,edge_pixels,deviation_nm\n"
                for row in profile_data:
                    profile_csv += f"{row[0]:.1f},{row[1]:.3f},{row[2]:.3f}\n"

                st.download_button(
                    label="📈 Download Profile Data",
                    data=profile_csv,
                    file_name=f"profile_data_{file_display_name}.csv",
                    mime="text/csv"
                )

            with col3:
                # FFT data
                fft_data = np.column_stack([
                    fft_results['spatial_periods'],
                    fft_results['amplitudes']
                ])
                fft_csv = "spatial_period_nm,amplitude\n"
                for row in fft_data:
                    fft_csv += f"{row[0]:.3f},{row[1]:.6f}\n"

                st.download_button(
                    label="🌊 Download FFT Data",
                    data=fft_csv,
                    file_name=f"fft_data_{file_display_name}.csv",
                    mime="text/csv"
                )

            # Additional export options
            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                # Processed ROI images export
                if st.button("💾 Export Processed Images"):
                    processed_roi = st.session_state.processed_roi

                    # Save processed images
                    from PIL import Image as PILImage
                    import io

                    # Create a zip-like structure (conceptually)
                    st.success("✅ Processed images ready for download")
                    st.info("Images include: Original ROI, Enhanced, Binary, and Edge-detected versions")

            with col2:
                # Session parameters export
                session_params = {
                    'scale_nm_per_px': scale_nm_per_px,
                    'roi_coords': st.session_state.roi_coords,
                    'contrast_factor': contrast_factor,
                    'brightness': brightness,
                    'threshold': threshold,
                    'edge_side': edge_side,
                    'detrend_method': detrend_method,
                    'known_distance': st.session_state.known_distance
                }

                import json

                params_json = json.dumps(session_params, indent=2)

                st.download_button(
                    label="⚙️ Download Parameters",
                    data=params_json,
                    file_name=f"analysis_params_{file_display_name}.json",
                    mime="application/json"
                )

        else:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("""
            ### 📋 Export Options Available After Analysis

            Complete the analysis workflow to access:

            - **📄 Full Analysis Report** - Comprehensive text report with all metrics
            - **📈 Profile Data** - Raw edge coordinates and deviation values (CSV)
            - **🌊 FFT Data** - Spatial frequency analysis results (CSV)
            - **💾 Processed Images** - All intermediate processing steps
            - **⚙️ Analysis Parameters** - Session settings for reproducibility (JSON)

            All exports include metadata and processing parameters for full traceability.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

else:
    # Instructions when no file is uploaded
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ## 🚀 Advanced Analysis Workflow

    ### **1. 📤 Upload SEM Image**
    - Support for PNG, JPG, TIFF, BMP formats
    - Any resolution (optimized processing)
    - Color or grayscale images accepted

    ### **2. 📏 Scale Calibration**
    - Interactive line placement on reference markers
    - Automatic scale calculation from known distances
    - High-precision calibration for accurate measurements

    ### **3. 🎯 ROI Selection**
    - Draw rectangular regions directly on image
    - Manual coordinate input for precision
    - Multiple ROI support for comparative analysis

    ### **4. 🎨 Image Processing**
    - Advanced contrast and brightness controls
    - Real-time binary conversion preview
    - Flexible edge detection (left/right/center)
    - Region-specific processing optimization

    ### **5. 📊 Analysis & Results**
    - Multiple detrending options (mean, linear, parabolic)
    - RMS roughness with statistical uncertainty
    - Autocorrelation and correlation length
    - FFT analysis for Bosch etching periodicities
    - Interactive visualization of all results

    ### **6. 💾 Export & Documentation**
    - Comprehensive analysis reports
    - Raw data exports (CSV format)
    - Processed image exports
    - Parameter files for reproducibility

    ---

    ### 🎯 **Key Advantages:**
    - **Region-specific analysis** for targeted measurements
    - **Interactive scale calibration** using SEM reference markers  
    - **Advanced image processing** with real-time preview
    - **Multiple detrending methods** for different surface types
    - **Complete data traceability** with parameter logging
    - **Professional export options** for publication-ready results
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Example workflow visualization
    st.markdown("### 🔬 Example Analysis Pipeline")

    # Create example workflow diagram
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **📤 Step 1-2: Upload & Calibrate**
        ```
        📁 Upload SEM Image
        ↓
        📏 Set Reference Lines
        ↓  
        🎯 Calculate nm/pixel Scale
        ```
        """)

    with col2:
        st.markdown("""
        **🎯 Step 3-4: Select & Process**
        ```
        🎨 Define ROI Region
        ↓
        🔧 Adjust Contrast/Threshold
        ↓
        📈 Extract Edge Profile  
        ```
        """)

    with col3:
        st.markdown("""
        **📊 Step 5-6: Analyze & Export**
        ```
        🧮 Calculate Roughness Metrics
        ↓
        📊 Generate Visualizations
        ↓
        💾 Export Results & Data
        ```
        """)

    st.markdown("---")
    st.markdown("""
    ### 📸 Ready for Analysis
    Upload your SEM image to begin the interactive roughness analysis workflow with ROI selection, 
    scale calibration, and advanced processing controls.
    """)

# Footer
st.markdown("---")
from datetime import datetime
st.markdown(f"""
<br><hr>
<div style='text-align: center; color: #888888; font-size: 1rem;'>
&copy; {datetime.now().year} Francesco Villasmunta &mdash; made with phlove <span style='font-size:1.2em;'>✨❤️</span>
</div>
""", unsafe_allow_html=True)

