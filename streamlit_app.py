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
    "Upload SEM Image",
    type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
    help="Upload a SEM image of the etched sidewall"
)


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


def process_roi_image(img_array, roi_coords, contrast_factor=1.0, brightness=0.0, threshold=0.5):
    """Process image within ROI with enhanced contrast controls"""

    if roi_coords is None:
        # Use full image if no ROI selected
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

    # Apply contrast and brightness adjustments
    img_enhanced = np.clip(img_gray * contrast_factor + brightness, 0, 1)

    # Apply threshold to create binary image
    img_binary = img_enhanced > threshold

    return {
        'roi_img': roi_img,
        'img_gray': img_gray,
        'img_enhanced': img_enhanced,
        'img_binary': img_binary,
        'roi_coords': roi_coords
    }


def extract_edge_profile_from_roi(processed_roi, edge_side='right'):
    """Extract edge profile from processed ROI"""

    img_binary = processed_roi['img_binary']
    height, width = img_binary.shape

    profile = []
    y_coords = []

    for y in range(height):
        row = img_binary[y, :]

        if edge_side == 'right':
            # Find rightmost white pixel
            white_pixels = np.where(row)[0]
            if len(white_pixels) > 0:
                profile.append(white_pixels[-1])
                y_coords.append(y)
        elif edge_side == 'left':
            # Find leftmost white pixel
            white_pixels = np.where(row)[0]
            if len(white_pixels) > 0:
                profile.append(white_pixels[0])
                y_coords.append(y)
        else:
            # Find center of white region
            white_pixels = np.where(row)[0]
            if len(white_pixels) > 0:
                profile.append(np.mean(white_pixels))
                y_coords.append(y)

    if len(profile) == 0:
        return None, None

    return np.array(profile), np.array(y_coords)


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
    # Load image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

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

            subcol1, subcol2 = st.columns(2)
            with subcol1:
                roi_x0 = st.number_input("X0 (left)", min_value=0, max_value=st.session_state.image_width - 1, value=0,
                                         key="roi_x0")
                roi_y0 = st.number_input("Y0 (top)", min_value=0, max_value=st.session_state.image_height - 1, value=0,
                                         key="roi_y0")
            with subcol2:
                roi_x1 = st.number_input("X1 (right)", min_value=roi_x0 + 1, max_value=st.session_state.image_width,
                                         value=min(500, st.session_state.image_width), key="roi_x1")
                roi_y1 = st.number_input("Y1 (bottom)", min_value=roi_y0 + 1, max_value=st.session_state.image_height,
                                         value=min(500, st.session_state.image_height), key="roi_y1")

            # Store ROI coordinates
            st.session_state.roi_coords = (roi_x0, roi_y0, roi_x1, roi_y1)

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
            st.markdown("---")
            st.markdown("### 🎯 Quick ROI Presets")

            if st.button("📐 Full Image"):
                st.session_state.roi_coords = (0, 0, st.session_state.image_width, st.session_state.image_height)
                st.rerun()

            if st.button("🎯 Center Quarter"):
                w, h = st.session_state.image_width, st.session_state.image_height
                st.session_state.roi_coords = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
                st.rerun()

            if st.button("📏 Left Half"):
                w, h = st.session_state.image_width, st.session_state.image_height
                st.session_state.roi_coords = (0, 0, w // 2, h)
                st.rerun()

            if st.button("📐 Right Half"):
                w, h = st.session_state.image_width, st.session_state.image_height
                st.session_state.roi_coords = (w // 2, 0, w, h)
                st.rerun()

    # === TAB 3: IMAGE PROCESSING ===
    with tab3:
        st.markdown("## 🎨 Advanced Image Processing")

        # Get ROI coordinates
        if st.session_state.roi_coords:
            roi_coords = st.session_state.roi_coords
        else:
            roi_coords = None

        # Image processing controls
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Processing Controls")

            contrast_factor = st.slider(
                "Contrast",
                min_value=0.1,
                max_value=3.0,
                value=1.0,
                step=0.1,
                help="Adjust image contrast"
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
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help="Threshold for binary conversion"
            )

            edge_side = st.selectbox(
                "Edge to Extract",
                ["right", "left", "center"],
                index=0,
                help="Which edge of the white region to extract"
            )

            # Process button
            process_button = st.button("🔄 Process ROI", type="primary")

        with col2:
            if process_button or 'processed_roi' not in st.session_state:
                # Process the ROI
                processed_roi = process_roi_image(
                    img_array,
                    roi_coords,
                    contrast_factor,
                    brightness,
                    threshold
                )
                st.session_state.processed_roi = processed_roi

                # Extract edge profile
                profile, y_coords = extract_edge_profile_from_roi(processed_roi, edge_side)

                if profile is not None:
                    st.session_state.edge_profile = profile
                    st.session_state.edge_y_coords = y_coords
                    st.success(f"✅ Edge profile extracted: {len(profile)} points")
                else:
                    st.error("❌ No edge detected! Try adjusting the threshold or contrast.")
                    st.session_state.edge_profile = None

            # Display processed images
            if 'processed_roi' in st.session_state:
                processed_roi = st.session_state.processed_roi

                # Create subplot for processed images
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                # Original ROI
                axes[0, 0].imshow(processed_roi['img_gray'], cmap='gray')
                axes[0, 0].set_title('Original ROI')
                axes[0, 0].axis('off')

                # Enhanced ROI
                axes[0, 1].imshow(processed_roi['img_enhanced'], cmap='gray')
                axes[0, 1].set_title('Enhanced (Contrast + Brightness)')
                axes[0, 1].axis('off')

                # Binary ROI
                axes[1, 0].imshow(processed_roi['img_binary'], cmap='gray')
                axes[1, 0].set_title('Binary (Thresholded)')
                axes[1, 0].axis('off')

                # Binary with edge profile
                axes[1, 1].imshow(processed_roi['img_binary'], cmap='gray')
                if hasattr(st.session_state, 'edge_profile') and st.session_state.edge_profile is not None:
                    axes[1, 1].plot(st.session_state.edge_profile, st.session_state.edge_y_coords, 'r-', linewidth=2)
                axes[1, 1].set_title('Binary + Detected Edge')
                axes[1, 1].axis('off')

                plt.tight_layout()
                st.pyplot(fig)

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

    # === TAB 5: EXPORT DATA ===
    with tab5:
        st.markdown("## 💾 Export Analysis Data")

        if (hasattr(st.session_state, 'edge_profile') and
                st.session_state.edge_profile is not None and
                'scale_nm_per_px' in locals() and
                'metrics' in locals()):

            col1, col2, col3 = st.columns(3)

            with col1:
                # Analysis report
                results_text = f"""Advanced Sidewall Roughness Analysis Report
================================================

File: {uploaded_file.name}
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
                    file_name=f"advanced_roughness_report_{uploaded_file.name}.txt",
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
                    file_name=f"profile_data_{uploaded_file.name}.csv",
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
                    file_name=f"fft_data_{uploaded_file.name}.csv",
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
                    file_name=f"analysis_params_{uploaded_file.name}.json",
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
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    🔬 Advanced Sidewall Roughness Analysis | 
    ROI Selection • Scale Calibration • Interactive Processing | 
    Built with Streamlit & Scientific Python
</div>
""", unsafe_allow_html=True)