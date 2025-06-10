import streamlit as st
import tempfile
import shutil
import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from CART.src.main import run_pipeline
import base64

 # --- Default project paths ---
DEFAULT_CD28_PATH = "/Users/mukulsherekar/pythonProject/CART-Project/fasta/cd28.fasta"
DEFAULT_CD3Z_PATH = "/Users/mukulsherekar/pythonProject/CART-Project/fasta/cd3z.fasta"
DEFAULT_CAR_PATH = "/Users/mukulsherekar/pythonProject/CART-Project/fasta/scFV_FMC63.fasta"
DEFAULT_CYTOX_PATH = "/Users/mukulsherekar/pythonProject/CART-Project/mutants/CAR_mutants_cytox.csv"
DEFAULT_UNIPROT_PATH = "/Users/mukulsherekar/pythonProject/CART-Project/fasta/uniprot_trembl.fasta"

# --- ESM Model Options ---
ESM_MODELS = {
    "facebook/esm2_t6_8M_UR50D": "8M (Fastest, smallest)",
    "facebook/esm2_t12_35M_UR50D": "35M (Good balance)",
    "facebook/esm2_t30_150M_UR50D": "150M (Larger, better accuracy)",
    "facebook/esm2_t33_650M_UR50D": "650M (High-performance)",
    "facebook/esm2_t36_3B_UR50D": "3B (Largest, best performance)"
}

PIPELINE_STEPS = [
    "augmentation", "mds", "mutants", "finetuning", "embeddings", "prediction", "evaluation", "score"
]

# Initialize session state for output directory if not already set
if 'output_dir' not in st.session_state:
    st.session_state.output_dir = str(Path("output").absolute())
if 'pipeline_ran' not in st.session_state:
    st.session_state.pipeline_ran = False
if 'use_default_files' not in st.session_state:
    st.session_state.use_default_files = False

st.set_page_config(page_title="CART Project App", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #4d8ceb;
        text-align: center;
    }
    .category-header {
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        color: #2c6cb8;
        border-bottom: 2px solid #2c6cb8;
        padding-bottom: 0.5rem;
    }
    .subcategory-header {
        font-size: 1.4rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
        color: #3a7ddf;
    }
    .plot-caption {
        font-size: 0.9rem;
        font-style: italic;
        margin-top: 0.3rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        font-size: 1rem;
        font-weight: 600;
        color: #4d8ceb;
    }
</style>
""", unsafe_allow_html=True)

# App mode selection

app_mode = st.sidebar.radio("", ["Run Pipeline", "View Results"])

# Helper functions for result visualization
def display_image(image_path, caption="", width=None):
    try:
        if os.path.exists(image_path):
            img = Image.open(image_path)
            if width:
                st.image(img, caption=caption, width=width)
            else:
                st.image(img, caption=caption)
        else:
            st.info(f"Image not found: {image_path}")
    except Exception as e:
        st.error(f"Error displaying image: {e}")

def read_csv(csv_path):
    try:
        if os.path.exists(csv_path):
            # Try to read CSV, handling potential issues
            try:
                # First attempt with default pandas settings
                df = pd.read_csv(csv_path)
            except Exception as e:
                st.warning(f"Standard CSV reading failed, trying with more options: {e}")
                # Try again with more flexible options
                df = pd.read_csv(csv_path, dtype=str, on_bad_lines='warn')
            
            # Aggressively convert columns to make Arrow compatibility easier
            for col in df.columns:
                # First check if this is the 'actual' column that's causing issues
                if col.lower() == 'actual':
                    df[col] = df[col].astype(str)
                # Convert all float columns to string to avoid arrow issues
                elif df[col].dtype == np.float64 or df[col].dtype == np.float32:
                    df[col] = df[col].astype(str)
                # Convert int64 to int32 for better Arrow compatibility
                elif df[col].dtype == np.int64:
                    try:
                        df[col] = df[col].astype(np.int32)
                    except:
                        df[col] = df[col].astype(str)
                # Convert object columns to string
                elif df[col].dtype.name == 'object':
                    df[col] = df[col].astype(str)
                    
            return df
        else:
            st.info(f"CSV file not found: {csv_path}")
            return None
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        # Try one last time with minimal options
        try:
            st.warning("Attempting to read CSV with minimal settings...")
            df = pd.read_csv(csv_path, dtype=str)
            return df
        except:
            return None

def display_dataframe(df):
    """Display is now hidden - we'll just show a message indicating data is available."""
    st.info("Data is available but not displayed. Check the CSV files in the output directory for details.")

def read_text(text_path):
    try:
        if os.path.exists(text_path):
            with open(text_path, 'r') as f:
                return f.read()
        else:
            st.info(f"Text file not found: {text_path}")
            return None
    except Exception as e:
        st.error(f"Error reading text file: {e}")
        return None

# Helper function for displaying fold results without CSV data
def show_fold_results(model_name, fold_selection, paths):
    """Show only the fold correlation plot without the CSV data"""
    # Extract the fold number from the selection (e.g., "Fold 1" -> "1")
    fold_num = fold_selection.split()[-1]
    
    fold_image = paths['RESULTS_DIR'] / f"{model_name}_embeddings_fold{fold_num}_correlation.png"
    display_image(fold_image, f"Correlation Plot for {fold_selection}")
    
    # Show message instead of CSV data
    fold_csv = paths['RESULTS_DIR'] / f"{model_name}_embeddings_fold{fold_num}_correlation.csv"
    if os.path.exists(fold_csv):
        st.info(f"Detailed data available in CSV: {fold_csv.name}")

# Define function to get dynamic output paths based on the current output directory
def get_output_paths(output_dir):
    output_path = Path(output_dir)
    return {
        'EMBEDDINGS_MDS_DIR': output_path / "mds",
        'EVALUATION_DIR': output_path / "evaluation",
        'MDS_DIR': output_path / "mds",
        'HIGH_MODEL_DIR': output_path / "models/high",
        'LOW_MODEL_DIR': output_path / "models/low",
        'PLOTS_DIR': output_path / "plots",
        'RESULTS_DIR': output_path / "results"
    }

# Suppress torch-related warnings that don't impact functionality
import warnings
warnings.filterwarnings("ignore", message=".*torch\._classes.*")

# PIPELINE MODE
if app_mode == "Run Pipeline":
    st.markdown('<div class="main-header">CAR-T Cytotoxicity Prediction Pipeline</div>', unsafe_allow_html=True)


    # --- Sidebar: File Uploads and Options ---
    st.sidebar.header("1. File Uploads")

    use_default = st.sidebar.checkbox("Use Default Project Files", value=st.session_state.use_default_files)
    st.session_state.use_default_files = use_default

    # File upload sections
    if use_default:
        # Check if default files exist and show status
        st.sidebar.header("Default Files Status:")
        if os.path.exists(DEFAULT_CD28_PATH):
            st.sidebar.success(f"✓ CD28: {os.path.basename(DEFAULT_CD28_PATH)}")
            domain_1_path = DEFAULT_CD28_PATH
            domain_1_file = None
        else:
            st.sidebar.error(f"✗ CD28 not found: {DEFAULT_CD28_PATH}")
            domain_1_path = None
            domain_1_file = None
        
        if os.path.exists(DEFAULT_CD3Z_PATH):
            st.sidebar.success(f"✓ CD3Z: {os.path.basename(DEFAULT_CD3Z_PATH)}")
            domain_2_path = DEFAULT_CD3Z_PATH
            domain_2_file = None
        else:
            st.sidebar.error(f"✗ CD3Z not found: {DEFAULT_CD3Z_PATH}")
            domain_2_path = None
            domain_2_file = None
        
        if os.path.exists(DEFAULT_CAR_PATH):
            st.sidebar.success(f"✓ CAR: {os.path.basename(DEFAULT_CAR_PATH)}")
            car_path = DEFAULT_CAR_PATH
            car_file = None
        else:
            st.sidebar.warning(f"? CAR not found: {DEFAULT_CAR_PATH}")
            car_path = None
            car_file = None
        
        if os.path.exists(DEFAULT_CYTOX_PATH):
            st.sidebar.success(f"✓ Cytotoxicity: {os.path.basename(DEFAULT_CYTOX_PATH)}")
            cytotox_path = DEFAULT_CYTOX_PATH
            cytotox_file = None
        else:
            st.sidebar.error(f"✗ Cytotoxicity not found: {DEFAULT_CYTOX_PATH}")
            cytotox_path = None
            cytotox_file = None
        
        if os.path.exists(DEFAULT_UNIPROT_PATH):
            st.sidebar.success(f"✓ Uniprot: {os.path.basename(DEFAULT_UNIPROT_PATH)}")
            uniprot_path = DEFAULT_UNIPROT_PATH
            uniprot_file = None
        else:
            st.sidebar.error(f"✗ Uniprot not found: {DEFAULT_UNIPROT_PATH}")
            uniprot_path = None
            uniprot_file = None
    else:
        # Manual file uploads
        
        domain_1_file = st.sidebar.file_uploader("CD28 FASTA", type=["fasta", "fa", "txt"], key="cd28_upload")
        domain_2_file = st.sidebar.file_uploader("CD3Z FASTA", type=["fasta", "fa", "txt"], key="cd3z_upload")
        car_file = st.sidebar.file_uploader("CAR FASTA", type=["fasta", "fa", "txt"], key="car_upload")
        cytotox_file = st.sidebar.file_uploader("Cytotoxicity CSV", type=["csv"], key="cytotox_upload")
        uniprot_file = st.sidebar.text_input("Uniprot Database FASTA Path", key="uniprot_upload")
        
        # Show validation status for uniprot path
        if uniprot_file and uniprot_file.strip():
            if os.path.exists(uniprot_file.strip()):
                st.sidebar.success(f"✓ Uniprot file found: {os.path.basename(uniprot_file.strip())}")
            else:
                st.sidebar.error(f"✗ Uniprot file not found: {uniprot_file.strip()}")
        
        # Set path variables to None when using uploads (except uniprot which is now a path input)
        domain_1_path = None
        domain_2_path = None
        car_path = None
        cytotox_path = None
        uniprot_path = uniprot_file if uniprot_file else None

    st.sidebar.header("2. Model and Parameters")
    model_choice = st.sidebar.selectbox("ESM Model", list(ESM_MODELS.keys()), format_func=lambda x: f"{x} - {ESM_MODELS[x]}")

    st.sidebar.header("3. Hyperparameters")
    batch_size = st.sidebar.number_input("Batch Size", min_value=1, max_value=256, value=32)
    learning_rate = st.sidebar.number_input("Learning Rate", min_value=1e-7, max_value=1e-3, value=5e-6, format="%e")
    max_epochs = st.sidebar.number_input("Max Epochs", min_value=1, max_value=200, value=50)
    patience = st.sidebar.number_input("Early Stopping Patience", min_value=1, max_value=20, value=5)

    st.sidebar.header("4. Pipeline Steps")
    steps = st.sidebar.multiselect("Which steps to run?", PIPELINE_STEPS, default=PIPELINE_STEPS)

    st.sidebar.header("5. Output Directory")
    default_output = st.session_state.output_dir
    output_dir = st.sidebar.text_input("Output Directory", value=default_output)
    st.session_state.output_dir = output_dir

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("Run Pipeline 🚀")

    # --- Main Area: Progress, Logs, Results ---
    if run_button:
        st.subheader("Pipeline Progress and Results")
        with st.spinner("Preparing files and running pipeline..."):
            # Use a temp dir for uploads
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                # Save uploaded files or use existing paths
                def save_file(uploaded, path, name):
                    if uploaded:
                        out_path = temp_dir / name
                        with open(out_path, "wb") as f:
                            f.write(uploaded.read())
                        return str(out_path)
                    elif path:
                        return path
                    else:
                        return None
                domain_1 = save_file(domain_1_file, domain_1_path, "cd28.fasta")
                domain_2 = save_file(domain_2_file, domain_2_path, "cd3z.fasta")
                wt_car = save_file(car_file, car_path, "car.fasta")
                cytotox_csv = save_file(cytotox_file, cytotox_path, "cytotox.csv")
                
                # Uniprot DB - now handles path input instead of file upload
                if uniprot_file and uniprot_file.strip():
                    # uniprot_file is now a path string, not an uploaded file
                    uniprot_db = uniprot_file.strip()
                    # Validate that the path exists
                    if not os.path.exists(uniprot_db):
                        st.error(f"Uniprot database file not found: {uniprot_db}")
                        uniprot_db = None
                elif uniprot_path:
                    uniprot_db = uniprot_path
                else:
                    uniprot_db = None
                
                # Validate
                missing = []
                if not domain_1: missing.append("CD28 FASTA")
                if not domain_2: missing.append("CD3Z FASTA")
                if not cytotox_csv: missing.append("Cytotoxicity CSV")
                if not uniprot_db: missing.append("Uniprot DB")
                if missing:
                    st.error(f"Missing required files: {', '.join(missing)}")
                else:
                    # Run pipeline
                    try:
                        run_pipeline(
                            steps=steps,
                            fasta_dir=temp_dir,
                            output_dir=Path(output_dir),
                            model_dir=Path(output_dir)/"models",
                            domain_1=domain_1,
                            domain_2=domain_2,
                            cytotox_csv=cytotox_csv,
                            esm_model=model_choice,
                            uniprot_db=uniprot_db,
                            wt_car=wt_car,
                        )
                        st.success("Pipeline completed! Check the output directory for results.")
                        st.session_state.pipeline_ran = True
                        # Just list output files without creating download links
                        if Path(output_dir).exists():
                            # Use a more efficient approach - only scan specific folders for important file types
                            important_extensions = ['.png', '.csv', '.txt', '.json']
                            important_folders = ['results', 'plots', 'evaluation', 'mds']
                            
                            # Create an expander to hide the file list by default
                            with st.expander("Show Output Files (click to expand)"):
                                st.markdown("### Key Output Files:")
                                st.markdown(f"Output directory: `{output_dir}`")
                                
                                for folder in important_folders:
                                    folder_path = Path(output_dir) / folder
                                    if not folder_path.exists():
                                        continue
                                        
                                    st.markdown(f"**{folder}/**")
                                    folder_files = []
                                    
                                    # Find files with important extensions
                                    for ext in important_extensions:
                                        folder_files.extend(list(folder_path.glob(f"**/*{ext}")))
                                    
                                    # Sort and limit files per folder
                                    folder_files = sorted(folder_files)[:10]
                                    
                                    if folder_files:
                                        file_list = "\n".join([f"- {file_path.name}" for file_path in folder_files])
                                        st.text(file_list)
                                    else:
                                        st.text("No files found")
                                    
                                st.info("Open these files directly from your file system")
                        # --- Grouped, Icon-Enhanced Plot Visualization ---
                        st.markdown("## Visualize Pipeline Plots")
                        PLOT_GROUPS = [
                            ("augmentation", "🧬 Augmentation", ["augmentation/plots"]),
                            ("mds", "🗺️ MDS", ["mds", "plots"]),
                            ("mutants", "🧪 Mutants", ["mutants"]),
                            ("finetuning", "🏋️ Finetuning", ["models/high", "models/low"]),
                            ("prediction", "📈 Prediction", ["results"]),
                            ("score", "🎯 Scoring", ["plots"]),
                            ("dunnet", "⚖️ Dunnett Test", ["plots", "results"]),
                            ("visualization", "🖼️ Visualization", ["plots"]),
                        ]
                        # Sidebar filter for steps
                        step_labels = {step: label for step, label, _ in PLOT_GROUPS}
                        selected_steps = st.sidebar.multiselect(
                            "Show plots for steps:",
                            options=[step for step, _, _ in PLOT_GROUPS],
                            default=[step for step, _, _ in PLOT_GROUPS],
                            format_func=lambda x: step_labels[x]
                        )
                        for step, label, folders in PLOT_GROUPS:
                            if step not in selected_steps:
                                continue
                            found = False
                            for folder in folders:
                                plot_dir = Path(output_dir) / folder
                                if plot_dir.exists():
                                    plot_files = list(plot_dir.glob("*.png"))
                                    if plot_files:
                                        found = True
                                        st.markdown(f"### {label}")
                                        for plot_path in sorted(plot_files):
                                            st.markdown(f"**{plot_path.name}**")
                                            st.image(str(plot_path))
                            if not found:
                                st.markdown(f"#### {label}\n_No plots found for this step._")
                    except Exception as e:
                        st.error(f"Pipeline failed: {e}")
        st.info("You can rerun with different files or parameters from the sidebar.")
    elif st.session_state.pipeline_ran:
        st.success("Pipeline was previously run. You can rerun with different parameters from the sidebar.")
        st.info(f"Results are available in: {st.session_state.output_dir}")
    else:
        st.info("")

# RESULTS VISUALIZATION MODE
else:
    st.markdown('<div class="main-header">CART Project Results Dashboard</div>', unsafe_allow_html=True)
    
    # Get the output directory from session state or let user specify
    output_dir = st.sidebar.text_input("Output Directory to Visualize", value=st.session_state.output_dir)
    st.session_state.output_dir = output_dir
    
    # Get dynamic output paths based on selected output directory
    paths = get_output_paths(output_dir)
    
    # Create tabs for different result categories
    tabs = st.tabs([
        "Overview", 
        "Embeddings & MDS", 
        "Model Evaluation", 
        "Plots & Visualizations", 
        "Detailed Results"
    ])
    
    # Tab 1: Overview
    with tabs[0]:
        st.markdown('<div class="category-header">Project Overview</div>', unsafe_allow_html=True)
        
        st.write("""
        
        - **Embeddings & MDS**: Visualizations of sequence embeddings and multi-dimensional scaling
        - **Model Evaluation**: Confusion matrices and performance metrics for different model variants
        - **Plots & Visualizations**: Various plots showing cytotoxicity, mutation counts, and precision-recall curves
        - **Detailed Results**: Correlation plots and detailed performance metrics for each model fold
        """)
        
        # Summary metrics if available
        prediction_summary = read_text(paths['RESULTS_DIR'] / "prediction_summary.txt")
        if prediction_summary:
            st.markdown('<div class="subcategory-header">Prediction Summary</div>', unsafe_allow_html=True)
            st.code(prediction_summary)
    
    # Tab 2: Embeddings & MDS
    with tabs[1]:
        st.markdown('<div class="category-header">Embeddings & MDS Analysis</div>', unsafe_allow_html=True)
        
        # MDS Visualizations
        st.markdown('<div class="subcategory-header">Multi-Dimensional Scaling (MDS)</div>', unsafe_allow_html=True)
        
        mds_cols = st.columns(3)
        with mds_cols[0]:
            display_image(paths['MDS_DIR'] / "mds.png", "MDS Visualization")
        with mds_cols[1]:
            display_image(paths['MDS_DIR'] / "levenshtein.png", "Levenshtein Distance")
            # Also try with alternative filenames that might exist
            if not os.path.exists(paths['MDS_DIR'] / "levenshtein.png"):
                display_image(paths['MDS_DIR'] / "distance.png", "Distance Visualization")
        with mds_cols[2]:
            display_image(paths['MDS_DIR'] / "lengths.png", "Sequence Lengths")
            # Also try with alternative filenames that might exist
            if not os.path.exists(paths['MDS_DIR'] / "lengths.png"):
                display_image(paths['MDS_DIR'] / "length.png", "Length Visualization")
    
    # Tab 3: Model Evaluation
    with tabs[2]:
        st.markdown('<div class="category-header">Model Evaluation</div>', unsafe_allow_html=True)
        
        # Confusion Matrices
        st.markdown('<div class="subcategory-header">Confusion Matrices</div>', unsafe_allow_html=True)
        conf_cols = st.columns(3)
        
        with conf_cols[0]:
            display_image(paths['EVALUATION_DIR'] / "confusion_matrix_pretrained_embeddings_best_fold.png", 
                        "Pretrained Model")
        with conf_cols[1]:
            display_image(paths['EVALUATION_DIR'] / "confusion_matrix_finetuned_low_embeddings_best_fold.png", 
                        "Finetuned Low Model")
        with conf_cols[2]:
            display_image(paths['EVALUATION_DIR'] / "confusion_matrix_finetuned_high_embeddings_best_fold.png", 
                        "Finetuned High Model")
        
        # Precision-Recall Curves
        st.markdown('<div class="subcategory-header">Precision & Recall Metrics</div>', unsafe_allow_html=True)
        
        # Precision-Recall at K
        display_image(paths['EVALUATION_DIR'] / "precision_recall_at_k.png", "Precision-Recall at K")
        
        # Spearman Correlation
        st.markdown('<div class="subcategory-header">Spearman Correlation Comparison</div>', unsafe_allow_html=True)
        display_image(paths['EVALUATION_DIR'] / "spearman_correlation_comparison.png")
    
    # Tab 4: Plots & Visualizations
    with tabs[3]:
        st.markdown('<div class="category-header">Plots & Visualizations</div>', unsafe_allow_html=True)
        
        # Cytotoxicity and Mutation Counts
        st.markdown('<div class="subcategory-header">Mutation Analysis</div>', unsafe_allow_html=True)
        mut_cols = st.columns(2)
        
        with mut_cols[0]:
            display_image(paths['PLOTS_DIR'] / "cytotoxicity.png", "Cytotoxicity Analysis")
        with mut_cols[1]:
            display_image(paths['PLOTS_DIR'] / "mutation_counts.png", "Mutation Counts")
        
        # Recall-Precision Curves
        st.markdown('<div class="subcategory-header">Recall-Precision Curves</div>', unsafe_allow_html=True)
        
        recall_cols = st.columns(3)
        with recall_cols[0]:
            display_image(paths['PLOTS_DIR'] / "pretrained_embeddings_recall_precision.png", 
                        "Pretrained Embeddings")
        with recall_cols[1]:
            display_image(paths['PLOTS_DIR'] / "finetuned_low_embeddings_recall_precision.png", 
                        "Finetuned Low Embeddings")
        with recall_cols[2]:
            display_image(paths['PLOTS_DIR'] / "finetuned_high_embeddings_recall_precision.png", 
                        "Finetuned High Embeddings")
    
    # Tab 5: Detailed Results
    with tabs[4]:
        st.markdown('<div class="category-header">Detailed Results</div>', unsafe_allow_html=True)
        
        # Create subtabs for different model types
        model_tabs = st.tabs(["Pretrained", "Finetuned Low", "Finetuned High"])
        
        # Pretrained Model Results
        with model_tabs[0]:
            st.markdown('<div class="subcategory-header">Pretrained Model Performance</div>', unsafe_allow_html=True)
            
            # Spearman Whisker Plot
            st.markdown('<div class="subcategory-header">Spearman Correlation Summary</div>', unsafe_allow_html=True)
            display_image(paths['RESULTS_DIR'] / "pretrained_embeddings_spearman_whisker.png")
            
            # Fold Results
            st.markdown('<div class="subcategory-header">Results by Cross-Validation Fold</div>', unsafe_allow_html=True)
            
            # Allow user to select which fold to display
            pretrained_fold = st.selectbox(
                "Select fold to view", 
                options=["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
                key="pretrained_fold"
            )
            
            show_fold_results("pretrained", pretrained_fold, paths)
        
        # Finetuned Low Model Results
        with model_tabs[1]:
            st.markdown('<div class="subcategory-header">Finetuned Low Model Performance</div>', unsafe_allow_html=True)
            
            # Spearman Whisker Plot
            st.markdown('<div class="subcategory-header">Spearman Correlation Summary</div>', unsafe_allow_html=True)
            display_image(paths['RESULTS_DIR'] / "finetuned_low_embeddings_spearman_whisker.png")
            
            # Fold Results
            st.markdown('<div class="subcategory-header">Results by Cross-Validation Fold</div>', unsafe_allow_html=True)
            
            # Allow user to select which fold to display
            low_fold = st.selectbox(
                "Select fold to view", 
                options=["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
                key="low_fold"
            )
            
            show_fold_results("finetuned_low", low_fold, paths)
        
        # Finetuned High Model Results
        with model_tabs[2]:
            st.markdown('<div class="subcategory-header">Finetuned High Model Performance</div>', unsafe_allow_html=True)
            
            # Spearman Whisker Plot
            st.markdown('<div class="subcategory-header">Spearman Correlation Summary</div>', unsafe_allow_html=True)
            display_image(paths['RESULTS_DIR'] / "finetuned_high_embeddings_spearman_whisker.png")
            
            # Fold Results
            st.markdown('<div class="subcategory-header">Results by Cross-Validation Fold</div>', unsafe_allow_html=True)
            
            # Allow user to select which fold to display
            high_fold = st.selectbox(
                "Select fold to view", 
                options=["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
                key="high_fold"
            )
            
            show_fold_results("finetuned_high", high_fold, paths)
    
    # Add a footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: gray; font-size: 0.8rem;">
            CART Project Results Dashboard | Created with Streamlit
        </div>
        """, 
        unsafe_allow_html=True
    )