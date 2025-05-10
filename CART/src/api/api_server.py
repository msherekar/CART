import logging
from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path
import subprocess
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/run_pipeline/")
async def run_pipeline(
    wt_cd28: UploadFile = File(...),
    wt_cd3z: UploadFile = File(...),
    cytotox_csv: UploadFile = File(...),
    esm_model: str = Form(...),
):
    logger.info("Received request to run_pipeline")
    
    # Log the names of the uploaded files
    logger.info(f"wt_cd28: {wt_cd28.filename}")
    logger.info(f"wt_cd3z: {wt_cd3z.filename}")
    logger.info(f"cytotox_csv: {cytotox_csv.filename}")
    logger.info(f"esm_model: {esm_model}")

    # Create a temp directory to store uploaded files
    temp_dir = Path("/tmp/cart_inputs")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded files
    wt_cd28_path = temp_dir / "wt_cd28.fasta"
    wt_cd3z_path = temp_dir / "wt_cd3z.fasta"
    uniprot_db_path = temp_dir / "uniprot_trembl.fasta"
    cytotox_csv_path = temp_dir / "CAR_mutants_cytox.csv"

    with open(wt_cd28_path, "wb") as f:
        f.write(await wt_cd28.read())
    with open(wt_cd3z_path, "wb") as f:
        f.write(await wt_cd3z.read())
    with open(cytotox_csv_path, "wb") as f:
        f.write(await cytotox_csv.read())

     # Hardcoded uniprot db inside container
    uniprot_db_path = Path("/app/CART/fasta/uniprot_trembl.fasta")


    # Log that files have been saved
    logger.info("Uploaded files saved successfully.")

    # Create output folders if not exist
    output_dir = Path("/tmp/cart_outputs")
    model_dir = output_dir / "models"
    output_dir.mkdir(exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Run the CART pipeline via subprocess
    cmd = [
        "cart", "run",
        "--steps", "all",
        "--fasta-dir", str(temp_dir),
        "--output-dir", str(output_dir),
        "--model-dir", str(model_dir),
        "--wt-cd28", str(wt_cd28_path),
        "--wt-cd3z", str(wt_cd3z_path),
        "--cytotox-csv", str(cytotox_csv_path),
        "--esm-model", esm_model
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Pipeline completed successfully.")
        return {"status": "Pipeline completed", "stdout": result.stdout}
    except subprocess.CalledProcessError as e:
        logger.error(f"Pipeline failed: {e.stderr}")
        return {"status": "Pipeline failed", "error": e.stderr}

