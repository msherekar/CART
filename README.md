# CAR-T Cell Activity Prediction System

A system for predicting CAR-T cell activity by fine-tuning ESM-2 protein language models with sequence augmentation. This project implements the methodology from *[Enhancing CAR-T cell activity prediction via fine-tuning protein language models with generated CAR sequences](https://www.biorxiv.org/content/10.1101/2025.03.27.645831v1.full)*. The system provides both research-grade experiment tools and a user-friendly web interface.

**Motivation**: The original paper proposing this framework did not release any source code, making this a challenging and rewarding exercise in protein machine learning and software engineering. It was also an opportunity to design an end-to-end AI system that could help immunologists apply machine learning to their own CAR constructs.

**Problem**: CAR (Chimeric Antigen Receptor) is a synthetic protein introduced into T cells to direct their activity against cancer. As per the paper, designing better CARs is limited by two major challenges:

  **1.)** The artificial nature of CARs means there's a lack of evolutionary information. This means standard pretraining approaches like evotuning don't apply.
  
  **2.)** There are no public large-scale CAR sequence databases due to proprietary restrictions, limiting data availability for machine learning.


**Solution**: To tackle these issues, Kei Yoshida et al. proposed a computational framework (published in April 2025) that uses protein language models (PLMs) fine-tuned with synthetically generated CAR sequences to predict cytotoxicity outcomes of CAR-T cells. This repository recreates that unpublished codebase as a reproducible AI system.

**Objective**: Create a codebase to implement paper methodology so that **any researcher can plug in their own CAR sequences and cytotoxicity data** to train and evaluate a prediction model tailored to their CAR-T constructs. The objective was not to reproduce the exact results from the paper because the authors have not provided the cytotoxicity data.

**Key Contributions**:

**1.)** Implemented a full training and evaluation pipeline using ESM-2 PLMs with support for model sizes ranging from 8M to 3B parameters.

**2.)** Developed a web-based interface to allow scientists to experiment with different CAR sequences, fine-tune models, assess the effect of training parameters, and visualize sequence diversity.

**3.)** Integrated the ability to run experiments from a module or a Docker container (with HMMER omitted due to >100GB size).

**Utility**: This system will enable researchers to develop personalized CAR constructs by providing a prediction mechanism for new mutations. The only inputs required are a wild-type CAR sequence, a set of mutants, their experimental cytotoxicity values, and access to a local UniProt(Trembl) database.

**Limitations**: Due to the size of HMMER UniProt database, full containerized end-to-end deployment is not included. I have provided the necessary code (see later) for dockerizing the model and deploying it on dockerhub. Users must locally manage this part of the pipeline.

**Learning Outcomes**: Through this project, I learnt masked language modeling (MLM), protein representation learning, model evaluation metrics (e.g., Spearman, Recall@K), and the role of sequence diversity and model size in fine-tuning performance.

## Original Paper

- **Title**: Enhancing CAR-T cell activity prediction via fine-tuning protein language models with generated CAR sequences
- **Authors**: Kei Yoshida, Shoji Hisada, Ryoichi Takase, Atsushi Okuma, Yoshihito Ishida, Taketo Kawara, Takuya Miura-Yamashita, Daisuke Ito
- **Published**: Biorxiv, 04/01/2025
- **Paper Link**: https://doi.org/10.1101/2025.03.27.645831
- **Original Code**: ["Not available"]

**Abstract Summary**: The original work developed a computational framework to predict CAR-T cell activity by fine-tuning ESM-2 with CAR sequences generated using sequence augmentation. The study addressed the challenge of applying protein language models to artificial CAR constructs by creating training data through recombining homologous CAR domains. Experimental validation showed that fine-tuned ESM-2 significantly improves prediction performance, with training parameters like sequence diversity, training steps, and model size substantially influencing results.

### Training Parameter Analysis

The framework enables systematic analysis of key training parameters identified in the original paper:
- **Sequence Diversity**: High vs. low diversity augmented training sets
- **Model Size**: Performance across ESM-2 variants (8M to 3B parameters)  
- **Training Steps**: Optimization of training duration and convergence
- **Cross-validation**: Robust evaluation across multiple folds

## Dataset

- **Dataset Type**: CAR-T cell cytotoxicity data with experimental validation
- **Input Sequences**: 
  - CAR constructs (including scFv, hinge, transmembrane, and signaling domains)
  - CD28 costimulatory domain sequences (FASTA format)
  - CD3ζ signaling domain sequences (FASTA format)
  - Wild-type CAR reference sequences (FASTA format)
- **Augmentation Strategy**: 
  - Homologous domain recombination to generate diverse CAR variants
  - High and low diversity sequence sets for training parameter analysis
  - UniProt database integration for domain homology identification
- **Experimental Data**: Cytotoxicity measurements (CSV format) for model validation
- **Challenge Addressed**: Overcoming sparse CAR sequence databases through computational augmentation
- **Task**: Regression/ranking for CAR-T cell activity prediction and optimization

## Model Architecture

- **Base Model**: ESM-2 (Evolutionary Scale Modeling) protein language model family
- **Supported Models**: 
  - facebook/esm2_t6_8M_UR50D (8M params - fastest, testing)
  - facebook/esm2_t12_35M_UR50D (35M params - **recommended balance**)
  - facebook/esm2_t30_150M_UR50D (150M params - better accuracy)
  - facebook/esm2_t33_650M_UR50D (650M params - high performance)
  - facebook/esm2_t36_3B_UR50D (3B params - best performance)
- **Fine-tuning Strategy**: Task-specific adaptation for CAR-T activity prediction
- **Key Innovation**: Addresses the challenge of applying PLMs to artificial CAR sequences
- **Training Approach**: 
  - Sequence augmentation to generate sufficient training data
  - Cross-validation for robust performance evaluation
  - Systematic analysis of model size effects on prediction accuracy
- **Task**: Regression for predicting CAR-T cell cytotoxicity with ranking capabilities

## Installation

### Package Installation (Recommended for Pipeline)
```bash
# Clone the repository
git clone https://github.com/msherekar/CART-Project.git
cd CART-Project

# Install the package
pip install -e .

# Virtual Environment
conda install environment.yml
```
## Usage

### Streamlit Web Application

Launch the interactive web interface for easy pipeline usage:

```bash
cd CAR-Project
streamlit run app.py
```

The web app provides:
- File upload interface for FASTA and CSV files
- Interactive parameter selection
- Real-time pipeline execution
- Visualization of results
- Model comparison tools

### Experiment Runner (Recommended for Batch Processing)

For comprehensive model training and evaluation with cross-validation:

```bash
python run_experiments.py --experiment all
```

#### Experiment Options
- `--experiment`: Type of experiment (`finetune`, `evaluate`, or `all`)
- `--high_fasta`: Path to high diversity FASTA file
- `--low_fasta`: Path to low diversity FASTA file
- `--output_dir`: Directory to save model checkpoints (default: checkpoints)
- `--plots_dir`: Directory to save plots (default: plots)
- `--n_folds`: Number of cross-validation folds
- `--batch_size`: Batch size for training
- `--max_length`: Maximum sequence length
- `--device`: Device to run on (`auto`, `cuda`, `mps`, or `cpu`)

#### Example Usage
```bash
# Run full experiment with custom settings
python run_experiments.py --experiment all \
  --high_fasta homologs/high_diversity.fasta \
  --low_fasta homologs/low_diversity.fasta \
  --output_dir checkpoints \
  --plots_dir plots \
  --n_folds 5

# Run fine-tuning only
python run_experiments.py --experiment finetune \
  --batch_size 16 \
  --max_length 512
```

### Command Line Pipeline

For individual pipeline components, use the `cart` command:

```bash
cart run \
  --wt-cd28 /path/to/cd28.fasta \
  --wt-cd3z /path/to/cd3z.fasta \
  --cytotox-csv /path/to/cytotox.csv \
  --esm-model facebook/esm2_t12_35M_UR50D \
  --uniprot-db /path/to/uniprot_trembl.fasta \
  --wt-car /path/to/car.fasta
```

#### Required Parameters
- `--wt-cd28`: Path to CD28 sequence in FASTA format
- `--wt-cd3z`: Path to CD3Z sequence in FASTA format  
- `--cytotox-csv`: Path to cytotoxicity CSV data
- `--esm-model`: ESM model identifier
- `--uniprot-db`: Path to UniProt database FASTA file

#### Optional Parameters
- `--wt-car`: Path to CAR sequence FASTA file
- `--steps`: Specify which pipeline steps to run (default: all)
- `--fasta-dir`: Directory containing FASTA files (default: fasta)
- `--output-dir`: Directory for output files (default: output)
- `--model-dir`: Directory for model files (default: output/models)

### Experiment Tracking

The system supports multiple experiment tracking platforms:

#### Weights & Biases
```bash
python run_experiments.py --experiment all \
  --use_wandb \
  --wandb_project esm2-cart
```

#### MLflow
```bash
python run_experiments.py --experiment all \
  --use_mlflow \
  --mlflow_experiment esm2-cart
```

### ESM Model Selection

| Model | Parameters | Memory | Description |
|-------|------------|---------|-------------|
| facebook/esm2_t6_8M_UR50D | 8M | Low | Fastest - good for testing |
| facebook/esm2_t12_35M_UR50D | 35M | Medium | **Recommended** - best balance |
| facebook/esm2_t30_150M_UR50D | 150M | Medium | Better accuracy |
| facebook/esm2_t33_650M_UR50D | 650M | High | High performance |
| facebook/esm2_t36_3B_UR50D | 3B | Very High | Best performance |

### Pipeline Steps

Run specific pipeline steps:
```bash
cart run \
  --steps augmentation mds mutants \
  --wt-cd28 /path/to/cd28.fasta \
  --wt-cd3z /path/to/cd3z.fasta \
  --cytotox-csv /path/to/cytotox.csv \
  --esm-model facebook/esm2_t12_35M_UR50D
```

Available steps: `augmentation`, `mds`, `mutants`, `finetuning`, `embeddings`, `prediction`, `evaluation`, `score`

## Experiments

### Training Details
- **Hardware**: [Apple Mac M1, 16GB]
- **Training Time**: [Hours/days]
- **Batch Size**: Configurable (default varies by model size)
- **Learning Rate**: [Rate and schedule]
- **Optimizer**: [Adam, AdamW, etc.]
- **Cross-Validation**: K-fold validation (configurable folds)
- **Early Stopping**: Based on validation loss
- **Checkpointing**: Automatic model saving at best performance

### Hyperparameters
Key hyperparameters used (can reference config files):
- Learning rate: [value]
- Batch size: [value]
- Number of epochs: [value]
- [Other relevant hyperparameters]

## Results Analysis

## Features

### Core Pipeline
- **Sequence Augmentation**: Generate diverse CAR variants through homologous domain recombination
- **MDS Analysis**: Multidimensional scaling for sequence space visualization and diversity assessment
- **Mutant Generation**: Create systematic CAR variants for improved cytotoxicity
- **Fine-tuning**: Adapt ESM-2 models for CAR-T activity prediction with cross-validation
- **Embedding Extraction**: Generate protein sequence representations for downstream analysis
- **Prediction & Evaluation**: Assess model performance using Spearman correlation and Recall@K metrics

### Experiment Management
- **Cross-Validation**: K-fold validation for robust model evaluation
- **Parameter Analysis**: Systematic study of sequence diversity, model size, and training steps
- **Experiment Tracking**: Integration with Weights & Biases and MLflow for reproducibility
- **Comprehensive Metrics**: Spearman correlation, Recall@K, Precision@K as per original paper
- **Automated Plotting**: Training curves, correlation plots, model comparisons, and parameter analysis
- **Checkpoint Management**: Automatic model saving and loading with metadata

### Web Interface
- **Interactive Dashboard**: User-friendly Streamlit application for non-technical users
- **File Management**: Easy upload and management of FASTA/CSV files
- **Parameter Tuning**: Interactive model and hyperparameter selection
- **Real-time Visualization**: Live plots and analysis during pipeline execution
- **Results Export**: Download predictions and analysis results for further analysis

### Model Flexibility
- **Multiple ESM-2 Variants**: Support for models from 8M to 3B parameters with performance analysis
- **Scalable Deployment**: From laptop testing to GPU cluster training
- **Modular Design**: Run individual pipeline steps as needed for targeted analysis
- **Docker Support**: Containerized deployment for reproducible research environments

## Files and Notebooks

- `streamlit_app/app.py`: Main Streamlit web application
- `run_experiments.py`: Main experiment entry point with full training pipeline
- `src/data/`: Data processing scripts
- `src/modeling/`: Model training and evaluation modules
- `src/visualization/`: Plotting utilities and visualization tools
- `notebooks/exploratory_analysis.ipynb`: Data exploration and visualization
- `notebooks/results_analysis.ipynb`: Detailed results comparison with original paper
- `scripts/train.py`: Main training script
- `scripts/evaluate.py`: Evaluation script
- `configs/`: Configuration files for different experiments
- `setup.py`: Package installation configuration

## Dependencies

Key dependencies (see `requirements.txt` for full list):
- Python >= 3.8
- PyTorch >= 1.12
- Transformers >= 4.20
- Streamlit >= 1.20
- ESM (Facebook's protein language model)
- BioPython
- Pandas, NumPy, Scikit-learn
- Plotly (for interactive visualizations)
- Weights & Biases (optional, for experiment tracking)
- MLflow (optional, for experiment tracking)

## Deployment

### Docker Support
The system supports containerized deployment for reproducible environments:

```bash
# Build Docker image
docker build -t cart-project .

# Run in container
docker run -v $(pwd)/data:/app/data cart-project python run_experiments.py --experiment all
```

### Production Deployment
- The main entry point `run_experiments.py` handles all path resolution relative to the project root
- Suitable for cloud deployment (AWS, GCP, Azure)
- Supports both CPU and GPU environments
- Environment variables can be used for configuration

## Reproducibility

To ensure reproducibility:
- All random seeds are fixed across experiments
- Exact package versions specified in requirements.txt
- Training configurations saved in config files and experiment logs
- Model checkpoints include hyperparameters and training metadata
- Cross-validation ensures robust performance estimates
- Experiment tracking provides full audit trail

## Citation

If you use this code, please cite both the original paper and this reproduction:

```bibtex
@article{original_paper,
  title={[Enhancing CAR-T cell activity prediction via fine-tuning protein language models with generated CAR sequences]},
  author={[Kei Yoshida, Shoji Hisada, Ryoichi Takase, Atsushi Okuma, Yoshihito Ishida, Taketo Kawara, Takuya Miura-Yamashita, Daisuke Ito]},
  journal={[Biorxiv]},
  year={[2025]},
  note={Computational framework for predicting CAR-T cell activity using fine-tuned ESM-2 with sequence augmentation}
}

@misc{my_reproduction,
  title={CART-Project: CAR-T Cell Activity Prediction System},
  author={Mukul Sherekar},
  year={2025},
  howpublished={\url{https://github.com/msherekar/CART-Project}},
  note={Implementation of computational framework for CAR-T sequence optimization using fine-tuned protein language models}
}
```

## License

MIT

## Acknowledgments

- Original authors of the CAR-T sequence optimization paper for the innovative methodology

## Contact

[Mukul Sherekar] - [mukulsherekar@gmail.com]

Project Link: [https://github.com/msherekar/CART-Project]