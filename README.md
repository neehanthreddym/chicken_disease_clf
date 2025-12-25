# Chicken Disease Classifier

A CNN-based image classification system for identifying chicken diseases using deep learning and computer vision techniques. This project uses **DVC (Data Version Control)** for pipeline management and reproducibility.

## Features
- **Deep Learning**: Uses a CNN (VGG16 based) for image classification.
- **DVC Pipeline**: Automated pipeline for data ingestion, model definition, training, and evaluation.
- **Web Interface**: User-friendly web app built with FastAPI and Bootstrap for easy interaction.
- **Reproducibility**: Global random seed (42) and GPU memory growth configuration for consistent results.
- **Experiment Tracking**: Tracks parameters (epochs, batch size, etc.) and metrics (accuracy, loss).


## System Configuration

The pipeline has been tested on the following configuration:
- **OS**: macOS 26.1
- **Model**: MacBook Pro
- **Chip**: Apple M3 Pro
- **Cores**: 11 (5 performance and 6 efficiency)
- **Memory**: 18 GB

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/neehanthreddym/chicken_disease_clf.git
   cd chicken_disease_clf
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage (DVC Pipeline)

To run the entire machine learning pipeline (Data Ingestion -> Model Definition -> Training -> Evaluation):

```bash
dvc repro
```

This will check for changes in dependencies and only run the necessary stages.

### Pipeline Stages
1. **Data Ingestion** (`stage01_data_ingestion.py`): Downloads and extracts the dataset.
2. **Model Definition** (`stage02_model_definition.py`): Prepares the VGG16 base model.
3. **Model Training** (`stage03_training.py`): Trains the model with augmented data.
4. **Model Evaluation** (`stage04_evaluation.py`): Evaluates the trained model and saves scores.

## Web Application

The project includes a FastAPI-based web interface to easily classify images.

1. Start the application:
   ```bash
   python app.py
   ```
2. Open your browser and navigate to `http://localhost:8000`.

## Reproducibility
- **Random Seed**: A global seed of `42` is set for Python, NumPy, and TensorFlow to ensure reproducible training runs.
- **GPU Config**: TensorFlow GPU memory growth is enabled to prevent allocation errors.


## Development Workflow

Follow these steps when making changes to the project:

1. **Update `config.yaml`**: Modify system configuration settings (paths, URLs).
2. **Update `secrets.yaml` (Optional)**: Add sensitive credentials like API keys.
3. **Update `params.yaml`**: Adjust parameters for model training/testing (Epochs, Batch Size, etc.).
4. **Update the Entity**: Modify data entities (dataclasses) in `src/entity` for accurate input/output.
5. **Update the Configuration Manager**: Adjust `src/config/configuration.py` to handle new configs.
6. **Update the Components**: Modify or add components in `src/components`.
7. **Update the Pipeline**: Update the processing steps in `src/pipeline`.
8. **Update `main.py`**: Modify the main script if necessary.
9. **Update `dvc.yaml`**: Update stage dependencies and outputs if the workflow changes.