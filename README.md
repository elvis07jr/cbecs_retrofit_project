# CBECS Heating Retrofit Potential Analyzer

This project implements a machine learning pipeline to identify commercial buildings with high heating energy retrofit potential based on CBECS 2018 data characteristics. It includes data processing, feature engineering, model training, and economic analysis.

## Project Structure

- `config.py`: Configuration constants (data paths, economic parameters, model settings).
- `data_processing.py`: Handles data loading and initial filtering.
- `feature_processing.py`: Manages feature engineering, economic calculations, and data preprocessing.
- `model.py`: Contains the machine learning model training, evaluation, and hyperparameter tuning logic.
- `pipeline.py`: The main script to run the full ML pipeline and save the trained model.
- `streamlit_app.py`: A web application built with Streamlit for interactive predictions.
- `requirements.txt`: Lists all Python package dependencies.
- `your_cbecs_data.csv` (or `.xlsx`): Placeholder for your actual CBECS 2018 dataset.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/cbecs-heating-retrofit.git](https://github.com/your-username/cbecs-heating-retrofit.git)
    cd cbecs-heating-retrofit
    ```
2.  **Place your data:**
    Download the CBECS 2018 data (or a relevant subset) and place it in the project's root directory, naming it `your_cbecs_data.csv` (or `.xlsx`).
3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the ML Pipeline

To train the models and generate the `heating_retrofit_model.pkl` artifact:

```bash
python pipeline.py