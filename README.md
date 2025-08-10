# Online Payments Fraud Detection

This project trains a model to detect fraudulent transactions (based on the PaySim-like dataset).
The code will try to download a public dataset CSV from GitHub. If you already have the CSV, place it in `data/transactions.csv`.

## Contents
- `train.py` - Downloads dataset (if not present), preprocesses, trains models, and saves best model to `model.joblib`.
- `app.py` - Flask app exposing a `/predict` endpoint and a simple form to test predictions.
- `requirements.txt` - Python dependencies.
- `templates/` - simple HTML for demo UI.
- `data/` - (empty) place to put dataset if you have it locally.

## How to run
1. Create and activate a virtualenv
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. (Optional) If you have the dataset CSV, place it at `data/transactions.csv`.
   Otherwise `train.py` will attempt to download a public CSV from GitHub automatically.
3. Train model:
   ```bash
   python train.py
   ```
   This will create `model.joblib` and `preprocessor.joblib`.
4. Run the Flask app:
   ```bash
   python app.py
   ```
   Open http://127.0.0.1:5000 and try the demo.

## Notes
- The training script uses SMOTE to handle class imbalance and evaluates models by F1-score on the fraud class.
- If the automatic dataset download fails, download the dataset manually from a PaySim source and save as `data/transactions.csv`.

## License
MIT
