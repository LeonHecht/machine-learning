# Scene Classification with Images

This repository's goal is to perform **scene classification** using pre‑computed image features and several supervised learning models (SVM and XGBoost).

## Repository Structure

- `ejercicio_1.ipynb` – Notebook for the first part of the assignment (e.g. exploratory analysis, baseline models, or feature handling).
- `ejercicio_2.ipynb` – Notebook for the second part of the assignment (e.g. alternative representations, PCA, model comparison, or evaluation).
- `svm_model.joblib` – Trained SVM model on the base feature representation.
- `svm_model_encoded.joblib` – Trained SVM model on an encoded / engineered feature space.
- `svm_model_pca.joblib` – Trained SVM model on PCA‑reduced features.
- `svm_model_rbc.joblib` – Trained SVM model with a region‑based coding representation (or similar variant, depending on the assignment).
- `xgboost_model.json` – Trained XGBoost model on the base feature representation.
- `xgboost_model_encoded.json` – Trained XGBoost model on the encoded / engineered feature space.
- `xgboost_model_pca.json` – Trained XGBoost model on PCA‑reduced features.
- `xgboost_model_rbc.json` – Trained XGBoost model with region‑based coding (or similar variant).
- `data_tarea/` – Dataset with pre‑computed features for each image.
  - `train/` – Training split.
  - `test/` – Test split.
  - Each split has one subfolder per scene class (`bedroom/`, `coast/`, `forest/`, `highway/`, `industrial/`, `insidecity/`, `kitchen/`, `livingroom/`, `mountain/`, `office/`, `opencountry/`, `store/`, `street/`, `suburb/`, `tallbuilding/`).
  - Inside each class folder, the `features/` directory contains `.npy` files with the pre‑computed features of each image.

## Requirements

The notebooks use standard Python ML and data libraries. A typical environment will include:

- Python 3.8+
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `xgboost`
- `joblib`

You can install these with:

```bash
pip install numpy pandas scikit-learn matplotlib xgboost joblib
```

(If your course provides an environment file, prefer using that instead.)

## How to Use

1. **Clone the repository**

   ```bash
   git clone https://github.com/<tu-usuario>/machine-learning.git
   cd machine-learning/Tarea4
   ```

2. **Create and activate a virtual environment (optional but recommended)**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Linux/macOS
   # .venv\\Scripts\\activate  # En Windows
   ```

3. **Install dependencies** (see Requirements section).

4. **Open the project in VS Code or Jupyter** and run the notebooks:

   - Start Jupyter:

     ```bash
     jupyter notebook
     ```

   - Open `ejercicio_1.ipynb` and `ejercicio_2.ipynb` and run the cells in order.

The notebooks assume the folder `data_tarea/` is located in the same directory as the notebooks and that the `.joblib` / `.json` model files are accessible there as well.

## Models

The saved models (`.joblib` and `.json`) allow you to:

- Reload the trained classifiers without retraining.
- Evaluate them on the test split.
- Compare different feature representations (raw, encoded, PCA, RBC, etc.).

Example of loading a saved SVM model in Python:

```python
import joblib

model = joblib.load("svm_model.joblib")
y_pred = model.predict(X_test)
```

For XGBoost models saved as JSON, you can use:

```python
from xgboost import XGBClassifier

model = XGBClassifier()
model.load_model("xgboost_model.json")
y_pred = model.predict(X_test)
```

(Adjust the paths according to where you run the code.)

## Notes

- This repository is mainly intended for educational purposes as part of a university course.
- If you change the folder structure or add new experiments, update this `README.md` accordingly.
