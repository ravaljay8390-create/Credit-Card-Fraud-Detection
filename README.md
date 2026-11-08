# Credit Card Fraud Detection

A beginner-friendly ML + DL capstone project that predicts whether a credit card transaction is fraudulent, using the Kaggle Credit Card Fraud Detection dataset (284,807 transactions, 0.17% fraud).

## Folder structure

```
fraud_detection/
├── notebook/
│   ├── Credit_Card_Fraud_Detection.ipynb   <- Jupyter notebook (EDA + ML + DL model training)
│   └── data/creditcard.csv                 <- dataset (used by the notebook)
├── models/                                 <- saved model files (created by the notebook)
│   ├── fraud_model.pkl                     <- final model (Random Forest)
│   ├── amount_scaler.pkl
│   ├── time_scaler.pkl
│   ├── columns.pkl
│   └── sample_transactions.csv             <- sample transactions used by the app's "Load Sample" tab
├── app/
│   └── app.py                              <- Streamlit web app (3 input modes)
├── docs/
│   └── Credit_Card_Fraud_Detection_Report.docx   <- project report
└── requirements.txt
```

## How to run it

### 1. Install the requirements
```bash
pip install -r requirements.txt
```
Note: this installs `tensorflow-cpu`, which is a fairly large package (~200-400MB) - it may take a few minutes.

### 2. Run the notebook (optional - the model is already trained and saved)
Open `notebook/Credit_Card_Fraud_Detection.ipynb` in Jupyter and run all cells top to bottom.
This will regenerate the model files inside `models/`. It trains a Random Forest on ~228K rows
and a small Neural Network, so it may take a few minutes to fully run.

```bash
jupyter notebook notebook/Credit_Card_Fraud_Detection.ipynb
```

### 3. Run the Streamlit app
Run this command from the `fraud_detection` project root folder (so it can find the `models/` folder):

```bash
streamlit run app/app.py
```

The app has 3 tabs:
- **Upload CSV** - upload a file with transactions (same columns as the original dataset) and get predictions for all rows.
- **Simple Form** - manually enter just Amount and Time (the other 28 PCA columns are auto-filled with typical values).
- **Load Sample Transaction** - instantly load a real genuine or fraud transaction from the test set and see if the model gets it right.

## ⚠️ Important note before pushing to GitHub

The dataset file (`notebook/data/creditcard.csv`) is about **99 MB**. GitHub blocks any file
over 100 MB, and even at 99 MB it makes your repo slow to clone and push. There are two good
options:

**Option A (recommended, simplest): don't push the CSV at all.**
Add this to your `.gitignore`:
```
notebook/data/creditcard.csv
```
Then in your README (or a note to whoever reviews it), mention that the dataset must be
downloaded separately from Kaggle and placed in `notebook/data/` before running the notebook.
The Streamlit app itself doesn't need the raw CSV to run (it only needs the files in `models/`,
which are small), so this won't break your deployment.

**Option B: use Git LFS** (Large File Storage) if your instructor specifically wants the dataset
committed. This is more setup than most college submissions need, so Option A is usually enough.

## Notes
- The final model is a **Random Forest** (Precision ~0.82, Recall ~0.81, F1 ~0.81, ROC-AUC ~0.99), chosen over Logistic Regression and a Neural Network because it had the best balance between catching fraud and not over-flagging genuine transactions.
- We used **class weighting** to handle the severe class imbalance (0.17% fraud), instead of oversampling techniques like SMOTE.
- Dataset source: Kaggle - Credit Card Fraud Detection: https://www.kaggle.com/datasets/arslanali4343/credit-card-cheating-detection-cccd
