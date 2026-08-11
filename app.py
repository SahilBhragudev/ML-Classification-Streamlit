import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score,roc_auc_score,precision_score,
    recall_score,f1_score,matthews_corrcoef,confusion_matrix,
    classification_report
)

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree Classifier": "model/decision_tree.pkl",
    "K- Nearest Neighbour": "model/knn.pkl",
    "Naive Bayes Classifier ": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_resource
def load_model_columns():
    try:
        return joblib.load("model/model_columns.pkl")
    except FileNotFoundError:
        st.error("model/model_columns.pkl not found. Make sure the model/ folder is in your repo.")
        st.stop()

st.set_page_config(page_title="Streamlit App",layout="wide")
st.title("Streamlit Assignment App")

st.sidebar.header("Controls")

model_choice = st.sidebar.selectbox("Choose a Model ",
        ["Logistic Regression" , "Decision Tree Classifier" , "K- Nearest Neighbour",
         "Naive Bayes Classifier ","Random Forest"])

st.sidebar.write(f"You selected : {model_choice}")

st.header("1. Upload a csv")
uploaded_file = st.file_uploader("Upload any csv file to preview it",type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data: ")
    st.dataframe(df.head())
else:
    st.info("No file uploaded yet - using dummy data below instead")
    df = pd.DataFrame(
        np.random.randn(20,4),
        columns=["feature_1","feature_2","feature_3","feature_4"]
    )
    df["target"] = np.random.choice([0,1],size=20)
    st.dataframe(df.head())


st.header("2. Evaluation Metrics")

if uploaded_file is not None and "Churn" in df.columns:
    model_columns = load_model_columns()
    model = load_model(MODEL_FILES[model_choice])

    X_uploaded = df.drop(columns=["Churn"])
    y_true = df["Churn"]

    X_uploaded = X_uploaded[model_columns]

    y_pred = model.predict(X_uploaded)
    y_proba = model.predict_proba(X_uploaded)[:,1]

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("AUC Score", f"{auc:.3f}")
    col3.metric("F1 Score", f"{f1:.3f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Precision", f"{prec:.3f}")
    col5.metric("Recall", f"{rec:.3f}")
    col6.metric("MCC", f"{mcc:.3f}")
else:
    st.warning("Upload test_data.csv (with a 'Churn' column) to see real metrics.")

st.header("3. Confusion Matrix")

if uploaded_file is not None and "Churn" in df.columns:
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred: No Churn", "Pred: Churn"],
                yticklabels=["True: No Churn", "True: Churn"])
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    st.pyplot(fig)

    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, target_names=["No Churn", "Churn"], output_dict=True)
    accuracy = report.pop("accuracy")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.3f}"))
    st.write(f"**Overall accuracy:** {accuracy:.3f}")
else:
    st.info("Upload test_data.csv to see the confusion matrix.")


st.header("4. Button")
if st.button("Click me"):
    st.success(f"Button clicked! Model selected was {model_choice}")
        
