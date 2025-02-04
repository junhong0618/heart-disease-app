import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import io

# Load dataset
def load_data():
    df = pd.read_csv("data/heart_disease_cleaned.csv")
    return df

df = load_data()

# Streamlit app layout
st.title("💓 Heart Disease EDA Dashboard")

# Manually define columns
numeric_columns = ['age', 'cigsPerDay', 'totChol', 'sysBP','diaBP','BMI','heartRate','glucose']
categorical_columns = ['Gender', 'education', 'currentSmoker', 'BPMeds','prevalentStroke','prevalentHyp','diabetes','Heart_ stroke']

# Sidebar Filters
st.sidebar.header("Filters")
selected_num_col = st.sidebar.selectbox("Select Numerical Feature", numeric_columns)
selected_cat_col = st.sidebar.selectbox("Select Categorical Feature", categorical_columns)

# Display dataset info
if st.sidebar.checkbox("Show Dataset Info"):
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)


if st.sidebar.checkbox("Show Summary Statistics"):
    st.write(df.describe())

if st.sidebar.checkbox("Show Missing Values"):
    st.write(df.isnull().sum())

# Visualizing distributions of numerical columns
st.subheader("Feature Distributions")
fig, ax = plt.subplots()
df[selected_num_col].hist(ax=ax, bins=20, edgecolor='black')
st.pyplot(fig)

# Correlation matrix
st.subheader("Correlation Matrix")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
st.pyplot(fig)

# Countplot for categorical features
st.subheader(f"Distribution of {selected_cat_col}")
fig, ax = plt.subplots()
sns.countplot(x=df[selected_cat_col], ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Boxplot for numerical features
st.subheader(f"Boxplot of {selected_num_col}")
fig, ax = plt.subplots()
sns.boxplot(y=df[selected_num_col], ax=ax)
st.pyplot(fig)

# Violin plot for numerical features
st.subheader(f"Violin Plot of {selected_num_col}")
fig, ax = plt.subplots()
sns.violinplot(y=df[selected_num_col], ax=ax)
st.pyplot(fig)

# Model Evaluation
st.subheader("Model Performance")
model_names = ["random_forest", "gradient_boosting", "logistic_regression", "support_vector_machine", "k_nearest_neighbors"]
model_results = []

for model_name in model_names:
    model_path = f"app/{model_name}.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        X = df.drop(columns=["Heart_ stroke"])
        y = df["Heart_ stroke"]
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        
        model_results.append({
            "Model": model_name.replace("_", " ").title(),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

if model_results:
    st.table(pd.DataFrame(model_results))
else:
    st.write("No models found. Please ensure trained models are saved in the 'app' directory.")