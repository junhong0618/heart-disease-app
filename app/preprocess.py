import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# Use absolute path
DATA_PATH = os.path.abspath("data/heart_disease.csv")

def preprocess_data():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Drop rows with too many missing values
    df_drop = df.dropna()

    label_encoder = LabelEncoder()
    df_drop['Heart_ stroke'] = label_encoder.fit_transform(df_drop['Heart_ stroke'])
    df_drop['education'] = label_encoder.fit_transform(df_drop['education'])
    df_drop['Gender'] = label_encoder.fit_transform(df_drop['Gender'])
    df_drop['prevalentStroke'] = label_encoder.fit_transform(df_drop['prevalentStroke'])

    # Fill missing values with median
    for col in ["totChol", "BMI", "glucose", "cigsPerDay", "BPMeds"]:
        df_drop[col].fillna(df_drop[col].median(), inplace=True)

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ["age", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
    df_drop[numerical_features] = scaler.fit_transform(df_drop[numerical_features])

    # Save preprocessed data & scaler
    joblib.dump(scaler, "app/scaler.pkl")
    df_drop.to_csv(os.path.abspath("data/heart_disease_cleaned.csv"), index=False)

    return df_drop

if __name__ == "__main__":
    df = preprocess_data()
    print("Preprocessing completed and data saved!")