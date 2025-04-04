import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Load your dataset
df = pd.read_csv("Career_Mapping_Dataset_with_New_Roles.csv")  # Adjust if needed

# Prepare features and labels
X = df.drop(columns=["Role"])
y = df["Role"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# Save everything again
model.save_model("xgb_career_model.json")
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Model and encoder files regenerated successfully.")
