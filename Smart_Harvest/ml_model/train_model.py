
# Crop Recommendation System - ML Training Script

# Command: python ml_model/train_model.py


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Step 1: Load or Create Dataset 

csv_path = os.path.join(os.path.dirname(__file__), "crop_dataset.csv")

print("Step 1: Loading dataset...")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"   Loaded existing CSV: {len(df)} rows, {df['label'].nunique()} crops")
else:
    print("   No CSV found — generating synthetic dataset...")
    crop_data = {
        "rice":        {"N":(60,100), "P":(30,60),  "K":(30,60),  "temp":(20,35), "hum":(80,95), "ph":(5.5,7.0), "rain":(150,300)},
        "maize":       {"N":(60,100), "P":(50,80),  "K":(60,90),  "temp":(18,35), "hum":(50,80), "ph":(5.5,7.5), "rain":(60,120)},
        "chickpea":    {"N":(10,40),  "P":(60,100), "K":(70,110), "temp":(10,30), "hum":(14,40), "ph":(5.5,7.0), "rain":(60,100)},
        "kidneybeans": {"N":(10,40),  "P":(60,100), "K":(70,110), "temp":(15,30), "hum":(18,50), "ph":(5.5,7.0), "rain":(80,120)},
        "mungbean":    {"N":(10,40),  "P":(40,80),  "K":(30,60),  "temp":(20,38), "hum":(80,90), "ph":(6.2,7.2), "rain":(30,80)},
        "blackgram":   {"N":(20,60),  "P":(40,70),  "K":(20,50),  "temp":(24,38), "hum":(65,80), "ph":(5.0,7.5), "rain":(60,100)},
        "lentil":      {"N":(10,40),  "P":(50,80),  "K":(20,50),  "temp":(15,30), "hum":(14,40), "ph":(6.0,7.5), "rain":(30,70)},
        "banana":      {"N":(80,120), "P":(70,110), "K":(40,80),  "temp":(25,40), "hum":(75,100),"ph":(5.5,7.0), "rain":(100,200)},
        "mango":       {"N":(10,40),  "P":(10,40),  "K":(30,50),  "temp":(24,40), "hum":(50,80), "ph":(5.5,7.5), "rain":(90,200)},
        "grapes":      {"N":(10,40),  "P":(120,200),"K":(180,250),"temp":(8,45),  "hum":(80,85), "ph":(5.5,7.0), "rain":(55,125)},
        "watermelon":  {"N":(80,120), "P":(10,50),  "K":(40,80),  "temp":(24,40), "hum":(80,90), "ph":(5.5,7.0), "rain":(50,150)},
        "apple":       {"N":(0,30),   "P":(10,50),  "K":(150,200),"temp":(0,25),  "hum":(90,95), "ph":(5.5,7.0), "rain":(100,200)},
        "orange":      {"N":(0,30),   "P":(10,30),  "K":(10,30),  "temp":(10,35), "hum":(90,95), "ph":(6.0,7.5), "rain":(100,200)},
        "papaya":      {"N":(40,80),  "P":(10,50),  "K":(40,80),  "temp":(25,45), "hum":(90,95), "ph":(6.0,7.0), "rain":(100,200)},
        "coconut":     {"N":(10,40),  "P":(10,30),  "K":(30,60),  "temp":(27,38), "hum":(80,95), "ph":(5.0,8.0), "rain":(100,300)},
        "cotton":      {"N":(100,150),"P":(30,70),  "K":(15,60),  "temp":(21,45), "hum":(55,85), "ph":(6.0,8.0), "rain":(60,110)},
        "jute":        {"N":(60,100), "P":(40,70),  "K":(40,80),  "temp":(24,40), "hum":(80,100),"ph":(6.0,7.0), "rain":(150,250)},
        "coffee":      {"N":(80,120), "P":(20,70),  "K":(20,70),  "temp":(15,28), "hum":(80,100),"ph":(6.0,6.5), "rain":(150,300)},
        "pomegranate": {"N":(10,40),  "P":(10,40),  "K":(40,80),  "temp":(18,40), "hum":(90,95), "ph":(5.5,7.2), "rain":(100,200)},
        "mothbeans":   {"N":(10,40),  "P":(40,70),  "K":(40,70),  "temp":(24,40), "hum":(25,60), "ph":(3.5,7.0), "rain":(30,80)},
        "pigeonpeas":  {"N":(10,40),  "P":(60,100), "K":(50,80),  "temp":(18,38), "hum":(40,70), "ph":(5.0,7.0), "rain":(60,100)},
        "muskmelon":   {"N":(80,120), "P":(10,50),  "K":(40,80),  "temp":(24,40), "hum":(90,95), "ph":(6.0,7.0), "rain":(20,60)},
    }
    rows = []
    np.random.seed(42)
    for crop_name, values in crop_data.items():
        for _ in range(100):
            rows.append({
                "N":           round(np.random.uniform(*values["N"]),    2),
                "P":           round(np.random.uniform(*values["P"]),    2),
                "K":           round(np.random.uniform(*values["K"]),    2),
                "temperature": round(np.random.uniform(*values["temp"]), 2),
                "humidity":    round(np.random.uniform(*values["hum"]),  2),
                "ph":          round(np.random.uniform(*values["ph"]),   2),
                "rainfall":    round(np.random.uniform(*values["rain"]), 2),
                "label":       crop_name
            })
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"   Dataset generated! Total rows: {len(df)}, Total crops: {df['label'].nunique()}")


# Step 2: Prepare Data 

print("\nStep 2: Preparing data...")

X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = df["label"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples:  {len(X_test)}")


#Step 3: Train 3 ML Models

print("\nStep 3: Training ML models...")

lr_model    = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test))
print(f"   Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")

knn_model    = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_accuracy = accuracy_score(y_test, knn_model.predict(X_test))
print(f"   K-Nearest Neighbors Accuracy: {knn_accuracy * 100:.2f}%")

rf_model    = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
print(f"   Random Forest Accuracy:       {rf_accuracy * 100:.2f}%")


#Step 4: Save Best Model 

print("\nStep 4: Saving best model...")

all_models = {
    "Logistic Regression": (lr_model, lr_accuracy),
    "K-Nearest Neighbors": (knn_model, knn_accuracy),
    "Random Forest":       (rf_model, rf_accuracy),
}

best_name     = max(all_models, key=lambda name: all_models[name][1])
best_model    = all_models[best_name][0]
best_accuracy = all_models[best_name][1]

save_dir = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(save_dir, exist_ok=True)

joblib.dump(best_model,    os.path.join(save_dir, "best_model.pkl"))
joblib.dump(scaler,        os.path.join(save_dir, "scaler.pkl"))
joblib.dump(label_encoder, os.path.join(save_dir, "label_encoder.pkl"))

with open(os.path.join(save_dir, "results.txt"), "w") as f:
    f.write(f"Best Model: {best_name}\n")
    f.write(f"Best Accuracy: {best_accuracy * 100:.2f}%\n\n")
    f.write("All Results:\n")
    for name, (model, acc) in all_models.items():
        f.write(f"  {name}: {acc * 100:.2f}%\n")

print(f"\n   Best Model: {best_name}")
print(f"   Best Accuracy: {best_accuracy * 100:.2f}%")
print(f"   Saved to: ml_model/saved_models/")
print("\n✅ Training complete! Now run: python manage.py runserver")
