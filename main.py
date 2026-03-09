import os
import zipfile
import glob
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import joblib  # For saving the models locally
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# 1. DATA PREPARATION & LABELING (LOCAL)
def prepare_medical_dataset():
    # Look for the data in the current folder
    extract_to = 'cardiac_data'
    zip_name = 'Reconstructed Data.zip'

    # Unzip only if the folder doesn't exist
    if not os.path.exists(extract_to):
        if os.path.exists(zip_name):
            print(f"Extracting {zip_name}...")
            with zipfile.ZipFile(zip_name, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            print(f"Error: '{zip_name}' not found in: {os.getcwd()}")
            return None, None, None

    # Use recursive search to find all .npy files
    file_list = glob.glob(f'{extract_to}/**/*.npy', recursive=True)
    file_list.sort()

    X, y_percent, y_class = [], [], []

    if len(file_list) == 0:
        print("Error: No .npy files found in the data folder.")
        return None, None, None

    print(f"Processing {len(file_list)} files for training...")
    for f in file_list:
        data = np.load(f)
        # Assuming 100x100 resolution; flattening to 10,000 features
        X.append(data.flatten())

        # Logic to calculate ground truth from pixels
        vessel_area = np.sum(data > 0.2)
        plaque_area = np.sum(data > 0.6)
        percentage = (plaque_area / vessel_area) * 100 if vessel_area > 0 else 0

        # Clinical Triage Logic
        if percentage > 80:
            triage = "IMMEDIATE SURGERY"
        elif percentage > 50:
            triage = "URGENT MONITORING"
        else:
            triage = "NON-URGENT / STABLE"

        y_percent.append(percentage)
        y_class.append(triage)

    return np.array(X), np.array(y_percent), np.array(y_class)


# 2. TRAINING THE DUAL-OUTPUT SYSTEM
try:
    X_data, y_p, y_c = prepare_medical_dataset()

    if X_data is not None:
        # 80/20 Split
        X_train, X_test, p_train, p_test, c_train, c_test = train_test_split(
            X_data, y_p, y_c, test_size=0.20, random_state=42
        )

        print("Training Random Forest Models... (This may take a moment)")
        # Model A: Regression
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        regressor.fit(X_train, p_train)

        # Model B: Classification
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, c_train)

        # SAVE MODELS for future use in UI
        joblib.dump(regressor, 'stenosis_regressor.joblib')
        joblib.dump(classifier, 'triage_classifier.joblib')
        print("✅ Models saved as .joblib files.")

        # 3. CLINICAL PREDICTION & VISUALIZATION
        p_pred = regressor.predict(X_test)
        c_pred = classifier.predict(X_test)

        # Pick one sample from the test set (Index 10)
        idx = 10
        # Reshape for 100x100 visualization
        sample_img = X_test[idx].reshape(100, 100)

        # Generate the AI Detection Mask
        _, ai_detection_mask = cv2.threshold(sample_img, 0.6, 1, cv2.THRESH_BINARY)

        
        print("AI CLINICAL DECISION SUPPORT")
        print(f"Predicted Blockage: {p_pred[idx]:.2f}%")
        print(f"Triage Category:    {c_pred[idx]}")
       

        # Visual Interface
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(sample_img, cmap='hot')
        plt.title("1. Phase 2 Reconstruction\n(Raw Clinical Data)")
        plt.colorbar(label='Acoustic Intensity')

        plt.subplot(1, 2, 2)
        plt.imshow(ai_detection_mask, cmap='jet')
        plt.title(f"2. AI Detection & Segmentation\n({p_pred[idx]:.1f}% Stenosis Identified)")
        plt.colorbar(label='Detection Confidence')

        plt.tight_layout()
        plt.show()

except Exception as e:
    print(f"An error occurred during execution: {e}")