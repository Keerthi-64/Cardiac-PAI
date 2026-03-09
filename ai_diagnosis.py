#Final Diagnosis
import numpy as np
import joblib
import cv2
import matplotlib.pyplot as plt
import os
import glob
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# --- STEP 1: SYNCED HIGH-RES TRAINING ---
def train_synced_model():
    print("Training Brain with Strict Signal Sync (100x100)...")
    X, y_reg, y_clf = [], [], []
    res = 100

    for _ in range(500): # Increased samples for better precision
        pct = np.random.uniform(5, 95)
        img = np.zeros((res, res))
        center, v_rad = res // 2, res // 3
        y, x = np.ogrid[:res, :res]
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Anatomy (Must match your Phase 1/2 logic exactly)
        img[dist <= v_rad] = 0.3
        p_rad = v_rad * np.sqrt(pct / 100)
        img[dist <= p_rad] = 0.9
        
        # Apply the EXACT Phase 2 Reconstruction (Smoothing)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = np.clip(img, 0, 1) # Normalization sync

        X.append(img.flatten())
        y_reg.append(pct)
        
        # Triage Logic
        if pct > 80: status = "IMMEDIATE SURGERY"
        elif pct > 50: status = "URGENT MONITORING"
        else: status = "NON-URGENT / STABLE"
        y_clf.append(status)

    reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_reg)
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y_clf)

    joblib.dump(reg, 'stenosis_regressor.joblib')
    joblib.dump(clf, 'triage_classifier.joblib')
    print(" Brain is now Synchronized with Phase 2 Physics.")
    return reg, clf

# --- STEP 2: ACCURATE DIAGNOSIS ---
def run_final_test():
    # FORCE RETRAIN to clear the 99% error memory
    reg, clf = train_synced_model()

    test_files = glob.glob("PHASE2_RECON_*.npy")
    if not test_files:
        print("❌ Error: No PHASE2_RECON files found!")
        return

    print("\n--- TESTABLE PATIENTS ---")
    for i, f in enumerate(test_files):
        print(f"[{i}] {f}")
    
    choice = int(input("\nSelect Patient Number: "))
    file_name = test_files[choice]

    # Load and Predict
    raw_data = np.load(file_name, allow_pickle=True)
    input_vec = raw_data.flatten().reshape(1, -1)

    pred_pct = reg.predict(input_vec)[0]
    status = clf.predict(input_vec)[0]

    print("\n" + "="*40)
    print(f" PATIENT REPORT: {file_name}")
    print("="*40)
    print(f"AI PREDICTED BLOCKAGE: {pred_pct:.2f}%")
    print(f"CLINICAL TRIAGE:       {status}")
    print("="*40)

    plt.imshow(raw_data, cmap='hot')
    plt.title(f" AI Diagnosis: {pred_pct:.1f}% ({status})")
    plt.show()

if __name__ == "__main__":
    run_final_test()