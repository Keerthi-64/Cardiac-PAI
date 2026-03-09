import numpy as np
import joblib
import cv2
import matplotlib.pyplot as plt
import os
import glob
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# --- 1. THE BRAIN (Strictly Synced) ---
def train_synced_model():
    print("Training AI with Spatial Awareness...")
    X, y_reg, y_clf = [], [], []
    res = 100
    for _ in range(500):
        pct = np.random.uniform(5, 95)
        img = np.zeros((res, res))
        center, v_rad = res // 2, res // 3
        y, x = np.ogrid[:res, :res]
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        img[dist <= v_rad] = 0.3
        p_rad = v_rad * np.sqrt(pct / 100)
        img[dist <= p_rad] = 0.9
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = np.clip(img, 0, 1)
        X.append(img.flatten())
        y_reg.append(pct)
        status = "IMMEDIATE SURGERY" if pct > 75 else "URGENT" if pct > 45 else "NON-URGENT"
        y_clf.append(status)
    
    reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_reg)
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y_clf)
    return reg, clf

# --- 2. THE LOCALIZATION ENGINE ---
def run_advanced_diagnosis():
    reg, clf = train_synced_model()
    test_files = glob.glob("PHASE2_RECON_*.npy")
    
    if not test_files:
        print("❌ Run your generator and reconstructor scripts first!")
        return

    print("\n--- SELECT PATIENT FOR ANALYSIS ---")
    for i, f in enumerate(test_files): print(f"[{i}] {f}")
    choice = int(input("\nSelect Patient: "))
    file_name = test_files[choice]

    # Load Data
    raw_data = np.load(file_name, allow_pickle=True)
    
    # AI Predictions
    pred_pct = reg.predict(raw_data.flatten().reshape(1, -1))[0]
    status = clf.predict(raw_data.flatten().reshape(1, -1))[0]

    # --- 3. BLOCKAGE LOCALIZATION LOGIC ---
    # We find where the signal is high (Plaque)
    mask_plaque = (raw_data > 0.6).astype(np.uint8)
    
    # Use OpenCV to find "Blobs" (Individual blockages)
    contours, _ = cv2.findContours(mask_plaque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_blockages = len(contours)

    # --- 4. VISUAL PROOF (The 10/10 Slide) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # View 1: Reconstructed PAI View
    ax1.imshow(raw_data, cmap='hot')
    ax1.set_title("Phase 2: Reconstructed Vascular Scan")
    ax1.axis('off')

    # View 2: AI Localization View
    # Highlight vessel in blue-ish and plaque in bright red
    ax2.imshow(raw_data, cmap='gray') 
    for cnt in contours:
        # Draw a bounding box or outline around the detected blockage
        x, y, w, h = cv2.boundingRect(cnt)
        rect = plt.Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=2)
        ax2.add_patch(rect)
    
    ax2.set_title(f"AI LOCALIZATION: {num_blockages} Blockage(s) Detected")
    ax2.axis('off')

    print("\n" + "="*45)
    print(f"REPORT: {file_name}")
    print(f"STENOSIS PERCENTAGE: {pred_pct:.2f}%")
    print(f"BLOCKAGE COUNT:      {num_blockages}")
    print(f"CLINICAL TRIAGE:     {status}")
    print("="*45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_advanced_diagnosis()
