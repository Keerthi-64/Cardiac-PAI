#To Detect Single and Mutiple Blockages(using distance transform and percision peak detection)
import numpy as np
import joblib
import cv2
import matplotlib.pyplot as plt
import os
import glob
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# --- 1. THE BRAIN: High-Resolution Training ---
def train_synced_model():
    print("Initializing AI Brain with Signal Synchronization...")
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
        
        # Physics Sync: Gaussian Smoothing
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = np.clip(img, 0, 1)

        X.append(img.flatten())
        y_reg.append(pct)
        
        if pct > 75: status = "IMMEDIATE SURGERY"
        elif pct > 45: status = "URGENT MONITORING"
        else: status = "NON-URGENT / STABLE"
        y_clf.append(status)
    
    reg = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_reg)
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y_clf)
    return reg, clf

# --- 2. THE DIAGNOSIS & PEAK DETECTION LOCALIZATION ---
def run_final_demo():
    reg, clf = train_synced_model()
    
    test_files = glob.glob("*.npy")
    test_files.sort()

    if not test_files:
        print("❌ No .npy files found!")
        return

    print("\n" + "="*35)
    print("   PRECISION AI DIAGNOSTIC STATION")
    print("="*35)
    for i, f in enumerate(test_files):
        print(f"[{i}] {f}")
    
    try:
        choice = int(input("\nSelect Patient: "))
        file_name = test_files[choice]
    except:
        return

    raw_data = np.load(file_name, allow_pickle=True)
    input_vec = raw_data.flatten().reshape(1, -1)

    pred_pct = reg.predict(input_vec)[0]
    status = clf.predict(input_vec)[0]

    # --- THE PEAK-DETECTION LOCALIZATION FIX ---
    # 1. Convert to 8-bit image
    img_8bit = (raw_data * 255).astype(np.uint8)
    _, thresh = cv2.threshold(img_8bit, 127, 255, cv2.THRESH_BINARY)

    # 2. Distance Transform to find centers
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

    # 3. Aggressive Thresholding (0.6) to separate merged blocks
    _, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 4. Find distinct contours from the isolated peaks
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_blocks = len(contours)

    print("\n" + "="*45)
    print(f"DIAGNOSTIC REPORT: {file_name}")
    print(f"BLOCKAGES DETECTED: {num_blocks}")
    print(f"PREDICTED STENOSIS: {pred_pct:.2f}%")
    print(f"CLINICAL TRIAGE:    {status}")
    print("="*45)

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(raw_data, cmap='hot')
    plt.title(f"Clinical View: {file_name}")

    plt.subplot(1, 2, 2)
    plt.imshow(raw_data, cmap='gray')
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # We expand the box slightly for better visualization
        plt.gca().add_patch(plt.Rectangle((x-2, y-2), w+4, h+4, edgecolor='red', fill=False, lw=2))
    
    plt.title(f"AI Detection: {num_blocks} Separate Lesion(s)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_final_demo()