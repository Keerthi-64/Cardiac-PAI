#Multiple Block Images
import numpy as np
import cv2
import os

def generate_multiblock_patient(num_blocks, res=100):
    """
    Generates a synthetic high-res phantom with 2 or 3 blockages.
    Follows Phase 1 (Simulation) & Phase 2 (DAS Smoothing) logic.
    """
    print(f"Simulating Complex Phantom: {num_blocks} Blockages...")
    
    # 1. Base Vessel Geometry (Phase 1.1)
    img = np.zeros((res, res))
    center, v_rad = res // 2, res // 3
    y, x = np.ogrid[:res, :res]
    vessel_dist = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Simulate US Signal (Anatomy/Vessel Wall)
    img[vessel_dist <= v_rad] = 0.3 

    # 2. Simulate Multiple PA Signals (Hemoglobin/Plaque) (Phase 1.2)
    # Define positions within the vessel for blockages
    if num_blocks == 2:
        # Side-by-side blockages
        positions = [(center - 15, center), (center + 15, center)]
        radii_pct = [40, 50] # Percent blockage contribution for each
    elif num_blocks == 3:
        # Triangular arrangement
        positions = [(center, center - 18), (center - 15, center + 10), (center + 15, center + 10)]
        radii_pct = [30, 35, 45]
    else:
        print("Error: Supports 2 or 3 blocks only.")
        return None

    # Draw the blockages
    for pos, pct in zip(positions, radii_pct):
        # Scale plaque radius based on percentage (follows previous data logic)
        p_rad = v_rad * np.sqrt(pct / 100) * 0.7 # 0.7 scales them down so they fit
        blockage_dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
        img[blockage_dist <= p_rad] = 0.9 # High-intensity PA signal

    # 3. Phase 2 Reconstruction (Task 2.1 DAS/Smoothing)
    # Apply the same Gaussian Blur used in your standard data
    final_img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Normalization (Task 2.3 Coregistration)
    final_img = np.clip(final_img, 0, 1)
    
    return final_img

# --- EXECUTION ---
if __name__ == "__main__":
    # Case A: 2 Blockages
    patient_2b = generate_multiblock_patient(2)
    file_2b = "MULTIBLOCK_PHASE2_2b.npy"
    np.save(file_2b, patient_2b)
    print(f"✅ Generated: {file_2b}")

    # Case B: 3 Blockages
    patient_3b = generate_multiblock_patient(3)
    file_3b = "MULTIBLOCK_PHASE2_3b.npy"
    np.save(file_3b, patient_3b)
    print(f"✅ Generated: {file_3b}")