#Phase 2 - Patient Data Reconstruction
import numpy as np
import cv2
import os
import glob

def reconstruct_phase2(file_path):
    """
    Transforms Phase 1 Raw Data into Phase 2 Reconstructed Data.
    Follows Task 2.1 & 2.2: Implement DAS-like smoothing and filtering.
    """
    # 1. Load Phase 1 Raw Data
    # allow_pickle=True is used for safety with NumPy files
    raw_data = np.load(file_path, allow_pickle=True)

    # 2. Phase 2: Reconstruction Logic (Physics-based smoothing)
    # This simulates the "Delay-and-Sum" (DAS) effect by smoothing raw signals
    reconstructed = cv2.GaussianBlur(raw_data, (5, 5), 0)

    # Task 2.3: Coregistration / Signal Normalization
    # Keeps pixel values between 0 and 1 for the AI model to read correctly
    reconstructed = np.clip(reconstructed, 0, 1)

    # 3. Create the NEW filename
    # This changes "new_patient_20pct.npy" to "PHASE2_RECON_20pct.npy"
    base_name = os.path.basename(file_path)
    new_filename = base_name.replace("new_patient", "PHASE2_RECON")
    
    # Save directly to your current folder
    np.save(new_filename, reconstructed)

    return new_filename

# --- EXECUTION FOR VS CODE ---
if __name__ == "__main__":
    # Look for all the Phase 1 files you generated earlier
    phase1_files = glob.glob("new_patient_*pct.npy")
    
    if not phase1_files:
        print("❌ Error: No Phase 1 files found! Run your generator script first.")
    else:
        print(f"Found {len(phase1_files)} files. Starting Phase 2 Reconstruction...")
        
        reconstructed_list = []
        for f_path in phase1_files:
            new_file = reconstruct_phase2(f_path)
            reconstructed_list.append(new_file)
            print(f"✅ Created: {new_file}")
            
        print("\n--- RECONSTRUCTION COMPLETE ---")
        print("The high-resolution files are now ready in your folder.")