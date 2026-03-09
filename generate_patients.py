#Phase 1 - New Patient Data
import numpy as np
import cv2
import os

def generate_high_res_patient(percent, res=100):
    # Phase 1.1: Design Synthetic Vascular Phantom
    image = np.zeros((res, res))
    center = res // 2
    vessel_radius = res // 3

    y, x = np.ogrid[:res, :res]
    dist = np.sqrt((x - center)**2 + (y - center)**2)

    # Phase 1.2 & 1.3: Simulate Hemoglobin/Anatomy Signal Intensity
    image[dist <= vessel_radius] = 0.3 # Vessel Wall (US Context)

    # Phase 1.4: Plaque scaling based on your 4 requested percentages
    plaque_radius = vessel_radius * np.sqrt(percent / 100)
    image[dist <= plaque_radius] = 0.9 # PA Signal (High Contrast)

    # Phase 2.1: Mimic DAS Reconstruction smoothing
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

# Create your 4 clinical test cases
blockages = [20, 48, 66, 92]
for b in blockages:
    data = generate_high_res_patient(b)
    np.save(f"new_patient_{b}pct.npy", data)
    print(f"Generated: new_patient_{b}pct.npy")