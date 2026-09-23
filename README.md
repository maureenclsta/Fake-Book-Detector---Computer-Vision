# Fake Book Cover Detector

A Computer Vision project that detects whether a book cover is **authentic**, **suspicious**, or **counterfeit** using classical local feature matching (ORB + SIFT) — no deep learning required.

---

## Description

This project analyzes a query book cover image and compares it against a reference cover to determine its authenticity. Instead of relying on OCR or a trained image classifier, it uses local feature detection and matching to measure visual similarity, then classifies the result using a tuned threshold and an ORB + SIFT ensemble decision.

The system was built and evaluated on a **custom dataset of 10 book covers**, each expanded with generated test variations (rotation, blur, noise, perspective, etc.) and generated suspicious/manipulated samples (watermarks, cropping, compression artifacts, etc.).

---

## Tech Stack

- **Language:** Python
- **Computer Vision:** OpenCV (ORB, SIFT, BFMatcher, KNN matching, Lowe's Ratio Test)
- **Data & Analysis:** NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib
- **Environment:** Jupyter Notebook

---

## Live Demo / Video

📽️ [Watch the demo video](https://drive.google.com/drive/folders/1XWjIl0jvE0stsUaxWSpwSh5C6AUIr0K0)

---

## Key Features

- 🔍 **Dual feature detection** — combines ORB (fast, binary) and SIFT (scale/rotation-invariant) descriptors for robust matching.
- 🧠 **Ensemble classification** — merges ORB and SIFT predictions into a single, more reliable decision.
- 🎯 **Threshold tuning via grid search** — authentic and suspicious thresholds are optimized on the training split rather than hardcoded.
- 🖼️ **Custom-built dataset** — 10 book covers with programmatically generated test and fake/suspicious variations.
- 📊 **Full evaluation pipeline** — outputs a confusion matrix, per-class metrics, and visualized feature-match results.

---

## ⚙️ What It Does

1. Extracts ORB and SIFT features from a reference and a query book cover.
2. Matches descriptors using OpenCV's `BFMatcher` (Hamming for ORB, L2 for SIFT) with KNN (`k=2`).
3. Filters matches using Lowe's Ratio Test to keep only reliable correspondences.
4. Computes a matching score and classifies it against tuned thresholds.
5. Combines ORB and SIFT results into a final ensemble label:
   - **AUTHENTIC** — matches its reference cover.
   - **SUSPICIOUS** — derived from the reference but visually altered/degraded.
   - **COUNTERFEIT** — does not correspond to the claimed reference book.

---

## 📈 Results

Evaluated on **150 test samples**, the final ORB + SIFT ensemble achieved:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| AUTHENTIC | 1.00 | 0.80 | 0.89 |
| SUSPICIOUS | 0.66 | 0.95 | 0.78 |
| COUNTERFEIT | 0.99 | 0.98 | 0.98 |

**Overall accuracy: 92.67%**

---

## 📁 Project Structure

```
CompVis_Fake-Book-Detector/
├── dataset1/               # Custom dataset: 10 book covers + generated variations
├── output_v6/               # Final results: confusion matrix, scores, match visualizations
├── src/
│   ├── generate_fake.py         # Generates suspicious/manipulated cover images
│   └── generate_test_images.py  # Generates test variations from reference covers
├── notebooks/
│   └── compvis.ipynb        # Main CV pipeline: feature extraction → matching → evaluation
└── README.md
```

---

## 👥 Team Members

- Angelina Jolie Candaya — 2802541644
- Isis Prianita — 2802572870
- Maureen Calista Surjo — 2802536392
