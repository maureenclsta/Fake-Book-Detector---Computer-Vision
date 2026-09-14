# Fake Book Cover Detection

A Computer Vision system for evaluating book cover authenticity using local feature matching with ORB and SIFT.

The system compares a query book cover against a reference cover and classifies it into three categories:

- **AUTHENTIC** — the cover matches its corresponding reference book.
- **SUSPICIOUS** — the cover is derived from the reference but has been visually manipulated or degraded.
- **COUNTERFEIT** — the cover does not correspond to the claimed reference book.

The project focuses on visual similarity rather than OCR or deep-learning image classification.

---

## Overview

Fake book covers can contain visual modifications such as cropping, blur, noise, brightness changes, perspective distortion, or watermarks. These changes can make direct image comparison unreliable.

This project uses local feature detection and feature matching to measure the visual similarity between a reference book cover and a query image.

Two complementary feature detection methods are used:

- **ORB** — fast binary feature descriptors suitable for efficient matching.
- **SIFT** — scale- and rotation-invariant local features for more robust matching.

Their results are combined through an ensemble decision process to improve classification reliability.

---

## Dataset

The project uses a **custom image dataset built from 10 book covers**. Instead of relying solely on an existing dataset, the images used for evaluation were generated and prepared specifically for this project.

The dataset is organized into three main types of samples:

### Reference Images

Each book starts with one original reference cover stored in the `reference/` directory.

These original covers serve as the baseline images for feature extraction and matching.

### Generated Test Images

Test samples were **generated independently from the original reference covers** using custom image augmentation scripts.

The augmentation process introduces realistic visual variations, including:

- Rotation
- Brightness adjustment
- Blur
- Gaussian noise
- Cropping and resizing
- Perspective transformation

These generated images are stored in the `test/` directory and are used to evaluate how well the feature-matching system recognizes the same book cover under different visual conditions.

The test dataset was generated using:

**`src/generate_test_images.py`**

This script automatically creates multiple variations for each reference book, allowing the project to build a controlled evaluation dataset from the original covers.

### Generated Suspicious Images

A separate set of manipulated images was **created from the original reference covers using custom image-generation scripts**.

These images simulate covers that are derived from the original but have been visually altered or degraded.

The transformations include:

- Contrast adjustment
- Blur
- Noise
- Cropping
- Watermarking
- JPEG compression artifacts
- Combined transformations

These samples are stored in the `fake/` directory and are evaluated as the **SUSPICIOUS** class.

The manipulated dataset was generated using:

**`src/generate_fake.py`**

This approach allows the project to create its own controlled set of suspicious samples rather than relying on a pre-existing labeled dataset.

### Counterfeit Samples

Counterfeit samples are created through **cross-book comparisons**.

A reference cover from one book is compared against a reference cover belonging to a different book. These samples represent cases where the query image does not correspond to the claimed reference book.

---

## Dataset Generation and Feature Matching

The dataset preparation and feature-matching stages are separated into two main phases:

**Original Book Covers → Custom Augmentation & Manipulation → Generated Dataset → ORB/SIFT Feature Extraction → Feature Matching → Classification**

First, the original book covers are used as reference images. Custom scripts then generate test variations and manipulated samples from these references.

The generated images are subsequently used as query images for the Computer Vision pipeline. ORB and SIFT features are extracted from both the reference and query images, followed by feature matching and threshold-based classification.

This approach provides a controlled experimental environment for evaluating the robustness of local feature matching under different types of image variation and manipulation.

---

## Methodology

The system follows a feature-matching and threshold-based classification pipeline:

**Input Image → ORB/SIFT Feature Detection → Feature Matching → Lowe's Ratio Test → Matching Score → Threshold Classification → ORB + SIFT Ensemble → Final Label**

### 1. Feature Detection

ORB and SIFT features are extracted from both the reference image and query image.

The system uses up to **4,000 features** for each method.

### 2. Feature Matching

Descriptors are matched using OpenCV's `BFMatcher`:

- ORB → Hamming distance
- SIFT → L2 distance

K-nearest-neighbor matching with `k=2` is used to obtain candidate matches.

### 3. Lowe's Ratio Test

The ratio between the best and second-best match is evaluated using Lowe's Ratio Test to filter unreliable feature matches.

The resulting good matches are used to calculate the matching score.

### 4. Threshold Tuning

Classification thresholds are tuned using a grid-search procedure over the available training split.

The system uses two thresholds:

- Authentic threshold
- Suspicious threshold

The resulting score is mapped to one of the three classes.

### 5. Ensemble Decision

ORB and SIFT predictions are combined into a final ensemble decision.

This allows the system to use complementary information from both feature detection methods.

---

## Results

The final ensemble was evaluated on **150 test samples** across three classes.

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| AUTHENTIC | 1.00 | 0.80 | 0.89 | 40 |
| SUSPICIOUS | 0.66 | 0.95 | 0.78 | 20 |
| COUNTERFEIT | 0.99 | 0.98 | 0.98 | 90 |
| **Accuracy** | | | **0.93** | **150** |
| Macro Avg | 0.88 | 0.91 | 0.88 | 150 |
| Weighted Avg | 0.95 | 0.93 | 0.93 | 150 |

**Overall accuracy: 92.67%**

The system achieved particularly strong performance in identifying **COUNTERFEIT** samples, while **SUSPICIOUS** samples remained more challenging because their manipulated appearance can still retain strong local similarities to the original reference.

---

## Project Structure

The repository is organized into separate directories for the dataset, experiment outputs, source scripts, and notebook.

- `dataset1/` — custom dataset containing 10 book covers and their image variations.
- `output_v6/` — final experiment outputs and matching visualizations.
- `src/` — Python scripts for generating test and manipulated images.
- `notebooks/` — main Computer Vision experiment notebook.

The main structure is:

CompVis_Fake-Book-Detector/
├── dataset1/
├── output_v6/
├── src/
│   ├── generate_fake.py
│   └── generate_test_images.py
├── notebooks/
│   └── compvis.ipynb
├── requirements.txt
└── README.md

### Source Files

**`src/generate_fake.py`**

Generates synthetically manipulated book cover images using visual transformations such as contrast adjustment, blur, noise, cropping, watermarking, JPEG artifacts, and combined transformations.

**`src/generate_test_images.py`**

Generates test variations from the original reference covers, including rotation, brightness changes, blur, noise, cropping, and perspective transformation.

**`notebooks/compvis.ipynb`**

Contains the main Computer Vision pipeline, including feature extraction, matching, threshold tuning, ensemble classification, evaluation, and matching visualizations.

---

## Output

The `output_v6/` directory contains the results produced by the final experiment, including:

- `confusion_matrix_v6.png` — confusion matrix of the ensemble classification results.
- `hasil_v6.csv` — classification results for the evaluated samples.
- `matches/` — visualizations of feature correspondences between reference and query images.

The matching visualizations show the local feature correspondences detected between reference and query images.

---

## Technologies

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

### Computer Vision Techniques

- ORB
- SIFT
- BFMatcher
- KNN feature matching
- Lowe's Ratio Test
- Local feature matching
- Threshold-based classification
- Ensemble decision

---

## Limitations

The system relies on local visual features and similarity thresholds. Performance can be affected by:

- Severe image distortion
- Large occlusions
- Significant changes in layout
- Very low image quality
- Covers with highly similar visual structures

The system is designed as a visual similarity-based authenticity analysis system rather than a production-grade counterfeit verification service.

---

## Key Takeaways

This project demonstrates the use of classical Computer Vision techniques to build an end-to-end image matching and classification pipeline without relying on deep learning.

The final ORB + SIFT ensemble achieved **92.67% accuracy on 150 test samples** across AUTHENTIC, SUSPICIOUS, and COUNTERFEIT categories.
