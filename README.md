# VisionCart-AI-Grocery-Recommendation-System

# 🛒 VisionCart: AI Grocery Recommendation System

VisionCart is an end-to-end AI-powered grocery recommendation system that combines **Computer Vision**, **Natural Language Processing**, and **Hybrid Recommendation Algorithms** to detect grocery products from images and generate personalized complementary product recommendations.

The system first detects grocery items using a fine-tuned **YOLO11** object detection model, maps detected categories to real Instacart products using **Sentence Transformers + FAISS**, and generates recommendations using a hybrid of **Association Rule Mining (FP-Growth)** and **Collaborative Filtering (Implicit ALS)**.

---

## Features

- Grocery product detection using YOLO11
- Semantic product matching with Sentence Transformers + FAISS
- Hybrid recommendation system
  - FP-Growth Association Rule Mining
  - Implicit Alternating Least Squares (ALS)
- Weighted hybrid score fusion
- Streamlit-based interactive web application
- Item-based recommendation evaluation
- Modular and scalable architecture

---

# System Architecture

```
                Grocery Image
                      │
                      ▼
              YOLO11 Object Detection
                      │
                      ▼
          Detected Grocery Categories
                      │
                      ▼
     Sentence Transformers + FAISS
         (Semantic Product Matching)
                      │
                      ▼
             Instacart Product ID
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   FP-Growth Rules          Implicit ALS
          │                       │
          └───────────┬───────────┘
                      ▼
            Weighted Score Fusion
                      ▼
          Top-K Product Recommendations
```

---

# Tech Stack

## Computer Vision

- YOLO11
- OpenCV
- Ultralytics

## Recommendation System

- Sentence Transformers
- FAISS
- Implicit ALS
- FP-Growth
- Scikit-learn

## Backend

- Python
- Pandas
- NumPy

## Interface

- Streamlit

---

# Dataset

The project uses the following datasets:

### Grocery Detection Dataset

Used for training the YOLO11 object detection model.

### Instacart Market Basket Dataset

Contains over **3 million grocery orders** used for:

- Collaborative Filtering
- Association Rule Mining
- Product Metadata

---

# Project Structure

```
VisionCart
│
├── app.py
├── requirements.txt
├── README.md
│
├── detection/
│   └── detect.py
│
├── recommendation/
│   ├── hybrid.py
│   ├── collaborative.py
│   ├── association_inference.py
│   ├── product_matcher.py
│   ├── content.py
│   ├── association.py
│   ├── train_als.py
│   ├── build_embeddings.py
│   └── prepare_data.py
│
├── evaluation/
│   ├── evaluate.py
│   └── metrics.py
│
├── models/
│
├── data/
│
└── outputs/
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/<username>/VisionCart.git

cd VisionCart
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

Start the Streamlit application

```bash
streamlit run app.py
```

Open the browser at

```
http://localhost:8501
```

Upload a grocery image to receive product recommendations.

---

# Recommendation Pipeline

### Step 1

Detect grocery products using YOLO11.

### Step 2

Match detected categories to the closest Instacart product using Sentence Transformers and FAISS.

### Step 3

Generate complementary recommendations using:

- FP-Growth Association Rules
- Implicit ALS Collaborative Filtering

### Step 4

Merge and rank recommendations using weighted score fusion.

---

# Evaluation

The recommendation system is evaluated using item-based recommendation metrics.

| Metric | Score |
|---------|-------|
| Precision@5 | 0.0382 |
| Recall@5 | 0.0096 |
| MAP@5 | 0.0164 |
| NDCG@5 | 0.0335 |
| HitRate@5 | 0.1520 |
| Coverage | 0.0949 |

Evaluation performed on **1000 randomly sampled products** from the Instacart dataset.

---

# Example Workflow

```
Input Image
      │
      ▼
Detected:
Milk
Coffee
Bread

      │
      ▼

Recommendations

Milk
• Granola
• Honey
• Pancake Mix

Coffee
• Cookies
• Creamer
• Sugar

Bread
• Butter
• Jam
• Cheese
```

---



# Results

- End-to-end grocery recommendation pipeline
- Real-time object detection
- Hybrid recommendation engine
- Interactive Streamlit application
- Modular architecture for future extensions
---

# Author

**Devesh Yadav**
