# Sentiment-Aware Product Recommendation System  
**Machine Learning | NLP | Applied Data Science**

## Project Summary
This project demonstrates an **end-to-end machine learning pipeline** that integrates **sentiment analysis of customer reviews** into a product recommendation workflow. It focuses on transforming **unstructured text data** into **actionable signals** that enhance recommendation quality.

The repository follows **production-oriented ML engineering practices**, with a clear separation between data processing, feature engineering, training, and evaluation. To keep the repository lightweight and maintainable, trained model artifacts are stored externally.

---

## Problem Statement
Conventional recommendation systems often rely primarily on ratings or interaction data. This project explores how **sentiment extracted from textual reviews** can:
- Better capture nuanced user preferences  
- Improve recommendations in cold-start scenarios  
- Enrich feature representations for ranking or recommendation tasks  

---

## Technical Approach

### Data Processing
- Text cleaning, normalization, and tokenization  
- Stopword handling and reproducible preprocessing steps  

### Feature Engineering
- TF-IDF and/or embedding-based representations  
- Sentiment-aware feature construction  
- Modular design to support experimentation  

### Modeling
- Supervised sentiment classification  
- Interpretable, efficient ML models  
- Clear separation between training and inference logic  

### Evaluation
- Standard classification metrics  
- Focus on robustness and generalization  

---

## Repository Structure
```text
├── data/                  # Sample or processed datasets
├── notebooks/             # EDA and experimentation
├── src/                   # Production-style source code
│   ├── preprocessing.py   # Text cleaning & normalization
│   ├── features.py        # Feature engineering
│   ├── train.py           # Model training pipeline
│   └── evaluate.py        # Evaluation & metrics
├── requirements.txt       # Reproducible dependencies
└── README.md              # Project documentation
````

---

## Trained Models (External Storage)

To follow **best practices for ML repositories**, trained model files are **not committed to GitHub**.

📦 **Download trained models here:**
[ADD GOOGLE DRIVE LINK HERE]

After downloading, place the files locally in:

```text
models/
```

All scripts are configured to load models from this directory.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
python src/preprocessing.py
python src/train.py
python src/evaluate.py
```

---

## What This Project Demonstrates to Recruiters

* Practical application of **NLP in recommender systems**
* Clean, maintainable **ML code structure**
* Awareness of **model versioning and repository hygiene**
* Ability to move from **raw text → features → trained models**
* Engineering decisions aligned with **real-world constraints**

---

## Future Improvements

* Integration with collaborative filtering or ranking models
* API deployment using FastAPI or Flask
* Experiment tracking with MLflow
* Model monitoring and drift detection

---

## Author

**Hishikesh Phukan**
Master of Data Science
GitHub: [https://github.com/hishikesh123](https://github.com/hishikesh123)

---

## License

Academic and portfolio use only.

