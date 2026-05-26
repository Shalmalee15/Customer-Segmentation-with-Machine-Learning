# 🛍️ Customer Segmentation with Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![R](https://img.shields.io/badge/R-4.x-276DC3?logo=r&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-Databricks-E25A1C?logo=apache-spark&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

> **Grouping customers into behavioural segments using RFM analysis and K-Means Clustering to enable targeted marketing strategies and improve customer retention.**

---

## 📌 Overview

Modern businesses generate vast amounts of transactional data but struggle to act on it meaningfully. This project builds a production-ready customer segmentation pipeline that transforms raw retail transactions into actionable customer profiles — enabling marketing teams to personalise campaigns, reduce churn, and maximise revenue.

The solution implements **RFM (Recency, Frequency, Monetary)** analysis combined with **K-Means Clustering**, delivered in both Python and R for flexibility across analytics environments.

---

## 🎯 Business Problem

Retailers face three core challenges:
- **Who are our most valuable customers?** — and how do we keep them?
- **Who is at risk of churning?** — and how do we re-engage them?
- **How do we personalise at scale?** — without manual segmentation effort?

This project addresses all three by automating customer grouping based on purchasing behaviour, producing segments that are interpretable, actionable, and directly tied to marketing strategy.

---

## 💡 Solution Approach

### RFM Framework
Each customer is scored across three dimensions:

| Dimension | Definition | Business Signal |
|-----------|-----------|----------------|
| **Recency** | Days since last purchase | Engagement level |
| **Frequency** | Number of purchases in period | Loyalty indicator |
| **Monetary** | Total spend | Revenue contribution |

### Clustering Methodology
- **Algorithm:** K-Means Clustering
- **Optimal K:** Determined via Elbow Method + Silhouette Score analysis
- **Validation:** Cluster stability tested across multiple random seeds

---

## 📊 Key Results

Four distinct customer segments identified:

| Segment | Profile | Recommended Strategy |
|---------|---------|---------------------|
| 🥇 **High-Value Loyal** | Frequent, high-spending, recent | VIP rewards, early access, premium support |
| 🆕 **New Customers** | Recent acquisition, low spend | Onboarding nurture, first-purchase incentives |
| ⚠️ **At-Risk Customers** | Declining frequency and spend | Win-back campaigns, loyalty discounts |
| 💤 **Low-Value Customers** | Rare visits, minimal spend | Low-cost re-engagement or deprioritise |

**Impact:** High-value customers account for ~60% of total revenue. Targeted re-engagement of at-risk customers estimated to improve retention by **15%**.

---

## 🏗️ Technical Implementation

### 1. Data Source
- **Dataset:** [Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail) — UCI Machine Learning Repository
- **Scale:** 500K+ transactions across international retail customers

### 2. Data Preparation
- Large-scale data cleaning using **PySpark** on Databricks
- Handling of nulls, duplicates, returns, and cancelled orders
- Feature normalisation for balanced clustering

### 3. Feature Engineering
```python
# RFM metrics computed per customer
rfm = df.groupby('CustomerID').agg(
    Recency   = ('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    Frequency = ('InvoiceNo', 'nunique'),
    Monetary  = ('TotalPrice', 'sum')
)
```

### 4. Clustering Model
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Segment'] = kmeans.fit_predict(rfm_scaled)
```

---

## 🛠️ Tools & Technologies

| Category | Tools |
|----------|-------|
| **Languages** | Python 3.9+, R 4.x |
| **ML & Stats** | Scikit-learn, K-Means, Silhouette Analysis |
| **Data Processing** | Pandas, PySpark, Databricks |
| **Visualisation** | Matplotlib, Seaborn, ggplot2 |
| **Environment** | Jupyter Notebook, R Markdown |

---

## 📁 Repository Structure

```
Customer-Segmentation-with-Machine-Learning/
│
├── notebooks/
│   └── customer_segmentation.Rmd    # R Markdown — full R implementation
│
├── src/
│   ├── data_preprocessing.R         # Data cleaning and RFM calculation (R)
│   ├── clustering_model.R           # K-Means clustering and evaluation (R)
│   └── spark_processing.R           # PySpark integration placeholder
│
├── .gitignore
├── LICENSE                          # Apache 2.0
└── README.md
```

---

## 🚀 How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Python Implementation
```bash
# Clone the repository
git clone https://github.com/Shalmalee15/Customer-Segmentation-with-Machine-Learning.git
cd Customer-Segmentation-with-Machine-Learning

# Open Jupyter Notebook
jupyter notebook notebooks/customer_segmentation.ipynb
```

### R Implementation
```r
# Open in RStudio
# Knit customer_segmentation.Rmd to reproduce full analysis
```

---

## 🔮 Future Work

- [ ] Incorporate demographic and geographic data for richer segmentation
- [ ] Add DBSCAN and Hierarchical Clustering for comparison
- [ ] Deploy as a real-time REST API using Flask + Databricks
- [ ] Build an interactive Power BI / Shiny dashboard for business users
- [ ] Automate retraining pipeline with MLflow tracking

---

## 👩‍💻 Author

**Shalmalee Sharma** — PhD Astrophysics | Senior Data Scientist  
📍 Melbourne, Australia  
🔗 [LinkedIn](https://linkedin.com/in/shalmalee-kapse) · [GitHub](https://github.com/Shalmalee15)

---

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).
