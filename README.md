# Customer-Segmentation-with-Machine-Learning

# Overview

This project focuses on creating a data-driven customer segmentation solution to enable targeted marketing strategies. By analysing transactional data, customers are grouped into segments based on their purchasing behaviour. This segmetation helps businesses improve customer engagement and optimise marketing campaigns. 

# Business Problem
Modern businesses need to understand  their customers to provide personalised experiences, The goal is to:
1. Identify distinct customer groups based on their behaviour
2. Recommend actionable strategies for each segment to maximise revenue and retention

# Solution Approach 
We implemented a K-means Clustering algorithm on customer transaction data to group customers into segments based on : 
1. Recenty: Time since the last purchase
2. Frequency: Number of purchase in a given perior
3. Monetary Value: Total spend by the customer

# Key Results
Identified 4 distinct customer segments:
1. High-Value Loyal Customers: Frequent shoppers with high spending
2. New Customers: Recently acuired, low spending so far
3. At-Risk Customers: Infrquent shoppers with declining spending
4. Low-Value Customers: Rare and low-spenders

Recommended tailored marketing strategoes for each segment, estimated to increase customer retention by 15%. 

# Technical Implementation
## 1. Data Preparation
1. Used the Online Retail Dataset.
2. Cleaned data using PySpark for handling large-scale datasets.

## 2. Feature Engineering
1. Created RFM (Recency, Frequency, Monetary) metrics using Python (Pandas).
2. Normalized features to ensure balanced clustering.


## 3. Clustering Model
1. Applied K-Means Clustering to segment customers.
2. Used the Elbow Method and Silhouette Score to determine the optimal number of clusters.


## 4. Tools & Technologies
1. Languages: Python, R
2. Libraries: Pandas, Scikit-learn, Matplotlib, ggplot2 (for R visualizations)
3. Big Data Tools: PySpark, Databricks
4. Visualization Tools: Matplotlib, Seaborn, ggplot2


# Repository Structure

```bash
├── data/
│   ├── online_retail.csv      # Input dataset
├── notebooks/
│   ├── customer_segmentation.ipynb  # Jupyter Notebook for Python implementation
│   ├── customer_segmentation.Rmd    # R Markdown for R implementation
├── src/
│   ├── data_preprocessing.py   # Python scripts for data cleaning
│   ├── clustering_model.py     # Python script for K-Means
│   ├── spark_processing.py     # PySpark script for large-scale data processing
├── README.md                  # Project documentation
```

# How to Run
## 1. Clone the repository:
   ```
git clone https://github.com/yourusername/customer-segmentation.git
cd customer-segmentation
```
## 2. Set up Environment:

```
pip install -r requirements.txt
```
Alternatiely, load the R environment by running customer_segmentation.Rmd

## 3. Run Python Implementation:
1. Open notebooks/customer_segmentation.ipynb in Jupyter Notebook.
2. Run all cells to reproduce the analysis.

## 4. Run R Implementation:
1. Open customer_segmentation.Rmd in RStudio or any R Markdown editor.
2. Knit the file to see the outputs.

# Results & Visualizations
## 1. Customer Segments:

(Add visualizations here)

## 2. Insights:
1. High-value customers account for 60% of total revenue.
2. At-risk customers can be re-engaged with loyalty discounts.

# Future Work
1. Enhance the model by incorporating demographic data.
2. Deploy the segmentation solution as a real-time service using Flask and Databricks.


# Acknowledgments
1. Dataset sourced from UCI Machine Learning Repository.
2. Inspired by practical use cases in customer analytics.


