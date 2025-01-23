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
Used the Online Retail Dataset.
Cleaned data using PySpark for handling large-scale datasets.
2. Feature Engineering
Created RFM (Recency, Frequency, Monetary) metrics using Python (Pandas).
Normalized features to ensure balanced clustering.
3. Clustering Model
Applied K-Means Clustering to segment customers.
Used the Elbow Method and Silhouette Score to determine the optimal number of clusters.
4. Tools & Technologies
Languages: Python, R
Libraries: Pandas, Scikit-learn, Matplotlib, ggplot2 (for R visualizations)
Big Data Tools: PySpark, Databricks
Visualization Tools: Matplotlib, Seaborn, ggplot2






