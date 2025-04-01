# Load necessary packages
install.packages(c("tidyverse", "cluster", "factoextra"))
library(tidyverse)
library(cluster)
library(factoextra)

# Load Cleaned Data
data_clean <- read.csv("../data/online_retail_clean.csv")

# Feature Engineering: Creating RFM (Recency, Frequency, Monetary) metrics
rfm_data <- data_clean %>%
  group_by(CustomerID) %>%
  summarise(
    Recency = as.numeric(difftime(max(InvoiceDate), max(InvoiceDate), units = "days")),
    Frequency = n(),
    Monetary = sum(TotalRevenue)
  ) %>%
  na.omit()  # Remove rows with missing CustomerID

# Scaling the data
rfm_scaled <- scale(rfm_data[, -1])

# Optimal number of clusters using Elbow Method
fviz_nbclust(rfm_scaled, kmeans, method = "wss") +
  labs(title = "Elbow Method for Optimal Clusters")

# Building K-Means Model
set.seed(123)
kmeans_model <- kmeans(rfm_scaled, centers = 4, nstart = 25)

# Add cluster labels to data
rfm_data$Cluster <- as.factor(kmeans_model$cluster)

# Save Model
saveRDS(kmeans_model, "../models/kmeans_model.rds")

# Visualize Clusters
fviz_cluster(kmeans_model, data = rfm_scaled, geom = "point", ellipse.type = "convex") +
  labs(title = "Customer Segmentation")

# Save segmented data
write.csv(rfm_data, "../data/customer_segments.csv", row.names = FALSE)
