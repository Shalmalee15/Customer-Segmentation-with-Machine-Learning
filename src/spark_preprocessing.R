# This is an optional code and only to be used for Big Data Processing

# Install Sparklyr if not installed
install.packages("sparklyr")
library(sparklyr)
library(dplyr)

# Connect to local Spark instance
sc <- spark_connect(master = "local")

# Load data into Spark
data_spark <- spark_read_csv(sc, name = "online_retail", path = "../data/online_retail.csv")

# Basic Data Processing using Spark
data_spark_clean <- data_spark %>%
  filter(Quantity > 0, UnitPrice > 0) %>%
  mutate(TotalRevenue = Quantity * UnitPrice)

# Write processed data back to CSV
spark_write_csv(data_spark_clean, "../data/online_retail_spark_clean.csv")

# Disconnect from Spark
spark_disconnect(sc)
