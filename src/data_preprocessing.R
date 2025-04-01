# Load necessary packages
install.packages(c("tidyverse", "lubridate"))
library(tidyverse)
library(lubridate)

# Load Data
data <- read.csv("../data/online_retail.csv")

# Inspect Data
glimpse(data)

# Data Cleaning & Preprocessing
data_clean <- data %>%
  # Remove rows with missing values
  drop_na() %>%
  # Remove negative or zero quantities and prices
  filter(Quantity > 0, UnitPrice > 0) %>%
  # Convert InvoiceDate to Date format
  mutate(InvoiceDate = as.Date(InvoiceDate, format = "%m/%d/%Y")) %>%
  # Create a new column for total revenue per transaction
  mutate(TotalRevenue = Quantity * UnitPrice)

# Summary of cleaned data
summary(data_clean)

# Save preprocessed data
write.csv(data_clean, "../data/online_retail_clean.csv", row.names = FALSE)
