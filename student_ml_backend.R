# ==============================================================================
# AI Student Success Predictor - Advanced ML Logic (SaaS Grade)
# ==============================================================================

# Load necessary libraries
library(randomForest)
library(caret)
library(dplyr)

cat("Starting Advanced Data Processing and Modeling...\n\n")

# 1. Load the dataset
# We read the dataset from the parent directory.
data <- read.csv("C:/Users/dell/PycharmProjects/Streamlit/Rpro/student_dataset_50000.csv", stringsAsFactors = FALSE)
data <- na.omit(data)

# 2. Advanced Feature Engineering
cat("Generating Derived Features and Interaction Effects...\n")
data <- data %>%
  mutate(
    # Engagement composite metric: balance of attendance, participation, and active scoring
    Engagement_Score = (Attendance + Participation + Assignments_Score) / 3,
    
    # Study Efficiency (Score output per hour of study) +1 mitigates div zero
    Study_Efficiency = Assignments_Score / (Study_Hours + 1),
    
    # Consistency index (similarity between past and current baseline)
    Consistency_Index = 100 - abs(Assignments_Score - Previous_Grade),
    
    # Interaction: Time * Engagement mapping non-linear compounding returns
    Time_Engagement_Idx = Study_Hours * (Attendance / 100),
    
    # Categorization classification grouping based on efficiency ratios
    Learner_Type = case_when(
      Study_Efficiency > 7 ~ "Fast Learner",
      Study_Efficiency < 3 ~ "Slow Learner",
      TRUE ~ "Average Learner"
    ),
    
    # Current theoretical performance band
    Performance_Band = case_when(
      Assignments_Score >= 90 ~ "A",
      Assignments_Score >= 80 ~ "B",
      Assignments_Score >= 70 ~ "C",
      Assignments_Score >= 60 ~ "D",
      TRUE ~ "F"
    )
  )

# Enforce strict factor bindings
data$Internet_Access <- as.factor(data$Internet_Access)
data$Learner_Type <- factor(data$Learner_Type, levels = c("Fast Learner", "Average Learner", "Slow Learner"))
data$Performance_Band <- factor(data$Performance_Band, levels = c("A", "B", "C", "D", "F"))
data$Risk_Label <- as.factor(ifelse(data$Final_Score < 40, "High Risk", "Safe"))

# Auto-convert remaining strings
str_cols <- sapply(data, is.character)
data[str_cols] <- lapply(data[str_cols], as.factor)

# 3. Data Splitting
cat("Splitting data into Training and Testing streams...\n")
set.seed(42)
train_index <- createDataPartition(data$Final_Score, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# 4. Tuned Random Forest Modeling
cat("Training tuned Random Forest model with engineered features...\n")
rf_formula <- Final_Score ~ Study_Hours + Attendance + Assignments_Score + 
                            Previous_Grade + Participation + Internet_Access +
                            Engagement_Score + Study_Efficiency + Consistency_Index + 
                            Time_Engagement_Idx + Learner_Type + Performance_Band

advanced_rf <- randomForest(
  formula = rf_formula,
  data = train_data,
  ntree = 150,
  mtry = 4, # optimized depth vector for expanded feature set
  importance = TRUE,
  keep.inbag = TRUE # Enables prediction error bounds dynamically
)

# 5. Model Evaluation
cat("Evaluating Advanced SaaS Model against validation matrix...\n")
predictions <- predict(advanced_rf, test_data)

rmse_val <- sqrt(mean((predictions - test_data$Final_Score)^2))
r2_val <- cor(predictions, test_data$Final_Score)^2

cat(sprintf("\n=== Advanced Model Evaluation ===\n"))
cat(sprintf("RMSE      : %.4f\n", rmse_val))
cat(sprintf("R-squared : %.4f\n", r2_val))
cat("=================================\n\n")

cat("Top Feature Influencers:\n")
importance_df <- as.data.frame(importance(advanced_rf))
importance_df <- importance_df[order(-importance_df$`%IncMSE`), ]
print(head(importance_df, 8))

# 6. Advanced K-Means for Granular Analytical Segmentation
cat("\nPerforming Advanced Behavioral Clustering on Derived Metrics...\n")
cluster_features <- scale(train_data[, c("Engagement_Score", "Consistency_Index", "Study_Efficiency")])
advanced_kmeans <- kmeans(cluster_features, centers = 4, nstart = 25)

# 7. Model Writing
saveRDS(advanced_rf, "C:/Users/dell/PycharmProjects/Streamlit/Rpro/backend/student_rf_model.rds")
saveRDS(advanced_kmeans, "C:/Users/dell/PycharmProjects/Streamlit/Rpro/backend/student_kmeans_model.rds")

cat("\nAdvanced Machine Learning pipeline execution complete. Check app environment natively!\n")
