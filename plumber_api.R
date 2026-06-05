# ==============================================================================
# AI Student Success Predictor - Plumber API Backend (Advanced SaaS v2)
# ==============================================================================
# Architecture:
#   backend/prediction_utils.R   → Model loading, advanced inference & insights
#   backend/recommendation_logic.R → Recommendation engine
#   plumber_api.R                → REST API routing logic
# ==============================================================================

library(plumber)

# Source Backend Modules (Centralized ML utilities wrapper)
BASE_API_DIR <- "C:/Users/dell/PycharmProjects/Streamlit/Rpro/backend"
source(file.path(BASE_API_DIR, "prediction_utils.R"))
source(file.path(BASE_API_DIR, "recommendation_logic.R"))

# 1. Initialization: Load the Advanced Analytics Random Forest model
model_path <- file.path(BASE_API_DIR, "student_rf_model.rds")

if(!file.exists(model_path)) {
  warning("Model file not found at API startup! Ensure student_ml_backend.R is run first.")
  rf_model <- NULL
} else {
  cat("Loading Advanced SaaS Machine Learning Model...\n")
  rf_model <- load_model_safe(model_path)
  cat("Model successfully loaded.\n")
}

#* @apiTitle Advanced AI Student Predictor Dashboard API
#* @apiDescription Deep Analytics API capable of generating SaaS-oriented categorization metrics natively.
#* @apiContact list(name = "SaaS Analytics Backend", email = "admin@example.com")

#* Predict Student Output and Derive Deep Behavioral Metrics
#* @param Study_Hours:numeric Number of hours studied conceptually per week
#* @param Attendance:numeric Student attendance percentage
#* @param Assignments_Score:numeric Current average score in assignments
#* @post /predict
function(res, req, Study_Hours, Attendance, Assignments_Score) {
  
  if (is.null(rf_model)) {
    res$status <- 500
    return(list(error = "Internal Server Error: Predictor model not loaded. Check backend file configuration."))
  }
  
  # Parse API params to native R structures correctly
  study_hrs <- as.numeric(Study_Hours)
  attendance_pct <- as.numeric(Attendance)
  assignments_scr <- as.numeric(Assignments_Score)
  
  if (is.na(study_hrs) || is.na(attendance_pct) || is.na(assignments_scr)) {
    res$status <- 400
    return(list(error = "Invalid API structure request. Params must resolve numerically natively."))
  }
  
  # 2. Leverage Shared Advanced Context Wrapper
  # Resolves standard inference + derivations natively
  pred_res <- predict_score(rf_model, study_hrs, attendance_pct, assignments_scr)
  
  # 3. Assess System Risk Logic Structurally
  risk_level <- classify_risk(pred_res$score)
  
  # 5. Build Comprehensive Deep Analytics SaaS Response Context
  return(list(
    Input_Signature = list(
      Study_Hours = study_hrs,
      Attendance = attendance_pct,
      Assignments_Score = assignments_scr
    ),
    Core_Prediction = list(
      Final_Score = pred_res$score,
      Risk_Assessment_Label = risk_level,
      Prediction_Confidence_Percentage = pred_res$confidence
    ),
    Advanced_Analytics_Behavior_Segmentation = list(
      Computed_Study_Efficiency = pred_res$efficiency,
      Learner_Type = pred_res$learner_type,
      Consistency_Profile = pred_res$consistency_profile,
      Projected_Performance_Band = pred_res$performance_band
    ),
    Actionable_Logic = list(
      Contextual_Recommendation_Text = pred_res$recommendation
    ),
    Service_Diagnostics = list(
      Execution_Timestamp = as.character(Sys.time())
    )
  ))
}

#* Deployment Liveness Matrix Check
#* @get /health
function() {
  list(
    status = "healthy", 
    service = "Deep Analytics Student Predictor Router",
    uptime = as.character(Sys.time())
  )
}
