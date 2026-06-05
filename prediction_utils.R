# ==============================================================================
# AI Student Success Predictor - Advanced Prediction Utilities
# Module: prediction_utils.R
# Purpose: Handles real-time inference, advanced feature engineering emulation,
#          and insights extrapolation (confidence, segmentations).
# ==============================================================================

#' Load Model Safely
load_model_safe <- function(path) {
  if (file.exists(path)) {
    return(readRDS(path))
  } else {
    warning(paste("Model not found at:", path, "- Run student_ml_backend.R first."))
    return(NULL)
  }
}

#' Predict Student Final Score & Advanced Insights
#'
#' Leverages the advanced trained model to produce predictions + generated metrics 
#' derived directly logically mimicking the backend training procedure.
#' @return A list containing: score, confidence, learner_type, performance_band, efficiency.
predict_score <- function(model, study_hours, attendance, assignments_score, prev_grade = 75) {
  # Safe failover mapping
  if (is.null(model)) {
    return(list(
      score = NA, 
      confidence = NA, 
      learner_type = "Unknown", 
      performance_band = "Unknown", 
      efficiency = NA
    ))
  }

  sh <- as.numeric(study_hours)
  att <- as.numeric(attendance)
  ascore <- as.numeric(assignments_score)
  
  # Base default injections for missing variables
  # Longitudinal Base Setup
  prev_grade <- as.numeric(prev_grade)
  participation <- 50
  
  ia_levels <- model$forest$xlevels$Internet_Access
  default_ia <- if (!is.null(ia_levels)) factor(ia_levels[length(ia_levels)], levels = ia_levels) else factor("Yes")
  
  # Advanced Feature Engineering mappings natively linked to inputs
  eng_score <- (att + participation + ascore) / 3
  efficiency <- ascore / (sh + 1)
  cons_index <- 100 - abs(ascore - prev_grade)
  time_eng <- sh * (att / 100)
  
  l_type <- if (efficiency > 7) "Fast Learner" else if (efficiency < 3) "Slow Learner" else "Average Learner"
  c_profile <- if (cons_index >= 70 && ascore >= 50 && sh >= 4) "Consistent Performer" else "Inconsistent/At-Risk"
  p_band <- if (ascore >= 90) "A" else if (ascore >= 80) "B" else if (ascore >= 70) "C" else if (ascore >= 60) "D" else "F"
  
  lt_levels <- model$forest$xlevels$Learner_Type
  if(is.null(lt_levels)) lt_levels <- c("Fast Learner", "Average Learner", "Slow Learner")
  l_type_fac <- factor(l_type, levels = lt_levels)
  
  pb_levels <- model$forest$xlevels$Performance_Band
  if(is.null(pb_levels)) pb_levels <- c("A", "B", "C", "D", "F")
  p_band_fac <- factor(p_band, levels = pb_levels)

  input_df <- data.frame(
    Study_Hours      = sh,
    Attendance       = att,
    Assignments_Score = ascore,
    Previous_Grade   = prev_grade,
    Participation    = participation,
    Internet_Access  = default_ia,
    Engagement_Score = eng_score,
    Study_Efficiency = efficiency,
    Consistency_Index = cons_index,
    Time_Engagement_Idx = time_eng,
    Learner_Type     = l_type_fac,
    Performance_Band = p_band_fac
  )

  # Purge unexpected columns safely avoiding R model dimension mismatch
  input_df <- input_df[, names(input_df) %in% rownames(model$importance), drop = FALSE]

  # Make advanced prediction mapped across the decision trees matrix
  pred_obj <- predict(model, input_df, predict.all = TRUE)
  predicted <- round(as.numeric(pred_obj$aggregate), 2)
  
  # Advanced Insight: Prediction Confidence Metric via Tree Deviation
  tree_preds <- as.numeric(pred_obj$individual)
  tree_sd <- sd(tree_preds)
  tree_mean <- mean(tree_preds)
  
  # Data-Driven dynamic confidence approach utilizing Coefficient of Variation explicitly
  cv <- ifelse(tree_mean == 0, 0, tree_sd / tree_mean)
  conf <- round(max(0, min(100, 100 - (cv * 100))), 1)
  
  # Determine risk status internally for prescriptive logic
  risk_status <- if (predicted < 40) "High Risk" else "Safe"
  
  # Calculate dynamic Predicted Performance Band mathematically
  predicted_p_band <- if (predicted >= 85) "A" else if (predicted >= 70) "B" else if (predicted >= 55) "C" else if (predicted >= 40) "D" else "F"

  # Time-Aware Target Trend Mathematics logically bounding 3-point slope vectors
  trend_status <- if (predicted > prev_grade + 3.5) "Performance is improving steadily" else if (predicted < prev_grade - 3.5) "Declining trend detected" else "Performance metric is stabilized"

  # Dynamic Semantic Persona Overrides for UI mapping requested explicitly!
  ui_persona <- if (predicted >= 80 && att >= 80) {
    "🚀 Top Performer"
  } else if (predicted < 50) {
    "⚠️ At Risk – Needs Intervention"
  } else if (sh <= 10 && ascore >= 60) {
    "🎯 High Potential but Low Effort"
  } else {
    "⚖️ Average Learner"
  }

  list(
    score = predicted,
    confidence = conf,
    learner_type = ui_persona,
    consistency_profile = c_profile,
    performance_band = predicted_p_band,
    learning_trend = trend_status,
    efficiency = round(efficiency, 2),
    recommendation = generate_ai_recommendation(model, input_df, predicted, risk_status, l_type, trend_status)
  )
}

#' Classify Risk Level
classify_risk <- function(score) {
  if (is.na(score)) return("Unknown")
  ifelse(score < 40, "High Risk", "Safe")
}

#' Extract Feature Importance as Data Frame
get_feature_importance <- function(model) {
  if (is.null(model)) return(data.frame(Feature = character(), Importance = numeric()))
  imp_matrix <- importance(model)
  imp_df <- data.frame(
    Feature    = rownames(imp_matrix),
    Importance = round(imp_matrix[, "%IncMSE"], 2),
    stringsAsFactors = FALSE
  )
  imp_df <- imp_df[order(imp_df$Importance, decreasing = TRUE), ]
  rownames(imp_df) <- NULL
  imp_df
}

#' Goal-Oriented Metric Optimization Simulator
optimize_for_target <- function(model, target_score, current_ascore, current_att, current_sh) {
  fail_res <- list(
    feasible = FALSE, 
    message = "Optimization model mapping statistically unavailable.", 
    current = list(Study_Hours = current_sh, Attendance = current_att, Assignments_Score = current_ascore),
    required = list(Study_Hours = NA, Attendance = NA, Assignments_Score = NA),
    probability = 0,
    range = "N/A"
  )
  if (is.null(model)) return(fail_res)
  
  sh_grid <- seq(max(0, current_sh), 50, by = 2) 
  att_grid <- seq(max(0, current_att), 100, by = 5)
  asc_grid <- seq(max(0, current_ascore), 100, by = 5)
  if(length(sh_grid)==0) sh_grid <- 50
  if(length(att_grid)==0) att_grid <- 100
  if(length(asc_grid)==0) asc_grid <- 100

  search_df <- expand.grid(Study_Hours = sh_grid, Attendance = att_grid, Assignments_Score = asc_grid)
  search_df$Previous_Grade <- 75
  search_df$Participation <- 50
  
  ia_levels <- model$forest$xlevels$Internet_Access
  search_df$Internet_Access <- if (!is.null(ia_levels)) factor(ia_levels[length(ia_levels)], levels = ia_levels) else factor("Yes")
  
  search_df$Engagement_Score <- (search_df$Attendance + search_df$Participation + search_df$Assignments_Score) / 3
  search_df$Study_Efficiency <- search_df$Assignments_Score / (search_df$Study_Hours + 1)
  search_df$Consistency_Index <- 100 - abs(search_df$Assignments_Score - search_df$Previous_Grade)
  search_df$Time_Engagement_Idx <- search_df$Study_Hours * (search_df$Attendance / 100)
  
  search_df$Learner_Type <- factor(ifelse(search_df$Study_Efficiency > 7, "Fast Learner", ifelse(search_df$Study_Efficiency < 3, "Slow Learner", "Average Learner")), levels = model$forest$xlevels$Learner_Type)
  search_df$Performance_Band <- factor(ifelse(search_df$Assignments_Score >= 90, "A", ifelse(search_df$Assignments_Score >= 80, "B", ifelse(search_df$Assignments_Score >= 70, "C", ifelse(search_df$Assignments_Score >= 60, "D", "F")))), levels = model$forest$xlevels$Performance_Band)

  trained_cols <- rownames(model$importance)
  trimmed_df <- if(!is.null(trained_cols)) search_df[, names(search_df) %in% trained_cols, drop=FALSE] else search_df

  search_df$Predicted_Score <- as.numeric(predict(model, trimmed_df))
  success_df <- search_df[search_df$Predicted_Score >= target_score - 1, ] 

  if (nrow(success_df) == 0) {
    best_idx <- which.max(search_df$Predicted_Score)[1]
    best_path <- search_df[best_idx, ]
    best_trimmed <- trimmed_df[best_idx, , drop=FALSE]
  } else {
    success_df$Cost <- (success_df$Study_Hours - current_sh)*3 + (success_df$Attendance - current_att)*1.5 + (success_df$Assignments_Score - current_ascore)*2 
    best_idx <- as.numeric(rownames(success_df[which.min(success_df$Cost), ]))
    best_path <- search_df[best_idx, ]
    best_trimmed <- trimmed_df[best_idx, , drop=FALSE]
  }

  pred_all <- predict(model, best_trimmed, predict.all=TRUE)
  trees <- as.numeric(pred_all$individual)
  
  prob_val <- round(mean(trees >= target_score) * 100, 1)
  mean_val <- round(mean(trees), 1)
  sd_val <- round(sd(trees), 1)
  range_str <- sprintf("%.1f - %.1f", mean_val - sd_val, mean_val + sd_val)
  conf_lvl <- round(max(0, min(100, 100 - (sd_val * 4.5))), 1)
  
  feas_text <- if (prob_val >= 70) "High" else if (prob_val >= 30) "Moderate" else "Low"
  
  delta_sh <- best_path$Study_Hours - current_sh
  delta_att <- best_path$Attendance - current_att
  delta_asc <- best_path$Assignments_Score - current_ascore
  
  sh_msg <- if(delta_sh > 0) sprintf("Optimize study volume: %d ➔ %d (+%d hrs/wk).", current_sh, best_path$Study_Hours, delta_sh) else ""
  att_msg <- if(delta_att > 0) sprintf("Stabilize attendance rate: %d%% ➔ %d%% (+%d%%).", current_att, best_path$Attendance, delta_att) else ""
  asc_msg <- if(delta_asc > 0) sprintf("Scale assignment quality: %d ➔ %d (+%d pts).", current_ascore, best_path$Assignments_Score, delta_asc) else ""
  
  delta_arr <- c(sh_msg, att_msg, asc_msg)
  delta_arr <- delta_arr[delta_arr != ""]
  
  if (length(delta_arr) == 0 && mean_val >= target_score) {
    msg <- sprintf("Your current behavioral profile naturally projects structural victory. The expected performance range is %s with %s%% ensemble confidence. Feasibility: %s (%.1f%% algorithmic probability). Maintain current metrics entirely.", range_str, conf_lvl, feas_text, prob_val)
  } else if (nrow(success_df) == 0) {
    msg <- sprintf("Your current profile limits mathematically achieving the absolute target outcome. Modifying your behavior at absolute peak margins shifts potential range to %s with %s%% confidence. Feasibility: %s (%.1f%% probability). The primary limiting factor across your index is strict structural assignment limitations.", range_str, conf_lvl, feas_text, prob_val)
  } else {
    msg <- sprintf("Your behavioral profile suggests structural modifications are required to safely pierce the target margin. Projected range safely shifts to %s with %s%% confidence. Feasibility: %s (%.1f%% probability). Execute the following AI-derived strategy dynamically: %s", 
                   range_str, conf_lvl, feas_text, prob_val, paste(delta_arr, collapse = " "))
  }

  list(
    feasible = nrow(success_df) > 0,
    message = msg,
    current = list(Study_Hours = current_sh, Attendance = current_att, Assignments_Score = current_ascore),
    required = list(Study_Hours = best_path$Study_Hours, Attendance = best_path$Attendance, Assignments_Score = best_path$Assignments_Score),
    probability = prob_val,
    range = range_str
  )
}
