# ==============================================================================
# AI Student Success Predictor - AI-Driven Recommendation Engine (SaaS v3)
# ==============================================================================
# Purpose: Generate AI-prescriptive recommendations using the trained ML model 
#          to perform What-If counterfactual analysis on the student's metrics.
# ==============================================================================

#' Generate Prescriptive AI-Driven Recommendations
#'
#' @param model The trained machine learning model
#' @param input_df Data.frame containing the student's baseline inputs (pre-trimming)
#' @param base_score The predicted score of the current student
#' @param risk_level "Safe" or "High Risk"
#' @param learner_type "Fast Learner", "Average Learner", "Slow Learner"
#' @param trend_status Formatted time-aware structural trend indicator natively evaluated
generate_ai_recommendation <- function(model, input_df, base_score, risk_level, learner_type, trend_status = "Unknown") {
  if (is.null(model)) return("Data unavailable for AI recommendations.")

  if (base_score >= 85) {
     return(paste0("Excellent! AI projection is top tier (", base_score, "). Maintain your current highly effective engagement behaviors."))
  }

  # What-If Analysis: Perturb inputs to find maximum marginal gain
  
  # A. +5 Study Hours
  df_study <- input_df
  df_study$Study_Hours <- min(50, df_study$Study_Hours + 5)
  df_study$Study_Efficiency <- df_study$Assignments_Score / (df_study$Study_Hours + 1)
  df_study$Time_Engagement_Idx <- df_study$Study_Hours * (df_study$Attendance / 100)
  
  # B. +15% Attendance
  df_att <- input_df
  df_att$Attendance <- min(100, df_att$Attendance + 15)
  df_att$Engagement_Score <- (df_att$Attendance + df_att$Participation + df_att$Assignments_Score) / 3
  df_att$Time_Engagement_Idx <- df_att$Study_Hours * (df_att$Attendance / 100)
  
  # C. +15 Assignment Score
  df_ass <- input_df
  df_ass$Assignments_Score <- min(100, df_ass$Assignments_Score + 15)
  df_ass$Engagement_Score <- (df_ass$Attendance + df_ass$Participation + df_ass$Assignments_Score) / 3
  df_ass$Study_Efficiency <- df_ass$Assignments_Score / (df_ass$Study_Hours + 1)
  df_ass$Consistency_Index <- 100 - abs(df_ass$Assignments_Score - df_ass$Previous_Grade)

  # Purge unexpected cols safely mirroring training matrix
  trained_cols <- rownames(model$importance)
  if(!is.null(trained_cols)) {
    df_study <- df_study[, names(df_study) %in% trained_cols, drop=FALSE]
    df_att <- df_att[, names(df_att) %in% trained_cols, drop=FALSE]
    df_ass <- df_ass[, names(df_ass) %in% trained_cols, drop=FALSE]
  }

  gains <- c(
    Study = as.numeric(predict(model, df_study)) - base_score,
    Attendance = as.numeric(predict(model, df_att)) - base_score,
    Assignments = as.numeric(predict(model, df_ass)) - base_score
  )

  best_gain <- max(gains)
  best_action <- names(gains)[which.max(gains)]

  # Construct Contextual Recommendation
  advice <- if (risk_level == "High Risk") c("⚠️ URGENT AI INSIGHT:") else c("💡 Data-Driven Insight:")

  if (best_gain > 1.5) {
    if (best_action == "Study") {
      advice <- c(advice, sprintf("Increasing study time by just 5 hours/week yields your highest ML impact, adding +%.1f to projected score.", best_gain))
    } else if (best_action == "Attendance") {
      advice <- c(advice, sprintf("Attendance is your key blocker. Improving by 15%% is mathematically projected to boost your outcome by +%.1f points.", best_gain))
    } else {
      advice <- c(advice, sprintf("Maximizing assignments is your optimal path. A 15-point increase will raise overall predictive margins by +%.1f points.", best_gain))
    }
  } else {
     advice <- c(advice, "Your behavioral metrics are heavily scaled already. Broad spectrum stability across all categories is required to ascend further.")
  }

  # Append Segment Intuitions
  if (learner_type == "Slow Learner" && best_action == "Assignments") {
     advice <- c(advice, "Due to detected low efficiency patterns, prioritize structural grading rubrics intensely rather than just passive reading.")
  } else if (learner_type == "Fast Learner") {
     advice <- c(advice, "You clearly exhibit fast learner patterns! Leverage this naturally high statistical efficiency to master coursework rapidly.")
  }

  if (grepl("Declining", trend_status)) {
     advice <- c(advice, "⚠️ URGENT LONGITUDINAL TREND: Declining trajectory detected against previous semester baseline! Implement optimal path immediately to reverse metric degradation.")
  } else if (grepl("improving", trend_status)) {
     advice <- c(advice, "📈 Excellent longitudinal trajectory: Performance is improving steadily over your historical baseline. Maintain trajectory!")
  }

  paste(advice, collapse = " ")
}
