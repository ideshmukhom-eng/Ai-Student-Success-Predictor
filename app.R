# ==============================================================================
# AI Student Success Predictor - Enhanced Shiny Dashboard (SaaS & Advanced ML)
# ==============================================================================
library(shiny)
library(ggplot2)
library(randomForest)
library(dplyr)
library(bslib)
library(scales)
library(caret)
library(plotly)

# ── Source Backend Modules ──────────────────────────────────────────────────
BASE       <- "C:/Users/dell/PycharmProjects/Streamlit/Rpro"
MODEL_PATH <- file.path(BASE, "backend/student_rf_model.rds")
KM_PATH    <- file.path(BASE, "backend/student_kmeans_model.rds")

source(file.path(BASE, "backend/prediction_utils.R"))
source(file.path(BASE, "backend/recommendation_logic.R"))

# ── Global Data Loading ──────────────────────────────────────────────────────
message("[App] Loading dataset...")
student_data <- tryCatch(
  read.csv(file.path(BASE, "student_dataset_50000.csv"), stringsAsFactors = FALSE),
  error = function(e) { warning("Dataset not found."); data.frame() }
)

# ── Safe Loading Logic (Fallback Omitted for SaaS Clarity) ───────────────
rf_model     <- load_model_safe(MODEL_PATH)
kmeans_model <- load_model_safe(KM_PATH)

if (nrow(student_data) > 0) {
  # Deterministically evaluate logical characteristics natively replacing meaningless numeric groupings
  student_data$Segment_Profile <- with(student_data, {
    ifelse(Final_Score >= 75 & Attendance >= 80, "Top Performers",
    ifelse(Final_Score < 50 & Study_Hours < 15, "At Risk",
    ifelse(Final_Score < 75 & Study_Hours >= 20, "High Potential", "Average")))
  })
  student_data$Segment_Profile <- factor(student_data$Segment_Profile, levels=c("Top Performers", "High Potential", "Average", "At Risk"))
}
if (nrow(student_data) > 0 && "Final_Score" %in% names(student_data)) {
  student_data$Risk <- ifelse(student_data$Final_Score < 40, "High Risk", "Safe")
}

feat_imp <- get_feature_importance(rf_model)
risk_color <- function(risk) if (risk == "High Risk") "#e74c3c" else "#27ae60"

theme_dashboard <- function() {
  theme_minimal(base_size = 13) +
    theme(
      plot.background  = element_rect(fill = "#f8f9fa", colour = NA),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = "#e9ecef"),
      plot.title       = element_text(face = "bold", size = 14, hjust = 0.5, colour = "#212529"),
      axis.title       = element_text(colour = "#495057"),
      legend.position  = "bottom"
    )
}

# ==============================================================================
# Global UI
# ==============================================================================
ui <- page_navbar(
  title = span(icon("graduation-cap"), " Advanced Analytics Predictor"),
  theme = bs_theme(version = 5, bootswatch = "flatly", primary = "#2c3e50"),
  bg = "#2c3e50", inverse = TRUE, fillable = TRUE,
  
  # Global Sidebar applicable to all tabs
  sidebar = sidebar(
    width = 300, bg = "#ecf0f1",
    title = span(icon("sliders"), " Target Metrics Validation"),
    p(class="text-muted", style="font-size:13px;", "Tune the sliders below to structurally test baseline inputs across all diagnostic tabs simultaneously."),
    sliderInput("study_hours", label = "Study Hours / Week", min = 0, max = 50, value = 15, step = 1),
    sliderInput("attendance", label = "Attendance (%)", min = 0, max = 100, value = 80, step = 1),
    sliderInput("assignment_score", label = "Assignment Score (Present)", min = 0, max = 100, value = 70, step = 1),
    sliderInput("previous_grade", label = "Previous Grade (Past Baseline)", min = 0, max = 100, value = 75, step = 1),
    hr(),
    actionButton("reset_btn", "Reset Metrics", class = "btn-outline-secondary btn-sm w-100")
  ),

  nav_panel(
    title = tagList(icon("chart-bar"), " Deep Insights"),
    # Top Row Outputs (Basic Metrics)
    layout_columns(
      col_widths = c(3, 3, 3, 3),
      card(
        card_header(tagList(icon("bullseye"), " Projected Score"), class = "bg-primary text-white"),
        card_body(div(style = "text-align:center;", uiOutput("score_badge")))
      ),
      card(
        card_header(tagList(icon("shield-alt"), " System Risk Assess"), class = "bg-primary text-white"),
        card_body(div(style = "text-align:center;", uiOutput("risk_badge")))
      ),
      card(
        card_header(tagList(icon("brain"), " Confidence Level"), class = "bg-primary text-white"),
        card_body(div(style = "text-align:center;", uiOutput("confidence_badge")))
      ),
      card(
        card_header(tagList(icon("chart-line"), " Longitudinal Trend"), class = "bg-primary text-white"),
        card_body(div(style = "text-align:center;", uiOutput("trend_badge")))
      )
    ),

    # Mid Row Outputs (Advanced SaaS Insights)
    layout_columns(
      col_widths = c(4, 4, 4),
      card(
        card_header(tagList(icon("user-tag"), " Behavioral Archetype"), class = "bg-info text-white"),
        card_body(div(style = "text-align:center;", uiOutput("learner_type_badge")))
      ),
      card(
        card_header(tagList(icon("layer-group"), " Performance Band"), class = "bg-info text-white"),
        card_body(div(style = "text-align:center;", uiOutput("perf_band_badge")))
      ),
      card(
        card_header(tagList(icon("lightbulb"), " Contextual Action"), class = "bg-dark text-white"),
        card_body(div(style = "text-align:center;", uiOutput("recommendation_badge")))
      )
    ),
    
    br(),
    card(
      card_header(tagList(icon("chart-area"), " Longitudinal Learning Curve Analytics"), class = "bg-secondary text-white"),
      card_body(plotOutput("learning_curve_plot", height = "320px"))
    )
  ),

  nav_panel(
    title = tagList(icon("bullseye"), " Advanced Goal Planner"),
    card(
      card_header(tagList(icon("crosshairs"), " Predictive Optimization Request"), class = "bg-primary text-white"),
      card_body(
        p("Execute rigorous multi-factor counterfactual simulations against the ML model to find the minimum required behavioral path to hit specific grade outcomes.", style="font-size:15px;"),
        layout_columns(
          col_widths = c(4, 8),
          numericInput("target_score", "Absolute Target Score:", value = 80, min = 40, max = 100),
          div(style="padding-top:30px;", actionButton("run_opt_btn", "Simulate Optimized Strategy Path", class = "btn-secondary", icon = icon("rocket")))
        )
      )
    ),
    uiOutput("goal_planner_dashboard")
  ),

  nav_panel(
    title = tagList(icon("ranking-star"), " Feature Importance"),
    layout_columns(
      col_widths = c(8, 4),
      card(
        card_header(tagList(icon("microchip"), " Interactive Diagnostic Matrix"), class = "bg-primary text-white"),
        card_body(plotlyOutput("importance_plot", height = "450px"))
      ),
      card(
        card_header(tagList(icon("info-circle"), " Global Transparency Array"), class = "bg-info text-white"),
        card_body(uiOutput("transparency_text"))
      )
    )
  ),

  nav_panel(
    title = tagList(icon("users"), " Deep Segmentation"),
    layout_columns(
      col_widths = c(6, 6),
      card(card_header("Clustering: Study vs Attendance", class="bg-primary text-white"), card_body(plotOutput("cluster_plot_1"))),
      card(card_header("Clustering: Attendance vs Score", class="bg-primary text-white"), card_body(plotOutput("cluster_plot_2")))
    )
  )
)

# ==============================================================================
# SERVER
# ==============================================================================
server <- function(input, output, session) {

  pred_result <- reactive({
    res <- predict_score(rf_model, input$study_hours, input$attendance, input$assignment_score, input$previous_grade)
    risk <- classify_risk(res$score)
    
    list(
      score = res$score, 
      risk = risk, 
      recommendation = res$recommendation,
      confidence = res$confidence,
      learner_type = res$learner_type,
      learning_trend = res$learning_trend,
      consistency = res$consistency_profile,
      perf_band = res$performance_band
    )
  })

  output$score_badge <- renderUI({
    r <- pred_result()
    if (is.na(r$score)) return(tags$span("Model Error", class = "badge bg-warning"))
    col <- if (r$score < 40) "#e74c3c" else if (r$score >= 75) "#27ae60" else "#f39c12"
    tagList(div(style=paste0("font-size:42px; font-weight:800; color:", col, ";"), r$score))
  })

  output$risk_badge <- renderUI({
    r <- pred_result()
    if (is.na(r$score)) return(HTML(""))
    col <- risk_color(r$risk)
    tagList(div(style=paste0("font-size:32px; font-weight:800; color:", col, "; margin-top:5px;"), r$risk))
  })
  
  output$confidence_badge <- renderUI({
    r <- pred_result()
    if (is.na(r$score)) return(HTML(""))
    col <- if (r$confidence > 80) "#27ae60" else if (r$confidence > 50) "#f39c12" else "#e74c3c"
    tagList(div(style=paste0("font-size:36px; font-weight:800; color:", col, "; margin-top:5px;"), paste0(r$confidence, "%")))
  })
  
  output$trend_badge <- renderUI({
    r <- pred_result()
    clr <- if (grepl("improving", r$learning_trend)) "#27ae60" else if (grepl("Declining", r$learning_trend)) "#e74c3c" else "#f39c12"
    icn <- if (grepl("improving", r$learning_trend)) icon("arrow-trend-up") else if (grepl("Declining", r$learning_trend)) icon("arrow-trend-down") else icon("minus")
    tagList(div(style=paste0("font-size:18px; font-weight:700; color:", clr, "; margin-top:15px; text-transform:uppercase;"), icn, " ", r$learning_trend))
  })
  
  output$learner_type_badge <- renderUI({
    r <- pred_result()
    c_class <- if (r$consistency == "Consistent Performer") "bg-success" else "bg-warning text-dark"
    tagList(
      div(class="badge bg-secondary", style="font-size:22px; margin-top:5px;", r$learner_type),
      br(),
      div(class=paste("badge", c_class), style="font-size:16px; margin-top:10px;", r$consistency)
    )
  })
  
  output$perf_band_badge <- renderUI({
    r <- pred_result()
    tagList(div(class="badge bg-dark", style="font-size:22px; margin-top:10px;", paste("Predicted Band:", r$perf_band)))
  })

  output$recommendation_badge <- renderUI({
    r <- pred_result()
    cls <- if (grepl("URGENT", r$recommendation)) "bg-danger" else "bg-primary text-white"
    tagList(div(style="margin-top:12px; font-size:16px;", tags$span(r$recommendation, class = paste("badge p-2 text-wrap shadow-sm", cls))))
  })

  # Goal Planner Output Execution
  opt_result <- eventReactive(input$run_opt_btn, {
    optimize_for_target(rf_model, input$target_score, input$assignment_score, input$attendance, input$study_hours)
  })

  output$goal_planner_dashboard <- renderUI({
    res <- opt_result()
    req(res)
    
    feas_color <- if (res$probability > 70) "success" else if (res$probability > 30) "warning" else "danger"
    
    tagList(
      card(
        card_header("AI Prescriptive Strategy Matrix", class="bg-dark text-white"),
        card_body(
          div(
            h3("Feasibility Diagnosis:", span(paste0(res$probability, "%"), class=paste0("text-", feas_color))),
            p(style="font-size:18px; font-weight:500; line-height: 1.6;", res$message)
          )
        )
      ),
      layout_columns(
        col_widths = c(4,4,4),
        card(
          card_header("Study Volume (Hrs/Wk)", class="bg-primary text-white"),
          card_body(
             h4(paste("Current:", res$current$Study_Hours)),
             h4(paste("Required:", res$required$Study_Hours), class="text-primary"),
             hr(),
             p(paste("Delta:", res$required$Study_Hours - res$current$Study_Hours), style="font-weight:bold; color:#e74c3c;")
          )
        ),
        card(
          card_header("Attendance Tracking (%)", class="bg-primary text-white"),
          card_body(
             h4(paste("Current:", res$current$Attendance)),
             h4(paste("Required:", res$required$Attendance), class="text-primary"),
             hr(),
             p(paste("Delta:", res$required$Attendance - res$current$Attendance), style="font-weight:bold; color:#e74c3c;")
          )
        ),
        card(
          card_header("Assignment Integrity (Pts)", class="bg-primary text-white"),
          card_body(
             h4(paste("Current:", res$current$Assignments_Score)),
             h4(paste("Required:", res$required$Assignments_Score), class="text-primary"),
             hr(),
             p(paste("Delta:", res$required$Assignments_Score - res$current$Assignments_Score), style="font-weight:bold; color:#e74c3c;")
          )
        )
      )
    )
  })

  observeEvent(input$reset_btn, {
    updateSliderInput(session, "study_hours", value = 15)
    updateSliderInput(session, "attendance", value = 80)
    updateSliderInput(session, "assignment_score", value = 70)
    updateSliderInput(session, "previous_grade", value = 75)
  })

  output$learning_curve_plot <- renderPlot({
    r <- pred_result()
    req(r$score)
    
    CurveData <- data.frame(
      Timepoint = factor(c("T-1 (Past Grade)", "T-0 (Present Assignment Avg)", "T+1 (ML Future Projection)"), 
                         levels = c("T-1 (Past Grade)", "T-0 (Present Assignment Avg)", "T+1 (ML Future Projection)")),
      Trajectory = c(input$previous_grade, input$assignment_score, r$score)
    )
    
    clr <- if (grepl("improving", r$learning_trend)) "#27ae60" else if (grepl("Declining", r$learning_trend)) "#e74c3c" else "#f39c12"
    
    ggplot(CurveData, aes(x = Timepoint, y = Trajectory, group = 1)) +
      geom_line(colour = clr, linewidth = 2, alpha = 0.8) +
      geom_point(colour = "#2c3e50", size = 5) +
      geom_text(aes(label = round(Trajectory, 1)), vjust = -1.5, fontface = "bold", size = 5) +
      scale_y_continuous(limits = c(min(CurveData$Trajectory) - 10, max(CurveData$Trajectory) + 15)) +
      labs(x = "Assessment Timeline", y = "Relative Academic Scoring") +
      theme_dashboard()
  })

  output$importance_plot <- renderPlotly({
    req(nrow(feat_imp) > 0)
    p <- ggplot(feat_imp, aes(
      x = reorder(Feature, Importance), 
      y = Importance, 
      fill = Importance,
      text = paste("Metric:", Feature, "<br>Impact Score:", Importance)
    )) +
      geom_col(width = 0.65, show.legend = FALSE) +
      scale_fill_gradient(low = "#85c1e9", high = "#1a5276") +
      coord_flip() +
      scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
      labs(x = NULL, y = "Feature Impact Weight (MSE Variation)") +
      theme_dashboard()
      
    ggplotly(p, tooltip = "text") %>% layout(hovermode = "y", margin = list(l = 20, r = 20, b = 20, t = 20))
  })
  
  output$transparency_text <- renderUI({
    req(nrow(feat_imp) > 0)
    top_feature <- feat_imp$Feature[1]
    top_val <- feat_imp$Importance[1]
    second_feature <- feat_imp$Feature[2]
    
    HTML(paste0(
      "<div style='font-size:16px; line-height:1.7;'>",
      "<p>This Deep Analytics dashboard processes <strong>Random Forest Gini node purities</strong> across 150 independent decision trees to mathematically quantify the weight of every behavioral factor.</p>",
      "<hr>",
      "<h5>Key Findings:</h5>",
      "<p>The strongest universal predictor determining student success is currently <strong style='color:#e74c3c;'>", top_feature, "</strong> (Impact score: ", top_val, ").</p>",
      "<p>This is closely followed by <strong style='color:#3498db;'>", second_feature, "</strong> as the secondary decision component.</p>",
      "<hr>",
      "<p><strong>Startup Analytics Best Practice:</strong><br/>",
      "Focus intervention techniques strictly on the uppermost features within the interactive matrix. Manipulating highly weighted signals produces exponential returns globally.</p>",
      "</div>"
    ))
  })

  output$cluster_plot_1 <- renderPlot({
    req(nrow(student_data) > 0)
    plot_data <- head(student_data, 2000)
    ggplot(plot_data, aes(x = Study_Hours, y = Attendance, colour = Segment_Profile)) +
      geom_point(alpha = 0.45, size = 2) + 
      scale_colour_manual(values = c("Top Performers"="#27ae60", "High Potential"="#3498db", "Average"="#f39c12", "At Risk"="#e74c3c")) + 
      theme_dashboard() + labs(x="Study Hours", y="Attendance", colour="Segment Label")
  })
  
  output$cluster_plot_2 <- renderPlot({
    req(nrow(student_data) > 0)
    plot_data <- head(student_data, 2000)
    ggplot(plot_data, aes(x = Attendance, y = Assignments_Score, colour = Segment_Profile)) +
      geom_point(alpha = 0.45, size = 2) + 
      scale_colour_manual(values = c("Top Performers"="#27ae60", "High Potential"="#3498db", "Average"="#f39c12", "At Risk"="#e74c3c")) + 
      theme_dashboard() + labs(x="Attendance", y="Assignments Score", colour="Segment Label")
  })
}

shinyApp(ui = ui, server = server)
