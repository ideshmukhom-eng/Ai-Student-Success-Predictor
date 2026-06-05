# ==============================================================================
# AI Student Success Predictor - API Launcher
# ==============================================================================

# Ensure plumber is installed
if (!require(plumber)) {
  install.packages("plumber")
  library(plumber)
}

# Source the plumber API script
api_file <- "C:/Users/dell/PycharmProjects/Streamlit/Rpro/backend/plumber_api.R"

cat("Initializing Student Success Predictor API...\n")
pr <- plumb(api_file)

# Run the API on port 8000
# Allows connections universally (0.0.0.0) making it SaaS deployment-ready
cat("Starting service on port 8000...\n")
pr$run(host = "0.0.0.0", port = 8000)
