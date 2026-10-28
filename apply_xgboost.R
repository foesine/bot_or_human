# setup ------------------------------------------------------------------------
rm(list=ls())
gc()

packages <- c("data.table", "ggplot2", "lubridate", "stringi", "recipes", "rsample", 
              "xgboost", "pROC", "PRROC", "knitr", "scales", "tidymodels")

install_if_missing <- function(p) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
  library(p, character.only = TRUE)}

invisible(lapply(packages, install_if_missing))

data_path <- "" #specify
figure_path <- "" #specify
result_path <- "" #specify

# Prep data --------------------------------------------------------------------

# read survey data
svy_nosi <- as.data.table(readRDS(file.path(data_path, "surveydata/01_surveydata_nosi.rds")))
svy_bots <- as.data.table(readRDS(file.path(data_path, "surveydata/06_surveydata_bots.rds")))

# read para data
para_nosi <- as.data.table(readRDS(file.path(data_path, "paradata/01_paradata_nosi.rds")))
para_bots <- as.data.table(readRDS(file.path(data_path, "paradata/06_paradata_bots.rds")))

# read vignette data
# leave out for now

# merge the data types

nosi <- merge(svy_nosi, para_nosi[, !c("sample_id", "status"), with = FALSE], by = "id", all.x = TRUE)
bots <- merge(svy_bots, para_bots[, !c("sample_id", "status"), with = FALSE], by = "id", all.x = TRUE)

# create new vars

# months in current employment status
nosi[, ERW_STARTDATE := as.Date(as.character(ERW_STARTDATE),format = "%Y-%m-%d")]
nosi[, employment_duration_months := (year(as.Date("2025-10-01")) - year(ERW_STARTDATE)) * 12L + month(as.Date("2025-10-01")) - month(ERW_STARTDATE)]

bots[, ERW_STARTDATE := as.Date(as.character(ERW_STARTDATE),format = "%Y-%m-%d")]
bots[, employment_duration_months := (year(as.Date("2026-10-01")) - year(ERW_STARTDATE)) * 12L + month(as.Date("2026-10-01")) - month(ERW_STARTDATE)]


# drop variables hard identifiers
drop_vars <- c(grep("^(start_|end_)", names(nosi), value = TRUE), 
               "int_start", "int_end", "int_duration", "int_datetime", "int_time", "int_hour", "int_weekday", 
               "id", "SURVEY_START",
               "device", "password", "voucher_code", "QUESTION_ORDER", "QUESTION_ORDER_ALL", "incentive", "reminder",
               "OPEN_1_answered") # drop to create in line with the other "answered" indicators
nosi[, (drop_vars) := NULL]
bots[, (drop_vars) := NULL]

# merge data
stopifnot(setequal(names(nosi), names(bots)))
df <- rbindlist(list(nosi, bots), use.names = TRUE, fill = FALSE)

# only compare completes
df <- df[status == 1]

# check variable types
df[, lapply(.SD, class), .SDcols = setdiff(names(df)[!vapply(df, is.numeric, logical(1))], "sample_id")]

# transform character vars to numerical information
for (var in c("ERW_1A13", "ERW_RETRO_2A13", "ERW_RETRO_1A13", "OPEN_1")) {
  text <- stri_trim_both(as.character(df[[var]]))
  unanswered <- is.na(text) | text == "" | text == "-98"
  clean_text <- fifelse(unanswered, "", text)
  
  set(df, j = paste0(var, "_words"),
      value = as.integer(stri_count_words(clean_text)))
  
  set(df, j = paste0(var, "_characters"),
      value = as.integer(stri_length(clean_text)))
  
  set(df, j = paste0(var, "_answered"),
        value = as.integer(!unanswered))
}

# drop variables that are problematic variable types for model
drop_vars <- c("BIRTHDATE", "ERW_STARTDATE" ,"ERW_1A13", "ERW_RETRO_2A13", "ERW_RETRO_1A13", "OPEN_1")
df[, (drop_vars) := NULL]


# Check response times ---------------------------------------------------------
duration_vars <- grep("^duration_", names(df), value = TRUE)

duration_long <- data.table::melt(
  df[sample_id %in% c("bots", "nosi")],
  id.vars = "sample_id",
  measure.vars = duration_vars,
  variable.name = "variable",
  value.name = "duration")

duration_long[, duration := as.numeric(as.character(duration))]

setDT(duration_long)

duration_long[, item := sub("^duration_", "", as.character(variable))]
duration_long[, respondent_type := factor(sample_id, levels = c("bots", "nosi"), labels = c("Bots", "Human respondents"))]

item_order <- duration_long[!is.na(duration), .(median_duration = median(duration)), by = .(item, sample_id)]
item_order <- dcast(item_order, item ~ sample_id, value.var = "median_duration")
item_order[, median_difference := bots - nosi]
setorder(item_order, median_difference)

duration_long[, item := factor(item, levels = item_order$item)]

bot_availability <- duration_long[sample_id == "bots", .(n_valid = sum(!is.na(duration)), n_unique = uniqueN(duration[!is.na(duration)])), by = item]
no_bot_items <- as.character(bot_availability[n_valid == 0 | n_unique <= 1, item])

boxplot_data <- duration_long[!(sample_id == "bots" & as.character(item) %in% no_bot_items)]
no_bot_data <- data.table(item = factor(no_bot_items, levels = levels(duration_long$item)), label = "No bot timing")

separator_positions <- seq(0.5, length(levels(duration_long$item)) + 0.5, by = 1)

duration_boxplot <- ggplot(boxplot_data[!is.na(duration)], aes(x = duration, y = item, fill = respondent_type)) +
  geom_hline(yintercept = separator_positions, color = "grey82", linewidth = 0.40) +
  geom_boxplot(position = position_dodge2(width = 0.70, preserve = "single", padding = 0.15), width = 0.48, alpha = 0.50, linewidth = 0.40, outlier.shape = NA) +
  geom_text(data = no_bot_data, aes(x = 79, y = item, label = label), inherit.aes = FALSE, color = "#D73027", size = 2.7, hjust = 1) +
  scale_fill_manual(values = c("Bots" = "#D73027", "Human respondents" = "#4575B4")) +
  scale_x_continuous(breaks = seq(0, 80, by = 10), expand = expansion(mult = c(0.01, 0.01))) +
  coord_cartesian(xlim = c(0, 80)) +
  labs(title = "Response durations of bots and human respondents", 
       subtitle = "Items are ordered by the difference between bot and human median response times", 
       x = "Response duration in seconds", y = NULL, fill = NULL, 
       caption = "Outliers are not displayed. Items without a varying bot timing distribution are marked explicitly.") +
  theme_minimal(base_size = 12) +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(color = "grey92", linewidth = 0.35),
        panel.grid.minor = element_blank(),
        legend.position = "top",
        plot.title = element_text(face = "bold"),
        plot.caption = element_text(hjust = 0.5, size = 9),
        axis.text.y = element_text(size = 9),
        plot.margin = margin(10, 15, 10, 10))


duration_boxplot

ggsave(file.path(figure_path, "response_duration_boxplots.png"), plot = duration_boxplot, width = 14, height = 15, units = "in", dpi = 300)

               
# Run classification -----------------------------------------------------

set.seed(123)

bot_label <- "bots"
response_time_vars <- grep("^duration_", names(df), value = TRUE)
persona_time_vars <- c("duration_ALT", "duration_GES", "duration_BULA")
model_order <- c("All response times", "Without persona response times", "Answers only")

# Check and prepare target

table(df$sample_id, useNA = "ifany")

if (anyNA(df$sample_id)) stop("The target variable sample_id contains missing values.")
if (length(unique(df$sample_id)) != 2) stop("sample_id must contain exactly two classes.")
if (!bot_label %in% unique(df$sample_id)) stop("The value specified in bot_label does not occur in sample_id.")

missing_persona_vars <- setdiff(persona_time_vars, names(df))
if (length(missing_persona_vars) > 0) warning("Missing persona response-time variables: ", paste(missing_persona_vars, collapse = ", "))

model_df <- as.data.frame(df)
model_df$target <- as.integer(model_df$sample_id == bot_label)
model_df$sample_id <- NULL

table(model_df$target)

# One stratified split for all specifications

set.seed(123)

split <- initial_split(model_df, prop = 0.80, strata = target)
train_df <- training(split)
test_df <- testing(split)

table(train_df$target)
table(test_df$target)

# Preprocess, estimate, and evaluate one specification

fit_xgboost <- function(train_data, test_data, drop_vars = character(), model_name) {
  drop_vars <- intersect(drop_vars, names(train_data))
  train_data <- train_data[, !names(train_data) %in% drop_vars, drop = FALSE]
  test_data <- test_data[, !names(test_data) %in% drop_vars, drop = FALSE]
  
  rec <- recipe(target ~ ., data = train_data) |>
    step_string2factor(all_nominal_predictors()) |>
    step_unknown(all_nominal_predictors(), new_level = "missing") |>
    step_novel(all_nominal_predictors(), new_level = "new") |>
    step_zv(all_predictors()) |>
    step_dummy(all_nominal_predictors(), one_hot = TRUE)
  
  rec_prep <- prep(rec, training = train_data, retain = TRUE)
  train_processed <- bake(rec_prep, new_data = NULL)
  test_processed <- bake(rec_prep, new_data = test_data)
  
  predictor_names <- setdiff(names(train_processed), "target")
  x_train <- as.matrix(train_processed[, predictor_names])
  x_test <- as.matrix(test_processed[, predictor_names])
  y_train <- train_processed$target
  y_test <- test_processed$target
  
  dtrain <- xgb.DMatrix(data = x_train, label = y_train, missing = NA)
  dtest <- xgb.DMatrix(data = x_test, label = y_test, missing = NA)
  scale_pos_weight <- sum(y_train == 0) / sum(y_train == 1)
  
  params <- list(objective = "binary:logistic", eval_metric = "aucpr", eta = 0.05, max_depth = 4, min_child_weight = 1,
                 subsample = 0.80, colsample_bytree = 0.80, lambda = 1, alpha = 0, scale_pos_weight = scale_pos_weight,
                 nthread = parallel::detectCores())
  
  set.seed(123)
  cv_fit <- xgb.cv(params = params, data = dtrain, nrounds = 1000, nfold = 5, stratified = TRUE,
                   early_stopping_rounds = 30, maximize = TRUE, verbose = 0)
  
  best_nrounds <- cv_fit$early_stop$best_iteration
  
  set.seed(123)
  xgb_fit <- xgb.train(params = params, data = dtrain, nrounds = best_nrounds, verbose = 0)
  
  probability <- predict(xgb_fit, dtest)
  threshold <- 0.50
  prediction <- as.integer(probability >= threshold)
  
  tn <- sum(prediction == 0 & y_test == 0)
  fp <- sum(prediction == 1 & y_test == 0)
  fn <- sum(prediction == 0 & y_test == 1)
  tp <- sum(prediction == 1 & y_test == 1)
  
  accuracy <- (tp + tn) / length(y_test)
  precision <- ifelse(tp + fp == 0, NA_real_, tp / (tp + fp))
  recall <- ifelse(tp + fn == 0, NA_real_, tp / (tp + fn))
  specificity <- ifelse(tn + fp == 0, NA_real_, tn / (tn + fp))
  f1 <- ifelse(is.na(precision) | is.na(recall) | precision + recall == 0, NA_real_, 2 * precision * recall / (precision + recall))
  
  roc_object <- pROC::roc(response = y_test, predictor = probability, levels = c(0, 1), direction = "<", quiet = TRUE)
  pr_object <- PRROC::pr.curve(scores.class0 = probability[y_test == 1], scores.class1 = probability[y_test == 0], curve = TRUE)
  
  metrics <- data.table(
    Model = model_name,
    Metric = c("Accuracy", "Bot precision", "Bot recall", "Specificity", "F1", "ROC AUC", "PR AUC"),
    Value = c(accuracy, precision, recall, specificity, f1, as.numeric(pROC::auc(roc_object)), pr_object$auc.integral)
  )
  
  confusion <- data.table(Model = model_name, Actual = ifelse(y_test == 1, "Bot", "Human"),
                          Predicted = ifelse(prediction == 1, "Bot", "Human"))[, .(n = .N), by = .(Model, Actual, Predicted)]
  
  importance <- as.data.table(xgb.importance(feature_names = colnames(x_train), model = xgb_fit))
  importance[, Model := model_name]
  
  predictions <- data.table(Model = model_name, observed = ifelse(y_test == 1, "bot", "human"),
                            probability_bot = probability, predicted = ifelse(prediction == 1, "bot", "human"))
  
  list(model = xgb_fit, recipe = rec_prep, cv = cv_fit, best_nrounds = best_nrounds, metrics = metrics,
       confusion = confusion, importance = importance, predictions = predictions, y_test = y_test)
}

# Fit all three specifications

fit_with_times <- fit_xgboost(train_df, test_df, drop_vars = character(), model_name = "All response times")

fit_without_persona_times <- fit_xgboost(train_df, test_df, drop_vars = persona_time_vars,
                                         model_name = "Without persona response times")

fit_without_times <- fit_xgboost(train_df, test_df, drop_vars = response_time_vars, model_name = "Answers only")

best_rounds <- data.table(
  Model = model_order,
  Best_rounds = c(fit_with_times$best_nrounds, fit_without_persona_times$best_nrounds, fit_without_times$best_nrounds)
)

best_rounds

# Performance comparison table

metrics_long <- rbind(fit_with_times$metrics, fit_without_persona_times$metrics, fit_without_times$metrics)
metric_order <- c("Accuracy", "Bot precision", "Bot recall", "Specificity", "F1", "ROC AUC", "PR AUC")

metrics_long[, Metric := factor(Metric, levels = metric_order)]
metrics_long[, Model := factor(Model, levels = model_order)]

performance_table <- dcast(metrics_long, Metric ~ Model, value.var = "Value")
performance_table[, Metric := factor(Metric, levels = metric_order)]

setorder(performance_table, Metric)
setcolorder(performance_table, c("Metric", model_order))

numeric_columns <- setdiff(names(performance_table), "Metric")
performance_table[, (numeric_columns) := lapply(.SD, round, digits = 3), .SDcols = numeric_columns]

performance_table

latex_table <- knitr::kable(performance_table, format = "latex", booktabs = TRUE, digits = 3,
                            caption = "Predictive performance of the XGBoost models on the test sample.",
                            label = "model_performance", align = c("l", rep("c", length(numeric_columns))))

latex_table

# Confusion-matrix figure

confusion_data <- rbind(fit_with_times$confusion, fit_without_persona_times$confusion, fit_without_times$confusion)
confusion_grid <- CJ(Model = model_order, Actual = c("Human", "Bot"), Predicted = c("Human", "Bot"))

confusion_data <- merge(confusion_grid, confusion_data, by = c("Model", "Actual", "Predicted"), all.x = TRUE)
confusion_data[is.na(n), n := 0L]
confusion_data[, Share := n / sum(n), by = .(Model, Actual)]
confusion_data[, Actual := factor(Actual, levels = c("Bot", "Human"))]
confusion_data[, Predicted := factor(Predicted, levels = c("Human", "Bot"))]
confusion_data[, Model := factor(Model, levels = model_order)]

facet_labels <- c("All response times" = "All response times",
                  "Without persona response times" = "Without persona\nresponse times",
                  "Answers only" = "Answers only")

confusion_plot <- ggplot(confusion_data, aes(x = Predicted, y = Actual, fill = Share)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = sprintf("%d\n(%.1f%%)", n, 100 * Share)), size = 4) +
  facet_wrap(~Model, nrow = 1, labeller = as_labeller(facet_labels)) +
  scale_fill_gradient(low = "#F2F6FA", high = "#4575B4", labels = scales::label_percent()) +
  guides(fill = guide_colorbar(title.position = "top", barwidth = grid::unit(5, "cm"), barheight = grid::unit(0.4, "cm"))) +
  coord_equal() +
  labs(title = "Classification results on the test sample", x = "Predicted respondent type",
       y = "Actual respondent type", fill = "Within-class share") +
  theme_minimal(base_size = 12) +
  theme(panel.grid = element_blank(),
        strip.text = element_text(face = "bold", size = 10, lineheight = 0.9, margin = margin(b = 6)),
        plot.title = element_text(face = "bold"), legend.position = "bottom",
        legend.title = element_text(hjust = 0.5), plot.margin = margin(10, 15, 10, 10))

confusion_plot

ggsave(file.path(figure_path, "xgboost_confusion_matrices.png"), plot = confusion_plot,
       width = 11, height = 5.5, units = "in", dpi = 300)

# Feature-importance figure

importance_data <- rbind(fit_with_times$importance, fit_without_persona_times$importance,
                         fit_without_times$importance, fill = TRUE)

importance_data[, Model := factor(Model, levels = model_order)]
importance_data <- importance_data[order(Model, -Gain), head(.SD, 10), by = Model]

importance_data[, Feature_label := fifelse(
  grepl("^duration_", Feature),
  paste0(sub("^duration_", "", Feature), " (response time)"),
  Feature
)]

importance_data <- importance_data[order(Model, Gain)]
importance_data[, Feature_panel := paste(Model, Feature_label, sep = "___")]
importance_data[, Feature_panel := factor(Feature_panel, levels = unique(Feature_panel))]

importance_plot <- ggplot(importance_data, aes(x = Gain, y = Feature_panel, fill = Model)) +
  geom_col(width = 0.7, alpha = 0.85, show.legend = FALSE) +
  facet_wrap(~Model, scales = "free_y", ncol = 1) +
  scale_y_discrete(labels = \(x) sub("^.*___", "", x)) +
  scale_x_continuous(labels = scales::label_percent(accuracy = 1), expand = expansion(mult = c(0, 0.05))) +
  scale_fill_manual(values = c("All response times" = "#D73027",
                               "Without persona response times" = "#E69F00",
                               "Answers only" = "#4575B4")) +
  labs(title = "Feature importance in the XGBoost models",
       subtitle = "Up to ten predictors with the highest gain are shown",
       x = "Share of total gain", y = NULL,
       caption = "Gain measures the relative improvement in the model objective attributable to splits involving each predictor.") +
  theme_minimal(base_size = 12) +
  theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
        strip.text = element_text(face = "bold", hjust = 0),
        plot.title = element_text(face = "bold"), plot.caption = element_text(hjust = 0))

importance_plot

ggsave(file.path(figure_path, "xgboost_feature_importance.png"), plot = importance_plot,
       width = 11, height = 11, units = "in", dpi = 300)

# Save respondent-level predictions -------------------------------------

stopifnot(
  identical(fit_with_times$predictions$observed, fit_without_persona_times$predictions$observed),
  identical(fit_with_times$predictions$observed, fit_without_times$predictions$observed)
)

test_predictions <- data.table(
  test_row = seq_along(fit_with_times$y_test),
  observed = fit_with_times$predictions$observed,
  probability_all_times = fit_with_times$predictions$probability_bot,
  predicted_all_times = fit_with_times$predictions$predicted,
  probability_without_persona_times = fit_without_persona_times$predictions$probability_bot,
  predicted_without_persona_times = fit_without_persona_times$predictions$predicted,
  probability_answers_only = fit_without_times$predictions$probability_bot,
  predicted_answers_only = fit_without_times$predictions$predicted
)

head(test_predictions)

# Distributions underlying the most important answer predictors

plot_df <- copy(as.data.table(df))
plot_df[, respondent_type := factor(sample_id, levels = c("nosi", "bots"), labels = c("Human respondents", "Bots"))]

plot_df[, bil_code := sub("^X", "", as.character(BIL_1))]
plot_df[bil_code %in% c("", "-98", "-99"), bil_code := NA_character_]

plot_df[, orient_4_plot := suppressWarnings(as.numeric(as.character(ORIENT_4)))]
plot_df[orient_4_plot < 0, orient_4_plot := NA_real_]

plot_df[, ver_4_plot := suppressWarnings(as.numeric(as.character(VER_4)))]
plot_df[ver_4_plot < 0, ver_4_plot := NA_real_]

distribution_data <- rbindlist(list(
  plot_df[, .(
    respondent_type,
    Feature = "BIL_1\nHighest general schooling qualification",
    Response = fcase(
      bil_code == "1", "No qualification",
      bil_code == "2", "Lower secondary",
      bil_code == "3", "Intermediate secondary",
      bil_code == "4", "University entrance",
      default = NA_character_
    ),
    Response_order = suppressWarnings(as.numeric(bil_code))
  )],
  plot_df[, .(
    respondent_type,
    Feature = "ORIENT_4\nImportance of a recognized and respected job",
    Response = fcase(
      orient_4_plot == 1, "Very important",
      orient_4_plot == 2, "Rather important",
      orient_4_plot == 3, "Rather unimportant",
      orient_4_plot == 4, "Not important",
      default = NA_character_
    ),
    Response_order = orient_4_plot
  )],
  plot_df[, .(
    respondent_type,
    Feature = "VER_4\nTrust in politicians",
    Response = as.character(ver_4_plot),
    Response_order = ver_4_plot
  )]
), use.names = TRUE)

feature_order <- c(
  "BIL_1\nHighest general schooling qualification",
  "ORIENT_4\nImportance of a recognized and respected job",
  "VER_4\nTrust in politicians"
)

distribution_data <- distribution_data[!is.na(Response) & !is.na(respondent_type)]
distribution_summary <- distribution_data[, .(n = .N), by = .(Feature, Response, Response_order, respondent_type)]
distribution_summary[, Share := n / sum(n), by = .(Feature, respondent_type)]
distribution_summary[, Feature := factor(Feature, levels = feature_order)]

setorder(distribution_summary, Feature, Response_order)

distribution_summary[, Response_plot := sprintf("%02d___%s", Response_order, Response)]
distribution_summary[, Response_plot := factor(Response_plot, levels = unique(Response_plot))]

answer_distribution_plot <- ggplot(distribution_summary, aes(x = Response_plot, y = Share, fill = respondent_type)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7, alpha = 0.85) +
  geom_text(aes(label = scales::percent(Share, accuracy = 1)), position = position_dodge(width = 0.8), vjust = -0.3, size = 3) +
  facet_wrap(~Feature, scales = "free_x", ncol = 1) +
  scale_x_discrete(labels = \(x) sub("^\\d+___", "", x)) +
  scale_y_continuous(labels = scales::label_percent(), expand = expansion(mult = c(0, 0.12))) +
  scale_fill_manual(values = c("Human respondents" = "#4575B4", "Bots" = "#D73027")) +
  labs(title = "Distributions of the most important survey items", subtitle = "Percentages are calculated separately within each respondent type", x = NULL, y = "Share of respondents", fill = NULL) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(), strip.text = element_text(face = "bold", hjust = 0), plot.title = element_text(face = "bold"), legend.position = "top", axis.text.x = element_text(size = 9))

answer_distribution_plot

ggsave(file.path(figure_path, "xgboost_answer_distributions.png"), 
       plot = answer_distribution_plot, width = 11, height = 9, units = "in", dpi = 300)

# Robustness checks ------------------------------------------------------------

# Settings

bot_label <- "bots"
split_seeds <- 1:50
bot_fractions <- c("100%" = 1, "50%" = 0.5, "20%" = 0.2, "10%" = 0.1)
threshold <- 0.50
force_refit <- FALSE
cache_version <- "v1"

detected_cores <- parallel::detectCores(logical = FALSE)
xgb_threads <- if (is.na(detected_cores)) 2L else max(1L, min(4L, detected_cores))

dir.create(result_path, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_path, recursive = TRUE, showWarnings = FALSE)

if (anyNA(df$sample_id)) stop("The target variable sample_id contains missing values.")
if (length(unique(df$sample_id)) != 2) stop("sample_id must contain exactly two classes.")
if (!bot_label %in% unique(df$sample_id)) stop("The value specified in bot_label does not occur in sample_id.")

# Prepare target and predictor specifications

model_df <- as.data.frame(df)
model_df$target <- factor(ifelse(model_df$sample_id == bot_label, "bot", "human"), levels = c("human", "bot"))
model_df$sample_id <- NULL

all_predictors <- setdiff(names(model_df), "target")
response_time_vars <- grep("^duration_", all_predictors, value = TRUE)
persona_response_time_vars <- intersect(c("duration_ALT", "duration_GES", "duration_BULA"), response_time_vars)

specifications <- list(
  "All response times" = all_predictors,
  "Without persona response times" = setdiff(all_predictors, persona_response_time_vars),
  "Answers only" = setdiff(all_predictors, response_time_vars))

model_order <- names(specifications)
fraction_order <- names(bot_fractions)

sample_size_table <- as.data.table(model_df)[, .N, by = target]
setnames(sample_size_table, c("target", "N"), c("respondent_type", "N"))

cat("\nAnalytical sample:\n")
print(sample_size_table)

cat("\nNumber of predictors:\n")
print(sapply(specifications, length))

# File paths

cache_path <- file.path(result_path, paste0("xgboost_run_cache_", cache_version))

result_rds <- file.path(result_path, "xgboost_robustness_results.rds")
metrics_file <- file.path(result_path, "xgboost_robustness_metrics.csv")
summary_file <- file.path(result_path, "xgboost_robustness_summary.csv")
importance_file <- file.path(result_path, "xgboost_robustness_importance.csv")
stability_file <- file.path(result_path, "xgboost_feature_stability.csv")
predictions_file <- file.path(result_path, "xgboost_robustness_predictions.csv")
sample_size_file <- file.path(result_path, "xgboost_sample_sizes.csv")
training_size_file <- file.path(result_path, "xgboost_training_bot_counts.csv")

performance_figure_file <- file.path(figure_path, "xgboost_robustness_performance.png")
stability_figure_file <- file.path(figure_path, "xgboost_feature_stability.png")

safe_save_rds <- function(object, path) {
  temporary_path <- paste0(path, ".tmp")
  saveRDS(object, temporary_path)
  
  if (!file.rename(temporary_path, path)) {
    file.copy(temporary_path, path, overwrite = TRUE)
    unlink(temporary_path)
  }
  
  invisible(path)
}

# Function for fitting one model

# XGBoost setup 

detected_cores <- parallel::detectCores(logical = FALSE)
xgb_threads <- if (is.na(detected_cores)) 2L else max(1L, min(4L, detected_cores))

# Create CV folds containing humans and bots 

make_stratified_folds <- function(y, nfolds, seed) {
  set.seed(seed)
  
  human_indices <- sample(which(y == 0))
  bot_indices <- sample(which(y == 1))
  
  human_fold <- rep(seq_len(nfolds), length.out = length(human_indices))
  bot_fold <- rep(seq_len(nfolds), length.out = length(bot_indices))
  
  folds <- lapply(seq_len(nfolds), function(fold) {
    indices <- c(
      human_indices[human_fold == fold],
      bot_indices[bot_fold == fold])
    
    sample(indices)
  })
  
  valid_folds <- vapply(
    folds,
    function(indices) length(unique(y[indices])) == 2,
    logical(1))
  
  if (!all(valid_folds)) stop("At least one cross-validation fold does not contain both classes.")
  
  folds
}

# Fit one XGBoost model

fit_xgb_model <- function(train_df, test_df, predictors, model_name, seed, bot_fraction, threshold = 0.50) {
  train_model <- train_df[, c("target", predictors), drop = FALSE]
  test_model <- test_df[, c("target", predictors), drop = FALSE]
  
  rec <- recipe(target ~ ., data = train_model) |>
    step_string2factor(all_nominal_predictors()) |>
    step_unknown(all_nominal_predictors(), new_level = "missing") |>
    step_novel(all_nominal_predictors(), new_level = "new") |>
    step_zv(all_predictors()) |>
    step_dummy(all_nominal_predictors(), one_hot = TRUE)
  
  rec_prep <- prep(rec, training = train_model, retain = TRUE)
  train_processed <- bake(rec_prep, new_data = NULL)
  test_processed <- bake(rec_prep, new_data = test_model)
  
  x_train <- data.matrix(
    train_processed[, setdiff(names(train_processed), "target"), drop = FALSE])
  
  x_test <- data.matrix(
    test_processed[, setdiff(names(test_processed), "target"), drop = FALSE])
  
  y_train <- as.integer(train_processed$target) - 1L
  y_test <- as.integer(test_processed$target) - 1L
  
  if (length(unique(y_train)) != 2) stop("The training data do not contain both classes.")
  if (length(unique(y_test)) != 2) stop("The test data do not contain both classes.")
  
  dtrain <- xgb.DMatrix(data = x_train, label = y_train, missing = NA)
  dtest <- xgb.DMatrix(data = x_test, label = y_test, missing = NA)
  
  n_training_humans <- sum(y_train == 0)
  n_training_bots <- sum(y_train == 1)
  scale_pos_weight <- n_training_humans / n_training_bots
  nfolds <- min(5L, n_training_humans, n_training_bots)
  
  if (nfolds < 2L) stop("Too few bot observations for cross-validation.")
  
  cv_folds <- make_stratified_folds(
    y = y_train, nfolds = nfolds, seed = seed)
  
  params <- list(
    objective = "binary:logistic", eval_metric = "aucpr", eta = 0.05,
    max_depth = 4, min_child_weight = 1, subsample = 0.80,
    colsample_bytree = 0.80, lambda = 1, alpha = 0,
    scale_pos_weight = scale_pos_weight, nthread = xgb_threads,
    seed = seed)
  
  set.seed(seed)
  
  cv_fit <- xgb.cv(
    params = params, data = dtrain, nrounds = 1000,
    nfold = nfolds, folds = cv_folds, stratified = FALSE,
    early_stopping_rounds = 30, maximize = TRUE, verbose = 0)
  
  best_nrounds <- cv_fit$early_stop$best_iteration
  
  if (is.null(best_nrounds) && !is.null(cv_fit$best_iteration)) {
    best_nrounds <- cv_fit$best_iteration
  }
  
  if (is.null(best_nrounds) || !is.finite(best_nrounds)) {
    test_metric <- cv_fit$evaluation_log$test_aucpr_mean
    
    if (all(is.na(test_metric))) {
      stop("Cross-validation did not produce a valid PR AUC value.")
    }
    
    best_nrounds <- which.max(test_metric)
  }
  
  set.seed(seed)
  
  xgb_fit <- xgb.train(
    params = params, data = dtrain,
    nrounds = best_nrounds, verbose = 0)
  
  probability_bot <- predict(xgb_fit, dtest)
  predicted_bot <- as.integer(probability_bot >= threshold)
  
  tn <- sum(predicted_bot == 0 & y_test == 0)
  fp <- sum(predicted_bot == 1 & y_test == 0)
  fn <- sum(predicted_bot == 0 & y_test == 1)
  tp <- sum(predicted_bot == 1 & y_test == 1)
  
  accuracy <- (tp + tn) / length(y_test)
  precision <- ifelse(tp + fp == 0, NA_real_, tp / (tp + fp))
  recall <- ifelse(tp + fn == 0, NA_real_, tp / (tp + fn))
  specificity <- ifelse(tn + fp == 0, NA_real_, tn / (tn + fp))
  
  f1 <- ifelse(
    is.na(precision) | is.na(recall) | precision + recall == 0,
    NA_real_, 2 * precision * recall / (precision + recall))
  
  roc_object <- pROC::roc(
    response = y_test, predictor = probability_bot,
    levels = c(0, 1), direction = "<", quiet = TRUE)
  
  roc_auc <- as.numeric(pROC::auc(roc_object))
  
  pr_object <- PRROC::pr.curve(
    scores.class0 = probability_bot[y_test == 1],
    scores.class1 = probability_bot[y_test == 0],
    curve = FALSE)
  
  pr_auc <- pr_object$auc.integral
  
  metrics <- data.table(
    model = model_name, bot_fraction = bot_fraction,
    Metric = c(
      "Accuracy", "Bot precision", "Bot recall",
      "Specificity", "F1", "ROC AUC", "PR AUC"),
    Value = c(
      accuracy, precision, recall, specificity,
      f1, roc_auc, pr_auc),
    best_nrounds = best_nrounds,
    n_training_humans = n_training_humans,
    n_training_bots = n_training_bots,
    n_test_humans = sum(y_test == 0),
    n_test_bots = sum(y_test == 1),
    TP = tp, FP = fp, TN = tn, FN = fn)
  
  importance <- as.data.table(
    xgb.importance(feature_names = colnames(x_train), model = xgb_fit))
  
  if (nrow(importance) > 0) {
    importance[, rank := frank(-Gain, ties.method = "min")]
  }
  
  predictions <- data.table(
    observed = factor(
      ifelse(y_test == 1, "bot", "human"),
      levels = c("human", "bot")),
    probability_bot = probability_bot,
    predicted = factor(
      ifelse(predicted_bot == 1, "bot", "human"),
      levels = c("human", "bot")))
  
  list(
    metrics = metrics,
    importance = importance,
    predictions = predictions)
}

# Functions for aggregating results

summarize_performance <- function(metrics) {
  summary <- metrics[, .(
    training_bots = round(median(n_training_bots)),
    minimum_training_bots = min(n_training_bots),
    maximum_training_bots = max(n_training_bots),
    median = median(Value, na.rm = TRUE),
    mean = mean(Value, na.rm = TRUE),
    p05 = quantile(Value, 0.05, na.rm = TRUE),
    p25 = quantile(Value, 0.25, na.rm = TRUE),
    p75 = quantile(Value, 0.75, na.rm = TRUE),
    p95 = quantile(Value, 0.95, na.rm = TRUE),
    minimum = min(Value, na.rm = TRUE),
    maximum = max(Value, na.rm = TRUE)
  ), by = .(model, bot_fraction, Metric)]
  
  setorder(summary, model, bot_fraction, Metric)
  summary
}

summarize_feature_stability <- function(metrics, importance) {
  importance_runs <- unique(metrics[, .(model, bot_fraction, split_seed)])
  importance_features <- unique(importance[, .(model, bot_fraction, Feature)])
  
  importance_grid <- merge(
    importance_runs, importance_features,
    by = c("model", "bot_fraction"), allow.cartesian = TRUE)
  
  importance_grid <- merge(
    importance_grid,
    importance[, .(model, bot_fraction, split_seed, Feature, Gain, rank)],
    by = c("model", "bot_fraction", "split_seed", "Feature"), all.x = TRUE)
  
  importance_grid[is.na(Gain), Gain := 0]
  importance_grid[is.na(rank), rank := .Machine$integer.max]
  
  stability <- importance_grid[, .(
    mean_gain = mean(Gain),
    median_gain = median(Gain),
    p05_gain = quantile(Gain, 0.05),
    p95_gain = quantile(Gain, 0.95),
    top_predictor_share = mean(rank == 1),
    top_five_share = mean(rank <= 5),
    top_ten_share = mean(rank <= 10)
  ), by = .(model, bot_fraction, Feature)]
  
  setorder(stability, model, bot_fraction, -top_predictor_share, -mean_gain)
  stability
}

make_run_key <- function(split_seed, bot_fraction, model) {
  paste(split_seed, bot_fraction, model, sep = "||")
}

# Load existing consolidated results 

robustness_metrics <- NULL
robustness_predictions <- NULL
robustness_importance <- NULL
feature_stability <- NULL

if (!force_refit && file.exists(result_rds)) {
  saved_results <- tryCatch(readRDS(result_rds), error = function(e) NULL)
  
  if (!is.null(saved_results)) {
    robustness_metrics <- saved_results$robustness_metrics
    robustness_predictions <- saved_results$robustness_predictions
    robustness_importance <- saved_results$robustness_importance
    feature_stability <- saved_results$feature_stability
    message("Existing consolidated results loaded from: ", result_rds)
  }
}

if (!force_refit && is.null(robustness_metrics) && file.exists(metrics_file)) robustness_metrics <- fread(metrics_file)
if (!force_refit && is.null(robustness_predictions) && file.exists(predictions_file)) robustness_predictions <- fread(predictions_file)
if (!force_refit && is.null(robustness_importance) && file.exists(importance_file)) robustness_importance <- fread(importance_file)
if (!force_refit && is.null(feature_stability) && file.exists(stability_file)) feature_stability <- fread(stability_file)

if (is.null(robustness_metrics)) robustness_metrics <- data.table()
if (is.null(robustness_predictions)) robustness_predictions <- data.table()
if (is.null(robustness_importance)) robustness_importance <- data.table()
if (is.null(feature_stability)) feature_stability <- data.table()

# Standardize loaded data before checking completed runs

if (nrow(robustness_metrics) > 0) {
  setDT(robustness_metrics)
  robustness_metrics[, `:=`(
    model = as.character(model),
    bot_fraction = as.character(bot_fraction))]
  
  robustness_metrics <- robustness_metrics[
    split_seed %in% split_seeds &
      model %in% model_order &
      bot_fraction %in% fraction_order]
}

if (nrow(robustness_predictions) > 0) {
  setDT(robustness_predictions)
  robustness_predictions[, `:=`(
    model = as.character(model),
    bot_fraction = as.character(bot_fraction))]
  
  robustness_predictions <- robustness_predictions[
    split_seed %in% split_seeds &
      model %in% model_order &
      bot_fraction %in% fraction_order]
}

if (nrow(robustness_importance) > 0) {
  setDT(robustness_importance)
  robustness_importance[, `:=`(
    model = as.character(model),
    bot_fraction = as.character(bot_fraction))]
  
  robustness_importance <- robustness_importance[
    split_seed %in% split_seeds &
      model %in% model_order &
      bot_fraction %in% fraction_order]
}

if (nrow(feature_stability) > 0) {
  setDT(feature_stability)
  feature_stability[, `:=`(
    model = as.character(model),
    bot_fraction = as.character(bot_fraction))]
  
  feature_stability <- feature_stability[
    model %in% model_order &
      bot_fraction %in% fraction_order]
}

# Identify completed and missing model runs

expected_runs <- CJ(
  split_seed = split_seeds,
  bot_fraction = fraction_order,
  model = model_order,
  unique = TRUE)

expected_runs[, run_key := make_run_key(split_seed, bot_fraction, model)]

if (nrow(robustness_metrics) > 0) {
  completed_runs <- unique(
    robustness_metrics[, .(split_seed, bot_fraction, model)]
  )
  
  completed_runs[, run_key := make_run_key(split_seed, bot_fraction, model)]
  completed_run_keys <- completed_runs$run_key
} else {
  completed_run_keys <- character()
}

missing_run_keys <- setdiff(expected_runs$run_key, completed_run_keys)

message(
  length(completed_run_keys), " of ", nrow(expected_runs),
  " model runs already available; ",
  length(missing_run_keys), " runs remain."
)

# Fit only missing runs or load them from individual cache files

new_metric_results <- list()
new_importance_results <- list()
new_prediction_results <- list()
new_run_number <- 0L

if (length(missing_run_keys) > 0) {
  for (split_seed in split_seeds) {
    set.seed(split_seed)
    split <- initial_split(model_df, prop = 0.80, strata = target)
    full_train_df <- training(split)
    test_df <- testing(split)
    
    human_rows <- which(full_train_df$target == "human")
    bot_rows <- which(full_train_df$target == "bot")
    
    set.seed(10000L + split_seed)
    bot_order <- sample(bot_rows)
    
    for (fraction_index in seq_along(bot_fractions)) {
      fraction_name <- names(bot_fractions)[fraction_index]
      fraction <- bot_fractions[[fraction_index]]
      n_bots_keep <- max(1L, round(length(bot_rows) * fraction))
      
      selected_rows <- c(human_rows, bot_order[seq_len(n_bots_keep)])
      train_df <- full_train_df[selected_rows, , drop = FALSE]
      
      set.seed(20000L + split_seed + fraction_index)
      train_df <- train_df[sample(seq_len(nrow(train_df))), , drop = FALSE]
      
      for (specification_index in seq_along(specifications)) {
        specification_name <- names(specifications)[specification_index]
        run_key <- make_run_key(split_seed, fraction_name, specification_name)
        
        if (!run_key %chin% missing_run_keys) next
        
        model_seed <- 100000L + split_seed * 100L + fraction_index * 10L + specification_index
        run_cache_file <- file.path(
          cache_path,
          sprintf("seed_%03d_fraction_%03d_spec_%02d.rds", split_seed, round(fraction * 100), specification_index))
        
        run_result <- NULL
        
        if (file.exists(run_cache_file) && !force_refit) {
          run_result <- tryCatch(readRDS(run_cache_file), error = function(e) NULL)
          if (!is.null(run_result)) message("Loaded cache: seed ", split_seed, ", bots ", fraction_name, ", ", specification_name)
        }
        
        if (is.null(run_result)) {
          new_run_number <- new_run_number + 1L
          
          message(
            "Fitting missing run ", new_run_number, " of ", length(missing_run_keys),
            ": seed ", split_seed, ", bots ", fraction_name,
            ", ", specification_name)
          
          fit <- fit_xgb_model(
            train_df = train_df, test_df = test_df,
            predictors = specifications[[specification_name]],
            model_name = specification_name, seed = model_seed,
            bot_fraction = fraction_name, threshold = threshold)
          
          fit$metrics[, split_seed := split_seed]
          
          if (nrow(fit$importance) > 0) {
            fit$importance[, `:=`(
              model = specification_name,
              bot_fraction = fraction_name,
              split_seed = split_seed)]
          }
          
          fit$predictions[, `:=`(
            model = specification_name,
            bot_fraction = fraction_name,
            split_seed = split_seed,
            test_row = seq_len(.N))]
          
          run_result <- list(
            metrics = fit$metrics,
            importance = fit$importance,
            predictions = fit$predictions)
          
          safe_save_rds(run_result, run_cache_file)
          rm(fit)
          gc(verbose = FALSE)
        }
        
        new_metric_results[[run_key]] <- run_result$metrics
        new_importance_results[[run_key]] <- run_result$importance
        new_prediction_results[[run_key]] <- run_result$predictions
        
        rm(run_result)
      }
    }
    
    gc(verbose = FALSE)
  }
}

# Combine existing and newly fitted results

if (length(new_metric_results) > 0) {
  robustness_metrics <- rbindlist(
    list(robustness_metrics, rbindlist(new_metric_results, fill = TRUE)),
    fill = TRUE)
  
  robustness_predictions <- rbindlist(
    list(robustness_predictions, rbindlist(new_prediction_results, fill = TRUE)),
    fill = TRUE)
  
  robustness_importance <- rbindlist(
    list(robustness_importance, rbindlist(new_importance_results, fill = TRUE)),
    fill = TRUE)
}

robustness_metrics <- unique(
  robustness_metrics,
  by = c("split_seed", "bot_fraction", "model", "Metric"))

robustness_predictions <- unique(
  robustness_predictions,
  by = c("split_seed", "bot_fraction", "model", "test_row"))

if (nrow(robustness_importance) > 0) {
  robustness_importance <- unique(
    robustness_importance,
    by = c("split_seed", "bot_fraction", "model", "Feature"))
}

# Verify that all requested runs are available 

completed_runs <- unique(
  robustness_metrics[, .(split_seed, bot_fraction, model)]
)

completed_runs[, run_key := make_run_key(split_seed, bot_fraction, model)]
still_missing <- setdiff(expected_runs$run_key, completed_runs$run_key)

if (length(still_missing) > 0) {
  stop(length(still_missing), " requested model runs are still missing.")
}

# Restore factor order

robustness_metrics[, model := factor(model, levels = model_order)]
robustness_metrics[, bot_fraction := factor(bot_fraction, levels = fraction_order)]

robustness_predictions[, model := factor(model, levels = model_order)]
robustness_predictions[, bot_fraction := factor(bot_fraction, levels = fraction_order)]

if (nrow(robustness_importance) > 0) {
  robustness_importance[, model := factor(model, levels = model_order)]
  robustness_importance[, bot_fraction := factor(bot_fraction, levels = fraction_order)]
}

# Create sample-size and performance tables

bot_training_sizes <- unique(
  robustness_metrics[, .(split_seed, bot_fraction, n_training_bots)]
)

bot_training_size_table <- bot_training_sizes[, .(
  median_training_bots = median(n_training_bots),
  minimum_training_bots = min(n_training_bots),
  maximum_training_bots = max(n_training_bots)
), by = bot_fraction]

setorder(bot_training_size_table, bot_fraction)

bot_training_size_table[, plot_label := ifelse(
  minimum_training_bots == maximum_training_bots,
  paste0(round(median_training_bots), " bots"),
  paste0(round(median_training_bots), " bots"))]

bot_count_labels <- setNames(
  bot_training_size_table$plot_label,
  as.character(bot_training_size_table$bot_fraction))

bot_count_levels <- unname(bot_count_labels[fraction_order])

robustness_summary <- summarize_performance(robustness_metrics)

# Recalculate feature stability where raw importance is complete

if (nrow(robustness_importance) > 0) {
  importance_coverage <- unique(
    robustness_importance[, .(model, bot_fraction, split_seed)]
  )[, .(available_runs = uniqueN(split_seed)), by = .(model, bot_fraction)]
  
  complete_importance_groups <- importance_coverage[
    available_runs == length(split_seeds),
    .(model, bot_fraction)]
  
  if (nrow(complete_importance_groups) > 0) {
    complete_importance <- robustness_importance[
      complete_importance_groups,
      on = .(model, bot_fraction),
      nomatch = 0]
    
    complete_importance_metrics <- robustness_metrics[
      complete_importance_groups,
      on = .(model, bot_fraction),
      nomatch = 0]
    
    recomputed_stability <- summarize_feature_stability(
      complete_importance_metrics,
      complete_importance)
    
    if (nrow(feature_stability) > 0) {
      feature_stability[, `:=`(
        model = factor(model, levels = model_order),
        bot_fraction = factor(bot_fraction, levels = fraction_order))]
      
      feature_stability <- feature_stability[
        !complete_importance_groups,
        on = .(model, bot_fraction)]
    }
    
    feature_stability <- rbindlist(
      list(feature_stability, recomputed_stability),
      fill = TRUE)
  }
}

if (nrow(feature_stability) == 0) {
  stop("Feature stability could not be calculated and no saved feature-stability table is available.")
}

feature_stability[, model := factor(as.character(model), levels = model_order)]
feature_stability[, bot_fraction := factor(as.character(bot_fraction), levels = fraction_order)]
setorder(feature_stability, model, bot_fraction, -top_predictor_share, -mean_gain)

# Print tables

cat("\nAnalytical sample:\n")
print(sample_size_table)

cat("\nNumber of bots retained in the training sample:\n")
print(bot_training_size_table)

cat("\nClassification performance across repeated splits:\n")
print(robustness_summary)

feature_stability_table <- feature_stability[, head(.SD, 10), by = .(model, bot_fraction)]
feature_stability_table[, training_bots := bot_count_labels[as.character(bot_fraction)]]
setcolorder(feature_stability_table, c("model", "bot_fraction", "training_bots"))

cat("\nTen most stable predictors by specification and training-sample size:\n")
print(feature_stability_table)

## Performance figure ----------------------------------------------------------

performance_plot_data <- robustness_metrics[
  Metric %in% c("Bot precision", "Bot recall", "F1", "PR AUC")]

performance_plot_data[, training_bots := factor(
  bot_count_labels[as.character(bot_fraction)],
  levels = bot_count_levels)]

plot_minimum <- min(performance_plot_data$Value, na.rm = TRUE)
plot_lower_limit <- max(0, floor((plot_minimum - 0.01) / 0.05) * 0.05)
plot_breaks <- seq(plot_lower_limit, 1, 0.05)

performance_plot <- ggplot(
  performance_plot_data,
  aes(x = training_bots, y = Value, fill = model)) +
  geom_boxplot(
    width = 0.70, outlier.alpha = 0.35,
    position = position_dodge2(width = 0.80, preserve = "single")) +
  facet_wrap(~Metric, ncol = 2) +
  scale_fill_manual(values = c(
    "All response times" = "#E34F47",
    "Without persona response times" = "#EDA923",
    "Answers only" = "#5B87BD")) +
  scale_y_continuous(
    breaks = plot_breaks,
    labels = scales::percent_format(accuracy = 1),
    expand = expansion(mult = c(0.02, 0.02))) +
  coord_cartesian(ylim = c(plot_lower_limit, 1.005)) +
  labs(
    title = "Classification performance across repeated data splits",
    subtitle = paste0(length(split_seeds), " stratified train-test splits"),
    x = "Number of bots retained in the training sample",
    y = "Test-sample performance",
    fill = "Model specification") +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold"))

print(performance_plot)

## Feature-stability figure ----------------------------------------------------

model_order <- c("All response times", "Without persona response times", "Answers only")
model_colors <- c("All response times" = "#E34F47", "Without persona response times" = "#EDA923", "Answers only" = "#5B87BD")
model_panel_labels <- c("All response times" = "All response\ntimes", "Without persona response times" = "Without persona\nresponse times", "Answers only" = "Answers only")

# Construct column labels from the previously calculated training-bot counts
if (!all(fraction_order %in% names(bot_count_labels))) {
  stop("bot_count_labels does not contain labels for all values in fraction_order.")
}

bot_count_levels <- unname(bot_count_labels[fraction_order])

# Keep only predictors that ranked first in at least one repeated split
top_feature_plot_data <- copy(feature_stability)[!is.na(top_predictor_share) & top_predictor_share > 0]

# Define panel order and labels
top_feature_plot_data[, model := factor(as.character(model), levels = model_order)]
top_feature_plot_data[, bot_fraction := factor(as.character(bot_fraction), levels = fraction_order)]
top_feature_plot_data[, bot_count := factor(as.character(bot_fraction), levels = fraction_order, labels = bot_count_levels)]
top_feature_plot_data[, model_panel := factor(as.character(model), levels = model_order, labels = unname(model_panel_labels[model_order]))]

# Convert duration_ITEM to ITEM (response time)
top_feature_plot_data[, feature_label := as.character(Feature)]
top_feature_plot_data[grepl("^duration_", feature_label), feature_label := paste0(sub("^duration_", "", feature_label), " (response time)")]

# Put predictors with the largest first-place shares at the top
feature_order <- top_feature_plot_data[, .(maximum_share = max(top_predictor_share, na.rm = TRUE)), by = feature_label][order(maximum_share, feature_label), feature_label]
top_feature_plot_data[, feature_label := factor(feature_label, levels = feature_order)]

# Create figure
feature_stability_plot <- ggplot(top_feature_plot_data, aes(x = top_predictor_share, y = feature_label, fill = model)) +
  geom_col(width = 0.72, show.legend = FALSE) +
  facet_grid(model_panel ~ bot_count, scales = "free_y", space = "free_y", switch = "y") +
  scale_fill_manual(values = model_colors, drop = FALSE) +
  scale_x_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1), labels = scales::label_percent(accuracy = 1), expand = expansion(mult = c(0, 0.04))) +
  labs(title = "Stability of the most important predictor",
       subtitle = paste0("Share of ", length(split_seeds), " repeated splits in which each predictor ranks first"),
       x = "Share of repeated splits", y = NULL) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.spacing.x = grid::unit(0.45, "cm"),
        panel.spacing.y = grid::unit(0.35, "cm"),
        plot.title = element_text(face = "bold", size = 15),
        plot.subtitle = element_text(size = 11, margin = margin(b = 12)),
        strip.placement = "outside",
        strip.background = element_blank(),
        strip.text.x = element_text(face = "bold", size = 10),
        strip.text.y.left = element_text(angle = 0, hjust = 1, face = "bold", size = 9, margin = margin(r = 10)),
        axis.text.x = element_text(size = 9),
        axis.text.y = element_text(size = 9),
        axis.title.x = element_text(size = 11),
        plot.margin = margin(t = 12, r = 15, b = 12, l = 20))

print(feature_stability_plot)

## Save tables, underlying results, and figures ---------------------------------

fwrite(sample_size_table, sample_size_file)
fwrite(bot_training_size_table, training_size_file)
fwrite(robustness_metrics, metrics_file)
fwrite(robustness_summary, summary_file)
fwrite(feature_stability, stability_file)
fwrite(robustness_predictions, predictions_file)

if (nrow(robustness_importance) > 0) fwrite(robustness_importance, importance_file)

saved_results <- list(
  robustness_metrics = robustness_metrics,
  robustness_summary = robustness_summary,
  feature_stability = feature_stability,
  robustness_predictions = robustness_predictions,
  robustness_importance = robustness_importance,
  sample_size_table = sample_size_table,
  bot_training_size_table = bot_training_size_table)

safe_save_rds(saved_results, result_rds)

ggsave(performance_figure_file, performance_plot, width = 11, height = 7.5, dpi = 300)
ggsave(stability_figure_file, feature_stability_plot, width = 12.5, height = 9, dpi = 300)

message("All requested runs, tables, and figures are complete and saved.")