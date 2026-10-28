# Bot or Human?

This repository contains a proof-of-concept workflow for generating automated survey responses and classifying completed interviews as **bot** or **human**. It combines:

- a Python survey bot using a local large language model through [Ollama](https://ollama.com/) and browser automation through Selenium; and
- an R analysis using XGBoost to compare classifiers based on survey answers, page-level response times, or both.

The project was developed for the study **“Bot or Human? Using ML Classification Models for Survey Data Observations.”** It is intended as a controlled methodological experiment, not as a general-purpose bot detector.

## Responsible use

The Python script opens a survey in a browser and submits responses. Run it only on surveys for which you have explicit authorization to conduct automated testing. Before running the script, replace the configured `SURVEY_URL` with an approved test or development survey. Do not use this code to interfere with live data collection, claim incentives, or submit unauthorized responses.

## Repository structure

| Path | Description |
| --- | --- |
| `SurvB_foesine.py` | Selenium/Ollama survey bot, including persona generation, answer selection, response-time simulation, validation, and run logging. |
| `apply_xgboost.R` | Data preparation, descriptive response-time analysis, XGBoost estimation, evaluation, robustness checks, and figure generation. |
| `survey_bot_runs.csv` | Run-level bot log containing persona characteristics, completion status, pages completed, and errors. |
| `survey_bot_answers.jsonl` | Detailed line-delimited JSON log with personas, generated answer weights, realized answers, timings, and execution information. |
| `xgboost_results/` | Aggregated metrics, predictions, feature importance, feature stability, and training-sample summaries from the robustness analysis. |

## Methodological overview

### Survey bot

The bot uses the local `qwen2.5:7b` model to answer survey questions. Each run receives a newly sampled respondent persona containing fixed demographic characteristics and randomly generated work and institutional-trust preferences. Fixed demographic questions are answered directly in Python. For subjective questions, the language model returns plausibility weights over the available response options, after which Python draws the realized answer.

The bot also simulates variation in response behavior through:

- respondent-specific answering pace and reading speed;
- item-level timing variation and familiarity effects for similar pages;
- probabilistic completion of optional open-text fields;
- probabilistic attention-check performance; and
- short answer histories supplied to the model to support consistency.

Every run produces structured logs that retain the assigned persona, response options and weights, realized answers, page timings, browser actions, completion status, and technical errors.

### XGBoost classification

The R analysis estimates three binary XGBoost specifications:

1. **All response times:** survey answers and all page-level response times.
2. **Without persona response times:** excludes response times for the age, gender, and federal-state items answered directly from the assigned persona.
3. **Answers only:** excludes all response-time variables.

Preprocessing is learned from the training data only. Nominal predictors are dummy-encoded, zero-variance predictors are removed, and missing or previously unseen factor levels are handled explicitly. The number of boosting rounds is selected through cross-validation, and the bot class is weighted to account for class imbalance.

The main evaluation reports accuracy, bot precision, bot recall, specificity, F1, ROC AUC, and PR AUC. The robustness analysis repeats the stratified train–test split for 50 random seeds and refits every specification using 100%, 50%, 20%, and 10% of the bots assigned to the training sample. Individual runs are cached so an interrupted analysis can resume without refitting completed models.

## Requirements

### Python survey bot

- Python 3
- Google Chrome or Chromium with a working Selenium WebDriver
- [Ollama](https://ollama.com/) running locally
- Ollama model `qwen2.5:7b`
- Python packages `requests` and `selenium`

Install the Python packages and model with:

```bash
python -m pip install requests selenium
ollama pull qwen2.5:7b
```

Start Ollama if it is not already running:

```bash
ollama serve
```

### R analysis

The analysis requires R 4.1 or later and the following packages:

```text
data.table, ggplot2, lubridate, stringi, recipes, rsample,
xgboost, pROC, PRROC, knitr, scales, tidymodels
```

`apply_xgboost.R` installs missing packages automatically. For a controlled or reproducible software environment, install and record the package versions separately before running the script.

## Running the survey bot

1. Open `SurvB_foesine.py` and set `SURVEY_URL` to an authorized survey.
2. Review the model, persona, behavioral, and timing settings near the beginning of the script.
3. Ensure that Ollama and Chrome are available.
4. Run one synthetic respondent:

```bash
python -c "from SurvB_foesine import run_bot; run_bot(1)"
```

To run ten synthetic respondents:

```bash
python -c "from SurvB_foesine import run_bot; run_bot(10)"
```

Important settings include:

| Setting | Purpose |
| --- | --- |
| `MODEL` | Ollama model used to generate response weights and text. |
| `SURVEY_URL` | Authorized survey or test environment to open. |
| `MAX_PAGES` | Safety limit for the number of pages processed in one run. |
| `OPTIONAL_TEXT_BLANK_PROBABILITY` | Probability that an optional text field is left blank. |
| `ATTENTION_CHECK_PASS_PROBABILITY` | Probability that a recognized attention check is passed. |
| Timing constants | Reading speed, response-time limits, respondent pace, page jitter, and familiarity effects. |

Generated files are written next to the Python script. In addition to the two consolidated logs included in the repository, each run can produce a Markdown report and a structured JSON record. Model errors are written to `survey_bot_model_errors.jsonl` when applicable.

## XGBoost analysis
Since the data cannot be released for the respondents privacy reasons, external replication is not possible.

The analysis creates descriptive figures, fits the three main classifiers, generates confusion matrices and feature-importance plots, runs the repeated-split robustness analysis, and writes the resulting tables and figures to the configured output folders.

## Included robustness outputs

The `xgboost_results/` directory contains:

| File | Contents |
| --- | --- |
| `xgboost_robustness_metrics.csv` | Performance metrics for each split, bot training-sample size, and model specification. |
| `xgboost_robustness_summary.csv` | Medians, means, percentiles, and ranges of the repeated-split performance metrics. |
| `xgboost_robustness_predictions.csv` | Test-sample probabilities and classifications for the repeated runs. |
| `xgboost_robustness_importance.csv` | Run-specific XGBoost feature-importance values and ranks. |
| `xgboost_feature_stability.csv` | Frequency with which predictors rank among the most important features. |
| `xgboost_sample_sizes.csv` | Numbers of human and bot observations in the analytical sample. |
| `xgboost_training_bot_counts.csv` | Numbers of bots retained under the four training-sample conditions. |


## Citation

If you use or adapt this repository, please cite the repository and the accompanying study:

> Fössing, Emma. *Bot or Human? Using ML Classification Models for Survey Data Observations.* 2026.

## License

No license is currently included. Unless a license is added, the repository remains subject to the default copyright restrictions that apply to published source code.
