import os
import csv
import json
import math
import time
import random
import re
import requests
import unicodedata
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from decimal import Decimal, InvalidOperation
from pathlib import Path
from uuid import uuid4

from datetime import datetime
from difflib import SequenceMatcher

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

# ============================================================
# SETTINGS
# ============================================================

BOT_VERSION = "2026-10-13.2"


def get_script_directory():
    # runpy, imports, normal execution, and exec(compile(..., filename, ...))
    # preserve the source filename. Plain exec(source_text) does not.
    source = globals().get("SURVEY_BOT_SCRIPT_PATH") or get_script_directory.__code__.co_filename
    if source and not str(source).startswith("<"):
        script_path = Path(source).expanduser().resolve()
        if script_path.is_file():
            return script_path.parent
    raise RuntimeError(
        "Cannot determine the survey bot script folder. "
        "Load the saved .py file with runpy.run_path(full_script_path), "
        "or set SURVEY_BOT_SCRIPT_PATH to its full path when using exec. "
        "The bot will not save output to an unknown working directory."
    )


SCRIPT_DIR = get_script_directory()

SURVEY_URL = "https://umfragen.iab.de/?i_survey=80__cf7e3508728fda0840a6fb931d338f15"

MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Safety limit for all pages, including instructions and vignettes.
MAX_PAGES = 200
CSV_FILE = str(SCRIPT_DIR / "survey_bot_runs.csv")
ANSWER_LOG_FILE = str(SCRIPT_DIR / "survey_bot_answers.jsonl")
MODEL_ERROR_LOG_FILE = str(SCRIPT_DIR / "survey_bot_model_errors.jsonl")

# Keep Ollama's model loaded between pages/runs.
OLLAMA_KEEP_ALIVE = "30m"

# This controls model wording/generation. Answer draws happen in Python.
OLLAMA_TEMPERATURE = 0.2
OLLAMA_NUM_CTX = 8192

# Keep model output short. This is one of the main speed improvements.
OLLAMA_NUM_PREDICT = 250
OLLAMA_MAX_NUM_PREDICT = 1600
AI_VALIDATION_ATTEMPTS = 2

# Independent synthetic behavior draws, once per eligible field/check.
# These are user-chosen simulation settings, not estimated human probabilities.
OPTIONAL_TEXT_BLANK_PROBABILITY = 0.9
ATTENTION_CHECK_PASS_PROBABILITY = 0.9

# An employment Other specification is a keyword label, not a comment.
EMPLOYMENT_TEXT_MAX_WORDS = 6
EMPLOYMENT_TEXT_MAX_CHARS = 60
FORWARD_BUTTON_TIMEOUT = 4.0
FORWARD_BUTTON_POLL_INTERVAL = 0.2

# Independent, uncalibrated preferences for synthetic testing, drawn once/run.
# Edit these dimensions to match another survey; they are not population data.
ADD_PREFERENCE_PROFILE = True
WORK_PREFERENCE_DIMENSIONS = (
    "job_security", "income", "promotion", "recognition", "free_time",
    "interesting_work", "independence", "conscientious_work", "responsibility",
    "social_contact", "helping_others", "usefulness_to_society"
)
TRUST_PREFERENCE_DIMENSIONS = (
    "parliament", "justice_system", "police", "politicians",
    "political_parties", "european_parliament"
)

# Timing limits.
MIN_RESPONSE_TIME = 0.5
MAX_RESPONSE_TIME = 30.0
MIN_READING_TIME = 0.8
MAX_READING_TIME = 90.0

# Synthetic timing settings for participation and data-consent pages.
# Apply this minimum before the answer click, even on a very short page.
MIN_AGREEMENT_READING_TIME = 5.0
AGREEMENT_DECISION_TIME_MIN = 0.8
AGREEMENT_DECISION_TIME_MAX = 2.0

# One respondent-specific pace is drawn per run.
RESPONDENT_PACE_SD = 0.18
PAGE_JITTER_SD = 0.08

# Reading speed is also drawn once per run rather than fixed.
READING_WPM_MEAN = 230
READING_WPM_SD = 18
READING_WPM_MIN = 170
READING_WPM_MAX = 290

# Small page-level processing overhead, also random.
READING_OVERHEAD_MIN = 0.4
READING_OVERHEAD_MAX = 1.4

# Similar pages/batteries become faster as respondents gain familiarity.
SIMILAR_PAGE_THRESHOLD = 0.60
MAX_FAMILIARITY_REDUCTION = 0.30
FAMILIARITY_LEARNING_RATE = 1.40

# Technical wait after navigation, not respondent behavior.
NAVIGATION_WAIT = 0.6

# Only a short realized-answer history is sent back to the model.
MAX_HISTORY_ITEMS_FOR_MODEL = 12

# ============================================================
# SURVEY PROMPT
# ============================================================

SURVEY_PROMPT = """
You are a synthetic respondent testing an online survey.

You have been assigned a fixed persona.
Answer every question as this person would answer it.

General rules:
- remain consistent with the assigned persona
- remain consistent with factual answers and earlier answers to the same question
- treat gender, age, age group, Bundesland and region as fixed characteristics
- understand each question and choose a plausible answer for the assigned persona
- consider previous answers when relevant
- do not invent answer options
- do not contradict earlier answers
- do not change fixed demographic characteristics
- if a demographic characteristic is not given by the persona or previous answers,
  choose a plausible value for this person
- a previous rating for a DIFFERENT item is not a template for this item
- fixed facts (including age and gender) require a direct answer, never a random draw
- for subjective single-choice questions return plausibility weights for the
  available options; Python draws the realized answer from those weights
- use the fixed preference_profile to distinguish this person's priorities and
  institution-specific trust; demographics alone do not determine these attitudes
- consider the particular item, its wording, polarity and scale anchors
- do not force answers to alternate or avoid a choice merely because it was used
- weights are synthetic plausibility judgments, not calibrated population estimates

Question-type rules:
- for every required single-choice group return one answer object
- use mode="direct" with a labeled choice for known facts or explicit instructions
- use mode="sample" with weights keyed by actual choice labels for uncertainty
- for checkbox questions choose all options that this persona would plausibly select
- for dropdowns use the same direct-answer versus weighted-answer distinction
- for open-text questions provide one short natural answer
- use listed factual categories when they fit; reserve Other for cases they do not cover
- an Other specification belongs only to its selected Other option
- an employment Other specification contains only job/status keywords, never a
  sentence, personal history, reasoning, or a list of existing answer categories
- for optional comments requested by Python, use brief, informal German rather than
  a polished explanation; minor everyday language imperfections are acceptable
- fill every required group, including each separate month and year dropdown
- use only the actual group names, field names and choice labels supplied to you

Response-time rule:
- estimate one plausible response time in seconds for the CURRENT PAGE
- this should reflect question complexity and the assigned persona
- do not simulate the wait yourself; Python will do that

Vignette rule:
- for income-fairness vignettes:
  - -5 means the income is much too low
  - 0 means the income is exactly fair
  - +5 means the income is much too high
  - intermediate values indicate intermediate judgments
- return weights over the actual available scale from this persona's perspective
- other scales have their own anchors: trust from 0 to 10 is not a fairness scale

Return only valid JSON. Do not output markdown or explanations outside JSON.
"""

# ============================================================
# PERSONA DISTRIBUTION
# ============================================================

PERSONA_DISTRIBUTION = [
    {"gender": "male", "region": "West", "age_group": "18-28", "probability": 0.0838},
    {"gender": "male", "region": "East", "age_group": "18-28", "probability": 0.0166},
    {"gender": "female", "region": "West", "age_group": "18-28", "probability": 0.07395},
    {"gender": "female", "region": "East", "age_group": "18-28", "probability": 0.01475},

    {"gender": "male", "region": "West", "age_group": "29-38", "probability": 0.101},
    {"gender": "male", "region": "East", "age_group": "29-38", "probability": 0.02165},
    {"gender": "female", "region": "West", "age_group": "29-38", "probability": 0.08615},
    {"gender": "female", "region": "East", "age_group": "29-38", "probability": 0.01995},

    {"gender": "male", "region": "West", "age_group": "39-48", "probability": 0.08905},
    {"gender": "male", "region": "East", "age_group": "39-48", "probability": 0.0235},
    {"gender": "female", "region": "West", "age_group": "39-48", "probability": 0.084},
    {"gender": "female", "region": "East", "age_group": "39-48", "probability": 0.02195},

    {"gender": "male", "region": "West", "age_group": "49-58", "probability": 0.09065},
    {"gender": "male", "region": "East", "age_group": "49-58", "probability": 0.02055},
    {"gender": "female", "region": "West", "age_group": "49-58", "probability": 0.09015},
    {"gender": "female", "region": "East", "age_group": "49-58", "probability": 0.02005},

    {"gender": "male", "region": "West", "age_group": "59-64", "probability": 0.05705},
    {"gender": "male", "region": "East", "age_group": "59-64", "probability": 0.01435},
    {"gender": "female", "region": "West", "age_group": "59-64", "probability": 0.0561},
    {"gender": "female", "region": "East", "age_group": "59-64", "probability": 0.01475}
]

WEST_STATES = [
    "Baden-Württemberg",
    "Bayern",
    "Bremen",
    "Hamburg",
    "Hessen",
    "Niedersachsen",
    "Nordrhein-Westfalen",
    "Rheinland-Pfalz",
    "Saarland",
    "Schleswig-Holstein"
]

EAST_STATES = [
    "Berlin",
    "Brandenburg",
    "Mecklenburg-Vorpommern",
    "Sachsen",
    "Sachsen-Anhalt",
    "Thüringen"
]

AGE_RANGES = {
    "18-28": (18, 28),
    "29-38": (29, 38),
    "39-48": (39, 48),
    "49-58": (49, 58),
    "59-64": (59, 64)
}

# ============================================================
# PERSONA
# ============================================================

def draw_persona():
    profile = random.choices(
        PERSONA_DISTRIBUTION,
        weights=[x["probability"] for x in PERSONA_DISTRIBUTION],
        k=1
    )[0]

    age_min, age_max = AGE_RANGES[profile["age_group"]]
    age = random.randint(age_min, age_max)

    states = WEST_STATES if profile["region"] == "West" else EAST_STATES
    bundesland = random.choice(states)

    # Birth month/year are generated once per respondent and remain fixed.
    # They are constructed so that they are compatible with the assigned age.
    today = datetime.now()
    birth_month = random.randint(1, 12)

    if birth_month < today.month:
        birth_year = today.year - age
    elif birth_month > today.month:
        birth_year = today.year - age - 1
    else:
        # The survey asks only for month/year, not the exact birthday.
        # Either year can be compatible within the current month.
        birth_year = (
            today.year - age
            if random.random() < 0.5
            else today.year - age - 1
        )

    persona = {
        "gender": profile["gender"],
        "age": age,
        "age_group": profile["age_group"],
        "region": profile["region"],
        "bundesland": bundesland,
        "birth_month": birth_month,
        "birth_year": birth_year,
        "quota_probability": profile["probability"]
    }

    if ADD_PREFERENCE_PROFILE:
        persona["preference_profile"] = {
            "work_priority_scale": "1 = low priority, 5 = high priority",
            "work_priorities": {
                dimension: random.randint(1, 5)
                for dimension in WORK_PREFERENCE_DIMENSIONS
            },
            "trust_scale": "0 = no trust, 10 = complete trust",
            "institutional_trust": {
                dimension: random.randint(0, 10)
                for dimension in TRUST_PREFERENCE_DIMENSIONS
            }
        }

    return persona

def print_persona(run_number, persona):
    print()
    print("=" * 70)
    print(f"RUN {run_number}: ASSIGNED PERSONA")
    print("=" * 70)
    print(json.dumps(persona, ensure_ascii=False, indent=2))
    print("=" * 70)
    print()

# ============================================================
# LOGGING
# ============================================================

def save_run_to_csv(
    run_number,
    start_time,
    end_time,
    persona,
    pages_completed,
    status,
    error_message=""
):
    file_exists = os.path.isfile(CSV_FILE)

    fieldnames = [
        "run_number",
        "start_time",
        "end_time",
        "gender",
        "age",
        "age_group",
        "region",
        "bundesland",
        "quota_probability",
        "pages_completed",
        "status",
        "error_message"
    ]

    row = {
        "run_number": run_number,
        "start_time": start_time,
        "end_time": end_time,
        "gender": persona["gender"],
        "age": persona["age"],
        "age_group": persona["age_group"],
        "region": persona["region"],
        "bundesland": persona["bundesland"],
        "quota_probability": persona["quota_probability"],
        "pages_completed": pages_completed,
        "status": status,
        "error_message": error_message
    }

    with open(
        CSV_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

def save_answer_log(run_number, start_time, end_time, persona, history, status, error_message):
    """Append enough information to inspect each weight and realized answer."""
    record = {
        "bot_version": BOT_VERSION, "model": MODEL,
        "run_number": run_number, "start_time": start_time, "end_time": end_time,
        "persona": persona, "status": status, "error_message": error_message,
        "behavior_probabilities": {
            "optional_text_blank": OPTIONAL_TEXT_BLANK_PROBABILITY,
            "attention_check_pass": ATTENTION_CHECK_PASS_PROBABILITY
        },
        "history": history
    }
    with open(ANSWER_LOG_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


class RunOutputTee:
    """Keep console output visible while recording the individual run."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def reserve_run_files(run_number):
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    base_id = f"{stamp}_run{run_number:03d}_{uuid4().hex[:8]}"
    suffix = 0
    while True:
        run_id = base_id if suffix == 0 else f"{base_id}_{suffix}"
        report_path = SCRIPT_DIR / f"survey_bot_run_{run_id}.md"
        data_path = report_path.with_suffix(".json")
        report_file = None
        try:
            report_file = report_path.open("x", encoding="utf-8")
            data_file = data_path.open("x", encoding="utf-8")
        except FileExistsError:
            if report_file is not None:
                report_file.close()
                report_path.unlink()
            suffix += 1
            continue
        except Exception:
            if report_file is not None:
                report_file.close()
                report_path.unlink()
            raise
        return run_id, report_path, data_path, report_file, data_file


def markdown_cell(value):
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def markdown_code_block(text, language="text"):
    text = str(text)
    longest = max((len(match.group()) for match in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def render_run_report(record):
    lines = [f"# Survey bot — run {record['run_number']}", "", "## Run summary", "",
             "| Field | Value |", "| --- | --- |"]
    fields = (
        ("Run ID", record["run_id"]), ("Status", record["status"]),
        ("Started", record["started_at"]), ("Ended", record["ended_at"]),
        ("Total elapsed time (seconds)", f"{record['duration_seconds']:.2f}"),
        ("Pages completed", record.get("pages_completed", 0)),
        ("Bot version", record["bot_version"]), ("Model", record["model"]),
        ("Output folder", record.get("output_directory", str(SCRIPT_DIR))),
        ("Respondent pace", record.get("timing_pace", "")),
        ("Reading speed (words/minute)", record.get("reading_wpm", "")),
        ("Error", record.get("error_message", ""))
    )
    lines.extend(f"| {key} | {markdown_cell(value)} |" for key, value in fields)
    lines.extend(["", "## Assigned respondent", "",
                  markdown_code_block(json.dumps(record.get("persona", {}), ensure_ascii=False, indent=2), "json"),
                  "", "## Pages", ""])
    page_types = {"forced_participation": "Participation", "fixed_birthdate": "Birth date",
                  "instruction": "Information / instruction", "question": "Question"}
    for entry in record.get("history", []):
        lines.extend([f"### Page {entry['page']} — {page_types.get(entry.get('type'), 'Page')}", ""])
        if entry.get("question_text"):
            lines.extend(["**Survey page**", "", markdown_code_block(entry["question_text"].strip()), ""])
        answer = entry.get("summary") or entry.get("answer")
        if entry.get("type") == "fixed_birthdate":
            answer = f"Birth month: {entry['birth_month']}; birth year: {entry['birth_year']}; age: {entry['age']}"
        if answer:
            lines.extend([f"**Answer:** {markdown_cell(answer)}", ""])
        if entry.get("behavior_draws"):
            lines.extend(["**Independent behavior draws**", "",
                          "| Decision | Target | Probability | Outcome |",
                          "| --- | --- | ---: | --- |"])
            for draw in entry["behavior_draws"]:
                lines.append(f"| {markdown_cell(draw['kind'])} | {markdown_cell(draw['target'])} | "
                             f"{draw['probability']:.2f} | {markdown_cell(draw['outcome'])} |")
            lines.append("")
        for correction in entry.get("answer_corrections", []):
            lines.extend([
                f"**Listed employment category used:** {markdown_cell(correction['listed_choice'])}",
                f"**Original Other specification:** {markdown_cell(correction['original_specification'])}",
                ""
            ])
        timing = entry.get("timing", {})
        time_fields = (
            ("Predicted response", entry.get("predicted_response_time")),
            ("Simulated total response", entry.get("actual_response_time")),
            ("Reading", entry.get("actual_reading_time", timing.get("actual_reading_time"))),
            ("Decision pause", entry.get("decision_time_seconds", timing.get("decision_time_seconds"))),
            ("Planned pause before answer", entry.get("pre_answer_delay_seconds", timing.get("pre_answer_delay_seconds"))),
            ("Measured pause before answer", timing.get("observed_pre_answer_wait_seconds")),
            ("Pause after answer", entry.get("post_answer_delay_seconds"))
        )
        times = [(label, value) for label, value in time_fields if value is not None]
        if times:
            lines.extend(["| Timing | Seconds |", "| --- | ---: |"])
            lines.extend(f"| {label} | {value:.2f} |" for label, value in times)
            lines.append("")
        if entry.get("execution"):
            lines.extend(["**Executed actions**", "", "| Action | Answer | Element | Successful |",
                          "| --- | --- | --- | --- |"])
            for action in entry["execution"]:
                values = (action.get("actual_action", action.get("requested_action", "")),
                          action.get("value", ""), action.get("element", ""), action.get("success", ""))
                lines.append("| " + " | ".join(markdown_cell(value) for value in values) + " |")
            lines.append("")
        for sample in entry.get("sampling", []):
            lines.extend([f"**Answer draw — {markdown_cell(sample['group_key'])}**", "",
                          "| Choice | Probability | Selected |", "| --- | ---: | --- |"])
            for option, probability in zip(sample["options"], sample["probabilities"]):
                selected = "Yes" if option["label"] == sample["chosen_label"] else ""
                lines.append(f"| {markdown_cell(option['label'])} | {probability:.4f} | {selected} |")
            lines.append("")
    last_page = record.get("last_page")
    if last_page and not any(entry.get("page") == last_page["page"] for entry in record.get("history", [])):
        lines.extend([f"## Last displayed page — {last_page['page']}", "",
                      markdown_code_block(last_page["question_text"].strip()), ""])
        if last_page.get("behavior_draws"):
            lines.extend(["**Behavior draws before the run stopped**", "",
                          markdown_code_block(json.dumps(last_page["behavior_draws"], ensure_ascii=False, indent=2), "json"), ""])
    if last_page and last_page.get("navigation_stop"):
        lines.extend(["## Navigation outcome", "",
                      markdown_code_block(json.dumps(last_page["navigation_stop"], ensure_ascii=False, indent=2), "json"), ""])
    lines.extend(["## Complete console output", "", markdown_code_block(record.get("console_output", "")), ""])
    return "\n".join(lines)


def run_one_survey(run_number):
    run_id, report_path, data_path, report_file, data_file = reserve_run_files(run_number)
    started = datetime.now().astimezone()
    started_clock = time.monotonic()
    record = {
        "schema_version": 1, "run_id": run_id, "run_number": run_number,
        "bot_version": BOT_VERSION, "model": MODEL, "survey_url": SURVEY_URL,
        "output_directory": str(SCRIPT_DIR),
        "behavior_probabilities": {
            "optional_text_blank": OPTIONAL_TEXT_BLANK_PROBABILITY,
            "attention_check_pass": ATTENTION_CHECK_PASS_PROBABILITY
        },
        "started_at": started.isoformat(timespec="seconds"), "status": "started",
        "pages_completed": 0, "persona": {}, "history": [], "error_message": "",
        "documentation_file": str(report_path), "data_file": str(data_path)
    }
    interrupted = False
    output = io.StringIO()
    with report_file, data_file:
        # Stream a readable transcript immediately so output survives an abrupt exit.
        report_file.write(f"# Survey bot — run {run_number}\n\nRun ID: {run_id}\n\n```text\n")
        report_file.flush()
        json.dump(record, data_file, ensure_ascii=False, indent=2)
        data_file.flush()
        with redirect_stdout(RunOutputTee(sys.stdout, output, report_file)), \
                redirect_stderr(RunOutputTee(sys.stderr, output, report_file)):
            print(f"RUN ID: {run_id}")
            try:
                _run_one_survey(run_number, record)
            except KeyboardInterrupt:
                record.update(status="interrupted", error_message="Run interrupted by the user.")
                interrupted = True
                print("Run interrupted. Saving partial documentation.")
            except Exception as error:
                record.update(status="error", error_message=str(error))
                print(f"ERROR IN RUN {run_number}: {error}")
        record.update(
            ended_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            duration_seconds=time.monotonic() - started_clock,
            console_output=output.getvalue()
        )
        report_file.seek(0)
        report_file.truncate()
        report_file.write(render_run_report(record))
        data_file.seek(0)
        data_file.truncate()
        json.dump(record, data_file, ensure_ascii=False, indent=2)
        data_file.write("\n")
    print(f"Run documentation saved to: {report_path}")
    print(f"Run data saved to: {data_path}")
    if interrupted:
        raise KeyboardInterrupt
    return record

# ============================================================
# OLLAMA
# ============================================================

def check_ollama():
    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=3
        )

        if response.status_code != 200:
            return False

        models = response.json().get("models", [])
        installed_names = {
            model.get("name", "")
            for model in models
        }

        return (
            MODEL in installed_names
            or any(
                name.startswith(MODEL)
                for name in installed_names
            )
        )

    except Exception:
        return False

def compact_history_for_model(history):
    compact = []

    for item in history[-MAX_HISTORY_ITEMS_FOR_MODEL:]:
        if item.get("type") != "question":
            continue

        successful_keys = {
            result.get("element") for result in item.get("execution", [])
            if result.get("success")
        }
        realized_actions = [
            action for action in item.get("actions", [])
            if action.get("element_key") in successful_keys
        ]
        if not realized_actions:
            continue

        compact.append({
            "page": item.get("page"),
            "question_text": item.get("question_text", "")[:1200],
            "answers": [
                {
                    "element_key": action.get("element_key"),
                    "value": action.get("value")
                }
                for action in realized_actions
            ]
        })

    return compact

def single_choice_groups(elements, visible_text=""):
    """Build one ordered list of actual options for each single-choice group."""
    groups = {}
    for element in elements:
        kind = element.get("type")
        key = element["element_key"]
        if kind == "radio":
            group_key = "radio::" + (element.get("group") or key)
            group = groups.setdefault(group_key, {
                "group_key": group_key, "type": "radio", "options": [],
                "question_text": element.get("question_text", "")
            })
            group["options"].append({
                "element_key": key,
                "label": element.get("label") or element.get("value", "")
            })
        elif kind == "dropdown":
            group_key = "dropdown::" + key
            groups[group_key] = {
                "group_key": group_key, "type": "dropdown",
                "options": [{
                    "element_key": key, "label": option["label"],
                    "value": option.get("value", option["label"]),
                    "option_key": option.get("option_key", f"option::{key}::{index}"),
                    "option_index": option.get("index", index)
                } for index, option in enumerate(element.get("options", []))
                if not (option.get("value") in (None, "") and re.search(
                    r"wählen|choose|select|^-+$", normalize_option_text(option["label"])
                ))]
            }
    # Some surveys implement Other as the text input itself, without a radio.
    # It is still an alternative to the listed categories, not an extra answer.
    radio_groups = [group for group in groups.values() if group["type"] == "radio"]
    unlabeled = [field for field in elements if is_free_text_field(field)
                 and not field.get("label") and not field.get("context_label")]
    other_lines = [line.strip() for line in visible_text.splitlines() if is_other_label(line.strip())]
    for field in elements:
        if (not is_other_text_field(field) and len(unlabeled) == 1
                and field is unlabeled[0] and len(radio_groups) == 1 and len(other_lines) == 1):
            field = dict(field, label=other_lines[0])
        if not is_other_text_field(field):
            continue
        linked_key = field.get("other_option_key")
        candidates = [group for group in radio_groups if any(
            option["element_key"] == linked_key for option in group["options"]
        )] if linked_key else []
        if not candidates:
            field_id = re.match(r"id::(Q\d+)A", field["element_key"], re.I)
            candidates = [group for group in radio_groups if any(
                re.match(r"id::" + re.escape(field_id[1]) + r"A", option["element_key"], re.I)
                for option in group["options"]
            )] if field_id else []
        if not candidates and len(radio_groups) == 1:
            candidates = radio_groups
        if len(candidates) != 1:
            continue
        group = candidates[0]
        other_options = [option for option in group["options"] if (
            option["element_key"] == linked_key if linked_key else is_other_label(option["label"])
        )]
        if len(other_options) == 1:
            other_options[0]["text_field_key"] = field["element_key"]
        elif not other_options:
            group["options"].append({
                "element_key": field["element_key"],
                "label": field.get("label") or field.get("context_label") or "Other",
                "text_field_key": field["element_key"], "input_mode": "other_text"
            })
    for group in groups.values():
        counts = {}
        for option in group["options"]:
            counts[option["label"]] = counts.get(option["label"], 0) + 1
        for index, option in enumerate(group["options"]):
            label = option["label"]
            option["choice_label"] = (
                label if counts[label] == 1 else f"{label} [choice {index + 1}]"
            )
    return list(groups.values())


def is_other_label(label):
    return bool(re.match(
        r"^(?:etwas anderes\b|sonstiges(?:\s*[,(:-]|\s+und zwar|$)|"
        r"andere[rs]?(?:\s*[,(:-]|\s+und zwar|$)|other(?:\s*[,(:-]|\s+specify|$)|something else\b)",
        normalize_option_text(label)
    ))


def is_free_text_field(element):
    return element.get("type") in {"text", "open_text", "textarea"} and (
        element.get("input_mode") not in {"numeric", "decimal"}
    )


def is_other_text_field(element):
    return is_free_text_field(element) and (
        bool(element.get("other_option_key"))
        or is_other_label(element.get("label", ""))
        or is_other_label(element.get("context_label", ""))
    )


def text_field_rules(field, visible_text="", other_group=None):
    """Identify keyword specifications without changing any behavior draw."""
    rules = {}
    if field.get("tag_name") == "input" or field.get("type") == "text":
        rules["single_line"] = True
    maximum = field.get("max_length")
    if isinstance(maximum, int) and maximum > 0:
        rules["max_chars"] = maximum
    if not is_free_text_field(field) or not other_group:
        return rules
    context = normalize_option_text(
        field.get("question_text") or other_group.get("question_text") or visible_text
    )
    employment_question = bool(re.search(
        r"erwerbsstatus|beschäftigungsstatus|haupttätigkeit|employment status|"
        r"berufliche.{0,15}(?:situation|tätigkeit)|"
        r"(?:derzeit|zurzeit|gegenwärtig|aktuell).{0,80}(?:beruf|erwerb|tätigkeit|beschäftig)|"
        r"(?:beruf|erwerb|tätigkeit|beschäftig).{0,80}(?:derzeit|zurzeit|gegenwärtig|aktuell)",
        context
    ))
    status_options = sum(bool(re.search(
        r"erwerbstätig|selbstständig|arbeitslos|arbeitssuchend|arbeitsuchend|"
        r"rentner|rente|pension|schüler|student|studium|ausbildung|hausfrau|hausmann",
        normalize_option_text(option["label"])
    )) for option in other_group["options"])
    # This survey's custom Other input has no separate radio button.
    if (field["element_key"].lower() == "id::q6a13"
            or employment_question or status_options >= 2):
        rules.update({
            "role": "employment_keywords", "single_line": True,
            "employment_status_group": other_group["group_key"],
            "max_words": EMPLOYMENT_TEXT_MAX_WORDS,
            "max_chars": min(rules.get("max_chars", EMPLOYMENT_TEXT_MAX_CHARS),
                             EMPLOYMENT_TEXT_MAX_CHARS)
        })
    return rules


def browser_text_value(value, single_line=False):
    """Never send Selenium special-key characters or implicit Enter to an input."""
    text = re.sub(r"[\uE000-\uF8FF]", "", str(value))
    return re.sub(r"\s+", " ", text).strip() if single_line else text


def validate_text_value(value, field_key, rules):
    if rules.get("role") == "employment_keywords" and value.strip():
        compact = re.sub(r"\s+", " ", value).strip()
        reasoning = re.search(
            r"\b(?:ich|wir|mein|meine|meinen|meinem|meiner|weil|deshalb|daher|damit|"
            r"um|bin|war|habe|hatte|arbeite|arbeitet|möchte|möchten)\b", compact, re.I
        )
        if (re.search(r"[\r\n.!?;\uE000-\uF8FF]", value) or reasoning
                or len(compact.split()) > rules["max_words"]
                or len(compact) > rules["max_chars"]
                or not re.search(r"[A-Za-zÄÖÜäöüß]", compact)):
            raise ValueError(
                f"{field_key} requires only a keyword job/status description: "
                f"1-{rules['max_words']} words, at most {rules['max_chars']} characters. "
                "No complete sentence, explanation, personal history, sentence punctuation, "
                "or line breaks. Use a listed category if it already describes the status."
            )
    value = browser_text_value(value, rules.get("single_line", False))
    if rules.get("max_chars") and len(value) > rules["max_chars"]:
        raise ValueError(f"{field_key} allows at most {rules['max_chars']} characters.")
    return value


def listed_employment_choice(specification, group):
    """Recognize category names; a genuine job-title phrase may still use Other."""
    text = normalize_option_text(specification.splitlines()[0]).strip(" .:;")
    listed = [option for option in group["options"] if not option.get("text_field_key")]
    exact = [option for option in listed if normalize_option_text(option["label"]) == text]
    if len(exact) == 1:
        return exact[0]
    aliases = (
        (r"angestellt(?:e|er)?(?: erwerbstätig)?", r"angestellt"),
        (r"arbeitslos|arbeitsuchend|arbeitssuchend", r"arbeitslos"),
        (r"(?:berufliche )?ausbildung(?:/lehre)?|lehre|azubi|auszubildende[rs]?", r"ausbildung|lehre"),
        (r"schule|studium|schule oder studium|student(?:in)?|schüler(?:in)?", r"schule|studium"),
        (r"selbst(?:ständig|ändig)(?: gemeldet)?|freiberuflich", r"selbst(?:ständig|ändig)"),
        (r"verbeamtet|beamter|beamtin", r"verbeamtet"),
        (r"altersteilzeit", r"altersteilzeit"),
        (r"vorruhestand(?:, rente)?|rente|rentner(?:in)?|pensioniert", r"vorruhestand|rente"),
        (r"umschulung", r"umschulung"),
        (r"freiwilliger wehrdienst|bundesfreiwilligendienst|freiwilliges soziales jahr|"
         r"freiwilligendienst|fsj|bfd", r"freiwillig|bundesfreiwilligendienst"),
        (r"mutterschutz|pflegezeit|sonstige beurlaubung", r"mutterschutz|pflegezeit|beurlaubung")
    )
    for source, target in aliases:
        if re.fullmatch(source, text):
            matches = [option for option in listed if re.search(target, normalize_option_text(option["label"]))]
            if len(matches) == 1:
                return matches[0]
    return None


def canonicalize_employment_choices(responses, others, groups, policy):
    """Use the listed control when the Other text merely repeats a known status."""
    corrections = []
    for group in groups:
        for other in group["options"]:
            key = other.get("text_field_key")
            if policy.get("text_rules", {}).get(key, {}).get("role") != "employment_keywords":
                continue
            value = others.get(key, "")
            if not isinstance(value, str) or not value.strip():
                continue
            listed = listed_employment_choice(value, group)
            response = responses.get(group["group_key"], {})
            if listed is None or not isinstance(response, dict):
                continue
            recoded = False
            if (set(response) == {"mode", "choice"} and response["mode"] == "direct"
                    and response["choice"] == other["choice_label"]):
                responses[group["group_key"]] = {"mode": "direct", "choice": listed["choice_label"]}
                recoded = True
            elif set(response) == {"mode", "weights"} and response["mode"] == "sample":
                weights = response["weights"]
                labels = {option["choice_label"] for option in group["options"]}
                if (isinstance(weights, dict) and not (set(weights) - labels)
                        and all(not isinstance(weight, bool) and isinstance(weight, (int, float))
                                and math.isfinite(weight) and 0 <= weight <= 100 for weight in weights.values())
                        and weights.get(other["choice_label"], 0) > 0):
                    merged = dict(weights)
                    merged[listed["choice_label"]] = (
                        merged.get(listed["choice_label"], 0) + merged.pop(other["choice_label"])
                    )
                    total = sum(merged.values())
                    responses[group["group_key"]] = {
                        "mode": "sample", "weights": {label: 100 * weight / total for label, weight in merged.items()}
                    }
                    recoded = True
            if recoded:
                others[key] = ""
                corrections.append({
                    "kind": "listed_employment_status", "field_key": key,
                    "original_specification": value, "listed_choice": listed["choice_label"],
                    "reason": "The specification names an employment status already offered in the list."
                })
    return corrections


def bernoulli_draw(probability):
    if (isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not math.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError("A Bernoulli probability must be a finite number between 0 and 1.")
    uniform = random.random()
    return uniform < probability, uniform


def attention_instruction(visible_text, group, group_count):
    text = normalize_option_text(visible_text if group_count == 1 else group.get("question_text", ""))
    if re.search(
        r"(?:bitte\s+)?wählen sie(?:\s+hier)?\s+keine antwort|"
        r"bitte lassen sie.{0,60}(?:unbeantwortet|leer)|"
        r"bitte beantworten sie.{0,40}nicht|"
        r"(?:please\s+)?leave (?:this|the) (?:item|question).{0,20}(?:blank|unanswered)|"
        r"(?:please\s+)?do not (?:answer|select)", text
    ):
        return {"kind": "leave_unanswered"}
    # Exact quoted options cover explicit select-a-particular-answer checks.
    instruction = re.search(r"(?:bitte wählen sie|please (?:select|choose)).{0,150}", text)
    if instruction:
        quoted = re.findall(r'["„“”«»\']([^"„“”«»\']+)["„“”«»\']', instruction[0])
        matches = [option for option in group["options"] if normalize_option_text(option["label"]) in quoted]
        if len(matches) == 1:
            return {"kind": "select_choice", "choice": matches[0]["choice_label"]}
    return None


def prepare_response_policy(visible_text, elements, groups):
    """Draw once before model generation; retries reuse this exact policy."""
    active = [dict(group, options=[dict(option) for option in group["options"]]) for group in groups]
    conditional = {option["text_field_key"]: group for group in groups
                   for option in group["options"] if option.get("text_field_key")}
    policy = {"draws": [], "blank_fields": [], "fill_fields": [],
              "skipped_groups": [], "fixed_attention": {}, "clear_actions": [],
              "text_rules": {}}
    for field in elements:
        if not is_free_text_field(field):
            continue
        key = field["element_key"]
        other_group = conditional.get(key)
        policy["text_rules"][key] = text_field_rules(field, visible_text, other_group)
        if (field.get("required") and not other_group) or (other_group and len(other_group["options"]) == 1):
            continue
        blank, uniform = bernoulli_draw(OPTIONAL_TEXT_BLANK_PROBABILITY)
        policy["draws"].append({
            "kind": "optional_text_blank", "target": key,
            "probability": OPTIONAL_TEXT_BLANK_PROBABILITY,
            "uniform_draw": uniform, "success": blank,
            "outcome": "leave blank" if blank else "allow conditional Other" if other_group else "write a short comment"
        })
        policy["blank_fields" if blank else "fill_fields"].append(key)
        if blank and other_group:
            for group in active:
                group["options"] = [option for option in group["options"] if option.get("text_field_key") != key]
                if not group["options"]:
                    raise ValueError(f"No listed choices remain for {group['group_key']}.")
    for group in active:
        instruction = attention_instruction(visible_text, group, len(groups))
        if not instruction:
            continue
        passed, uniform = bernoulli_draw(ATTENTION_CHECK_PASS_PROBABILITY)
        key = group["group_key"]
        skip = instruction["kind"] == "leave_unanswered" and passed
        choices = [option for option in group["options"] if option.get("input_mode") != "other_text"]
        if instruction["kind"] == "select_choice" and passed:
            selected = instruction["choice"]
        elif not passed:
            wrong = [option for option in choices if instruction["kind"] == "leave_unanswered"
                     or option["choice_label"] != instruction.get("choice")]
            selected = random.choice(wrong)["choice_label"] if wrong else None
            skip = selected is None
        else:
            selected = None
        if skip:
            policy["skipped_groups"].append(key)
        else:
            policy["fixed_attention"][key] = selected
        policy["draws"].append({
            "kind": "attention_check_pass", "target": key,
            "probability": ATTENTION_CHECK_PASS_PROBABILITY,
            "uniform_draw": uniform, "success": passed, "instruction": instruction,
            "outcome": "pass: leave unanswered" if passed and skip else
                       "pass: select instructed choice" if passed else "fail: answer contrary to instruction",
            "selected_choice": selected
        })
        if skip:
            group_keys = {option["element_key"] for option in group["options"]}
            for option in group["options"]:
                field_key = option.get("text_field_key")
                if field_key and field_key not in policy["blank_fields"]:
                    policy["blank_fields"].append(field_key)
            for element in elements:
                if element["element_key"] in group_keys and element.get("selected"):
                    policy["clear_actions"].append({
                        "element_key": element["element_key"], "action": "clear_selection",
                        "value": "", "answer_mode": "attention_pass"
                    })
    for field in elements:
        if field["element_key"] in policy["blank_fields"] and field.get("value"):
            policy["clear_actions"].append({
                "element_key": field["element_key"], "action": "clear", "value": "",
                "answer_mode": "optional_blank"
            })
    active = [group for group in active if group["group_key"] not in policy["skipped_groups"]]
    return active, policy

def normalize_option_text(value):
    """Normalize formatting, while retaining words such as 'nicht' and numbers."""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.translate(str.maketrans({"–": "-", "—": "-", "−": "-"}))
    return " ".join(text.split()).casefold()

def numeric_option_value(value):
    """Compare scalar numbers exactly, including 30.0 versus 30 or '30 Stunden'."""
    text = normalize_option_text(value)
    match = re.fullmatch(
        r"([+-]?\d+(?:[.,]\d+)?)(?:\s*(?:stunden?|std\.?|hours?|h))?", text
    )
    if not match:
        return None
    try:
        return Decimal(match.group(1).replace(",", "."))
    except InvalidOperation:
        return None

def match_dropdown_option(action, options):
    """Resolve an actual option identifier first; tolerate equivalent legacy text."""
    requested_key = action.get("option_key")
    value = action.get("value")
    matches = []
    if requested_key is not None:
        matches = [option for option in options if option["option_key"] == requested_key]
    elif value is not None:
        text = str(value)
        numeric = numeric_option_value(text)
        tests = [
            lambda option: text == str(option["label"]),
            lambda option: text == str(option.get("value", "")),
            lambda option: normalize_option_text(text) == normalize_option_text(option["label"]),
            lambda option: text.strip() == str(option.get("value", "")).strip()
        ]
        if numeric is not None:
            tests.append(lambda option: numeric == numeric_option_value(option["label"]))
            if re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", normalize_option_text(text)):
                tests.append(lambda option: numeric == numeric_option_value(option.get("value", "")))
        for test in tests:
            matches = [option for option in options if test(option)]
            if matches:
                break
    if len(matches) == 1:
        return matches[0]
    allowed = [{"option_key": option["option_key"], "label": option["label"],
                "value": option.get("value", "")} for option in options]
    raise ValueError(
        f"Dropdown {action.get('element_key')}: returned value={value!r}, "
        f"option_key={requested_key!r}; "
        f"{'multiple options match' if matches else 'no option matches'}. "
        "Choose one exact option_key from: " + json.dumps(allowed, ensure_ascii=False)
    )

def dropdown_action(option, mode):
    """Carry DOM identity through validation, logging and Selenium execution."""
    return {
        "element_key": option["element_key"], "action": "select",
        "value": option["label"], "answer_mode": mode,
        "option_key": option["option_key"], "option_value": option.get("value", ""),
        "option_index": option["option_index"]
    }

def fixed_group_choices(persona, visible_text, groups):
    """Answer recognizable assigned demographics locally; never sample them."""
    text = normalize_option_text(visible_text)
    fixed = {}
    gender_labels = {
        "male": ("männlich", "mann", "male"),
        "female": ("weiblich", "frau", "female"),
        "diverse": ("divers", "non-binary", "nonbinary")
    }
    gender_question = bool(re.search(
        r"welches geschlecht (?:haben|hatten) sie|ihr geschlecht|your gender", text
    ))
    state_question = "bundesland" in text and bool(re.search(
        r"wohnen sie|leben sie|wohnort|in welchem bundesland", text
    ))
    age_question = bool(re.search(r"wie alt sind sie|how old are you|ihr alter", text))
    for group in groups:
        candidates = []
        for option in group["options"]:
            label = normalize_option_text(option["label"])
            if gender_question and label in gender_labels.get(persona.get("gender"), ()):
                candidates.append(option)
            elif state_question and label == normalize_option_text(persona.get("bundesland", "")):
                candidates.append(option)
            elif age_question and "age" in persona:
                age = persona["age"]
                numeric = numeric_option_value(label)
                interval = re.fullmatch(r"(\d+)\s*(?:-|bis)\s*(\d+)(?:\s+jahre)?", label)
                if numeric == age or (interval and int(interval[1]) <= age <= int(interval[2])):
                    candidates.append(option)
        if len(candidates) == 1:
            fixed[group["group_key"]] = candidates[0]["choice_label"]
        elif len(candidates) > 1:
            raise ValueError(f"Assigned demographic matches multiple options: {group['group_key']}")
    return fixed

def other_answer_elements(elements):
    return [element for element in elements if element.get("type") not in ("radio", "dropdown")]

def answer_schema(groups, other_elements, fixed, policy=None):
    """Require every field and constrain choices to actual labels through Ollama."""
    answers = {}
    for group in groups:
        if group["group_key"] in fixed:
            continue
        labels = [option["choice_label"] for option in group["options"]]
        if not labels:
            raise ValueError(f"No selectable options: {group['group_key']}")
        direct = {
            "type": "object",
            "properties": {"mode": {"type": "string", "enum": ["direct"]},
                           "choice": {"type": "string", "enum": labels}},
            "required": ["mode", "choice"], "additionalProperties": False
        }
        sampled = {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["sample"]},
                "weights": {"type": "object", "properties": {
                    label: {"type": "number", "minimum": 0, "maximum": 100}
                    for label in labels
                }, "minProperties": 1, "additionalProperties": False}
            }, "required": ["mode", "weights"], "additionalProperties": False
        }
        answers[group["group_key"]] = {"anyOf": [direct, sampled]}
    conditional = {option["text_field_key"] for group in groups
                   for option in group["options"] if option.get("text_field_key")}
    fill_fields = set((policy or {}).get("fill_fields", []))
    others = {}
    for element in other_elements:
        key = element["element_key"]
        if element.get("type") == "checkbox":
            others[key] = {"type": "boolean"}
        else:
            nonblank = key not in conditional and (element.get("required") or key in fill_fields)
            others[key] = {"type": "string", "minLength": 1 if nonblank else 0}
            rules = (policy or {}).get("text_rules", {}).get(key, {})
            if rules.get("max_chars"):
                others[key]["maxLength"] = rules["max_chars"]
            if rules.get("role") == "employment_keywords":
                others[key]["pattern"] = r"^[^\r\n.!?;\uE000-\uF8FF]*$"
    return {
        "type": "object", "properties": {
            "answers": {"type": "object", "properties": answers,
                        "required": list(answers), "additionalProperties": False},
            "other_answers": {"type": "object", "properties": others,
                              "required": list(others), "additionalProperties": False},
            "response_time_seconds": {"type": "number", "minimum": MIN_RESPONSE_TIME,
                                      "maximum": MAX_RESPONSE_TIME}
        }, "required": ["answers", "other_answers", "response_time_seconds"],
        "additionalProperties": False
    }

def resolve_grouped_answer(answer, elements, groups, fixed, policy=None):
    """Convert labeled model answers to canonical DOM actions before drawing."""
    policy = policy or {"draws": [], "blank_fields": [], "fill_fields": [],
                        "skipped_groups": [], "clear_actions": []}
    if not isinstance(answer, dict):
        raise ValueError("The response must be a JSON object.")
    responses = answer.get("answers")
    model_others = answer.get("other_answers")
    if not isinstance(responses, dict) or not isinstance(model_others, dict):
        raise ValueError("answers and other_answers must be objects.")
    expected = {group["group_key"] for group in groups} - set(fixed)
    if set(responses) != expected:
        raise ValueError(f"Required answer groups: {sorted(expected)}; returned: {sorted(responses)}")
    extra_elements = other_answer_elements(elements)
    blank_fields = set(policy["blank_fields"])
    expected_fields = {element["element_key"] for element in extra_elements} - blank_fields
    if set(model_others) != expected_fields:
        raise ValueError(f"Required other fields: {sorted(expected_fields)}; returned: {sorted(model_others)}")
    others = {key: "" for key in blank_fields}
    others.update(model_others)
    responses = dict(responses)
    corrections = canonicalize_employment_choices(responses, others, groups, policy)
    conditional = {option["text_field_key"] for group in groups
                   for option in group["options"] if option.get("text_field_key")}
    for element in extra_elements:
        key, value = element["element_key"], others[element["element_key"]]
        if element.get("type") == "checkbox":
            if not isinstance(value, bool):
                raise ValueError(f"Checkbox {key} requires true or false.")
        elif not isinstance(value, str):
            raise ValueError(f"Text field {key} requires a string.")
        elif (key not in blank_fields and key not in conditional
              and (element.get("required") or key in policy["fill_fields"]) and not value.strip()):
            raise ValueError(f"Text field {key} requires a nonempty answer for this draw.")
        if isinstance(value, str):
            others[key] = validate_text_value(value, key, policy.get("text_rules", {}).get(key, {}))

    plans = []
    for group in groups:
        key = group["group_key"]
        response = {"mode": "direct", "choice": fixed[key]} if key in fixed else responses[key]
        if not isinstance(response, dict):
            raise ValueError(f"Answer for {key} must be an object.")
        options = {option["choice_label"]: option for option in group["options"]}
        if response.get("mode") == "direct":
            if (set(response) != {"mode", "choice"}
                    or not isinstance(response.get("choice"), str)
                    or response["choice"] not in options):
                raise ValueError(f"Choose an actual labeled option for {key}: {list(options)}")
            option = options[response["choice"]]
            plans.append((group, option, None))
        elif response.get("mode") == "sample":
            weights = response.get("weights")
            if set(response) != {"mode", "weights"} or not isinstance(weights, dict):
                raise ValueError(f"Supply an object of labeled weights for {key}.")
            if not weights or set(weights) - set(options):
                raise ValueError(f"Weights for {key} must name actual choice labels: {list(options)}")
            values = [
                weights.get(option["choice_label"], 0) for option in group["options"]
            ]
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or not 0 <= value <= 100 for value in values):
                raise ValueError(f"Weights must be finite numbers between 0 and 100: {key}")
            total = sum(values)
            if total <= 0:
                raise ValueError(f"At least one option needs a positive weight: {key}")
            plans.append((group, None, [value / total for value in values]))
        else:
            raise ValueError(f"Answer mode for {key} must be direct or sample.")

    # Validate every possible Other specification before any answer draw.
    for group, direct_option, probabilities in plans:
        possible = [direct_option] if direct_option is not None else [
            option for option, probability in zip(group["options"], probabilities) if probability > 0
        ]
        for option in possible:
            field_key = option.get("text_field_key")
            if field_key and not others.get(field_key, "").strip():
                raise ValueError(f"An Other choice needs a short specification in {field_key}.")
    seconds = answer.get("response_time_seconds", 3.0)
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        raise ValueError("response_time_seconds must be a finite number.")
    actions = [dict(action) for action in policy["clear_actions"]]
    sampling, selected_other, other_labels = [], set(), {}
    for group, option, probabilities in plans:
        mode = "direct"
        if probabilities is not None:
            option = random.choices(group["options"], weights=probabilities, k=1)[0]
            mode = "sampled"
            sampling.append({"group_key": group["group_key"], "options": group["options"],
                             "probabilities": probabilities, "chosen_element_key": option["element_key"],
                             "chosen_label": option["label"]})
        field_key = option.get("text_field_key")
        if field_key:
            selected_other.add(field_key)
            other_labels[field_key] = option["label"]
        if option.get("input_mode") == "other_text":
            continue
        if group["type"] == "dropdown":
            actions.append(dropdown_action(option, mode))
        else:
            actions.append({"element_key": option["element_key"], "action": "click",
                            "value": option["label"], "answer_mode": mode})
    for element in extra_elements:
        key, value = element["element_key"], others[element["element_key"]]
        if key in conditional and key not in selected_other:
            if element.get("value") and not any(action["element_key"] == key for action in actions):
                actions.insert(0, {"element_key": key, "action": "clear", "value": "",
                                   "answer_mode": "inactive_other"})
            continue
        if element.get("type") == "checkbox":
            if value:
                actions.append({"element_key": key, "action": "click",
                                "value": element.get("label", ""), "answer_mode": "direct"})
        elif value.strip():
            action = {"element_key": key, "action": "type", "value": value, "answer_mode": "direct"}
            if policy.get("text_rules", {}).get(key):
                action["text_rules"] = policy["text_rules"][key]
            if key in other_labels:
                action["choice_label"] = other_labels[key]
            actions.append(action)
    omissions = [{"kind": "optional_text", "target": key} for key in policy["blank_fields"]]
    omissions.extend({"kind": "attention_check", "target": key} for key in policy["skipped_groups"])
    summaries = [f"{action.get('choice_label', '') + ': ' if action.get('choice_label') else ''}"
                 f"{action['value']} ({action['answer_mode']})" for action in actions if action["action"] not in {"clear", "clear_selection"}]
    if policy["blank_fields"]:
        summaries.append("Optional text left blank")
    if policy["skipped_groups"]:
        summaries.append("Attention check: item intentionally left unanswered")
    return {"actions": actions, "sampling": sampling, "response_time_seconds": seconds,
            "assessment": "", "summary": "; ".join(summaries) or "No answer action needed.",
            "behavior_draws": policy["draws"], "intentional_omissions": omissions,
            "answer_corrections": corrections}

def resolve_answer(answer, elements, groups):
    """Validate everything before sampling; derive clicked labels from the DOM."""
    if not isinstance(answer, dict):
        raise ValueError("The response must be a JSON object.")
    direct = answer.get("actions", [])
    distributions = answer.get("choice_distributions", [])
    if not isinstance(direct, list) or not isinstance(distributions, list):
        raise ValueError("actions and choice_distributions must be arrays.")

    by_key = {element["element_key"]: element for element in elements}
    by_group = {group["group_key"]: group for group in groups}
    key_to_group = {
        option["element_key"]: group["group_key"]
        for group in groups for option in group["options"]
    }
    actions, plans, covered, seen_keys = [], [], set(), set()
    for action in direct:
        if not isinstance(action, dict):
            raise ValueError("Each direct action must be an object.")
        key = action.get("element_key")
        if not isinstance(key, str) or key not in by_key or key in seen_keys:
            raise ValueError(f"Invalid or duplicate action key: {key}")
        seen_keys.add(key)
        element = by_key[key]
        kind = element.get("type")
        value = str(action.get("value", ""))
        if kind in ("radio", "dropdown"):
            group_key = key_to_group.get(key)
            if group_key is None or group_key in covered:
                raise ValueError(f"Single-choice group has no options or duplicates: {key}")
            covered.add(group_key)
            if kind == "radio":
                label = element.get("label") or element.get("value", "")
                if value and normalize_option_text(value) != normalize_option_text(label):
                    raise ValueError(f"For {key}, value must match its label: {label}")
                value = label
            else:
                option = match_dropdown_option(action, by_group[group_key]["options"])
                actions.append(dropdown_action(option, "direct"))
                continue
        elif kind == "checkbox":
            value = element.get("label") or element.get("value", "")
        elif action.get("action") != "type":
            raise ValueError(f"Text element requires action='type': {key}")
        actions.append({
            "element_key": key,
            "action": "select" if kind == "dropdown" else (
                "click" if kind in ("radio", "checkbox") else "type"
            ),
            "value": value, "answer_mode": "direct"
        })

    for distribution in distributions:
        if not isinstance(distribution, dict):
            raise ValueError("Each choice_distribution must be an object.")
        group_key = distribution.get("group_key")
        if not isinstance(group_key, str) or group_key not in by_group or group_key in covered:
            raise ValueError(f"Unknown or duplicate distribution group: {group_key}")
        group = by_group[group_key]
        weights = distribution.get("weights")
        if not isinstance(weights, list) or len(weights) != len(group["options"]):
            raise ValueError(f"Return one weight per option, in order, for {group_key}")
        if any(isinstance(weight, bool) or not isinstance(weight, (int, float))
               or not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError(f"Weights must be finite nonnegative numbers: {group_key}")
        # Scale first so summing very large valid weights cannot overflow.
        largest = max(weights, default=0)
        if largest <= 0:
            raise ValueError(f"At least one option needs a positive weight: {group_key}")
        scaled = [weight / largest for weight in weights]
        total = sum(scaled)
        probabilities = [weight / total for weight in scaled]
        covered.add(group_key)
        plans.append((group, probabilities))

    missing = set(by_group) - covered
    if missing:
        raise ValueError(f"Missing single-choice answer(s): {sorted(missing)}")

    sampling = []
    for group, probabilities in plans:
        chosen = random.choices(group["options"], weights=probabilities, k=1)[0]
        if group["type"] == "dropdown":
            actions.append(dropdown_action(chosen, "sampled"))
        else:
            actions.append({
                "element_key": chosen["element_key"], "action": "click",
                "value": chosen["label"], "answer_mode": "sampled"
            })
        sampling.append({
            "group_key": group["group_key"],
            "options": group["options"], "probabilities": probabilities,
            "chosen_element_key": chosen["element_key"],
            "chosen_label": chosen["label"]
        })

    return {
        "actions": actions, "sampling": sampling,
        "response_time_seconds": answer.get("response_time_seconds", 3.0),
        "assessment": answer.get("summary", ""),
        "summary": "; ".join(
            f"{action['value']} ({action['answer_mode']})" for action in actions
        ) or "No checkbox options selected."
    }

def ask_ai(persona, history, visible_text, elements, response_context=None):
    groups = single_choice_groups(elements, visible_text)
    groups, policy = prepare_response_policy(visible_text, elements, groups)
    if response_context is not None:
        response_context.update(behavior_draws=policy["draws"], response_policy=policy)
    fixed = fixed_group_choices(persona, visible_text, groups)
    fixed.update(policy["fixed_attention"])
    other_elements = [element for element in other_answer_elements(elements)
                      if element["element_key"] not in policy["blank_fields"]]
    model_groups = [group for group in groups if group["group_key"] not in fixed]
    if not model_groups and not other_elements:
        return resolve_grouped_answer({
            "answers": {}, "other_answers": {}, "response_time_seconds": 2.0
        }, elements, groups, fixed, policy)

    schema = answer_schema(groups, other_elements, fixed, policy)
    required_groups = [{
        "group_key": group["group_key"], "type": group["type"],
        "response_role": "employment_status" if any(
            rule.get("employment_status_group") == group["group_key"]
            for rule in policy["text_rules"].values()
        ) else "single_choice",
        "choices": [option["choice_label"] for option in group["options"]]
    } for group in model_groups]
    other_fields = [{
        "field_key": element["element_key"], "type": element.get("type"),
        "label": element.get("label", ""), "required": bool(element.get("required")),
        "must_fill_for_this_draw": element["element_key"] in policy["fill_fields"],
        "response_constraints": policy["text_rules"].get(element["element_key"], {}),
        "conditional_choices": [{"group_key": group["group_key"], "choice": option["choice_label"]}
                                for group in groups for option in group["options"]
                                if option.get("text_field_key") == element["element_key"]]
    } for element in other_elements]
    prompt = f"""
{SURVEY_PROMPT}

ASSIGNED PERSONA:
{json.dumps(persona, ensure_ascii=False)}

PREVIOUS REALIZED ANSWERS (each belongs to its own question):
{json.dumps(compact_history_for_model(history), ensure_ascii=False)}

CURRENT SURVEY PAGE:
{visible_text}

FIXED ANSWERS ALREADY HANDLED BY PYTHON (do not return these groups):
{json.dumps(fixed, ensure_ascii=False)}

BEHAVIOR DECISIONS ALREADY DRAWN BY PYTHON (do not override or redraw):
{json.dumps(policy, ensure_ascii=False)}

REQUIRED SINGLE-CHOICE GROUPS:
{json.dumps(required_groups, ensure_ascii=False)}

REQUIRED OTHER FIELDS:
{json.dumps(other_fields, ensure_ascii=False)}

Return one JSON object containing answers, other_answers and response_time_seconds.
In answers, include EVERY group_key from REQUIRED SINGLE-CHOICE GROUPS exactly once.
For a known fact or explicit instruction, its answer object is:
{{"mode": "direct", "choice": "one exact choice label listed for that group"}}
For uncertainty or a subjective judgment, its answer object is:
{{"mode": "sample", "weights": {{"one exact choice label": 70, "another exact choice label": 30}}}}
Weights are an OBJECT keyed by choice labels, never a positional array. List only
plausible choices; omitted choices have weight zero. At least one weight must be
positive. For uncertain judgments, include several plausible choices when appropriate.

Copy labels exactly, including line breaks where present. A year is a choice such
as "2013". Do not construct option identifiers, DOM positions or element keys.
If this page requests a month and year, answer BOTH separate groups. Consider the
previously realized employment situation and this person's age when choosing the date.
Do not confuse employment dates with the person's birth date.

In other_answers, include every field_key from REQUIRED OTHER FIELDS, and no other
fields. Checkbox answers are true or false; text answers are short strings. A field
with must_fill_for_this_draw=true needs a brief informal German comment when it has
no conditional_choices. Required factual fields need a valid factual answer.
For a conditional Other field, prefer listed categories when any applies. Use an
empty string if its Other choice cannot be selected. If you select Other directly,
or give its choice positive weight, supply a short applicable specification. Python
will type it only if that Other choice is actually selected. Do not repeat the field
label, write a formal explanation, or fill a separate Other field alongside a listed
employment status. Skipped attention items require no model answer at all.
Employment status is different from a job title. Normally select the listed status
even when the person's specific occupation is not named in the list. Freelance and
self-employed work belong to the listed self-employment category; voluntary military,
federal or social service belongs to the listed voluntary-service category. The
optional-text draw only permits Other; it never requires you to choose it. Do not
paste an existing status name into Other. Prefer the appropriate listed control.
For response_constraints.role="employment_keywords", give only a keyword job/status
description, within max_words and max_chars. Never write a sentence, explanation,
personal history, line break, or copied list of existing options. Keep it to one
applicable role. Other text fields may still use short informal comments where asked.
Estimate response_time_seconds for this page.
The explanation is not needed; Python reports the actual realized answers.
Return JSON only and follow the supplied response schema.
"""
    option_count = sum(min(len(group["options"]), 5) for group in model_groups)
    token_limit = min(OLLAMA_MAX_NUM_PREDICT, max(
        OLLAMA_NUM_PREDICT, 250 + 80 * len(model_groups) + 55 * option_count + 80 * len(other_elements)
    ))
    validation_error, previous_raw = "", ""
    for attempt in range(AI_VALIDATION_ATTEMPTS):
        repair = ""
        if validation_error:
            repair = ("\nPrevious invalid response:\n" + previous_raw[:4000]
                      + "\nCorrect this validation error:\n" + validation_error)
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL, "prompt": prompt + repair, "stream": False,
            "format": schema, "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {"temperature": OLLAMA_TEMPERATURE,
                        "num_predict": token_limit, "num_ctx": OLLAMA_NUM_CTX}
        }, timeout=180)
        response.raise_for_status()
        payload = response.json()
        raw = payload.get("response", "")
        try:
            if payload.get("done_reason") == "length":
                raise ValueError("Model output was truncated; finish the complete answer object.")
            return resolve_grouped_answer(json.loads(raw), elements, groups, fixed, policy)
        except (ValueError, KeyError, TypeError) as error:
            validation_error, previous_raw = str(error), raw
            with open(MODEL_ERROR_LOG_FILE, "a", encoding="utf-8") as file:
                file.write(json.dumps({
                    "timestamp": datetime.now().isoformat(), "model": MODEL,
                    "attempt": attempt + 1, "question_text": visible_text,
                    "elements": elements, "raw_model_answer": raw,
                    "response_policy": policy,
                    "validation_error": validation_error
                }, ensure_ascii=False) + "\n")
            if attempt + 1 < AI_VALIDATION_ATTEMPTS:
                token_limit = min(OLLAMA_MAX_NUM_PREDICT, token_limit * 2)
                print(f"Retrying invalid model answer: {validation_error}")
    raise RuntimeError("Could not validate model answer: " + validation_error)


# ============================================================
# HTML HELPERS
# ============================================================

def form_control_metadata(driver, element):
    """Read required markers and local Other-option relationships from the DOM."""
    metadata = {
        "required": element.get_attribute("required") is not None
                    or element.get_attribute("aria-required") == "true",
        "input_mode": (element.get_attribute("inputmode") or "").lower(),
        "tag_name": (element.tag_name or "").lower(),
        "max_length": int(element.get_attribute("maxlength"))
                      if str(element.get_attribute("maxlength") or "").isdigit() else None
    }
    try:
        details = driver.execute_script(r"""
            const e = arguments[0];
            const q = e.closest('.question-wrapper, .question-container, .question, fieldset, [id^="question"]');
            const row = e.closest('li, .answer-item, .radio-item, td, label') || e.parentElement;
            const radios = row ? row.querySelectorAll('input[type="radio"]') : [];
            const context = row ? (row.innerText || row.textContent || '').trim() : '';
            const otherContext = /(?:\betwas anderes\b|\bsonstiges\b|\bother(?:\s|,|:|$)|\bsomething else\b)/i.test(context);
            const linked = e.getAttribute('data-other-for') ||
                (radios.length === 1 && otherContext ? radios[0].id : '');
            const prompt = q && q.querySelector('.question-text, .questiontext, .ls-label-question, legend');
            const questionRequired = q && (q.matches('.mandatory, .required, .question-mandatory') ||
                q.querySelector('.asterisk, .mandatory-marker, [aria-label="Required"]'));
            return {
                required: Boolean(e.required || e.getAttribute('aria-required') === 'true' || questionRequired),
                input_mode: (e.getAttribute('inputmode') || '').toLowerCase(),
                other_option_key: linked ? (linked.startsWith('id::') ? linked : 'id::' + linked) : '',
                context_label: context.length <= 200 ? context : '',
                question_text: prompt ? (prompt.innerText || prompt.textContent || '').trim() : ''
            };
        """, element)
        if isinstance(details, dict):
            metadata.update(details)
    except Exception:
        pass
    return metadata


def get_label(driver, element):
    element_id = element.get_attribute("id")

    if element_id:
        try:
            label = driver.find_element(
                By.CSS_SELECTOR,
                f'label[for="{element_id}"]'
            )

            text = label.text.strip()

            if text:
                return text

        except Exception:
            pass

    try:
        label = element.find_element(
            By.XPATH,
            "./ancestor::label[1]"
        )

        text = label.text.strip()

        if text:
            return text

    except Exception:
        pass

    # Accessible labels may be separate nodes rather than <label for=...>.
    try:
        label_ids = (element.get_attribute("aria-labelledby") or "").split()
        text = " ".join(
            driver.find_element(By.ID, label_id).text.strip()
            for label_id in label_ids
        ).strip()
        if text:
            return text
    except Exception:
        pass

    for attribute in [
        "aria-label",
        "placeholder",
        "title"
    ]:
        try:
            value = element.get_attribute(attribute)

            if value:
                return value.strip()

        except Exception:
            pass

    return ""

def make_element_key(
    element,
    element_type,
    index
):
    element_id = element.get_attribute("id")

    if element_id:
        return f"id::{element_id}"

    name = element.get_attribute("name")

    if name:
        return f"name::{name}::{index}"

    return f"{element_type}::{index}"

def visible_numeric_scale(visible_text, group_size):
    """Use an explicitly displayed sequence, never infer anchors from its size."""
    if group_size < 3:
        return []
    lines = [line.strip() for line in visible_text.splitlines() if line.strip()]
    for start in range(len(lines) - group_size, -1, -1):
        candidate = lines[start:start + group_size]
        matches = [re.fullmatch(r"([+-]?\d+)(?:\s+.*)?", line) for line in candidate]
        if not all(matches):
            continue
        values = [int(match.group(1)) for match in matches]
        step = values[1] - values[0]
        if abs(step) == 1 and all(
            right - left == step for left, right in zip(values, values[1:])
        ):
            return candidate
    return []

def get_radio_label(
    driver,
    element,
    index_in_group,
    group_size,
    visible_text=""
):
    label = get_label(
        driver,
        element
    )

    if label:
        return label

    # Trust (0..10) and fairness (-5..+5) can both have eleven options.
    scale = visible_numeric_scale(visible_text, group_size)
    if scale and 0 <= index_in_group < len(scale):
        return scale[index_in_group]

    return (
        element.get_attribute("value")
        or ""
    )

# ============================================================
# EXTRACT CURRENT PAGE
# ============================================================

def extract_page(driver):
    try:
        visible_text = driver.find_element(
            By.TAG_NAME,
            "body"
        ).text
    except Exception:
        visible_text = ""

    elements = []

    # ----------------------------
    # INPUTS
    # ----------------------------

    inputs = driver.find_elements(
        By.CSS_SELECTOR,
        "input"
    )

    radio_groups = {}

    for element in inputs:
        try:
            kind = (
                element.get_attribute("type")
                or ""
            ).lower()

            if (
                element.is_displayed()
                and kind == "radio"
            ):
                name = (
                    element.get_attribute("name")
                    or ""
                )

                radio_groups.setdefault(
                    name,
                    []
                ).append(element)

        except Exception:
            continue

    for i, element in enumerate(inputs):
        try:
            if not element.is_displayed():
                continue

            kind = (
                element.get_attribute("type")
                or "text"
            ).lower()

            if kind in [
                "hidden",
                "submit",
                "button",
                "reset",
                "file",
                "image"
            ]:
                continue

            name = (
                element.get_attribute("name")
                or ""
            )

            if kind == "radio":
                group = radio_groups.get(
                    name,
                    []
                )

                try:
                    index_in_group = group.index(
                        element
                    )
                except ValueError:
                    index_in_group = 0

                label = get_radio_label(
                    driver=driver,
                    element=element,
                    index_in_group=index_in_group,
                    group_size=len(group),
                    visible_text=visible_text
                )

            else:
                label = get_label(
                    driver,
                    element
                )

            elements.append({
                "element_key": make_element_key(
                    element,
                    "input",
                    i
                ),
                "type": kind,
                "group": name,
                "label": label,
                "value": (
                    element.get_attribute("value")
                    or ""
                ),
                "selected": (
                    element.is_selected()
                    if kind in [
                        "radio",
                        "checkbox"
                    ]
                    else False
                ),
                **form_control_metadata(driver, element)
            })

        except Exception:
            continue

    # ----------------------------
    # DROPDOWNS
    # ----------------------------

    selects = driver.find_elements(
        By.TAG_NAME,
        "select"
    )

    for i, element in enumerate(selects):
        try:
            if not element.is_displayed():
                continue

            options = []
            key = make_element_key(element, "select", i)

            for option_index, option in enumerate(Select(element).options):
                text = option.text.strip()

                if text and option.is_enabled():
                    options.append({
                        "label": text,
                        "value": option.get_attribute("value") or "",
                        "index": option_index,
                        "option_key": f"option::{key}::{option_index}"
                    })

            elements.append({
                "element_key": key,
                "type": "dropdown",
                "group": (
                    element.get_attribute("name")
                    or element.get_attribute("id")
                    or f"select_{i}"
                ),
                "label": get_label(
                    driver,
                    element
                ),
                "options": options,
                **form_control_metadata(driver, element)
            })

        except Exception:
            continue

    # ----------------------------
    # TEXT INPUTS
    # ----------------------------

    for i, element in enumerate(inputs):
        try:
            if not element.is_displayed():
                continue

            kind = (
                element.get_attribute("type")
                or ""
            ).lower()

            if kind not in [
                "text",
                "number",
                "email",
                "tel",
                "url"
            ]:
                continue

            key = make_element_key(
                element,
                "input",
                i
            )

            if not any(
                item.get("element_key") == key
                for item in elements
            ):
                elements.append({
                    "element_key": key,
                    "type": "open_text",
                    "group": (
                        element.get_attribute("name")
                        or element.get_attribute("id")
                        or f"input_{i}"
                    ),
                    "label": get_label(
                        driver,
                        element
                    ),
                    "value": element.get_attribute("value") or "",
                    **form_control_metadata(driver, element)
                })

        except Exception:
            continue

    # ----------------------------
    # TEXTAREAS
    # ----------------------------

    textareas = driver.find_elements(
        By.TAG_NAME,
        "textarea"
    )

    for i, element in enumerate(textareas):
        try:
            if not element.is_displayed():
                continue

            elements.append({
                "element_key": make_element_key(
                    element,
                    "textarea",
                    i
                ),
                "type": "open_text",
                "group": (
                    element.get_attribute("name")
                    or element.get_attribute("id")
                    or f"textarea_{i}"
                ),
                "label": get_label(
                    driver,
                    element
                ),
                "value": element.get_attribute("value") or "",
                **form_control_metadata(driver, element)
            })

        except Exception:
            continue

    return visible_text, elements

# ============================================================
# FIND ELEMENT AGAIN
# ============================================================

def find_element(driver, key):
    if key.startswith("id::"):
        value = key.split("::", 1)[1]

        try:
            return driver.find_element(
                By.ID,
                value
            )
        except Exception:
            return None

    if key.startswith("name::"):
        parts = key.split("::")

        if len(parts) != 3:
            return None

        name = parts[1]
        index = int(parts[2])

        matches = driver.find_elements(
            By.NAME,
            name
        )

        if index < len(matches):
            return matches[index]

        return None

    if key.startswith("input::"):
        index = int(
            key.split("::")[1]
        )

        elements = driver.find_elements(
            By.CSS_SELECTOR,
            "input"
        )

        if index < len(elements):
            return elements[index]

        return None

    if key.startswith("select::"):
        index = int(
            key.split("::")[1]
        )

        elements = driver.find_elements(
            By.TAG_NAME,
            "select"
        )

        if index < len(elements):
            return elements[index]

        return None

    if key.startswith("textarea::"):
        index = int(
            key.split("::")[1]
        )

        elements = driver.find_elements(
            By.TAG_NAME,
            "textarea"
        )

        if index < len(elements):
            return elements[index]

        return None

    return None

# ============================================================
# DETERMINISTIC SURVEY PAGES
# ============================================================

def wait_before_answer(seconds, action_description):
    print(
        f"Waiting {seconds:.2f} seconds before {action_description}...",
        flush=True
    )
    started = time.monotonic()
    time.sleep(seconds)
    observed_wait = time.monotonic() - started
    print(
        f"Wait completed after {observed_wait:.2f} seconds; "
        f"{action_description} now.",
        flush=True
    )
    return observed_wait


def is_data_consent_question(visible_text, elements):
    text = " ".join(visible_text.lower().split())
    markers = (
        "einwilligung", "einverständniserklärung", "daten zusammenführen",
        "zusammenführung ihrer daten", "datenzusammenführung"
    )
    if not any(marker in text for marker in markers):
        return False
    return any(
        item.get("type") == "radio"
        and (
            str(item.get("label", "")).strip().lower() in {"ja", "nein", "yes", "no"}
            or "einverstanden" in str(item.get("label", "")).lower()
        )
        for item in elements
    )


def handle_participation_question(
    driver,
    visible_text,
    elements,
    timing
):
    """
    Read a recognized participation page before choosing Ja/Yes.
    Return the reading-time details if handled, otherwise None.
    """
    text = " ".join(visible_text.lower().split())

    participation_markers = [
        "möchten sie teilnehmen",
        "möchten sie an der befragung teilnehmen",
        "an der befragung teilnehmen",
        "teilnahme an der befragung",
        "an der umfrage teilnehmen",
        "teilnahme an der umfrage"
    ]

    if not any(
        marker in text
        for marker in participation_markers
    ):
        return None

    for item in elements:
        if item.get("type") != "radio":
            continue

        label = str(
            item.get("label", "")
        ).strip().lower()

        if label in ["ja", "yes"]:
            element = find_element(
                driver,
                item["element_key"]
            )

            if element is None:
                continue

            pre_answer_delay, timing_info = timing.agreement_time(
                visible_text
            )
            actual_reading_time = timing_info["actual_reading_time"]

            print(
                f"LOCAL PARTICIPATION READING TIME: "
                f"{actual_reading_time:.2f} seconds"
            )
            print(
                "READING FACTORS: "
                f"{timing_info['word_count']} words, "
                f"{timing_info['reading_wpm']:.0f} wpm, "
                f"pace={timing_info['respondent_pace']:.2f}, "
                f"similar pages seen={timing_info['similar_pages_seen']}, "
                f"familiarity={timing_info['familiarity_multiplier']:.2f}"
            )

            print(
                f"DECISION PAUSE: {timing_info['decision_time_seconds']:.2f} seconds"
            )
            timing_info["observed_pre_answer_wait_seconds"] = wait_before_answer(
                pre_answer_delay, "selecting Ja"
            )

            try:
                element.click()
            except Exception:
                driver.execute_script(
                    "arguments[0].click();",
                    element
                )

            print(
                "Participation question detected: "
                "automatically selected YES."
            )
            return {
                "answer": "Ja",
                "actual_reading_time": actual_reading_time,
                "decision_time_seconds": timing_info["decision_time_seconds"],
                "pre_answer_delay_seconds": pre_answer_delay,
                "timing": timing_info
            }

    return None

def handle_birthdate_question(
    driver,
    visible_text,
    elements,
    persona
):
    """
    Handle month/year of birth locally from the fixed persona.
    No Ollama call is made.

    Returns a dictionary describing the selections if handled,
    otherwise None.
    """
    text = visible_text.lower()

    age_page = (
        "geburtsdatum" in text
        or (
            "geburtsmonat" in text
            and "geburtsjahr" in text
        )
        or (
            "wie alt sind sie" in text
            and "monat" in text
            and "jahr" in text
        )
    )

    if not age_page:
        return None

    dropdowns = [
        item
        for item in elements
        if item.get("type") == "dropdown"
    ]

    if len(dropdowns) < 2:
        return None

    month_dropdown = None
    year_dropdown = None

    for item in dropdowns:
        option_labels = [
            str(option.get("label", "")).strip()
            for option in item.get("options", [])
        ]

        numeric_options = []

        for label in option_labels:
            try:
                numeric_options.append(int(label))
            except Exception:
                pass

        if (
            len(numeric_options) >= 12
            and set(range(1, 13)).issubset(
                set(numeric_options)
            )
        ):
            month_dropdown = item
            continue

        four_digit_years = [
            value
            for value in numeric_options
            if 1900 <= value <= datetime.now().year
        ]

        if len(four_digit_years) >= 10:
            year_dropdown = item

    if month_dropdown is None or year_dropdown is None:
        return None

    birth_month = str(
        persona["birth_month"]
    )
    birth_year = str(
        persona["birth_year"]
    )

    month_element = find_element(
        driver,
        month_dropdown["element_key"]
    )
    year_element = find_element(
        driver,
        year_dropdown["element_key"]
    )

    if month_element is None or year_element is None:
        return None

    Select(
        month_element
    ).select_by_visible_text(
        birth_month
    )

    Select(
        year_element
    ).select_by_visible_text(
        birth_year
    )

    print(
        "Birthdate question detected: "
        f"selected month={birth_month}, year={birth_year} "
        f"for persona age={persona['age']}."
    )

    return {
        "birth_month": int(birth_month),
        "birth_year": int(birth_year),
        "month_element_key": month_dropdown["element_key"],
        "year_element_key": year_dropdown["element_key"]
    }

# ============================================================
# TIMING
# ============================================================

def safe_float(
    value,
    default=0.0
):
    try:
        return float(value)
    except Exception:
        return default

def clamp(
    value,
    minimum,
    maximum
):
    return max(
        minimum,
        min(
            maximum,
            value
        )
    )

class RespondentTiming:
    """
    One timing profile per survey run.

    Reading speed, general pace and page-level jitter differ across runs.
    Similar pages/batteries become gradually faster.
    """

    def __init__(self, persona):
        self.persona = persona

        self.pace = clamp(
            random.lognormvariate(
                0.0,
                RESPONDENT_PACE_SD
            ),
            0.70,
            1.55
        )

        # Draw a different reading speed for every respondent/run.
        self.reading_wpm = clamp(
            random.gauss(
                READING_WPM_MEAN,
                READING_WPM_SD
            ),
            READING_WPM_MIN,
            READING_WPM_MAX
        )

        # Mild age adjustment; the random respondent draw remains dominant.
        age = persona.get("age", 40)

        if age >= 55:
            self.reading_wpm *= 0.94
        elif age <= 30:
            self.reading_wpm *= 1.03

        self.reading_wpm = clamp(
            self.reading_wpm,
            READING_WPM_MIN,
            READING_WPM_MAX
        )

        self.seen_pages = []

    @staticmethod
    def normalize_text(text):
        text = str(text).lower()
        text = re.sub(
            r"\d+(?:[.,]\d+)?",
            "#",
            text
        )
        words = re.findall(
            r"[a-zäöüß#]+",
            text
        )

        return " ".join(
            words
        )

    def familiarity_multiplier(
        self,
        visible_text
    ):
        normalized = self.normalize_text(
            visible_text
        )

        similarities = [
            SequenceMatcher(
                None,
                normalized,
                previous
            ).ratio()
            for previous in self.seen_pages
            if normalized and previous
        ]

        repeat_count = sum(
            similarity >= SIMILAR_PAGE_THRESHOLD
            for similarity in similarities
        )

        familiarity = (
            1.0
            - MAX_FAMILIARITY_REDUCTION
            * (
                1.0
                - math.exp(
                    -FAMILIARITY_LEARNING_RATE
                    * repeat_count
                )
            )
        )

        self.seen_pages.append(
            normalized
        )

        return (
            familiarity,
            repeat_count,
            max(
                similarities,
                default=0.0
            )
        )

    def page_jitter(self):
        return random.lognormvariate(
            0.0,
            PAGE_JITTER_SD
        )

    def reading_time(
        self,
        visible_text
    ):
        """
        Local reading-time model.
        No Ollama call is needed for instruction pages.
        """
        words = len(
            re.findall(
                r"\b[\wÄÖÜäöüß'-]+\b",
                visible_text
            )
        )

        base_seconds = (
            words
            / self.reading_wpm
            * 60.0
        )

        overhead = random.uniform(
            READING_OVERHEAD_MIN,
            READING_OVERHEAD_MAX
        )

        familiarity, repeat_count, max_similarity = (
            self.familiarity_multiplier(
                visible_text
            )
        )

        jitter = self.page_jitter()

        actual = (
            (base_seconds + overhead)
            * self.pace
            * familiarity
            * jitter
        )

        actual = clamp(
            actual,
            MIN_READING_TIME,
            MAX_READING_TIME
        )

        info = {
            "word_count": words,
            "reading_wpm": self.reading_wpm,
            "respondent_pace": self.pace,
            "page_jitter": jitter,
            "similar_pages_seen": repeat_count,
            "max_previous_similarity": max_similarity,
            "familiarity_multiplier": familiarity
        }

        return (
            actual,
            info
        )

    def agreement_time(self, visible_text):
        reading_seconds, info = self.reading_time(visible_text)
        actual_reading_time = max(MIN_AGREEMENT_READING_TIME, reading_seconds)
        decision_time = random.uniform(
            AGREEMENT_DECISION_TIME_MIN,
            AGREEMENT_DECISION_TIME_MAX
        ) * self.pace
        total = actual_reading_time + decision_time
        info.update({
            "timing_type": "agreement",
            "reading_time_model_seconds": reading_seconds,
            "actual_reading_time": actual_reading_time,
            "decision_time_seconds": decision_time,
            "pre_answer_delay_seconds": total
        })
        return total, info

    def response_time(
        self,
        predicted_seconds,
        visible_text
    ):
        predicted_seconds = clamp(
            safe_float(
                predicted_seconds,
                3.0
            ),
            MIN_RESPONSE_TIME,
            MAX_RESPONSE_TIME
        )

        familiarity, repeat_count, max_similarity = (
            self.familiarity_multiplier(
                visible_text
            )
        )

        jitter = self.page_jitter()

        actual = (
            predicted_seconds
            * self.pace
            * familiarity
            * jitter
        )

        actual = clamp(
            actual,
            MIN_RESPONSE_TIME,
            MAX_RESPONSE_TIME
        )

        info = {
            "respondent_pace": self.pace,
            "page_jitter": jitter,
            "similar_pages_seen": repeat_count,
            "max_previous_similarity": max_similarity,
            "familiarity_multiplier": familiarity
        }

        return (
            actual,
            info
        )

# ============================================================
# EXECUTE ACTIONS
# ============================================================

def execute_actions(
    driver,
    actions
):
    results = []

    for action in actions:
        key = action.get(
            "element_key"
        )

        requested_action = action.get(
            "action"
        )

        value = action.get(
            "value",
            ""
        )

        element = find_element(
            driver,
            key
        )

        if element is None:
            results.append({
                "element": key,
                "success": False,
                "error": "Element not found"
            })
            continue

        try:
            driver.execute_script(
                """
                arguments[0].scrollIntoView({
                    block: 'center'
                });
                """,
                element
            )

            time.sleep(
                0.05
            )

            tag_name = (
                element.tag_name
                or ""
            ).lower()

            input_type = (
                element.get_attribute(
                    "type"
                )
                or ""
            ).lower()

            # ==================================================
            # RADIO BUTTON
            # ==================================================

            if (
                tag_name == "input"
                and input_type == "radio"
            ):
                if requested_action == "clear_selection":
                    driver.execute_script("""
                        const e = arguments[0];
                        e.checked = false;
                        e.dispatchEvent(new Event('input', {bubbles: true}));
                        e.dispatchEvent(new Event('change', {bubbles: true}));
                    """, element)
                    if element.is_selected():
                        raise RuntimeError("Attention-check selection could not be cleared.")
                    actual_action = "clear_selection"
                elif not element.is_selected():
                    try:
                        element.click()

                    except Exception:
                        driver.execute_script(
                            "arguments[0].click();",
                            element
                        )

                if requested_action != "clear_selection" and not element.is_selected():
                    raise RuntimeError(
                        "Radio button was not selected "
                        "after click."
                    )

                if requested_action != "clear_selection":
                    actual_action = "click"

            # ==================================================
            # CHECKBOX
            # ==================================================

            elif (
                tag_name == "input"
                and input_type == "checkbox"
            ):
                if not element.is_selected():
                    try:
                        element.click()

                    except Exception:
                        driver.execute_script(
                            "arguments[0].click();",
                            element
                        )

                if not element.is_selected():
                    raise RuntimeError(
                        "Checkbox was not selected "
                        "after click."
                    )

                actual_action = "click"

            # ==================================================
            # DROPDOWN
            # ==================================================

            elif tag_name == "select":
                dropdown = Select(
                    element
                )

                selected = False

                # Resolved actions carry the original DOM index and identity.
                # Recheck both before selecting so changed options cannot cause
                # a different answer to be clicked.
                if "option_index" in action:
                    index = action["option_index"]
                    options = dropdown.options
                    if not isinstance(index, int) or not 0 <= index < len(options):
                        raise ValueError(f"Dropdown option index is invalid: {index!r}")
                    option = options[index]
                    raw_value = option.get_attribute("value") or ""
                    if (normalize_option_text(option.text) != normalize_option_text(value)
                            or raw_value != str(action.get("option_value", ""))):
                        raise ValueError("Dropdown options changed after the page was read.")
                    if not option.is_enabled():
                        raise ValueError("The chosen dropdown option is disabled.")
                    dropdown.select_by_index(index)
                    if not option.is_selected():
                        raise RuntimeError("Dropdown option was not selected after selection.")
                    selected = True

                # First try visible label
                if not selected:
                    try:
                        dropdown.select_by_visible_text(
                            str(value)
                        )

                        selected = True

                    except Exception:
                        pass

                # Then try raw option value
                if not selected:
                    try:
                        dropdown.select_by_value(
                            str(value)
                        )

                        selected = True

                    except Exception:
                        pass

                # Then case-insensitive label matching
                if not selected:
                    for option in dropdown.options:
                        if (
                            option.text
                            .strip()
                            .lower()
                            == str(value)
                            .strip()
                            .lower()
                        ):
                            option.click()
                            selected = True
                            break

                if not selected:
                    raise ValueError(
                        f"Dropdown value not found: {value}"
                    )

                actual_action = "select"

            # ==================================================
            # TEXT INPUT / TEXTAREA
            # ==================================================

            elif (
                tag_name == "textarea"
                or (
                    tag_name == "input"
                    and input_type in [
                        "text",
                        "number",
                        "email",
                        "tel",
                        "url"
                    ]
                )
            ):
                if requested_action != "clear":
                    value = browser_text_value(value, single_line=tag_name == "input")
                    value = validate_text_value(value, key, action.get("text_rules", {}))
                    action["value"] = value
                element.clear()

                if requested_action == "clear":
                    if element.get_attribute("value"):
                        raise RuntimeError("Text field could not be left blank.")
                    actual_action = "clear"
                else:
                    element.send_keys(value)
                    entered = element.get_attribute("value") or ""
                    if entered != value:
                        raise RuntimeError("The text field did not retain the intended answer.")
                    # Commit change/blur handlers before looking up a fresh forward button.
                    driver.execute_script("arguments[0].blur();", element)
                    actual_action = "type"

            # ==================================================
            # GENERIC BUTTON / CLICKABLE ELEMENT
            # ==================================================

            elif requested_action == "click":
                try:
                    element.click()

                except Exception:
                    driver.execute_script(
                        "arguments[0].click();",
                        element
                    )

                actual_action = "click"

            else:
                raise ValueError(
                    "Unsupported element/action combination: "
                    f"tag={tag_name}, "
                    f"type={input_type}, "
                    f"requested_action={requested_action}"
                )

            results.append({
                "element": key,
                "requested_action": requested_action,
                "actual_action": actual_action,
                "value": value,
                "success": True
            })

        except Exception as error:
            results.append({
                "element": key,
                "requested_action": requested_action,
                "value": value,
                "success": False,
                "error": str(error)
            })

    return results

# ============================================================
# NEXT BUTTON
# ============================================================

def click_next(driver):
    """Wait for a usable forward control and read its label before clicking."""
    try:
        driver.execute_script(
            "if (document.activeElement) document.activeElement.blur();"
        )
    except Exception:
        pass
    deadline = time.monotonic() + FORWARD_BUTTON_TIMEOUT
    last_error = ""
    forward_words = ("weiter", "next", "continue", "fortfahren", "nächste", "naechste")
    backward_words = ("zurück", "zurueck", "back", "previous", "prev")
    while True:
        candidates = []
        try:
            candidates.append(driver.find_element(By.ID, "btn_send_ahead"))
        except Exception:
            pass
        try:
            for button in driver.find_elements(
                By.CSS_SELECTOR,
                "button, input[type='submit'], input[type='button'], "
                "input[type='image'], a[role='button'], a[id='btn_send_ahead']"
            ):
                if button not in candidates:
                    candidates.append(button)
        except Exception as error:
            last_error = str(error)
        for button in candidates:
            try:
                if not button.is_displayed() or not button.is_enabled():
                    continue
                label = (
                    button.get_attribute("value") or button.text
                    or button.get_attribute("aria-label")
                    or button.get_attribute("title") or ""
                ).strip()
                normalized = label.lower()
                if any(word in normalized for word in backward_words):
                    continue
                preferred = button.get_attribute("id") == "btn_send_ahead"
                if not preferred and not any(word in normalized for word in forward_words):
                    continue
                driver.execute_script(
                    "arguments[0].scrollIntoView({block:'center'});", button
                )
                button.click()
            except Exception as error:
                last_error = str(error)
                continue
            # A successful click may already have replaced the button. Never read it now.
            print("Clicked forward button:", label or "(no label)")
            time.sleep(NAVIGATION_WAIT)
            return True
        if time.monotonic() >= deadline:
            break
        time.sleep(FORWARD_BUTTON_POLL_INTERVAL)
    print("Forward navigation unavailable: no usable forward button.")
    if last_error:
        print("Last forward-button error:", last_error)
    return False


def note_final_page_advance(run_record, page_number, visible_text):
    """Remember a successful forward click from a page with an explicit end heading."""
    lines = [line.strip() for line in visible_text.splitlines() if line.strip()]
    heading = next((line for line in lines if re.fullmatch(
        r"ende der (?:befragung|umfrage)|end of (?:the )?(?:survey|questionnaire)",
        normalize_option_text(line).rstrip(".!:")
    )), None)
    if heading:
        run_record["final_page_advance"] = {
            "page": page_number, "question_text": visible_text,
            "end_heading": heading,
            "survey_title": lines[0] if lines[0] != heading else "",
            "forward_clicked": True
        }
    else:
        run_record.pop("final_page_advance", None)


def classify_navigation_stop(driver, run_record):
    """Recognize an explicit terminal page or the closing screen after final submission."""
    visible_text, elements = extract_page(driver)
    text = normalize_option_text(visible_text)
    explicit_terminal = not elements and bool(re.search(
        r"vielen dank für (?:ihre|deine|die) teilnahme|"
        r"(?:befragung|umfrage)(?:(?!\b(?:nicht|noch)\b).){0,30}\b(?:beendet|abgeschlossen)\b|"
        r"ende der (?:befragung|umfrage)|"
        r"(?:sie können|du kannst).{0,30}(?:fenster|browser).{0,30}schließen|"
        r"thank you for (?:your )?participation|"
        r"(?:survey|questionnaire)(?:(?!\bnot\b).){0,30}\b(?:completed|complete|finished)\b",
        text
    ))
    final_advance = run_record.get("final_page_advance", {})
    title = normalize_option_text(final_advance.get("survey_title", ""))
    closing_screen = bool(
        not elements and final_advance.get("forward_clicked")
        and (not text or (title and text == title))
    )
    terminal = explicit_terminal or closing_screen
    diagnostics = {
        "question_text": visible_text, "elements": elements,
        "completion_confirmed": terminal,
        "completion_basis": "explicit_terminal_text" if explicit_terminal else
                            "closing_screen_after_final_page" if closing_screen else "unconfirmed"
    }
    if closing_screen:
        diagnostics["final_page_advance"] = final_advance
    try:
        diagnostics["validation_messages"] = driver.execute_script("""
            return Array.from(document.querySelectorAll('input, select, textarea'))
                .filter(e => e.willValidate && !e.validity.valid)
                .map(e => ({id: e.id, message: e.validationMessage}));
        """)
    except Exception:
        pass
    run_record.setdefault("last_page", {})["navigation_stop"] = diagnostics
    if terminal:
        print("Survey completion confirmed on the final page.")
        return "finished", ""
    message = (
        "Could not advance the survey: no usable forward button. "
        "Survey completion was not confirmed."
    )
    print(message)
    return "error", message

# ============================================================
# PRINT PAGE INFORMATION
# ============================================================

def print_question_page_info(
    page_number,
    visible_text,
    answer,
    execution,
    predicted_response_time,
    actual_response_time,
    timing_info
):
    print()
    print("=" * 70)
    print(f"PAGE {page_number}")
    print("=" * 70)

    print()
    print("SURVEY PAGE:")
    print(
        visible_text[:2500]
    )

    print()
    print("BOT ANSWER:")
    print(
        answer.get(
            "summary",
            ""
        )
    )

    print()
    print("ACTIONS:")

    for action in answer.get(
        "actions",
        []
    ):
        print(
            f"  {action.get('action')} | "
            f"{action.get('element_key')} | "
            f"{action.get('value')}"
        )

    if answer.get("behavior_draws"):
        print()
        print("INDEPENDENT BEHAVIOR DRAWS:")
        for draw in answer["behavior_draws"]:
            print(f"  {draw['kind']} | {draw['target']} | "
                  f"p={draw['probability']:.2f} | {draw['outcome']}")

    for correction in answer.get("answer_corrections", []):
        print("Used listed employment category:", correction["listed_choice"])

    if answer.get("sampling"):
        print()
        print("PYTHON ANSWER DRAWS:")
        for sample in answer["sampling"]:
            print(f"  {sample['group_key']}: {sample['chosen_label']}")
            print("  " + "; ".join(
                f"{option['label']}={probability:.3f}"
                for option, probability in zip(sample["options"], sample["probabilities"])
            ))

    print()
    print("EXECUTION:")

    for item in execution:
        print(
            f"  {item}"
        )

    print()
    print(
        f"PREDICTED RESPONSE TIME: "
        f"{predicted_response_time:.2f} seconds"
    )
    print(
        f"DRAWN RESPONSE TIME: "
        f"{actual_response_time:.2f} seconds"
    )
    print(
        "TIMING FACTORS: "
        f"pace={timing_info['respondent_pace']:.2f}, "
        f"similar pages seen={timing_info['similar_pages_seen']}, "
        f"familiarity={timing_info['familiarity_multiplier']:.2f}"
    )
    print()

# ============================================================
# RUN ONE SURVEY
# ============================================================

def _run_one_survey(run_number, run_record):
    persona = draw_persona()
    timing = RespondentTiming(persona)

    print_persona(
        run_number,
        persona
    )

    print(
        f"RESPONDENT TIMING PACE: {timing.pace:.2f} "
        "(below 1 = faster, above 1 = slower)"
    )
    print(
        f"RESPONDENT READING SPEED: "
        f"{timing.reading_wpm:.0f} words/minute"
    )

    start_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    pages_completed = 0
    status = "started"
    error_message = ""
    history = []
    seen_pages = set()

    options = webdriver.ChromeOptions()

    options.add_argument(
        "--start-maximized"
    )

    driver = None

    try:
        driver = webdriver.Chrome(
            options=options
        )

        driver.get(
            SURVEY_URL
        )

        time.sleep(
            NAVIGATION_WAIT
        )

        for page_number in range(
            1,
            MAX_PAGES + 1
        ):
            visible_text, elements = extract_page(
                driver
            )
            run_record["last_page"] = {
                "page": page_number, "question_text": visible_text, "elements": elements
            }
            if elements:
                # A remaining question cannot inherit completion from an earlier page.
                run_record.pop("final_page_advance", None)

            # ------------------------------------------------
            # DETERMINISTIC PARTICIPATION PAGE
            # ------------------------------------------------
            participation_result = handle_participation_question(
                driver,
                visible_text,
                elements,
                timing
            )

            if participation_result is not None:
                post_answer_delay = random.uniform(0.4, 1.0) * timing.pace

                history.append({
                    "page": page_number,
                    "type": "forced_participation",
                    "question_text": visible_text,
                    **participation_result,
                    "post_answer_delay_seconds": post_answer_delay,
                    "actual_response_time": (
                        participation_result["pre_answer_delay_seconds"]
                        + post_answer_delay
                    )
                })

                pages_completed += 1

                # Short pause between selecting Ja and clicking Weiter.
                time.sleep(post_answer_delay)

                if not click_next(
                    driver
                ):
                    status = "error"
                    error_message = (
                        "Participation question answered, "
                        "but no forward button was found."
                    )
                    break

                continue

            # ------------------------------------------------
            # DETERMINISTIC BIRTH MONTH / YEAR
            # ------------------------------------------------
            birthdate_result = handle_birthdate_question(
                driver=driver,
                visible_text=visible_text,
                elements=elements,
                persona=persona
            )

            if birthdate_result is not None:
                # The answer is known from the persona, so there is no
                # reason to ask Ollama. Only add a short human-like delay.
                actual_response_time = clamp(
                    random.lognormvariate(
                        math.log(1.6),
                        0.18
                    )
                    * timing.pace,
                    0.7,
                    3.5
                )

                print(
                    f"LOCAL AGE RESPONSE TIME: "
                    f"{actual_response_time:.2f} seconds"
                )

                history.append({
                    "page": page_number,
                    "type": "fixed_birthdate",
                    "question_text": visible_text,
                    "age": persona["age"],
                    "birth_month": persona["birth_month"],
                    "birth_year": persona["birth_year"],
                    "actual_response_time": actual_response_time
                })

                pages_completed += 1

                time.sleep(
                    actual_response_time
                )

                if not click_next(
                    driver
                ):
                    status = "error"
                    error_message = (
                        "Birthdate question answered, "
                        "but no forward button was found."
                    )
                    break

                continue

            # ------------------------------------------------
            # INFORMATION / INSTRUCTION PAGE
            # Zero Ollama calls: timing is calculated locally.
            # ------------------------------------------------
            if not elements:
                print()
                print("=" * 70)
                print(
                    f"PAGE {page_number}: INFORMATION / INSTRUCTION"
                )
                print("=" * 70)
                print()
                print(
                    visible_text[:2500]
                )
                print()

                actual_reading_time, timing_info = timing.reading_time(
                    visible_text
                )

                print(
                    f"LOCAL READING TIME: "
                    f"{actual_reading_time:.2f} seconds"
                )
                print(
                    "READING FACTORS: "
                    f"{timing_info['word_count']} words, "
                    f"{timing_info['reading_wpm']:.0f} wpm, "
                    f"pace={timing_info['respondent_pace']:.2f}, "
                    f"similar pages seen={timing_info['similar_pages_seen']}, "
                    f"familiarity={timing_info['familiarity_multiplier']:.2f}"
                )

                time.sleep(
                    actual_reading_time
                )

                if click_next(
                    driver
                ):
                    pages_completed += 1

                    history.append({
                        "page": page_number,
                        "type": "instruction",
                        "question_text": visible_text,
                        "actual_reading_time": actual_reading_time,
                        "timing": timing_info
                    })
                    note_final_page_advance(run_record, page_number, visible_text)

                    print(
                        "Instruction page completed. Continuing."
                    )

                    continue

                status, error_message = classify_navigation_stop(driver, run_record)
                break

            # ------------------------------------------------
            # LOOP PROTECTION
            # ------------------------------------------------
            page_signature = (
                visible_text.strip(),
                tuple(
                    (
                        element.get("type"),
                        element.get("group"),
                        element.get("label"),
                        str(
                            element.get(
                                "options",
                                ""
                            )
                        )
                    )
                    for element in elements
                )
            )

            if page_signature in seen_pages:
                print(
                    "This page has already been visited. "
                    "Stopping to avoid a navigation loop."
                )

                status = "navigation_loop"
                break

            seen_pages.add(
                page_signature
            )

            # ------------------------------------------------
            # ONE OLLAMA CALL; RETRY ONLY IF ITS ANSWER IS INVALID
            # ------------------------------------------------
            try:
                answer = ask_ai(
                    persona=persona,
                    history=history,
                    visible_text=visible_text,
                    elements=elements,
                    response_context=run_record["last_page"]
                )
            except Exception:
                print(f"\nUNANSWERED SURVEY PAGE {page_number}:\n{visible_text[:4000]}")
                raise

            actions = answer.get(
                "actions",
                []
            )

            predicted_response_time = clamp(
                safe_float(
                    answer.get(
                        "response_time_seconds",
                        3.0
                    ),
                    3.0
                ),
                MIN_RESPONSE_TIME,
                MAX_RESPONSE_TIME
            )

            if is_data_consent_question(visible_text, elements):
                actual_response_time, timing_info = timing.agreement_time(
                    visible_text
                )
                print(
                    f"LOCAL CONSENT READING TIME: "
                    f"{timing_info['actual_reading_time']:.2f} seconds; "
                    f"decision pause: {timing_info['decision_time_seconds']:.2f} seconds"
                )
            else:
                actual_response_time, timing_info = timing.response_time(
                    predicted_seconds=predicted_response_time,
                    visible_text=visible_text
                )

            timing_info["observed_pre_answer_wait_seconds"] = wait_before_answer(
                actual_response_time, "selecting answers"
            )

            execution = execute_actions(
                driver,
                actions
            )

            print_question_page_info(
                page_number=page_number,
                visible_text=visible_text,
                answer=answer,
                execution=execution,
                predicted_response_time=predicted_response_time,
                actual_response_time=actual_response_time,
                timing_info=timing_info
            )

            history.append({
                "page": page_number,
                "type": "question",
                "question_text": visible_text,
                "summary": answer.get(
                    "summary",
                    ""
                ),
                "actions": actions,
                "execution": execution,
                "sampling": answer.get("sampling", []),
                "behavior_draws": answer.get("behavior_draws", []),
                "answer_corrections": answer.get("answer_corrections", []),
                "intentional_omissions": answer.get("intentional_omissions", []),
                "assessment": answer.get("assessment", ""),
                "predicted_response_time": predicted_response_time,
                "actual_response_time": actual_response_time,
                "timing": timing_info
            })

            if any(not result.get("success") for result in execution):
                status = "error"
                error_message = "An answer action failed; stopped before advancing."
                break

            pages_completed += 1

            if not click_next(
                driver
            ):
                status, error_message = classify_navigation_stop(driver, run_record)
                break
            note_final_page_advance(run_record, page_number, visible_text)

        if status == "started":
            status = "max_pages_reached"
            error_message = (
                f"Stopped at MAX_PAGES={MAX_PAGES} before survey completion "
                "was detected. Increase MAX_PAGES for a longer survey."
            )

        print()
        if status == "finished":
            print(f"Run {run_number} finished.")
        else:
            print(f"Run {run_number} stopped ({status}).")
            if error_message:
                print(error_message)

    except KeyboardInterrupt:
        status = "interrupted"
        error_message = "Run interrupted by the user."
        raise

    except Exception as error:
        status = "error"
        error_message = str(
            error
        )

        print()
        print(
            f"ERROR IN RUN {run_number}:"
        )
        print(
            error_message
        )

    finally:
        end_time = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        run_record.update({
            "run_number": run_number, "persona": persona, "status": status,
            "start_time": start_time, "end_time": end_time,
            "pages_completed": pages_completed, "error_message": error_message,
            "timing_pace": timing.pace, "reading_wpm": timing.reading_wpm,
            "history": history
        })

        try:
            save_run_to_csv(
                run_number=run_number, start_time=start_time, end_time=end_time,
                persona=persona, pages_completed=pages_completed,
                status=status, error_message=error_message
            )
        finally:
            try:
                save_answer_log(run_number, start_time, end_time, persona, history,
                                status, error_message)
            finally:
                if driver is not None:
                    driver.quit()

    return run_record

# ============================================================
# RUN BOT
# ============================================================

def run_bot(n_runs=1):
    print(f"Survey bot version {BOT_VERSION} (labeled choices)")
    print(f"Output folder: {SCRIPT_DIR}")
    if not check_ollama():
        raise RuntimeError(
            f"Ollama is not running or model '{MODEL}' is not installed."
        )

    results = []

    for run_number in range(
        1,
        n_runs + 1
    ):
        print()
        print("#" * 70)
        print(
            f"STARTING RUN {run_number} OF {n_runs}"
        )
        print("#" * 70)

        result = run_one_survey(
            run_number
        )

        results.append(
            result
        )

    print()
    print("=" * 70)
    finished_runs = sum(result["status"] == "finished" for result in results)
    print(
        f"{finished_runs} OF {n_runs} SURVEY RUN(S) FINISHED"
    )
    if finished_runs < n_runs:
        print(f"{n_runs - finished_runs} run(s) stopped before survey completion.")
    print(
        f"Documentation saved to: {CSV_FILE}"
    )
    print(f"Answer weights and realized answers saved to: {ANSWER_LOG_FILE}")
    print(f"Separate Markdown reports and JSON data saved beside the script: {SCRIPT_DIR}")
    print("=" * 70)

    return results

# ============================================================
# OPTIONAL DEBUGGING
# ============================================================

def inspect_page(driver):
    print("\n" + "=" * 80)
    print("PAGE TITLE:")
    print(driver.title)

    print("\nCURRENT URL:")
    print(driver.current_url)

    print("\nVISIBLE TEXT:")
    print(
        driver.find_element(
            By.TAG_NAME,
            "body"
        ).text[:5000]
    )

    print("\nALL INPUTS:")
    inputs = driver.find_elements(
        By.TAG_NAME,
        "input"
    )

    for i, el in enumerate(inputs):
        try:
            print({
                "i": i,
                "type": el.get_attribute("type"),
                "id": el.get_attribute("id"),
                "name": el.get_attribute("name"),
                "value": el.get_attribute("value"),
                "class": el.get_attribute("class"),
                "displayed": el.is_displayed(),
                "enabled": el.is_enabled(),
                "aria-label": el.get_attribute("aria-label")
            })
        except Exception as error:
            print(i, error)

    print("\nALL BUTTONS:")
    buttons = driver.find_elements(
        By.TAG_NAME,
        "button"
    )

    for i, el in enumerate(buttons):
        try:
            print({
                "i": i,
                "text": el.text,
                "id": el.get_attribute("id"),
                "name": el.get_attribute("name"),
                "class": el.get_attribute("class"),
                "displayed": el.is_displayed(),
                "enabled": el.is_enabled()
            })
        except Exception as error:
            print(i, error)

    print("\nALL SELECTS:")
    selects = driver.find_elements(
        By.TAG_NAME,
        "select"
    )

    for i, el in enumerate(selects):
        print({
            "i": i,
            "id": el.get_attribute("id"),
            "name": el.get_attribute("name"),
            "class": el.get_attribute("class"),
            "displayed": el.is_displayed()
        })

    print("\nALL IFRAMES:")
    frames = driver.find_elements(
        By.TAG_NAME,
        "iframe"
    )

    for i, el in enumerate(frames):
        print({
            "i": i,
            "src": el.get_attribute("src"),
            "id": el.get_attribute("id"),
            "name": el.get_attribute("name"),
            "displayed": el.is_displayed()
        })

    print("=" * 80)

# ============================================================
# START
# ============================================================

# One survey respondent:
# run_bot()

# Ten survey respondents:
# run_bot(10)
