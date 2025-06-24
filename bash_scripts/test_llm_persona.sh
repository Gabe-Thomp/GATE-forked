#!/bin/bash

# === Set parameters here ===
ENGINE="gpt-4"
AGENT="questions"         # or edge_cases or pool
EVAL_CONDITION="at_end" # or per_minute or at_end or per_turn
POOL_CLUSTERS=-1 
TASK="website_preferences" # moral_reasoning or email_regex or website_preferences
NUM_INTERACTIONS=5
QUESTION_MODES="questions_open"
NUM_CANDIDATE_QUESTIONS=10
WRITE_FILENAME="10_candidate"

# === Call Python script ===
python run_model_evaluation.py \
    --engine "$ENGINE" \
    --agent "$AGENT" \
    --eval_condition "$EVAL_CONDITION" \
    --pool_diversity_num_clusters "$POOL_CLUSTERS" \
    --task "$TASK" \
    --question_modes "$QUESTION_MODES" \
    --num_interactions $NUM_INTERACTIONS \
    --num_candidate_questions $NUM_CANDIDATE_QUESTIONS \
    --write_filename "$WRITE_FILENAME" \