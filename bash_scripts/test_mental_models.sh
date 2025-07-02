#!/bin/bash

# === Set parameters here ===
ENGINE="gpt-4"

# Use questionsmm for mental model
AGENT=null         # [edge_cases, pool, questionsmm, questions]
EVAL_CONDITION="at_end" # or per_minute or at_end or per_turn
POOL_CLUSTERS=-1 
# Use to specify the folder inside of gpt_prompts to use as profiles for the LM agent
TASK="website_preferences/test" # moral_reasoning or email_regex or website_preferences
NUM_INTERACTIONS=5
QUESTION_MODES="questionsmm_open"
NUM_CANDIDATE_QUESTIONS=1
WRITE_FILENAME="mental_models_profile8_07_02_2025"

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