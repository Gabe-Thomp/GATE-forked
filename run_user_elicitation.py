import glob
import json
import os
import random
from copy import deepcopy

from tap import Tap
from tqdm import tqdm

from interactive_user_agent import InteractiveUserAgent
from utils import update_metrics, update_test_responses


def run_user_problem_instance(
    problem_instance_filename,
    engine,
    openai_cache_file,
    num_interactions,
    temperature=0.0,
    outputs_save_file=None,
):
    """Runs an interactive elicitation session where a human queries an LLM."""
    agent = InteractiveUserAgent(
        problem_instance_filename,
        engine,
        openai_cache_file=openai_cache_file,
        temperature=temperature,
    )

    if outputs_save_file:
        outputs_save_file.write(f"0. {agent.persona}\n\n")

    test_xs = agent.get_interaction_features()
    test_score, test_responses = agent.score_test_cases()
    print(test_score)
    all_test_xs = update_metrics({}, test_xs)
    test_scores = update_metrics({}, test_score)
    start_test_scores = deepcopy(test_scores)
    all_test_responses = update_test_responses([], test_responses)

    for i in tqdm(range(num_interactions)):
        query = input("Your question (or 'quit' to exit): ")
        if query.strip().lower() in {"quit", "exit", "stop"}:
            break
        answer = agent.generate_oracle_response(query)
        print("Model:", answer)
        if outputs_save_file:
            outputs_save_file.write(f"{i}. {query}\n{answer}\n\n")

        test_xs = agent.get_interaction_features()
        test_score, test_responses = agent.score_test_cases(start_metrics=start_test_scores)
        print(test_score)
        all_test_xs = update_metrics(all_test_xs, test_xs)
        test_scores = update_metrics(test_scores, test_score)
        all_test_responses = update_test_responses(all_test_responses, test_responses)

    if outputs_save_file:
        outputs_save_file.write(
            f"===TEST RESPONSES===\n{json.dumps(all_test_responses, indent=2)}\n\n"
        )

    return all_test_xs, test_scores


def main(args):
    if args.no_cache:
        openai_cache_file = None
    else:
        openai_cache_file = f"{args.engine}-cache-seed-{args.seed}.jsonl"

    problem_instance_filename = random.choice(glob.glob(f"gpt_prompts/{args.task}/*.json"))
    os.makedirs(f"user_model_results/{args.task}", exist_ok=True)
    outputs_save_file = open(
        f"user_model_results/{args.task}/{args.engine}_{args.seed}.txt",
        "w",
    )

    run_user_problem_instance(
        problem_instance_filename=problem_instance_filename,
        engine=args.engine,
        openai_cache_file=openai_cache_file,
        num_interactions=args.num_interactions,
        temperature=args.temperature,
        outputs_save_file=outputs_save_file,
    )


class ArgumentParser(Tap):
    num_interactions: int = 5
    engine: str = "gpt-4"
    task: str = "email_regex"
    no_cache: bool = False
    seed: int = 0
    temperature: float = 0.0


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    main(args)
