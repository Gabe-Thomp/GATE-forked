import argparse
import glob
import json
from utils import query_api, load_openai_cache

DEFAULT_PROMPT = "Are you interested in the following article?\n{article}"


def run_profile(profile_path, engine, cache, cache_file):
    data = json.load(open(profile_path))
    persona = data["persona"]
    final_prompt = data.get("final_prompt", DEFAULT_PROMPT)

    results = []
    for article, _ in data.get("test_cases", []):
        prompt = final_prompt.format(article=article)
        messages = [
            {"role": "system", "content": persona},
            {"role": "user", "content": prompt},
        ]
        response, _ = query_api(messages, engine, cache, cache_file)
        results.append({"article": article, "response": response.strip()})
    return results


def main():
    parser = argparse.ArgumentParser(description="Refresh website preference test cases via OpenAI API")
    parser.add_argument("--engine", default="gpt-3.5-turbo", help="OpenAI model")
    parser.add_argument("--cache_file", default="openai_cache.jsonl", help="Path to OpenAI cache file")
    parser.add_argument("--profiles_glob", default="gpt_prompts/website_preferences/profile*.json", help="Glob pattern for profile files")
    parser.add_argument("--output_file", default="website_preferences_answers.json", help="Where to write responses")
    args = parser.parse_args()

    cache = load_openai_cache(args.cache_file)

    all_results = {}
    for profile_path in sorted(glob.glob(args.profiles_glob)):
        all_results[profile_path] = run_profile(profile_path, args.engine, cache, args.cache_file)

    with open(args.output_file, "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
