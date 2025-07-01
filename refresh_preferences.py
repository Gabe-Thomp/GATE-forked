'''
This script take in a set of profiles for the website preferences task and queries the OpenAI API to get responses for each article in the test cases. Each profile should be equipped with a persona.
'''

import argparse
import glob
import json
from utils import query_api, load_openai_cache

DEFAULT_PROMPT = "Are you interested in the following article?\n{article}"


def run_profile(profile_path, engine, cache, cache_file):
    with open(profile_path, "r") as f:
        # Load the profile JSON file
        data = json.load(f)
    
    persona = data["persona"]
    yes_no_append = data["yes_no_append"]
    persona = persona + " " + yes_no_append if yes_no_append else persona

    final_prompt = data.get("final_prompt", DEFAULT_PROMPT)

    print(f"Running profile: {profile_path}")
    print(f"EVALUATING Persona: {persona}")
    
    for i, (article, _) in enumerate(data.get("test_cases", [])):
        prompt = final_prompt.format(article=article)
        messages = [
            {"role": "system", "content": persona},
            {"role": "user", "content": prompt},
        ]
        response, _ = query_api(messages, engine, cache, cache_file)
        
        response_map = {"yes": True, "no": False}
        response = response_map[response.strip().lower()]
        # Changing the boolean evaluation from the LM 
        data["test_cases"][i] = [data["test_cases"][i][0], response]
        print(f"\nPrompt: {prompt}\nResponse: {response}")
    
    with open(profile_path, "w") as f:
        json.dump(data, f, indent=4)
    


def main():
    parser = argparse.ArgumentParser(description="Refresh website preference test cases via OpenAI API")
    parser.add_argument("--engine", default="gpt-4", help="OpenAI model")
    parser.add_argument("--cache_file", default="openai_cache.jsonl", help="Path to OpenAI cache file")
    parser.add_argument("--profiles_glob", default="gpt_prompts/website_preferences/profile*.json", help="Glob pattern for profile files")
    parser.add_argument("--output_file", default="website_preferences_answers.json", help="Where to write responses")
    args = parser.parse_args()

    cache = load_openai_cache(args.cache_file)

    all_results = {}
    for profile_path in sorted(glob.glob(args.profiles_glob)):
        run_profile(profile_path, args.engine, cache, args.cache_file)




if __name__ == "__main__":
    main()
