from base_active_learning_agent import BaseActiveLearningAgent

class InteractiveUserAgent(BaseActiveLearningAgent):
    """Active learning agent that takes queries from a human user."""

    def __init__(self, target_specification_file, engine, openai_cache_file=None, **kwargs):
        super().__init__(target_specification_file, engine, openai_cache_file, **kwargs)

    def get_hypothesis_prompt(self, interaction_history, broken_regexes=None):
        # Not used for interactive mode
        pass

    def generate_active_query(self):
        # Queries are provided by the user via input
        return None

    def generate_oracle_response(self, query):
        answer = self.query_oracle_api(query, None)
        self.interaction_history.append((query, answer))
        return answer
