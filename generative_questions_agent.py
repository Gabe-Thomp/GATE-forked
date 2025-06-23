import textwrap
import re

from base_active_learning_agent import BaseActiveLearningAgent
from utils import query_api

QUESTION_TYPES = ["yn", "open"]
IMPLEMENTATION = "Python regex"  #["Python regex", "system"]


class GenerativeQuestionsAgent(BaseActiveLearningAgent):
    """Active learning agent that generates questions to identify the target regex."""

    def __init__(self, target_specification_file, engine, openai_cache_file=None, question_type="open", num_candidate_questions=1, **kwargs):
        super().__init__(target_specification_file, engine, openai_cache_file, **kwargs)
        self.question_type = question_type
        self.num_candidate_questions = num_candidate_questions
        assert self.question_type in QUESTION_TYPES, f"Invalid question type: {self.question_type}. Must be one of {QUESTION_TYPES}."

    def get_hypothesis_prompt(self, task_description, interaction_history, broken_regexes=None):
        hypothesis_prompt = textwrap.dedent('''\
            Your task is to collaboratively help someone design a regex that will {task_description}.

            Help them come up with a hypothesis for the regex that they should try, consistent with the previous questions and answers.

            Previous questions and answers:
            {interaction_history}
            
            Previous invalid attempts (these regexes failed to compile):
            {broken_regexes}

            Generate the hypothesis regex without quotes and nothing else:'''
        ).format(
            task_description=task_description,
            interaction_history=self.format_questions_and_answers(interaction_history),
            broken_regexes='\n'.join(broken_regexes),
        )
        print(hypothesis_prompt)
        return [{"role": "user", "content": hypothesis_prompt}]

    def get_question_prompt(self, task_description, question_type, implementation, interaction_history, num_questions=1):
        if question_type == "yn":
            question_type_insert = "yes/no question"
        elif question_type == "open":
            question_type_insert = "open-ended question"
        else:
            raise ValueError(f"Invalid question type: {question_type}")

        if num_questions > 1:
            final_instruction = f"Generate {num_questions} {question_type_insert}s, each on a separate line and numbered. Output only the questions."
        else:
            final_instruction = f"Generate the {question_type_insert} and nothing else:"

        question_prompt = textwrap.dedent('''\
            Your task is to {task_description}.

            Previous questions:
            {interaction_history}

            Generate the most informative {question_type_insert} that, when answered, will reveal the most about the desired behavior beyond what has already been queried for above. Make sure your question addresses different aspects of the {implementation} than the questions that have already been asked. At the same time however, the question should be bite-sized, and not ask for too much at once. {additional_prompt}{final_instruction}'''
            ).format(
                implementation=implementation,
                task_description=task_description,
                additional_prompt=getattr(self, "additional_query_note", ""),
                question_type_insert=question_type_insert,
                interaction_history=self.format_questions_and_answers(interaction_history),
                final_instruction=final_instruction
            )
        print(question_prompt)
        print("===")
        return [{"role": "user", "content": question_prompt}]
    
    def get_query_prompt(self):
        return self.get_question_prompt(self.task_description, self.question_type, self.implementation, [["[Q]", "[A]"]], num_questions=self.num_candidate_questions)

    def generate_active_query(self):
        '''Generates a question for the oracle.'''
        question_prompt = self.get_question_prompt(
            self.task_description,
            self.question_type,
            self.implementation,
            self.interaction_history,
            num_questions=self.num_candidate_questions,
        )
        question_text, _ = query_api(
            question_prompt,
            self.engine,
            self.openai_cache,
            self.openai_cache_file,
            temperature=self.temperature,
        )
        if self.num_candidate_questions == 1:
            return question_text

        candidate_questions = self.parse_candidate_questions(question_text)
        if not candidate_questions:
            return question_text.strip()
        return self.select_best_question(candidate_questions[: self.num_candidate_questions])

    @staticmethod
    def parse_candidate_questions(raw_text):
        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
        questions = []
        for line in lines:
            line = re.sub(r"^[-\d\).]*\s*", "", line)
            line = line.strip('"').strip("'")
            if line:
                questions.append(line)
        return questions

    def select_best_question(self, candidate_questions):
        questions_str = "\n".join(f"{i+1}. {q}" for i, q in enumerate(candidate_questions))
        prompt = textwrap.dedent(
            '''\
            Your task is to {task_description}.

            Previous questions and answers:
            {interaction_history}

            Here are candidate questions:
            {questions}

            Choose the single question that will reveal the most new information. Output only that question.'''
        ).format(
            task_description=self.task_description,
            interaction_history=self.format_questions_and_answers(self.interaction_history),
            questions=questions_str,
        )
        messages = [{"role": "user", "content": prompt}]
        best_question, _ = query_api(
            messages,
            self.engine,
            self.openai_cache,
            self.openai_cache_file,
            temperature=self.temperature,
        )
        return best_question.strip().strip('"').strip("'")
       
    def generate_oracle_response(self, question):
        '''Generates an oracle response for the question'''
        answer = self.query_oracle_api(question, self.question_type)
        self.interaction_history.append((question, answer))
        return answer
    
    def query_type(self):
        return f"question_{self.question_type}"
