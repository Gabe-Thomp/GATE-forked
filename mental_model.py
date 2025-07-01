import textwrap
import re
from utils import query_api
from generative_questions_agent import GenerativeQuestionsAgent

class MentalModel(BaseActiveLearningAgent):
    def __init__(self, target_specification_file, engine, openai_cache_file=None, question_type="open", num_candidate_questions=1, **kwargs):
        super().init(target_specification_file, engine, openai_cache_file=None, question_type="open", num_candidate_questions=1, **kwargs)
    
    def update_mental_model(self, task_description, interaction_history):
        '''
        This is a method to update the running mental model of the user based on the past question(s) asked
        and the answers received. Whether that is done given ALL history or just the last question and answer
        is TBD.
        '''
        question_prompt = textwrap.dedent('''\
                Your task is to {task_description}.

                Previous questions:
                {interaction_history}

                Current mental model of {implementation}:
                {mental_model}

                Generate the most informative {question_type_insert} that, when answered, will reveal the most about the desired behavior beyond what has already been queried for above. Make sure your question addresses different aspects of the {implementation} than the questions that have already been asked. At the same time however, the question should be bite-sized, and not ask for too much at once. {additional_prompt}Generate the {question_type_insert} and nothing else:'''
            ).format(
                implementation=implementation,
                task_description=task_description,
                additional_prompt=getattr(self, "additional_query_note", ""),
                question_type_insert=question_type_insert,
                interaction_history=self.format_questions_and_answers(interaction_history),
                mental_model=self.mental_model,
            )
        # TODO: Change the prompt above to include the mental model and update it based on the response.
        pass

    def get_question_prompt(self, task_description, question_type, implementation, 
    interaction_history, num_candidate_questions=1):
        '''
        Want to add the functionality of the mental model here. 
        TODO: Need to format the prompt specifically for the mental model!! Possibly can use JSON...
        '''
        
        if question_type == "yn":
            question_type_insert = "yes/no question"
        elif question_type == "open":
            question_type_insert = "open-ended question"
        else:
            raise ValueError(f"Invalid question type: {question_type}")

        if num_candidate_questions == 1:
            question_prompt = textwrap.dedent('''\
                Your task is to {task_description}.

                Previous questions:
                {interaction_history}

                Current mental model of {implementation}:
                {mental_model}

                Generate the most informative {question_type_insert} that, when answered, will reveal the most about the desired behavior beyond what has already been queried for above. Make sure your question addresses different aspects of the {implementation} than the questions that have already been asked. At the same time however, the question should be bite-sized, and not ask for too much at once. {additional_prompt}Generate the {question_type_insert} and nothing else:'''
            ).format(
                implementation=implementation,
                task_description=task_description,
                additional_prompt=getattr(self, "additional_query_note", ""),
                question_type_insert=question_type_insert,
                interaction_history=self.format_questions_and_answers(interaction_history),
                mental_model=self.mental_model,
            )
        else:
            question_prompt = textwrap.dedent('''\
                Your task is to {task_description}.

                Previous questions:
                {interaction_history}

                Current mental model of {implementation}:
                {mental_model}

                Generate {num_candidate_questions} candidate {question_type_insert}s that, when answered, will reveal the most about the desired behavior beyond what has already been queried for above. Make sure each question addresses different aspects of the {implementation} than the questions that have already been asked. At the same time however, the question should be bite-sized, and not ask for too much at once. {additional_prompt}List each question on a new line and nothing else:'''
            ).format(
                implementation=implementation,
                task_description=task_description,
                additional_prompt=getattr(self, "additional_query_note", ""),
                question_type_insert=question_type_insert,
                interaction_history=self.format_questions_and_answers(interaction_history),
                num_candidate_questions=num_candidate_questions,
                mental_model=self.mental_model,
            )
        print(question_prompt)
        print("===")
        return [{"role": "user", "content": question_prompt}]

