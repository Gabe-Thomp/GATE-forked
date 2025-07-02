import textwrap
import re
from utils import query_api
from generative_questions_agent import GenerativeQuestionsAgent

class MentalModel(GenerativeQuestionsAgent):
    def __init__(self, target_specification_file, engine, openai_cache_file=None, question_type="open", num_candidate_questions=1, **kwargs):
        super().__init__(target_specification_file, engine, openai_cache_file=None, question_type="open", num_candidate_questions=1, **kwargs)
        self.mental_model = "None"
        self.with_history = kwargs.get("with_history", True)
    def get_mental_model_prompt(self, task_description_mental_model, interaction_history, implementation, with_history=True):
        '''
        This is a method to update the running mental model of the user based on the past question(s) asked
        and the answers received. Whether that is done given ALL history or just the last question and answer
        is TBD.
        '''
        if with_history:
            mental_model_prompt = textwrap.dedent('''\
                    Your task is to {task_description_mental_model}. Specifically, you are in charge of updating a \"mental model\" of the {implementation} based on their responses to questions.  Do not assume a user has given a complete answer to any question. Strike a balance between incorporating useful information for the task without overfitting to specific answers.

                    Current mental model of {implementation}:
                    {mental_model}

                    Previous questions and responses:
                    {interaction_history}

                    Update the current mental model based on the latest question and answer. Generate the mental model and nothing else:'''
                ).format(
                    implementation=implementation,
                    task_description_mental_model=task_description_mental_model,
                    interaction_history=self.format_questions_and_answers(interaction_history),
                    mental_model=self.mental_model,
                )
        else:
            mental_model_prompt = textwrap.dedent('''\
                    Your task is to {task_description_mental_model}. Specifically, you are in charge of updating a \"mental model\" of the {implementation} based on their responses to questions.  Do not assume a user has given a complete answer to any question. Strike a balance between incorporating useful information for the task without overfitting to specific answers.

                    Current mental model of {implementation}:
                    {mental_model}

                    Previous question and response:
                    {interaction_history}

                    Update the current mental model based on the latest question and answer. Generate the mental model and nothing else:'''
                ).format(
                    implementation=implementation,
                    task_description_mental_model=task_description_mental_model,
                    # Selecting only last interaction
                    interaction_history=self.format_questions_and_answers(interaction_history)[-1],
                    mental_model=self.mental_model,
                )
        
        print(f"\n\nPrompt to Get Mental Model:\n{mental_model_prompt}\n\n")
        return [{"role": "user", "content": mental_model_prompt}]
    
    def get_update_mental_model(self, implementation, with_history):
        # Gets prompt asking LLM to update the mental model
        mental_model_prompt = self.get_mental_model_prompt(
            task_description_mental_model=self.task_description_mental_model,
            interaction_history=self.interaction_history,
            with_history=with_history,
            implementation=implementation,
        )
        # Queries the API to get the updated mental model
        new_mental_model, _ = query_api(
            mental_model_prompt,
            self.engine,
            self.openai_cache,
            self.openai_cache_file,
            temperature=self.temperature,
        )

        return new_mental_model

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
                Your task is to {task_description}. {task_description_mental_model_append}

                Previous questions and responses:
                {interaction_history}

                Current mental model of {implementation}:
                {mental_model}

                Generate the most informative {question_type_insert} that, when answered, will reveal the most about the desired behavior beyond what has already been queried for above. Make sure your question addresses different aspects of the {implementation} than the questions that have already been asked. At the same time however, the question should be bite-sized, and not ask for too much at once. {additional_prompt}Generate the {question_type_insert} and nothing else:'''
            ).format(
                implementation=implementation,
                task_description=task_description,
                task_description_mental_model_append=getattr(self, "task_description_mental_model_append", ""),
                additional_prompt=getattr(self, "additional_query_note", ""),
                question_type_insert=question_type_insert,
                interaction_history=self.format_questions_and_answers(interaction_history),
                mental_model=self.mental_model,
            )
        else:
            question_prompt = textwrap.dedent('''\
                Your task is to {task_description}. {task_description_mental_model_append}

                Previous questions and responses:
                {interaction_history}

                Current mental model of {implementation}:
                {mental_model}

                Generate {num_candidate_questions} candidate {question_type_insert}s that, when answered, will reveal the most about the desired behavior beyond what has already been queried for above. Make sure each question addresses different aspects of the {implementation} than the questions that have already been asked. At the same time however, the question should be bite-sized, and not ask for too much at once. {additional_prompt}List each question on a new line and nothing else:'''
            ).format(
                implementation=implementation,
                task_description=task_description,
                task_description_mental_model_append=getattr(self, "task_description_mental_model_append", ""),
                additional_prompt=getattr(self, "additional_query_note", ""),
                question_type_insert=question_type_insert,
                interaction_history=self.format_questions_and_answers(interaction_history),
                num_candidate_questions=num_candidate_questions,
                mental_model=self.mental_model,
            )
        print(f"\n\nPrompt to Get Question:\n{question_prompt}\n\n")
        return [{"role": "user", "content": question_prompt}]

    def generate_active_query(self):
        question = super().generate_active_query()
        # Updates the mental model after generating the question
        
        return question

    
    def update_mental_model(self, implementation, with_history=True):
        '''
        Updates the mental model based on the interaction history.
        '''
        new_mental_model = self.get_update_mental_model(implementation = implementation, with_history=with_history)
        self.mental_model = new_mental_model
        print(f"\nUpdated Mental Model: {self.mental_model}\n")

    def generate_oracle_response(self, query):
        '''
        Overload the original method in generative_questions_agent.py 
        to add in the mental model update.
        '''
        answer = super().generate_oracle_response(query)
        # Updates the mental model after generating the answer
        self.update_mental_model(self.implementation, with_history=self.with_history)
        
        return answer