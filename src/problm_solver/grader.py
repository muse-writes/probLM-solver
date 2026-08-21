"""Grading utilities for evaluating model answers to dataset questions.

This module contains only the utilities shared between all grading processes. User interface with
this module should only be required for grading custom questions via a neutral LLM. Grading a
dataset already supported by this library, such as MATH500, is handled separately.

This library makes use of simple Ollama cloud compute resources for grading.
"""

import json
import logging

import requests
from ollama import Client, GenerateResponse

_logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = ("You are grading another LLM's answer to a dataset question."
    ' Think as much as you need. All of your final responses ought to be'
    ' one-word True or False statements, depending on the validity of the'
    ' answer to the given question.')

class CloudGrader(Client):
    """Ollama cloud client configured for grading dataset responses."""

    def __init__(self, model: str, system_prompt: str | None = None) -> None:
        """Initialise."""
        super().__init__()

        # Catch invalid model early.
        response = requests.get('https://ollama.com/api/tags', timeout=30)
        models_list_json: list[dict] = json.loads(response.text)['models']
        model_names: list[str] = [remote_model['model'] for remote_model in models_list_json]
        if model not in model_names:
            msg = 'Model string not found in Ollama cloud models list.'
            raise ValueError(msg)
        self._model = model

        # Set system prompt.
        self._system_prompt: str = (system_prompt if system_prompt is not None
            else DEFAULT_SYSTEM_PROMPT)

    def grade(self, question: str, answer: str, response: str) -> tuple[bool,str|None]:
        """Request a boolean grade for a given answer."""
        response: GenerateResponse = self.generate(
            model=self._model,
            prompt=(f"Question: {question}, Correct Answer: {answer}, Model's Answer: {response}."
                "Grade the model's response as either True or False."),
            system=self._system_prompt,
            think=True
        )
        if response.response == 'True':
            return True, response.thinking
        if response.response == 'False':
            return False, response.thinking
        msg: str = ('Grader LLM erroneously returned something other than "True" or "False".'
            f'    Response:\n        {response.response},'
            f'    Thinking:\n        {response.thinking}')
        raise RuntimeError(msg)
