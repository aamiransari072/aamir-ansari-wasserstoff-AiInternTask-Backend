from openai import OpenAI as OpenAIClient
from typing import Dict, Any


class OpenAI:
    def __init__(self, api_key='your_openai_api_key', model='gpt-4', temperature=0.2, **kwargs):
        self.client = OpenAIClient(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs

    def generate(self, prompt: str):
        """
        Generate content using OpenAI Chat model.

        Args:
            prompt (str): Prompt to send to the model.

        Returns:
            OpenAIResponse: Wrapper object for OpenAI responses.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                **self.kwargs
            )
            return OpenAIResponse(response)
        except Exception as e:
            return OpenAIResponse(error=str(e))


class OpenAIResponse:
    def __init__(self, response=None, error=None):
        self.raw_response = response
        self.error = error

    @property
    def text(self) -> str:
        """
        Get the text content of the response.

        Returns:
            The generated text or error message.
        """
        if self.error:
            return f"Error: {self.error}"
        try:
            return self.raw_response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error extracting text: {str(e)}"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the response to a dictionary.
        """
        return {
            "text": self.text,
            "raw": self.raw_response if not self.error else None,
            "error": self.error
        }
