import litellm
import os
import json

class LanguageModel:
    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        system_prompt: str = "You are a helpful assistant",
        json_mode: bool = False,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.json_mode = json_mode
        self.system_prompt_formatted = [
            {"content": self.system_prompt, "role": "system"}
        ]
        self.message_history = self.system_prompt_formatted

            # There should be a file called 'tokens.json' inside the same folder as this file
        openAI_key = ""
        token_path = 'tokens.json'
        if not os.path.isfile(token_path):
            raise Exception(f"{token_path} not found!")
        with open(token_path) as f:
            # If you get an error here, it means your token is formatted incorrectly. Did you put it in quotes?
            tokens = json.load(f)
            openAI_key = tokens['opeanai_api_key']
        os.environ["OPENAI_API_KEY"] = openAI_key

    def generate_response(
        self, prompt: str, maintain_message_history: bool = True, **kwargs
    ):
        # Add the user's prompt to the message history
        if maintain_message_history:
            self.message_history = [
                *self.message_history,
                {"content": f"{prompt}", "role": "user"},
            ]
        else:
            self.message_history = [{"content": f"{prompt}", "role": "user"}]
        # Generate a response
        response = litellm.completion(
            model=self.model_name,
            messages=self.message_history,
            response_format={"type": "json_object"} if self.json_mode else None,
            **kwargs,
        )
        # Add the response to the message history
        if maintain_message_history:
            self.message_history = [
                *self.message_history,
                {"content": response.choices[0].message.content, "role": "system"},
            ]

        return response.choices[0].message.content


class LLMEngine:
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        system_prompt: str = "You are a helpful assistant",
        json_mode: bool = False,
    ):

        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.json_mode = json_mode
        self.model = LanguageModel(
            model_name, temperature, system_prompt, json_mode=json_mode
        )

    def generate_response(
        self, prompt: str, maintain_message_history: bool = True, **kwargs
    ):
        return self.model.generate_response(prompt, maintain_message_history, **kwargs)
