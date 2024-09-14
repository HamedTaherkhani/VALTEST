from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
class OpenaiRequester:
    def __init__(self):
        self.key = os.getenv('openai_key')
        self.client = OpenAI(api_key=self.key)

    def get_completion(self,
            messages: list[dict[str, str]],
            model: str = "gpt-4",
            max_tokens=500,
            temperature=0,
            stop=None,
            seed=123,
            tools=None,
            logprobs=None,
            # whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message..
            top_logprobs=None
    ) -> str:
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop,
            "seed": seed,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
        }
        if tools:
            params["tools"] = tools

        completion = self.client.chat.completions.create(**params)
        return completion
