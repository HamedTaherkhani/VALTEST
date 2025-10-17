import time
import anthropic
from openai import OpenAI
import os
import requests
import json
from dotenv import load_dotenv
import torch.nn.functional as F
load_dotenv()
# from swebench import generate_test_cases_for_swebench
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
# import vertexai
# from vertexai.generative_models._generative_models import ResponseValidationError
# from vertexai.generative_models import GenerativeModel, ChatSession, GenerationConfig
# Import necessary libraries for different LLMs
from abc import ABC, abstractmethod
from llamaapi import LlamaAPI
# import google.generativeai as genai
class LLMRequester(ABC):
    @abstractmethod
    def get_completion(self, messages, **kwargs):
        pass

class LLamaAPIRequester(LLMRequester):
    def __init__(self, name):
        self.key = os.getenv("llama_api_key")
        self.client = LlamaAPI(self.key)
        self.name = name
    def get_completion(self, messages, **kwargs):
        # prompt = ''.join([message['content'] for message in messages])

        api_request_json = {
            "model": self.name,
            # "max_tokens": 2000,
            'messages': messages,
            "stream": False,
        }
        try:
            response = self.client.run(api_request_json)
            response = response.json()
        except Exception as e:
            print('here')
            print(e)
            return {
                'text': ' ',
                'logprobs': []
            }
        content = response['choices'][0]['message']['content']
        print(content)
        time.sleep(1)
        return {
            'text': content,
            'logprobs': []
        }


class FireworksAPIRequester(LLMRequester):
    def __init__(self, name):
        self.name = name
        self.key = os.getenv("fireworks_key")
    def get_completion(self, messages, **kwargs):
        prompt = ''.join([message['content'] for message in messages])
        # print(prompt)
        url = "https://api.fireworks.ai/inference/v1/chat/completions"
        payload = {
            "model": f"accounts/fireworks/models/{self.name}",
            # "max_tokens": 10000,
            "logprobs":5,
            # "top_p": 1,
            # "top_k": 40,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "temperature": kwargs["temperature"],
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "n":kwargs['n'],
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}"
        }
        res = requests.request("POST", url, headers=headers, data=json.dumps(payload))
        if kwargs['n'] != 1:
            res = res.json()
            return {
                'text': [res['message']['content'] for res in res['choices']]
            }
        try:
            res = res.json()
            # print(res)
            # print(res.keys())
            # print(res['choices'][0])
            # print(res['choices'][0].keys())
            tokens_with_logprobs = []
            for log in res['choices'][0]['logprobs']['top_logprobs']:
                keys = list(log.keys())
                # print(keys)
                tokens_with_logprobs.append((keys[0], log[keys[0]], [(a,log[a]) for a in keys[1:]]))
            # print(res.json()['choices'][0]['message']['content'])
            return {
                'text': res['choices'][0]['message']['content'],
                'logprobs': tokens_with_logprobs
            }
        except requests.exceptions.JSONDecodeError:
            print('here')
            return {
                'text': ' ',
                'logprobs': []
            }


class OpenaiRequester(LLMRequester):
    def __init__(self, name, backend=None):
        self.key = os.getenv('openai_key')
        if backend is None:
            self.client = OpenAI(api_key=self.key)
        else:
            if backend == "https://api.aimlapi.com/v1":
                self.key = os.getenv('aimlapi_key')
            elif backend == "https://api.deepinfra.com/v1/openai":
                self.key = os.getenv('deepinfraapi_key')
            else:
                self.key = os.getenv('deepseek_key')
            self.client = OpenAI(api_key=self.key, base_url=backend)
        self.name = name

    def get_completion(self,
            messages: list[dict[str, str]],
            max_tokens=4000,
            temperature=0,
            stop=None,
            seed=123,
            tools=None,
            logprobs=True,
            # whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message..
            top_logprobs=5,
            n=1,
    ) -> dict[str, str]:
        params = {
            "model": self.name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # "stop": stop,
            "seed": seed,
            "logprobs": logprobs,
            # "top_logprobs": top_logprobs,
            "n": n
        }
        if logprobs:
            params['top_logprobs'] = top_logprobs
        # if tools:
        #     params["tools"] = tools

        completion = self.client.chat.completions.create(**params)
        if logprobs:
            lps = completion.choices[0].logprobs.content
            text = completion.choices[0].message.content
            tokens_with_logprobs = []
            for lp in lps:
                tokens_with_logprobs.append((lp.token, lp.logprob, [(l.token,l.logprob) for l in lp.top_logprobs]))
            return {
                'text': text,
                'logprobs': tokens_with_logprobs
            }
        else:
            return {
                'text': [choice.message.content for choice in completion.choices]
            }
    # time.sleep(3)

class AntropicRequester(LLMRequester):
    def __init__(self, name):
        self.client = anthropic.Anthropic(api_key=os.getenv("anthropic_key"))
        self.name = name
    def get_completion(self, messages, **kwargs):
        prompt = ''.join([message['content'] for message in messages])
        response = self.client.messages.create(
            model=self.name,
            max_tokens=4000,
            # temperature=1,
            system="You are an expert python developer who writes good test cases.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        text = response.content[0].text
        return {
            'text': text,
            'logprobs': []
        }

class GeminiRequester(LLMRequester):
    def __init__(self, candidates=1):
        genai.configure(api_key=os.getenv('gemini_key'))
        self.client = genai.GenerativeModel("models/gemini-1.5-pro")
        self.candidates = candidates
    def get_completion(self, messages, **kwargs):
        try:
            prompt = ''.join([message['content'] for message in messages])
            response = self.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    # Only one candidate for now.
                    candidate_count=self.candidates,
                    # stop_sequences=["x"],
                    # max_output_tokens=20,
                    # temperature=0.3,
                    # top_k=8,
                    # response_logprobs=True,
                    # logprobs=2
                ),
            )
            candidates = []
            for idx, candidate in enumerate(response.candidates):
                a = {
                    'text': candidate.content.parts[0].text,
                    'logprobs': []
                }
                candidates.append(a)
            return candidates
        except Exception as e:
            return [{
                'text': '',
                'logprobs': []
            }]
# CodeLlama Requester
class HuggingfaceRequester(LLMRequester):
    def __init__(self, model_name):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            # load_in_8bit_fp32_cpu_offload=True
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_map = {
            "": self.device.type  # Automatically handles the best placement based on your setup
        }
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)

    def get_completion(self, messages, **kwargs):
        prompt = ''.join([message['content'] for message in messages])
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        max_length = kwargs.get('max_tokens', 2000) + input_ids.shape[1]
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",  # Return as PyTorch tensors
            padding=True,  # Enable padding
            truncation=True,  # Enable truncation
            max_length=max_length,  # Limit sequence length
            return_attention_mask=True  # Generate the attention mask
        ).to(self.device)



        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            max_length=max_length,
            temperature=kwargs.get('temperature', 0.0),
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            attention_mask=inputs["attention_mask"],
        )
        generated_sequence = outputs.sequences[0]
        generated_tokens = generated_sequence[input_ids.shape[1]:]  # Exclude prompt tokens
        completion = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Access the scores (logits) for each step
        scores = outputs.scores
        # List to hold results: (token, log_prob, all_log_probs)
        tokens_with_logprobs = []

        top_k = 5

        # Iterate through each generated token
        for i, token_id in enumerate(generated_tokens):
            # Get the logits for the i-th step
            logits = scores[i]  # Shape: (batch_size, vocab_size)

            # Convert logits to log probabilities
            log_probs = F.log_softmax(logits, dim=-1)[0]  # Get log probs for the current step

            # Extract the log probability of the generated token
            generated_log_prob = log_probs[token_id].item()

            # Get the top-k log probabilities and their indices
            top_k_log_probs, top_k_indices = torch.topk(log_probs, top_k)

            # Convert top-k indices and log probs to tuples (token, log_prob)
            top_k_candidates = [(self.tokenizer.decode([tid]), top_k_log_probs[j].item()) for j, tid in
                                enumerate(top_k_indices)]

            # Append the generated token, its log prob, and top-k candidate log probs
            tokens_with_logprobs.append((self.tokenizer.decode([token_id]), generated_log_prob, top_k_candidates))

        return {
            'text': completion,
            'logprobs': tokens_with_logprobs
            }
