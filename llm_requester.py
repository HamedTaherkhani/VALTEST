from openai import OpenAI
import os
from dotenv import load_dotenv
import torch.nn.functional as F
load_dotenv()
# from swebench import generate_test_cases_for_swebench
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
# Import necessary libraries for different LLMs
from abc import ABC, abstractmethod

class LLMRequester(ABC):
    @abstractmethod
    def get_completion(self, messages, **kwargs):
        pass


class OpenaiRequester(LLMRequester):
    def __init__(self, name):
        self.key = os.getenv('openai_key')
        self.client = OpenAI(api_key=self.key)
        self.name = name

    def get_completion(self,
            messages: list[dict[str, str]],
            max_tokens=1400,
            temperature=0,
            stop=None,
            seed=123,
            tools=None,
            logprobs=None,
            # whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message..
            top_logprobs=5 ## change this to generate top logprobs
    ) -> dict[str, str]:
        params = {
            "model": self.name,
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
        lps = completion.choices[0].logprobs.content
        text = completion.choices[0].message.content
        tokens_with_logprobs = []
        for lp in lps:
            tokens_with_logprobs.append((lp.token, lp.logprob, [(l.token,l.logprob) for l in lp.top_logprobs]))
        return {
            'text': text,
            'logprobs': tokens_with_logprobs
        }


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
        max_length = kwargs.get('max_tokens', 1000) + input_ids.shape[1]
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",  # Return as PyTorch tensors
            padding=True,  # Enable padding
            truncation=True,  # Enable truncation
            max_length=max_length,  # Limit sequence length
            return_attention_mask=True  # Generate the attention mask
        )



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
