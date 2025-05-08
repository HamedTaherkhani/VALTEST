
import json
class LiveCodeBenchLoader2:
    def __init__(self, instances=None):
        prompts = []
        solutions = []
        data = []
        with open('/home/hamed/PycharmProjects/hallucination/only_prompt_benchmarks/LCB.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                prompts.append(json.loads(line)['prompt'])
                solutions.append('')
        self.prompt = prompts
        self.solution = solutions
    def get_prompts(self):
        return self.prompt
    def get_solutions(self):
        return self.solution