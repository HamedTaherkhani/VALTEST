import re
import pickle
class MBPPLoader:
    def __init__(self):
        from datasets import load_dataset
        self.dataset = []
        dataset_full = load_dataset("google-research-datasets/mbpp", "sanitized")
        for key in dataset_full.keys():
            for item in dataset_full[key]:
                self.dataset.append(item)
        self.prompts = [a['prompt'] for a in self.dataset]
        self.tests = ["\n".join(a['test_list']) for a in self.dataset]
        self.func_names = []
        # self.funcname = [a['test_list'][0].split("(")[0].replace('assert', '').replace(' ', '') for a in self.dataset]
        self.prompts_ = []
        self.solutions = [a['code'] for a in self.dataset]
        for index, a in enumerate(self.dataset):
            entry = re.findall(r'def .*\(.*\)', a['code'])
            if len(entry) == 1:
                entry = entry[0]
            elif len(entry) > 1:
                entry = entry[-1]
                # print(index)

            ##correct func name
            to_be_replaces = entry.split("(")[0].replace("def ", '')
            splitted = a['test_list'][0].split("(")
            # print(a['test_list'][0].split("("))
            # print("-----------------------------------")

            if index == 82:
                func_name = "Diff"
            elif index == 99:
                func_name = "text_match_three"
            elif index == 113:
                func_name = "is_perfect_square"
            elif index == 149:
                func_name = "volume_sphere"
            elif index == 152:
                func_name = "surfacearea_sphere"
            elif index == 165:
                func_name = "multiply_num"
            elif index == 176:
                func_name = "common_in_nested_lists"
            elif index == 185:
                func_name = "angle_complex"
            elif index == 196:
                func_name = "zero_count"
            elif index == 198:
                func_name = "circle_circumference"
            elif index == 199:
                func_name = "extract_singly"
            elif index == 207:
                func_name = "area_polygon"
            elif index == 224:
                func_name = "larg_nnum"
            elif index == 225:
                func_name = "lateralsuface_cylinder"
            elif index == 235:
                func_name = "babylonian_squareroot"
            elif index == 237:
                func_name = "harmonic_sum"
            elif index == 261:
                func_name = "volume_cylinder"
            elif index == 282:
                func_name = "count_binary_seq"
            elif index == 292:
                func_name = "volume_cone"
            elif index == 420:
                func_name = "similar_elements"
            elif index == 424:
                func_name = "find_char_long"
            else:
                func_name = splitted[0].replace('assert', '').replace(' ', '')
            self.func_names.append(func_name)
            entry = entry.replace(to_be_replaces, func_name)
            assert func_name in a['test_list'][0]
            entry += ":\n  \"\"\"" + a['prompt'] + "\"\"\""
            self.prompts_.append(entry)

    def get_prompts(self):
        return self.prompts_

    def get_tests(self):
        return self.tests

    def get_func_names(self):
        return self.func_names

    def get_dataset(self):
        return self.dataset

    def get_solutions(self):
        return self.solutions