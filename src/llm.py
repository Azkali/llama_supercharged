import os
import json
from llama_cpp import Llama

# Parametrized model instance
# json -> str: json path
# cache_dir -> str: cache directory path
# **kwargs: additional parameters dict
class Llm:
    def __init__(self, json_file: str, cache_dir: str = "cache", **kwargs):
        self.model_dir = cache_dir + "/models/"
        os.makedirs(self.model_dir, exist_ok=True)

        with open(json_file, 'r') as f:
            self.data = json.load(f)

        params = self.data.get("params", {})
        merged = {**params, **kwargs}

        self.llm = Llama(**merged)

    def __call__(self):
        return self.llm(prompt=self.data["instruction"], **self.data["prompt"])

    def download_model(self, **kwargs):
        Llama.from_pretrained(local_dir=self.model_dir, **kwargs)
