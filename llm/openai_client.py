from openai import OpenAI
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import mlflow

PLAIN_TEMPLATE = Path("prompts/v1_plain.txt").read_text()
COT_TEMPLATE = Path("prompts/v2_cot.txt").read_text()

@dataclass
class OpenAIClient:
    model:str = "gpt-4.1-mini"
    api_key:str | None = None

    def __post_init__(self):
        self._client = OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_CLIENT")) 

    def build_prompt(self,query:str,context:str,mode:Literal["plain","cot"] = "plain") -> str:
        template = PLAIN_TEMPLATE if mode == "plain" else COT_TEMPLATE
        return template.format(context=context,query=query)
    
    @mlflow.trace
    def chat(self,query:str,context:str,mode:Literal["plain","cot"] = "plain") -> str:
        prompt = self.build_prompt(query,context,mode)
        response = self._client.responses.create(model=self.model,input=prompt)

        return response.output[0].content[0].text
    
    