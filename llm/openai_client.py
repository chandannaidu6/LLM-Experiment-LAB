from openai import OpenAI
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal,Optional
import mlflow

PLAIN_TEMPLATE = Path("prompts/v1_plain.txt").read_text()
COT_TEMPLATE = Path("prompts/v2_cot.txt").read_text()
SHORT_TEMPLATE = Path("prompts/hyde_short.txt").read_text()
LONG_TEMPLATE = Path("prompts/hyde_long.txt").read_text()



@dataclass
class OpenAIClient:
    model:str = "gpt-4.1-mini"
    api_key:str | None = None

    def __post_init__(self):
        self._client = OpenAI(api_key=self.api_key or os.getenv("OPENAI_API_CLIENT")) 

    def build_prompt(self,query:str,context:Optional[str] = None,mode:Literal["plain","cot","short","long"] = "plain") -> str:
        if mode == "plain":
            template = PLAIN_TEMPLATE
            return template.format(context=context,query=query)
        elif mode == "cot":
            template = COT_TEMPLATE
            return template.format(context=context,query=query)
        elif mode == "short":
            template = SHORT_TEMPLATE
            return template.format(query=query)
        else:
            template = LONG_TEMPLATE
            return template.format(query=query)

    @mlflow.trace
    def generate_context(self,query:str,mode:Literal["short","long"] = "short")->str:
        prompt = self.build_prompt(query=query,context=None,mode=mode)
        response = self._client.responses.create(model=self.model,input=prompt)
        return response.output[0].content[0].text
    
    @mlflow.trace
    def chat(self,query:str,context:str,mode:Literal["plain","cot"] = "plain") -> str:
        prompt = self.build_prompt(query,context,mode)
        response = self._client.responses.create(model=self.model,input=prompt)

        return response.output[0].content[0].text
    
    