from openai import OpenAI
import os

class OpenAI:
    def __init__(self,query):
        self.query = query

    def openai_chat(self):
        client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
        response = client.responses.create(model="gpt-4.1-mini",input= self.query)

        return response.output[0].content[0].text
