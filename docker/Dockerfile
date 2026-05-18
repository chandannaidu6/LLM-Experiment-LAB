FROM python:3.13-slim

WORKDIR /code

COPY requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY main.py /code/main.py
COPY fastapi_app.py /code/fastapi_app.py
COPY api /code/api
COPY chunkers /code/chunkers
COPY retrievers /code/retrievers
COPY llm /code/llm
COPY reranker /code/reranker
COPY evaluation /code/evaluation
COPY Data /code/Data
COPY embeddings /code/embeddings
COPY prompts /code/prompts
COPY storage /code/storage
COPY tests /code/tests
COPY preprocessing.py /code/preprocessing.py
COPY randomize.py /code/randomize.py
COPY tracking.py /code/tracking.py

CMD ["uvicorn","fastapi_app:app","--host","0.0.0.0","--port","8000"]
