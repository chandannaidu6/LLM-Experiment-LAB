import mlflow 
import os

class MLFlow:
    def __init__(self):
        self.tracking_uri = os.getenv("MLFLOW_URI")
        self.experiment_name = os.getenv("MLFLOW_APP")
        self.configure()

    def configure(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        mlflow.openai.autolog(log_traces=True)

    def start_run(self,run_name:str | None = None,nested:bool = False):
        return mlflow.start_run(run_name=run_name,nested=nested)

    def log_params(self,params:dict):
        mlflow.log_params(params)

    def log_metrics(self,metrics:dict):
        mlflow.log_metrics(metrics)

    def log_tags(self,tags:dict):
        mlflow.set_tags(tags)

    def log_artifact(self,file_path:str):
        mlflow.log_artifact(file_path)

    def end_run(self):
        mlflow.end_run()
        





