
import mlflow
from dllib.config import logger_cfg

cfg = logger_cfg()
logged_model = 'runs:/15e81f31a9f14835940840e6b7364448/model_best'

# Load model as a PyFuncModel.
mlflow.set_tracking_uri(cfg.log_uri)
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(data))