import mlflow
from dllib.config import logger_cfg
from dllib.domain.dataset import get_dataset

cfg_logger = logger_cfg()
log_uri = cfg_logger.log_uri
data_uri = "/mnt/d/data/birdclef-2024/test_soundscapes"

logged_model = 'runs:/8dd8760901ab4af48d8c6f1920514ff2/model_best'

# Load model as a PyFuncModel.
mlflow.set_tracking_uri(log_uri)
model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import numpy as np

x = np.random.randn(10,3,256,256).astype(np.float32)
model.predict(x)