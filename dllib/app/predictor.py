import mlflow
import pandas as pd
from dllib.config import logger_cfg,dataset_cfg,dataloader_cfg
from dllib.domain.dataset import get_dataset,get_dataloader
from torch import nn
import torch
import numpy as np
import os

cfg_logger = logger_cfg()
log_uri = cfg_logger.log_uri
data_uri = "/mnt/d/data/birdclef-2024/test_soundscapes"
phase = "test"
logged_model = 'runs:/8dd8760901ab4af48d8c6f1920514ff2/model_best'


# Load model as a PyFuncModel.
mlflow.set_tracking_uri(log_uri)
model = mlflow.pyfunc.load_model(logged_model)
# model = model.to(device)


cfg_dataset=dataset_cfg()
dataset = get_dataset(cfg_dataset,phase=phase)

cfg_dataloader = dataloader_cfg(batch_size_eval=4)
dataloader = get_dataloader(dataset,cfg_dataloader,phase)

sigmoid = nn.Sigmoid()


for j,img in enumerate(dataloader):
    img = img.detach().cpu().numpy()
    pred = model.predict(img)

    prob_ = sigmoid(torch.from_numpy(pred))
    prob_= prob_.detach().cpu().numpy()

    if j == 0:
        prob = prob_
    else:
        prob = np.concatenate([prob, prob_], 0)


# id_s = np.argmax(prob,axis=1)
# label_s = [dataset.id2bird[id_] for id_ in  id_s]

df_submission = pd.DataFrame()
df_submission["row_id"] = [os.path.basename(path).replace(".ogg","") for path in dataset.df["path"].to_list()]

for i,bird in enumerate(dataset.target_columns):
    df_submission[bird] = list(prob[:,i])

print(df_submission)
df_submission.to_csv('submission.csv',index=False)