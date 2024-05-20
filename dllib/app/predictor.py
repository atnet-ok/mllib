import pandas as pd
from dllib.config import logger_cfg,dataset_cfg,dataloader_cfg
from dllib.domain.dataset import get_dataset,get_dataloader
from torch import nn
import torch
import numpy as np
import os


def predict(model_url,cfg_dataset,cfg_dataloader):
    phase = "test"
    device = "cuda:0"

    # Load model as a PyFuncModel.
    # mlflow.set_tracking_uri(log_uri)
    # model = mlflow.pyfunc.load_model(model_url)
    # model = model.to(device)

    model = torch.load(model_url)


    dataset = get_dataset(cfg_dataset,phase=phase)
    dataloader = get_dataloader(dataset,cfg_dataloader,phase)

    sigmoid = nn.Sigmoid().to(device)


    for j,img in enumerate(dataloader):
        img = img.to(device)
        pred = model.predict(img)

        prob_ = sigmoid(pred)
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

    return df_submission

if __name__=="__main__":
    model_url = "/mnt/d/log/birdclef2024/mlruns/487278736873082663/2d527a266c6c4b66a6045b1c70eb8b8f/artifacts/model_best/data/model.pth"
    cfg_dataset=dataset_cfg(root_dir="/mnt/d/data")
    cfg_dataloader = dataloader_cfg(batch_size_eval=4)

    df_submission = predict(model_url)
    print(df_submission)
    df_submission.to_csv('submission.csv',index=False)