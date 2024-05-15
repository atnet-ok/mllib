from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
import torch
from  torch import nn
import torchvision

def get_metrics(task):

    if task == "classification":
        criterion = FocalLossBCE()
        score = "accuracy"
        best_score = 0
        score_direction = 1
        metrics =  classification_metrics
    

    elif task == "generation":
        criterion = nn.MSELoss()
        score = "MSE"
        best_score = 100000
        score_direction = -1
        metrics = generation_metrics

    elif task == "regression":
        criterion = nn.MSELoss()
        score = "r2"
        best_score = 0
        score_direction = 1

    elif task == "semaseg":
        #https://zenn.dev/aidemy/articles/a43ebe82dfbb8b
        criterion = nn.BCELoss()
        score = "IoU"
        best_score = 0
        score_direction = 1
        metrics = semaseg_metrics    

    return metrics, criterion




def regression_metrics(y_pred, y_true):  
    return 0  

def generation_metrics(y_pred, y_true):  
    return 0  

def semaseg_metrics(y_pred, y_true):  
    return 0  

def classification_metrics(y_pred, y_true):
    def summary_report(metrics_dict:dict, save_acc_per_class=None):
        summary = dict()
        for key,value in metrics_dict.items():

            if key =='accuracy':
                summary.update(
                    {key:value}
                )
            elif key =='macro avg':
                for key_t,value_t in value.items():
                    summary.update(
                        {key_t:value_t}
                    )      
            elif key =='weighted avg':
                pass
            else:
                if save_acc_per_class:
                    for key_t,value_t in value.items():
                        if key_t=='f1-score':
                            summary.update(
                                {key+'_'+key_t:value_t}
                            )      
                else:
                    pass 
            
        return summary
    
    cls_true = [np.argmax(i) for i in y_true]
    cls_pred = [np.argmax(i) for i in y_pred]

    metrics_dict = classification_report(
        cls_true, 
        cls_pred, 
        output_dict=True,
        zero_division=0
        )
    

    metrics_dict = summary_report(metrics_dict)

    return metrics_dict["accuracy"]

class FocalLossBCE(torch.nn.Module):
    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2,
            reduction: str = "mean",
            bce_weight: float = 1.0,
            focal_weight: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        focall_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=logits,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        bce_loss = self.bce(logits, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focall_loss


