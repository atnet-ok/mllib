from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
from  torch import nn


def get_metrics(task):

    if task == "classification":
        criterion = nn.CrossEntropyLoss()
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

    return metrics_dict