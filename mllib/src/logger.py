from sklearn.metrics import classification_report

def calc_metrics(y_pred,y_true,additional_metrics:dict=None):
    metrics_dict = classification_report(
        y_true, 
        y_pred, 
        output_dict=True,
        zero_division=0
        )
    if additional_metrics:
        metrics_dict.update(additional_metrics)

    for key,value in metrics_dict.items():
        if key in ["accuracy","loss_total"]:
            print(f"{key}:{value}")