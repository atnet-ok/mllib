import datetime
import yaml

def yaml2dct(yaml_path):
    with open(yaml_path) as file:
        dct = yaml.safe_load(file)
    return dct

def dct2yaml(dct, yaml_path):
    with open(yaml_path, 'w') as file:
        yaml.dump(dct, file)

def date2str():
    dt_now = datetime.datetime.now()
    return dt_now.strftime('%Y%m%d_%H%M_%S')