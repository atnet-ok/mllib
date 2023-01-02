from torch import nn
import torch.nn.functional as F
import timm

class TimmClassifier(nn.Module):
    def __init__(self, model_name, class_num, pre_train,in_chans):
        super(TimmClassifier, self).__init__()

        # for timm
        self.backbone =  timm.create_model(
            model_name, 
            pretrained=pre_train, 
            num_classes= 0,
            in_chans=in_chans
        )
        in_features = self.backbone.num_features

        self.head = nn.Linear(in_features, class_num)

    def forward(self, x):
        features = self.backbone(x)
        y = self.head(features)

        return y, features

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3) # 28x28x32 -> 26x26x32
        self.conv2 = nn.Conv2d(32, 64, 3) # 26x26x64 -> 24x24x64 
        self.pool = nn.MaxPool2d(2, 2) # 24x24x64 -> 12x12x64
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 12 * 12 * 64)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        y = self.fc2(x)
        return y, x

def get_model(cfg):
    if cfg.model.name in timm.list_models():
        model = TimmClassifier(
            model_name=cfg.model.name,
            class_num = cfg.data.class_num,
            pre_train = cfg.model.pre_train,
            in_chans = cfg.model.in_chans
            )
    elif cfg.model.name=="TestNet":
        model = TestNet()

    else:
        raise Exception(f'{cfg.model.name} in not implemented')

    return model