from torch import nn
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
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits ,x

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