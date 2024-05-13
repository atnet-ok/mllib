from torch import nn
import torch.nn.functional as F
import timm
import torch

from dllib.config import model_cfg

class TimmClassifier(nn.Module):
    def __init__(self, model_name, out_dim, pre_train,in_chans):
        super(TimmClassifier, self).__init__()

        # for timm
        self.backbone =  timm.create_model(
            model_name, 
            pretrained=pre_train, 
            num_classes= 0,
            in_chans=in_chans
        )
        in_features = self.backbone.num_features

        self.head = nn.Linear(in_features, out_dim)

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

class CNN1d(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN1d, self).__init__()

        input_channels = input_channels
        mid_channels = 10
        final_out_channels = 20
        kernel_size = 8
        stride =4
        dropout =True
        features_len=256

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(
                input_channels, 
                mid_channels, 
                kernel_size=kernel_size,
                stride=stride, 
                bias=False, 
                padding=(kernel_size // 2)),
            # nn.BatchNorm1d(
            #     mid_channels
            #     ),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=2, 
                stride=2, 
                padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(
                mid_channels, 
                mid_channels * 2, 
                kernel_size=8, 
                stride=1, 
                bias=False, 
                padding=4),
            # nn.BatchNorm1d(mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=2, 
                stride=2, 
                padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(
                mid_channels * 2, 
                final_out_channels,
                kernel_size=8, 
                stride=1,
                bias=False,
                padding=4),
            # nn.BatchNorm1d(final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=2, 
                stride=2, 
                padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(features_len)
        self.logits = nn.Linear(
            features_len * final_out_channels, 
            num_classes
            )

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        y = self.logits(x_flat)
        return y, x_flat

class DoubleConv(nn.Module):
    """DoubleConv is a basic building block of the encoder and decoder components. 
    Consists of two convolutional layers followed by a ReLU activation function.
    """    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class Down(nn.Module):
    """Downscaling.
    Consists of two consecutive DoubleConv blocks followed by a max pooling operation.
    """    
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):
    """Upscaling.
    Performed using transposed convolution and concatenation of feature maps from the corresponding "Down" operation.
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input tensor shape: (batch_size, channels, height, width)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        
        self.down4 = Down(512,1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x

def get_model(model_cfg:model_cfg):

    if model_cfg.name in timm.list_models()+timm.list_models(pretrained=True):
        model = TimmClassifier(
            model_name = model_cfg.name,
            out_dim = model_cfg.out_dim,
            pre_train = model_cfg.pre_train,
            in_chans = model_cfg.in_chans
            )

    else:
        raise Exception(f'{model_cfg.name} in not implemented')
    
    return model