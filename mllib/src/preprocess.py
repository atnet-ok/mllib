from torchvision import transforms

def get_transform(img_size,phase=True):

    if phase:
        transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(
                    degrees=[-15, 15], translate=(0.1, 0.1), scale=(0.5, 1.5)
                ),
                transforms.RandomAutocontrast(),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                transforms.RandomAdjustSharpness(sharpness_factor=2,p=0.2),
                transforms.RandomAdjustSharpness(sharpness_factor=0,p=0.2),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], 
                    [0.229, 0.224, 0.225])
            ])

    else:
        transform =  transforms.Compose([
                transforms.Resize(round(img_size*256/224)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], 
                    [0.229, 0.224, 0.225])
            ])

    return  transform 