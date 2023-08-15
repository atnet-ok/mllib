from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

def create_mask_image(image, masks):
    """Create a mask image from RLE.

    Args:
        image (numpy.ndarray): array of images.
        masks (list): List with RLE-encoded mask information.
        b (int): brightness of the pixel. Default to 1.
    
    Returns:
        numpy.ndarray: 1(b) - mask, 0 - background.
    """    
    
    s = image.shape
    h = s[0]
    w = s[1]
    mask_image = np.zeros((h,w))
    for mask in masks:
        mask_image += decode_rle(mask, h, w)
    mask_image = mask_image.clip(0, 1)
    return mask_image

def decode_rle(rle, height, width):
    """RLE to image
    modified from: https://www.kaggle.com/paulorzp/run-length-encode-and-decode

    Args:
        rle (str): mask with run length encoding.
        height (int): return image height.
        width (int): return image width.
        brightness (int): brightness of the pixel. Default to 1.

    Returns:
        np.ndarray: 1(b) - mask, 0 - background.
    """    
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1 # brightness
    return img.reshape((height, width)) 

def get_transform(img_size,phase='train',task="classification"):

    if task == "classification":
        if phase=='train':
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
            return  transform 
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
    
    elif task == "semaseg":
        if phase=='train':
            transform = [
                A.Resize(img_size,img_size,p=1),
                A.HorizontalFlip(p=0.5),
                A.Transpose(p=0.5),
                ToTensorV2(p=1)
            ]
            return A.Compose(transform)


        # Validation images undergo only resizing.
        else:
            transform = [
                A.Resize(img_size,img_size,p=1),
                ToTensorV2(p=1)
            ]
            return A.Compose(transform)