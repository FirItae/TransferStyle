import torch 
import numpy as np
from PIL import Image
from torchvision import transforms 

class STTWrapper:
    """
    Style Transformer Transfer Wrapper for telegram bot inference
    """
    def __init__(self, checkpoint='/app/checkpoint/epoch=1-step=14598.pt'):
        self.std =[0.229, 0.224, 0.225]
        self.mean =[0.485, 0.456, 0.406]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean , self.std )
        ])
        self.model = torch.load(checkpoint)

    def predict(self, content, style):
        content_f, size = self._preprocessing(content)
        style_f, _ = self._preprocessing(style)
        output = self.model(content_f, style_f)
        result = self._postprocessing(content, output, size)
        return result

    def __denorm(self, tensor):
        std_ = torch.Tensor(self.std).reshape(-1, 1, 1)
        mean_ = torch.Tensor(self.mean ).reshape(-1, 1, 1)
        res = torch.clamp(tensor * std_ + mean_, 0, 1)
        return res
        
    def __img_resize(self,im,div=32):#32
        """
        Resize the image to the desired size while maintaining the aspect ratio.

        Args:
            im (PIL.Image): Input image.
            div (int): Divisor for rounding the new size.

        Returns:
            noise_new_im (numpy.ndarray): Resized and converted image array.
        """
        desired_size=256
        h, w = im.size

        if h>w:
            new_h=desired_size
            new_w=int(new_h/h*w)
        else:
            new_w=desired_size
            new_h=int(new_w/w*h)
            
        new_w = (new_w%div==0) and new_w or (new_w + (div-(new_w%div)))
        new_h = (new_h%div==0) and new_h or (new_h + (div-(new_h%div)))
        new_im  = im.resize((new_h,new_w), Image.ANTIALIAS).convert("RGB")

        size = new_im.size
        new_im=np.array(new_im)
        return new_im, size
    
    def _preprocessing(self, image):
        image, size = self.__img_resize(image)
        image = torch.unsqueeze(self.transform(image), 0)
        return image, size


    def _postprocessing(self, content, output, size):
        output = self.__denorm(output)
        output = torch.squeeze(output, 0)
        output = transforms.ToPILImage()(output)
        # Crop the source content to the size of the target image
        cropped_image = output.crop((0, 0, size[0], size[1]))
        # Upsample the cropped image to match the size of the target image
        upsampled_image = cropped_image.resize((content.width, content.height), Image.LANCZOS)
        return upsampled_image

