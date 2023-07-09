import argparse
import pytorch_lightning as pl
import  torch


from models.backbone import Backbone_50
from models.position_encoding import build_position_encoding
from models.transformer import Transformer
from models.model import ISTT_NOFOLD, SetCriterion, PostProcess
from models.matcher import HungarianMatcher

from torchvision import transforms
from torchvision.transforms import functional as TF
import wandb
from pytorch_lightning.loggers import WandbLogger

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    # Model parameters

    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")

    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")

    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')


    #################### 

    parser.add_argument('--model_type', default='nofold', type=str,
                        help="type of model")
    parser.add_argument('--fold_k', default=8, type=int,
                        help="Size of fold kernels")
    parser.add_argument('--fold_stride', default=6, type=int,
                        help="Size of fold kernels")
    parser.add_argument('--img_size', default=256, type=int)
   
    parser.add_argument('--cbackbone_layer', type=int, default=2,
                        help="")
    parser.add_argument('--sbackbone_layer', type=int, default=4,
                        help="")
    

    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    

     # * Loss coefficients
    parser.add_argument('--content_loss_coef', default=1.0, type=float)
    parser.add_argument('--style_loss_coef', default=8, type=float)
    parser.add_argument('--tv_loss_coef', default=0, type=float) # was 0
 
    # dataset parameters
    parser.add_argument('--style_folder', default='/data/my_data/style')
    parser.add_argument('--content_folder', default='/data/my_data/content')

    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--wikiart_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default="output_nofold_fix512",
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda:5',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default="/STT/TransferStyle/checkpoint_model/epoch=1-step=14598.ckpt", help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--num_classes', default=134, type=int)


    return parser


    
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_image_with_labels(content, style, out):
    """
    Create an image with labels.

    Args:
        content (PIL.Image): Content image.
        style (PIL.Image): Style image.
        out (PIL.Image): Output image.

    Returns:
        result_image (PIL.Image): Image with labels.
    """
    # Converting tensors to PIL images
    i_content = wandb.Image(content)
    i_style = wandb.Image(style)
    i_out = wandb.Image(out)

    # Determining the size of the image window
    width = max(i_content.image.size[0], i_style.image.size[0], i_out.image.size[0]) 
    height = max(i_content.image.size[1], i_style.image.size[1], i_out.image.size[1]) 

    border_size = 10  # Adjust this value to control the size of the borders
    total_width = width * 3+ border_size * 2 +30
    total_height = height +30


    # Creating a new image
    result_image = wandb.Image(
        np.zeros((height, width, 3), dtype=np.uint8)
    )
    result_image = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    # Copying the original images to a new image
    result_image.paste(i_content.image, (0, 30))
    result_image.paste(i_style.image, ( width + border_size, 30))
    result_image.paste(i_out.image, (width * 2 + border_size * 2, 30))

    # Creating an object to draw
    draw = ImageDraw.Draw(result_image)

    # Determining the font and text size
    font = ImageFont.truetype("/data/arial.ttf", 20)

    # Adding captions to each image
    draw.text((0, 0), "Content", font=font, fill=(0, 0, 0))
    draw.text(( width + border_size + 10, 0), "Style", font=font, fill=(0, 0, 0))
    draw.text((width * 2 + border_size * 2 + 10, 0), "Merge", font=font, fill=(0, 0, 0))

    # Returning the image as a tensor
    return result_image


import torchvision.transforms as transforms
class MyModel(pl.LightningModule):
    """
    MyModel class extends the PyTorch Lightning `LightningModule` and implements the main model for the task.
    """
    def __init__(self, args):
        """
        Initializes an instance of MyModel.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        """
        super(MyModel, self).__init__()
        # model
        self.args = args
        
        num_classes = args.num_classes
        train_backbone = args.lr_backbone > 0
        backbonec = Backbone_50(args.backbone,args.cbackbone_layer, train_backbone,  args.dilation)
        backbones = Backbone_50(args.backbone,args.sbackbone_layer, train_backbone,  args.dilation)
        position_embedding = build_position_encoding(args)
        transformer = Transformer(
                            d_model=args.hidden_dim,
                            dropout=args.dropout,
                            nhead=args.nheads,
                            dim_feedforward=args.dim_feedforward,
                            num_encoder_layers=args.enc_layers,
                            num_decoder_layers=args.dec_layers,
                
                            return_intermediate_dec=True,
                        )
        self.model = ISTT_NOFOLD(
                    backbonec,
                    backbones,
                    position_embedding,
                    transformer,
                    num_classes=num_classes,
                    num_queries=args.num_queries,
                    fold_k=args.fold_k,
                )
    

        # criterion
        matcher = HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
        weight_dict = {'loss_content': args.content_loss_coef, 'loss_style': args.style_loss_coef, 'loss_tv':args.tv_loss_coef}
        losses = ['content', 'style','tv']
        self.criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                                losses=losses)
        
        # postprocessors 
        self.postprocessors = {'bbox': PostProcess()}
        

    def forward(self, content_image, style_image):
        """
        Perform forward pass on the model.

        Args:
            content_image (Tensor): Content image tensor.
            style_image (Tensor): Style image tensor.

        Returns:
            output (Tensor): Output tensor.
        """

        content_image, _ = content_image.decompose()
        style_image, _ = style_image.decompose()
        output = self.model(content_image,style_image)
        return output

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch: Batch of data.
            batch_idx: Index of the batch.

        Returns:
            dict: Dictionary containing the loss value.
        """

        content_images, style_images, target =  batch

        content_images, _ = content_images.decompose()
        style_images, _ = style_images.decompose()
        outputs = self.model(content_images,style_images)
        loss_dict = self.criterion(outputs, content_images, style_images)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # Logging of the loss function and samples
        self.log('train_loss', losses, on_step=True, on_epoch=True, prog_bar=True,batch_size=2)
 
        self.logger.experiment.log(
                {"samples": wandb.Image(create_image_with_labels(content_images[0], style_images[0], outputs[0]), caption= f'Sample on step {self.global_step}')}
                )

        
        return {'loss': losses}
        

    
    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            batch: Batch of data.
            batch_idx: Index of the batch.

        Returns:
            dict: Dictionary containing the loss value.
        """
        content_images, style_images, _ =  batch
        content_images, _ = content_images.decompose()
        style_images, _ = style_images.decompose()
        outputs = self.model(content_images,style_images)  
        loss_dict = self.criterion(outputs, content_images, style_images)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # Logging of the loss function and samples
        self.log('val_loss', losses, on_step=True, on_epoch=True, prog_bar=True, batch_size=2)
        self.logger.experiment.log(
                {"samples_val": wandb.Image(create_image_with_labels(content_images[0], style_images[0], outputs[0]), caption= f'Sample on step {self.global_step}')}
                )
      
        return  {'loss': losses}
        
        

    def configure_optimizers(self):
        """
        Configure the optimizer.

        Returns:
            optimizer (torch.optim.Optimizer): Optimizer.
        """

        return torch.optim.AdamW(self.parameters(), lr=self.args.lr,
                                  weight_decay=self.args.weight_decay)
