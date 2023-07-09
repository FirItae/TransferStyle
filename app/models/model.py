
from util.misc import NestedTensor, nested_tensor_from_tensor_list, accuracy
from util.box_ops import box_cxcywh_to_xyxy
from models.backbone import ResBlock_nonorm,ResBlock
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg19


class VGGEncoder(nn.Module):
    """
    VGGEncoder module that extracts features from input images using VGG19 pretrained model.
    """
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        self.slice4 = vgg[12: 21]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images, output_last_feature=False):
        """
        Forward pass of the VGGEncoder module.

        Args:
            images (Tensor): Batch of input images.
            output_last_feature (bool): Whether to output only the last feature or all intermediate features.

        Returns:
            features (List[Tensor] or Tensor): Extracted features from the input images.
        """
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        if output_last_feature:
            return h4
        else:
            return [h1, h2, h3, h4]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict,  losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.vgg_encoder = VGGEncoder()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        self.register_buffer('empty_weight', empty_weight)
        
    def calc_mean_std(self,features):
        """
        Calculate the mean and standard deviation of features.

        Args:
            features (Tensor): Features tensor of shape [batch_size, c, h, w].

        Returns:
            features_mean (Tensor): Mean tensor of shape [batch_size, c, 1, 1].
            features_std (Tensor): Standard deviation tensor of shape [batch_size, c, 1, 1].
        """

        batch_size, c = features.size()[:2]
        features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
        features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
        return features_mean, features_std

    def loss_style_adain(self,content_middle_features, style_middle_features):
        """
        Calculate the style loss using the Adaptive Instance Normalization (AdaIN) technique.

        Args:
            content_middle_features (List[Tensor]): List of content intermediate features.
            style_middle_features (List[Tensor]): List of style intermediate features.

        Returns:
            loss (Tensor): Style loss.
        """
        loss = 0
        for c, s in zip(content_middle_features, style_middle_features):

            c_mean, c_std = self.calc_mean_std(c)
            s_mean, s_std = self.calc_mean_std(s)
            loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)
        return loss
    
    def loss_content_last(self,out_features, t):
        """
        Calculate the content loss between the output features and the target features.

        Args:
            out_features (Tensor): Output features.
            t (Tensor): Target features.

        Returns:
            loss (Tensor): Content loss.
        """
        return F.mse_loss(out_features, t)
    def tv_loss(self,img):
        """
        Compute total variation loss.
        Inputs:
        - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
        - tv_weight: Scalar giving the weight w_t to use for the TV loss.
        Returns:
        - loss: PyTorch Variable holding a scalar giving the total variation loss
          for img weighted by tv_weight.
        """
        # Your implementation should be vectorized and not require any loops!
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        N,C,H,W = img.shape
        x1 = img[:,:,0:H-1,:]
        x2 = img[:,:,1:H,:]
        y1 = img[:,:,:,0:W-1]
        y2 = img[:,:,:,1:W]
        loss = ((x2-x1).pow(2).sum() + (y2-y1).pow(2).sum()) 
        return loss
    

    def forward(self, outputs, targets_content,targets_style): #_hybrid
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        
#         print("outputs.shape, targets_content.shape,targets_style.shape:",outputs.shape, targets_content.shape,targets_style.shape)
        
#         output_middle_features = self.vgg_encoder(outputs, output_last_feature=False)
#         content_middle_features = self.vgg_encoder(targets_content.tensors, output_last_feature=False)
#         loss_c = self.loss_content(output_middle_features, content_middle_features)
        
        
        content_features = self.vgg_encoder(targets_content, output_last_feature=True)
        output_features = self.vgg_encoder(outputs, output_last_feature=True)
        loss_c = self.loss_content_last(output_features, content_features)
        
        
        # adain loss:
        style_middle_features = self.vgg_encoder(targets_style, output_last_feature=False)
        output_middle_features = self.vgg_encoder(outputs, output_last_feature=False)
        loss_s = self.loss_style_adain(output_middle_features, style_middle_features)
        
        
        loss_tv = self.tv_loss(outputs)
        
        
        losses = {
            'loss_content':loss_c,
            'loss_style':loss_s,
            'loss_tv':loss_tv
        
        }
        return losses
    

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class ISTT_NOFOLD(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbonec,backbones, position_embedding,transformer, num_classes, num_queries, fold_k,aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        
        self.backbone_content = backbonec
        self.backbone_style = backbones
        self.position_embedding=position_embedding
        self.aux_loss = aux_loss
        
        self.input_proj_c = nn.Conv2d(self.backbone_content.num_channels, hidden_dim, kernel_size=1)
        self.input_proj_s = nn.Conv2d(self.backbone_style.num_channels, hidden_dim, kernel_size=1)
        self.output_proj = nn.Conv2d(hidden_dim, self.backbone_content.num_channels, kernel_size=1)
        
        
        tail_layers = []
        res_block=ResBlock 
        for ri in range(self.backbone_content.reduce_times):
            times=2**ri
            content_c=self.backbone_content.num_channels
            out_c=3 if ri==self.backbone_content.reduce_times-1 else int(content_c/(times*2))
            tail_layers.extend([
                res_block(int(content_c/times), int(content_c/(times*2))),
                nn.Upsample(scale_factor = 2, mode='bilinear'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(int(content_c/times),out_c,
                          kernel_size=3, stride=1, padding=0),
            ])
        self.tail = nn.Sequential(*tail_layers)
        
        
        
    
    def forward(self, samples: NestedTensor,style_images: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)  
            style_images = nested_tensor_from_tensor_list(style_images)
            
        B,C,out_h,out_w=samples.tensors.shape  
        
        src_features = self.backbone_content(samples)  # feature: [N,B,2048,H/32,W/32] ;  pos: [N,B,256,H/32,W/32] 
        style_features = self.backbone_style(style_images)  # feature: [N,B,2048,H/32,W/32] ;  pos: [N,B,256,H/32,W/32] 
        
        
        src_features, mask = src_features["0"].decompose()
        style_features, style_mask = style_features["0"].decompose()
        B,C,f_h,f_w=src_features.shape  
        
        
        pos = self.position_embedding(NestedTensor(src_features, mask)).to(src_features.dtype)
        style_pos = self.position_embedding(NestedTensor(style_features, style_mask)).to(style_features.dtype)
        
        assert mask is not None
        
        hs,mem = self.transformer(self.input_proj_s(style_features), style_mask, self.input_proj_c(src_features),pos,style_pos) # hs: [6, 2, 100, 
    
        
        B,h_w,C=hs[-1].shape        #[B, h*w=L, C]
        hs = hs[-1].permute(0,2,1).reshape(B,C,f_h,f_w)    # [B,C,h,w]

        res = self.output_proj(hs)   # [B,256*k*k,h*w=L]   L=[(H − k + 2P )/S+1] * [(W − k + 2P )/S+1]  k=16,P=2,S=32

        
        res = self.tail(res)# [B,3,H,W] 
        
        return res