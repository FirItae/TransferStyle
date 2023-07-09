import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):
    """
    Transformer module that consists of an encoder and a decoder.
    """

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu",
                 return_intermediate_dec=False):
        """
        Initialize the Transformer module.

        Args:
            d_model (int): The dimension of the input and output feature vectors.
            nhead (int): The number of heads in the multihead attention mechanism.
            num_encoder_layers (int): The number of encoder layers.
            num_decoder_layers (int): The number of decoder layers.
            dim_feedforward (int): The dimension of the feedforward network.
            dropout (float): The dropout probability.
            activation (str): The activation function to use.
            return_intermediate_dec (bool): Whether to return intermediate decoder outputs.
        """
        super().__init__()

        
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        encoder_norm = nn.LayerNorm(d_model) 
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        decoder_norm = nn.LayerNorm(d_model) 
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        
        ########### fold options
        self.fold_k=5
        self.fold_stride=self.fold_k-int(self.fold_k/4*3)
        self.fold_p=0
        

    def unfold_ours(self,tensor):
        """
        Unfold the input tensor.

        Args:
            tensor (Tensor): The input tensor of shape (B, C, in_h, in_w).

        Returns:
            Tensor: The unfolded tensor of shape (B, C * k * k, out_h * out_w).
        """
        B,C,in_h,in_w=tensor.shape
        out_h = (in_h - self.fold_k + 2 * self.fold_p) // self.fold_stride + 1
        out_w = (in_w - self.fold_k + 2 * self.fold_p) // self.fold_stride + 1

        tensor = F.unfold(tensor, kernel_size=(self.fold_k, self.fold_k), padding=self.fold_p,stride=self.fold_stride)  #[B,C*k*k,out_h*out_w]
        tensor = tensor.reshape(B,C,self.fold_k,self.fold_k,out_h,out_w).permute(0,1,4,5,2,3).reshape(B,C*out_h*out_w,self.fold_k*self.fold_k)
        return tensor 
    
    def _reset_parameters(self):
        """
        Reset the parameters of the model.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



    def forward(self, style_src, mask, src, query_pos_embed,pos_embed):
        """
        Perform forward pass of the Transformer module.

        Args:
            style_src (Tensor): The style source tensor.
            mask (Tensor): The mask tensor.
            src (Tensor): The source tensor.
            query_pos_embed (Optional[Tensor]): The query position embedding tensor.
            pos_embed (Optional[Tensor]): The position embedding tensor.

        Returns:
            Tuple[Tensor, Tensor]: The output tensor and the memory tensor.
        """
        src = src.flatten(2).permute(2, 0, 1) # [320, 2, 256])  [H/32*W/32,B,C] 
        
        if len(style_src.shape)==4:
            bs, c, h, w = style_src.shape

        style_src = style_src.flatten(2).permute(2, 0, 1)

        
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1) if pos_embed is not None else None#[H*W,B,C]
        query_pos_embed = query_pos_embed.flatten(2).permute(2, 0, 1)  if query_pos_embed is not None else None  #[H*W,B,C]
        mask = mask.flatten(1)
        tgt = src
        memory = self.encoder(style_src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_pos_embed)
        
        if len(src.shape)==4:
            return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
        else:
            return hs.transpose(1, 2), memory.permute(1, 2, 0)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder module.
    """


    def __init__(self, encoder_layer, num_layers, norm=None):
        """
        Initialize the Transformer encoder.

        Args:
            encoder_layer (nn.Module): The encoder layer module.
            num_layers (int): The number of encoder layers.
            norm (Optional[nn.Module]): The normalization module.
        """
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """
        Perform forward pass of the Transformer encoder.

        Args:
            src (Tensor): The source tensor.
            mask (Optional[Tensor]): The mask tensor.
            src_key_padding_mask (Optional[Tensor]): The source key padding mask tensor.
            pos (Optional[Tensor]): The position tensor.

        Returns:
            Tensor: The output tensor.
        """
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output



    
class TransformerDecoder(nn.Module):
    """
    Transformer decoder module.
    """

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        """
        Initialize the Transformer decoder.

        Args:
            decoder_layer (nn.Module): The decoder layer module.
            num_layers (int): The number of decoder layers.
            norm (Optional[nn.Module]): The normalization module.
            return_intermediate (bool): Whether to return intermediate decoder outputs.
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """
        Perform forward pass of the Transformer decoder.

        Args:
            tgt (Tensor): The target tensor.
            memory (Tensor): The memory tensor.
            tgt_mask (Optional[Tensor]): The target mask tensor.
            memory_mask (Optional[Tensor]): The memory mask tensor.
            tgt_key_padding_mask (Optional[Tensor]): The target key padding mask tensor.
            memory_key_padding_mask (Optional[Tensor]): The memory key padding mask tensor.
            pos (Optional[Tensor]): The position tensor.
            query_pos (Optional[Tensor]): The query position tensor.

        Returns:
            Tensor: The output tensor.
        """
        output = tgt

        intermediate = []

        for li,layer in enumerate(self.layers):
            output,att = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                if self.norm is not None:
                    intermediate.append(self.norm(output))
                else:
                    intermediate.append(output)
            
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer module.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        """
        Initialize the Transformer encoder layer.

        Args:
            d_model (int): The dimension of the input and output feature vectors.
            nhead (int): The number of heads in the multihead attention mechanism.
            dim_feedforward (int): The dimension of the feedforward network.
            dropout (float): The dropout probability.
            activation (str): The activation function to use.
            normalize_before (bool): Whether to normalize before the residual connection.
        """
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
    
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """
        Add positional embedding to the input tensor.

        Args:
            tensor (Tensor): The input tensor.
            pos (Optional[Tensor]): The positional embedding tensor.

        Returns:
            Tensor: The tensor with positional embedding.
        """
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        """
        Perform forward pass of the Transformer encoder layer.

        Args:
            src (Tensor): The source tensor.
            src_mask (Optional[Tensor]): The source mask tensor.
            src_key_padding_mask (Optional[Tensor]): The source key padding mask tensor.
            pos (Optional[Tensor]): The positional embedding tensor.

        Returns:
            Tensor: The output tensor.
        """
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
    
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
    
        src = self.norm2(src)
        return src


    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """
        Perform forward pass of the Transformer encoder layer.

        Args:
            src (Tensor): The source tensor.
            src_mask (Optional[Tensor]): The source mask tensor.
            src_key_padding_mask (Optional[Tensor]): The source key padding mask tensor.
            pos (Optional[Tensor]): The positional embedding tensor.

        Returns:
            Tensor: The output tensor.
        """
   
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer module.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        """
        Initialize the Transformer decoder layer.

        Args:
            d_model (int): The dimension of the input and output feature vectors.
            nhead (int): The number of heads in the multihead attention mechanism.
            dim_feedforward (int): The dimension of the feedforward network.
            dropout (float): The dropout probability.
            activation (str): The activation function to use.
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
       
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """
        Add positional embedding to the input tensor.

        Args:
            tensor (Tensor): The input tensor.
            pos (Optional[Tensor]): The positional embedding tensor.

        Returns:
            Tensor: The tensor with positional embedding.
        """
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        """
        Perform forward pass of the Transformer decoder layer.

        Args:
            tgt (Tensor): The target tensor.
            memory (Tensor): The memory tensor.
            tgt_mask (Optional[Tensor]): The target mask tensor.
            memory_mask (Optional[Tensor]): The memory mask tensor.
            tgt_key_padding_mask (Optional[Tensor]): The target key padding mask tensor.
            memory_key_padding_mask (Optional[Tensor]): The memory key padding mask tensor.
            pos (Optional[Tensor]): The positional embedding tensor.
            query_pos (Optional[Tensor]): The query position tensor.

        Returns:
            Tensor: The output tensor.
        """
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        
        tgt = self.norm1(tgt)
        tgt2_ = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt2=tgt2_[0]
        tgt2_att=tgt2_[1]
        
        tgt = tgt + self.dropout2(tgt2)
        
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        
        tgt = self.norm3(tgt)
        return tgt,tgt2_att


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """
        Perform forward pass of the Transformer decoder layer.

        Args:
            tgt (Tensor): The target tensor.
            memory (Tensor): The memory tensor.
            tgt_mask (Optional[Tensor]): The target mask tensor.
            memory_mask (Optional[Tensor]): The memory mask tensor.
            tgt_key_padding_mask (Optional[Tensor]): The target key padding mask tensor.
            memory_key_padding_mask (Optional[Tensor]): The memory key padding mask tensor.
            pos (Optional[Tensor]): The positional embedding tensor.
            query_pos (Optional[Tensor]): The query position tensor.

        Returns:
            Tensor: The output tensor.
        """
    
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

def _get_clones(module, N):
    """
    Create a list of cloned modules.

    Args:
        module (nn.Module): The module to clone.
        N (int): The number of clones.

    Returns:
        nn.ModuleList: The list of cloned modules.
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """
    Return an activation function given a string.

    Args:
        activation (str): The activation function name.

    Returns:
        Callable: The activation function.
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

