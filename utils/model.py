# --------------------------------------------------------
# ALPS
# Copyright (c)
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from sam_vit import ImageEncoderViT
from functools import partial
import torch

# build sam vit huge
def build_sam_vit(mode_type):
    image_encoder = None
    if mode_type == 'vit_h':
        encoder_embed_dim=1280
        encoder_depth=32
        encoder_num_heads=16
        encoder_global_attn_indexes=[7, 15, 23, 31]
    elif mode_type == 'vit_l':
        encoder_embed_dim=1024
        encoder_depth=24
        encoder_num_heads=16
        encoder_global_attn_indexes=[5, 11, 17, 23]
    elif mode_type == 'vit_b':
        encoder_embed_dim=768
        encoder_depth=12
        encoder_num_heads=12
        encoder_global_attn_indexes=[2, 5, 8, 11]
    else:
        raise Exception('{mode_type} is not in [\'vit_b\', \'vit_l\', \'vit_h\'')
    
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )
    return image_encoder