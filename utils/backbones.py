'''
All the logic and functions related extracting robust features with pre-trained backbones is found here. 
This way, training, eval and the model itself can all use the same code.
'''

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer
import open_clip
from PIL import Image
import numpy as np

# Paths to the pre-trained models
PATH_CKPT_CLIP14 = '/home/gridsan/manderson/ovdsat/weights/clip-vit-large-patch14'
PATH_CKPT_CLIP32 = '/home/gridsan/manderson/ovdsat/weights/clip-vit-base-patch32'
PATH_CKPT_GEORSCLIP_32 = '/home/gridsan/manderson/ovdsat/weights/RS5M_ViT-B-32.pt'
PATH_CKPT_GEORSCLIP_14 = '/home/gridsan/manderson/ovdsat/weights/RS5M_ViT-L-14.pt'
PATH_CKPT_REMOTECLIP_32 = '/home/gridsan/manderson/ovdsat/weights/RemoteCLIP-ViT-B-32.pt'
PATH_CKPT_REMOTECLIP_14 = '/home/gridsan/manderson/ovdsat/weights/RemoteCLIP-ViT-L-14.pt'
PATH_CKPT_OPENCLIP14_REMOTE_FMOW = '/home/gridsan/manderson/ovdsat/weights/vlm4rs/openclip-remote-fmow.pt'
PATH_CKPT_OPENCLIP14_GEORS_FMOW = '/home/gridsan/manderson/ovdsat/weights/vlm4rs/openclip-geors-fmow.pt'


def load_backbone(backbone_type):
    '''
    Load a pre-trained backbone model.

    Args:
        backbone_type (str): Backbone type
    '''
    if backbone_type == 'dinov2':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        print('loaded dinov2!')
    elif backbone_type == 'dinov2-reg':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', force_reload=True)
    elif backbone_type == 'clip-32':
        model = CLIPModel.from_pretrained(PATH_CKPT_CLIP32).vision_model
        print('loaded clip-32!')
    elif backbone_type == 'clip-14':
        model = CLIPModel.from_pretrained(PATH_CKPT_CLIP14).vision_model
        print('loaded clip-14!')
    elif backbone_type == 'openclip-32':
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        model = model.visual
        model.output_tokens = True
        print('loaded openclip-32!')
    elif backbone_type == 'openclip-14':
        model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        model = model.visual
        model.output_tokens = True
        print('loaded openclip-14!')
    elif backbone_type == 'georsclip-32':
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32')
        ckpt = torch.load(PATH_CKPT_GEORSCLIP_32, map_location="cpu")
        model.load_state_dict(ckpt)
        model = model.visual
        model.output_tokens = True
        print('loaded georsclip-32!')
    elif backbone_type == 'georsclip-14':
        model, _, _ = open_clip.create_model_and_transforms('ViT-L-14')
        ckpt = torch.load(PATH_CKPT_GEORSCLIP_14, map_location="cpu")
        model.load_state_dict(ckpt)
        model = model.visual
        model.output_tokens = True
        print('loaded georsclip-14!')
    elif backbone_type == 'remoteclip-32':
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32')
        ckpt = torch.load(PATH_CKPT_REMOTECLIP_32, map_location="cpu")
        model.load_state_dict(ckpt)
        model = model.visual
        model.output_tokens = True
        print('loaded remoteclip-32!')
    elif backbone_type == 'remoteclip-14':
        model, _, _ = open_clip.create_model_and_transforms('ViT-L-14')
        ckpt = torch.load(PATH_CKPT_REMOTECLIP_14, map_location="cpu")
        model.load_state_dict(ckpt)
        model = model.visual
        model.output_tokens = True
        print('loaded remoteclip-14!')
    elif backbone_type == 'openclip-14-remote-fmow':
        model, _, _ = open_clip.create_model_and_transforms('ViT-L-14')
        ckpt = torch.load(PATH_CKPT_OPENCLIP14_REMOTE_FMOW, map_location="cpu")
        model.load_state_dict(ckpt)
        model = model.visual
        model.output_tokens = True
    elif backbone_type == 'openclip-14-geors-fmow':
        model, _, _ = open_clip.create_model_and_transforms('ViT-L-14')
        ckpt = torch.load(PATH_CKPT_OPENCLIP14_GEORS_FMOW, map_location="cpu")
        model.load_state_dict(ckpt)
        model = model.visual
        model.output_tokens = True
    else:
        print(f'Warning: {backbone_type} not in list!')

    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
    return model

def load_backbone_and_tokenizer_and_preprocess(backbone_type):
    '''
    Load backbone model and tokenizer for VL models (CLIP).

    Args:
        backbone_type (str): Backbone type
    '''
    preprocess = None
    if backbone_type == 'clip-32':
        model = CLIPModel.from_pretrained(PATH_CKPT_CLIP32)
        tokenizer = CLIPTokenizer.from_pretrained(PATH_CKPT_CLIP32)
    elif backbone_type == 'clip-14':
        model = CLIPModel.from_pretrained(PATH_CKPT_CLIP14)
        tokenizer = CLIPTokenizer.from_pretrained(PATH_CKPT_CLIP14)
    elif backbone_type == 'openclip-32':
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
    elif backbone_type == 'openclip-14':
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
    elif backbone_type == 'georsclip-32':
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32')
        ckpt = torch.load(PATH_CKPT_GEORSCLIP_32, map_location="cpu")
        model.load_state_dict(ckpt)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
    elif backbone_type == 'georsclip-14':
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14')
        ckpt = torch.load(PATH_CKPT_GEORSCLIP_14, map_location="cpu")
        model.load_state_dict(ckpt)
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
    elif backbone_type == 'remoteclip-32':
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32')
        ckpt = torch.load(PATH_CKPT_REMOTECLIP_32, map_location="cpu")
        model.load_state_dict(ckpt)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
    elif backbone_type == 'remoteclip-14':
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14')
        ckpt = torch.load(PATH_CKPT_REMOTECLIP_14, map_location="cpu")
        model.load_state_dict(ckpt)
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
    elif backbone_type == 'openclip-14-remote-fmow':
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14')
        ckpt = torch.load(PATH_CKPT_OPENCLIP14_REMOTE_FMOW, map_location="cpu")
        model.load_state_dict(ckpt)
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
    elif backbone_type == 'openclip-14-geors-fmow':
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14')
        ckpt = torch.load(PATH_CKPT_OPENCLIP14_GEORS_FMOW, map_location="cpu")
        model.load_state_dict(ckpt)
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
    else:
        print(f'Warning: {backbone_type} not in list!')

    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
    return model, tokenizer, preprocess

def prepare_image_for_backbone(input_tensor, backbone_type):
    '''
    Preprocess an image for the backbone model given an input tensor and the backbone type.

    Args:
        input_tensor (torch.Tensor): Input tensor with shape (B, C, H, W)
        backbone_type (str): Backbone type
    '''
    
    if input_tensor.shape[1] == 4:
        input_tensor = input_tensor[:, :3, :, :]  # Discard the alpha channel (4th channel)
    
    # Define mean and std for normalization depending on the backbone type
    if 'dinov2' in backbone_type:
        mean = torch.tensor([0.485, 0.456, 0.406]).to(input_tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(input_tensor.device)
    else:
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(input_tensor.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(input_tensor.device)

    # Scale the values to range from 0 to 1
    input_tensor /= 255.0

    # Normalize the tensor
    normalized_tensor = (input_tensor - mean[:, None, None]) / std[:, None, None]

    return normalized_tensor

def get_backbone_params(backbone_type):
    '''
    Get the parameters patch size and embedding dimensionality of the backbone model given the backbone type.

    Args:
        backbone_type (str): Backbone type
    '''
    if '14' in backbone_type or 'dinov2' in backbone_type:
        patch_size = 14
        D = 1024
    else:
        patch_size = 32
        D = 768
    return patch_size, D


def get_feats(jpg_path, model, backbone_type, device, feat_type='cls'):
    with torch.no_grad():
        img = Image.open(jpg_path).convert("RGB")

        # (1, C, H, W) float32 on device
        x = (
            torch.from_numpy(np.array(img))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(device)
        )
        # Resize to 224x224
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        #print(x.shape)

        x = prepare_image_for_backbone(x, backbone_type)

        if feat_type == 'patch':
            feats = extract_backbone_features(x, model, backbone_type)
        elif feat_type == 'cls':
            dtype = next(model.parameters()).dtype
            feats = model(x.to(dtype))[0].squeeze()

        #print(jpg_path.name, feats.shape)

    return feats


def get_caption_sim(jpg_path, caption, model, preprocess, tokenizer, device):
    with torch.no_grad():
        img = Image.open(jpg_path).convert("RGB")

        image = preprocess(img).unsqueeze(0).to(device)
        text = tokenizer([caption]).to(device)

        image_feat = model.encode_image(image)
        text_feat = model.encode_text(text)

        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        sim = image_feat @ text_feat.T   # shape: (1, 1)

    return sim.squeeze()  # scalar