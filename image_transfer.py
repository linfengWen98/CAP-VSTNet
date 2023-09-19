import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
import torchvision.utils as utils
from utils.utils import img_resize, load_segment
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='photorealistic')
parser.add_argument('--ckpoint', type=str, default='checkpoints/photo_image.pt')

# data
parser.add_argument('--content', type=str, default='data/content/01.jpg')
parser.add_argument('--style', type=str, default='data/style/01.jpg')

parser.add_argument('--out_dir', type=str, default="output")
parser.add_argument('--max_size', type=int, default=1280)
parser.add_argument('--alpha_c', type=float, default=None)

# segmentation
parser.add_argument('--content_seg', type=str, default=None)
parser.add_argument('--style_seg', type=str, default=None)
parser.add_argument('--auto_seg', action='store_true', default=False)
parser.add_argument('--save_seg_label', action='store_true', default=True)
parser.add_argument('--save_seg_color', action='store_true', default=True)
parser.add_argument('--label_mapping', type=str, default='models/segmentation/ade20k_semantic_rel.npy')
parser.add_argument('--palette', type=str, default='models/segmentation/ade20k_palette.npy')
parser.add_argument('--min_ratio', type=float, default=0.02)

args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
out_dir = args.out_dir



# Reversible Network
from models.RevResNet import RevResNet
if args.mode.lower() == "photorealistic":
    RevNetwork = RevResNet(nBlocks=[10, 10, 10], nStrides=[1, 2, 2], nChannels=[16, 64, 256], in_channel=3, mult=4, hidden_dim=16, sp_steps=2)
elif args.mode.lower() == "artistic":
    RevNetwork = RevResNet(nBlocks=[10, 10, 10], nStrides=[1, 2, 2], nChannels=[16, 64, 256], in_channel=3, mult=4, hidden_dim=64, sp_steps=1)
else:
    raise NotImplementedError()

state_dict = torch.load(args.ckpoint)
RevNetwork.load_state_dict(state_dict['state_dict'])
RevNetwork = RevNetwork.to(device)
RevNetwork.eval()


# Transfer module
from models.cWCT import cWCT
cwct = cWCT()


content = Image.open(args.content).convert('RGB')
style = Image.open(args.style).convert('RGB')

ori_csize = content.size

content = img_resize(content, args.max_size, down_scale=RevNetwork.down_scale)
style = img_resize(style, args.max_size, down_scale=RevNetwork.down_scale)


# Segmentation
if args.auto_seg:
    # You can use any 'ade20k' segmentation model
    # -----------------------------
    # An example of using SegFormer
    print("Building Segmentation Model SegFormer")
    from mmseg.apis import inference_segmentor, init_segmentor
    config = 'models/segmentation/SegFormer/local_configs/segformer/B5/segformer.b5.640x640.ade.160k.py'
    checkpoint = 'models/segmentation/SegFormer/segformer.b5.640x640.ade.160k.pth'
    seg_model = init_segmentor(config, checkpoint, device=device)

    # Inference
    content_BGR = np.array(content, dtype=np.uint8)[..., ::-1]
    content_seg = inference_segmentor(seg_model, content_BGR)[0]  # shape:[H, W], value from 0 to 149 indicating the class of pixel
    style_BGR = np.array(style, dtype=np.uint8)[..., ::-1]
    style_seg = inference_segmentor(seg_model, style_BGR)[0]
    # -----------------------------


    # Post-processing segmentation results
    from models.segmentation.SegReMapping import SegReMapping
    label_remapping = SegReMapping(args.label_mapping, min_ratio=args.min_ratio)
    content_seg = label_remapping.self_remapping(content_seg)  # eliminate noisy class
    style_seg = label_remapping.self_remapping(style_seg)
    content_seg = label_remapping.cross_remapping(content_seg, style_seg)
    # style_seg = label_remapping.styl_merge(content_seg, style_seg)

    content_seg = np.asarray(content_seg).astype(np.uint8)
    style_seg = np.asarray(style_seg).astype(np.uint8)

    # Save the class label of segmentation results
    if args.save_seg_label:
        if not os.path.exists(os.path.join(out_dir, "segmentation")):
            os.makedirs(os.path.join(out_dir, "segmentation"))
        Image.fromarray(content_seg).save(os.path.join(out_dir, "segmentation", 'content_seg_label.png'))
        Image.fromarray(style_seg).save(os.path.join(out_dir, "segmentation", 'style_seg_label.png'))

    # Save the visualization of segmentation results
    if args.save_seg_color:
        palette = np.load(args.palette)
        content_seg_color = np.zeros((content_seg.shape[0], content_seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            content_seg_color[content_seg == label, :] = color  # RGB
        Image.fromarray(content_seg_color).save(os.path.join(out_dir, "segmentation", 'content_seg_color.png'))

        style_seg_color = np.zeros((style_seg.shape[0], style_seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            style_seg_color[style_seg == label, :] = color  # RGB
        Image.fromarray(style_seg_color).save(os.path.join(out_dir, "segmentation", 'style_seg_color.png'))

    content_seg = content_seg[None, ...]    # shape: [B, H, W]
    style_seg = style_seg[None, ...]

elif args.content_seg is not None and args.style_seg is not None:
    content_seg = load_segment(args.content_seg, content.size)
    style_seg = load_segment(args.style_seg, style.size)
    content_seg = content_seg[None, ...]     # shape: [B, H, W]
    style_seg = style_seg[None, ...]
else:
    content_seg = None     # default
    style_seg = None     # default


content = transforms.ToTensor()(content).unsqueeze(0).to(device)
style = transforms.ToTensor()(style).unsqueeze(0).to(device)


# Stylization
with torch.no_grad():
    # Forward inference
    z_c = RevNetwork(content, forward=True)
    z_s = RevNetwork(style, forward=True)

    # Transfer
    if args.alpha_c is not None and content_seg is None and style_seg is None:
        # interpolation between content and style, mask is not supported
        assert 0.0 <= args.alpha_c <= 1.0
        z_cs = cwct.interpolation(z_c, styl_feat_list=[z_s], alpha_s_list=[1.0], alpha_c=args.alpha_c)
    else:
        z_cs = cwct.transfer(z_c, z_s, content_seg, style_seg)

    # Backward inference
    stylized = RevNetwork(z_cs, forward=False)


# save stylized
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
cn = os.path.basename(args.content)
sn = os.path.basename(args.style)
file_name = "%s_%s.png" % (cn.split(".")[0], sn.split(".")[0])
path = os.path.join(out_dir, file_name)

# stylized = transforms.Resize((ori_csize[1], ori_csize[0]), interpolation=Image.BICUBIC)(stylized)    # Resize to original size
grid = utils.make_grid(stylized.data, nrow=1, padding=0)
ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
out_img = Image.fromarray(ndarr)

out_img.save(path, quality=100)
print("Save at %s" % path)

