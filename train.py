import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import datetime
from utils.dataset import get_data_loader_folder
from utils.utils import prepare_sub_folder, write_2images, write_html, print_params, adjust_learning_rate
from utils.MattingLaplacian import laplacian_loss_grad

torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"


parser = argparse.ArgumentParser()
parser.add_argument('--base_name', default=None, help='Directory name to save')
parser.add_argument('--mode', type=str, default='photorealistic')
parser.add_argument('--vgg_ckpoint', type=str, default='checkpoints/vgg_normalised.pth')

# Dataset
parser.add_argument('--train_content', default='/home/wenlinfeng/Downloads/unlabeled2017', help='Directory to dataset A')
parser.add_argument('--train_style', default='/home/wenlinfeng/Downloads/unlabeled2017', help='Directory to dataset B')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--new_size', type=int, default=512)
parser.add_argument('--crop_size', type=int, default=256)

parser.add_argument('--use_lap', type=bool, default=True)
parser.add_argument('--win_rad', type=int, default=1, help='The larger the value, the more detail in the generated image and the higher the CPU and memory requirements (proportional to the win_rad**2)')

# Training options
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)

parser.add_argument('--style_weight', type=float, default=1)
parser.add_argument('--content_weight', type=float, default=0)
parser.add_argument('--lap_weight', type=float, default=1500)
parser.add_argument('--rec_weight', type=float, default=10)
parser.add_argument('--temporal_weight', type=float, default=60)

parser.add_argument('--training_iterations', type=int, default=160000)
parser.add_argument('--fine_tuning_iterations', type=int, default=10000)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument('--resume_iter', type=int, default=-1)

# Log
parser.add_argument('--logs_directory', default='logs', help='Directory to log')
parser.add_argument('--display_size', type=int, default=16)
parser.add_argument('--image_display_iter', type=int, default=1000)
parser.add_argument('--image_save_iter', type=int, default=10000)
parser.add_argument('--model_save_interval', type=int, default=10000)

args = parser.parse_args()
if args.base_name is None:
    args.base_name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
total_iterations = args.training_iterations + args.fine_tuning_iterations
current_iter = -1


# Logs directory
logs_directory = os.path.join(args.logs_directory, args.base_name)
print("Logs directory:", logs_directory)
checkpoint_directory, image_directory = prepare_sub_folder(logs_directory)


# Dataset
batch_size = args.batch_size
num_workers = args.batch_size
new_size = args.new_size
crop_size = args.crop_size
win_rad = args.win_rad
train_loader_a = get_data_loader_folder(args.train_content, batch_size, new_size, crop_size, crop_size, use_lap=True, win_rad=win_rad)
train_loader_b = get_data_loader_folder(args.train_style, batch_size, new_size, crop_size, crop_size, use_lap=False)


# Reversible Network
from models.RevResNet import RevResNet
if args.mode.lower() == "photorealistic":
    RevNetwork = RevResNet(nBlocks=[10, 10, 10], nStrides=[1, 2, 2], nChannels=[16, 64, 256], in_channel=3, mult=4, hidden_dim=16, sp_steps=2)
elif args.mode.lower() == "artistic":
    RevNetwork = RevResNet(nBlocks=[10, 10, 10], nStrides=[1, 2, 2], nChannels=[16, 64, 256], in_channel=3, mult=4, hidden_dim=64, sp_steps=1)
else:
    raise NotImplementedError()

RevNetwork = RevNetwork.to(device)
RevNetwork.train()
print_params(RevNetwork)

# Optimizer
optimizer = torch.optim.Adam(RevNetwork.parameters(), lr=args.lr)


# Transfer module
from models.cWCT import cWCT
cwct = cWCT()


# VGG for style loss
from models.VGG import VGG19
vgg_enc = VGG19(args.vgg_ckpoint)
# encoder = nn.DataParallel(encoder)
vgg_enc.to(device)


# Resume
if args.resume:
    state_dict = torch.load(os.path.join(checkpoint_directory, "last.pt"))
    RevNetwork.load_state_dict(state_dict['state_dict'])
    optimizer.load_state_dict(state_dict['optimizer'])
    current_iter = args.resume_iter
    print('Resume from %s. Resume iter is %d' % (os.path.join(checkpoint_directory, "last.pt"), args.resume_iter))


# Loss
l1_loss = torch.nn.L1Loss()
if args.temporal_weight > 0:
    from utils.TemporalLoss import TemporalLoss
    Temporal_loss = TemporalLoss()


# Training
iter_loader_a, iter_loader_b = iter(train_loader_a), iter(train_loader_b)
while current_iter < total_iterations:
    images_a, images_b = next(iter_loader_a), next(iter_loader_b)

    lap_list = []
    if args.lap_weight > 0:
        for M in images_a['laplacian_m']:
            indices = torch.from_numpy(np.vstack((M.row, M.col))).long().to(device)
            values = torch.from_numpy(M.data).to(device)
            shape = torch.Size(M.shape)
            laplacian_m = torch.sparse_coo_tensor(indices, values, shape, device=device)
            lap_list.append(laplacian_m)

    images_a, images_b = images_a['img'].to(device), images_b['img'].to(device)

    # Optimizer
    adjust_learning_rate(optimizer, args.lr, args.lr_decay, current_iter)
    optimizer.zero_grad()

    # Forward inference
    z_c = RevNetwork(images_a, forward=True)
    z_s = RevNetwork(images_b, forward=True)

    # Transfer
    z_cs = cwct.transfer(z_c, z_s)

    # Backward inference
    stylized = RevNetwork(z_cs, forward=False)


    # Style loss
    loss_c, loss_s = vgg_enc(images_a, images_b, stylized, n_layer=4, content_weight=args.content_weight)

    # Cycle reconstruction
    if args.rec_weight > 0:
        z_cs = RevNetwork(stylized, forward=True)
        z_csc = cwct.transfer(z_cs, z_c)
        rec = RevNetwork(z_csc, forward=False)
        loss_rec = l1_loss(rec, images_a)
    else:
        loss_rec = 0

    # Matting Laplacian loss
    if args.lap_weight > 0:
        bn = stylized.size(0)
        lap_loss = list()
        grad = list()
        for i in range(bn):
            l, g = laplacian_loss_grad(stylized[i], lap_list[i])
            lap_loss.append(l)
            grad.append(g)
        grad = torch.stack(grad, dim=0)
        grad = grad * args.lap_weight
        grad = grad.clamp(-0.05, 0.05)
        stylized.backward(grad, retain_graph=True)  # We can directly backward gradient

        loss_lap = torch.mean(torch.stack(lap_loss, dim=0))
    else:
        loss_lap = 0

    # Temporal loss
    if args.temporal_weight > 0 and current_iter > args.training_iterations:
        SecondFrame, ForwardFlow = Temporal_loss.GenerateFakeData(images_a)
        z_c2 = RevNetwork(SecondFrame, forward=True)
        z_cs2 = cwct.transfer(z_c2, z_c)
        stylizedSecondFrame = RevNetwork(z_c2.clone(), forward=False)

        loss_tmp, FakeStyledSecondFrame_1 = Temporal_loss(stylized, stylizedSecondFrame, ForwardFlow)

        loss_tmp_GT, _ = Temporal_loss(images_a, SecondFrame, ForwardFlow)
    else:
        loss_tmp = 0
        loss_tmp_GT = 0.


    # Total loss
    loss = args.content_weight * loss_c + args.style_weight * loss_s + args.rec_weight * loss_rec + args.temporal_weight * loss_tmp

    loss.backward()
    nn.utils.clip_grad_norm_(RevNetwork.parameters(), 5)
    optimizer.step()


    # Dump training stats in log file
    if (current_iter + 1) % 10 == 0:
        message = "Iteration: %08d/%08d  content_loss:%.4f  lap_loss:%.4f  rec_loss:%.4f  style_loss:%.4f  loss_tmp:%.4f  loss_tmp_GT:%.4f" % (
            current_iter + 1, total_iterations,
            args.content_weight * loss_c,
            args.lap_weight * loss_lap,
            args.rec_weight * loss_rec,
            args.style_weight * loss_s,
            args.temporal_weight * loss_tmp,
            args.temporal_weight * loss_tmp_GT,
        )
        print(message)
        with open(logs_directory + "/loss.log", "a") as log_file:
            log_file.write('%s\n' % message)

        # Log sample
        if (current_iter + 1) % args.image_save_iter == 0:
            with torch.no_grad():
                index = torch.randint(low=0, high=len(train_loader_a.dataset), size=[args.display_size])
                train_display_images_a = torch.stack([train_loader_a.dataset[i]['img'] for i in index])
                index = torch.randint(low=0, high=len(train_loader_b.dataset), size=[args.display_size])
                train_display_images_b = torch.stack([train_loader_b.dataset[i]['img'] for i in index])
                train_image_outputs = RevNetwork.sample(cwct, train_display_images_a, train_display_images_b, device)
            write_2images(train_image_outputs, args.display_size, image_directory, 'train_%08d' % (current_iter + 1))
            # HTML
            write_html(logs_directory + "/index.html", current_iter + 1, args.image_save_iter, 'images')

        if (current_iter + 1) % args.image_display_iter == 0:
            with torch.no_grad():
                index = torch.randint(low=0, high=len(train_loader_a.dataset), size=[args.display_size])
                train_display_images_a = torch.stack([train_loader_a.dataset[i]['img'] for i in index])
                index = torch.randint(low=0, high=len(train_loader_b.dataset), size=[args.display_size])
                train_display_images_b = torch.stack([train_loader_b.dataset[i]['img'] for i in index])
                image_outputs = RevNetwork.sample(cwct, train_display_images_a, train_display_images_b, device)
            write_2images(image_outputs, args.display_size, image_directory, 'train_current')

        # Save network weights
        if (current_iter + 1) % args.model_save_interval == 0:
            ckpoint_file = os.path.join(checkpoint_directory, 'last.pt')
            torch.save({'state_dict': RevNetwork.state_dict(), 'optimizer': optimizer.state_dict()}, ckpoint_file)

        if (current_iter + 1) == args.training_iterations:
            ckpoint_file = os.path.join(checkpoint_directory, 'model_image.pt')
            torch.save({'state_dict': RevNetwork.state_dict()}, ckpoint_file)
        elif (current_iter + 1) == total_iterations:
            ckpoint_file = os.path.join(checkpoint_directory, 'model_video.pt')
            torch.save({'state_dict': RevNetwork.state_dict()}, ckpoint_file)

    current_iter += 1

print("Finishing training. Model save at %s" % checkpoint_directory)
