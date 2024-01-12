import json
import sys
import os
import torch
from pathlib import Path
import shutil
import cv2
from torchvision import transforms
from tqdm import tqdm
import numpy as np

sys.path.append("..")

from options.train_options import TrainOptions
from models import gan_networks
import multiprocessing as mp

# IMAGE_DIRPATH = "../tmp/samples"
# DESTINATION_DIRPATH = "../tmp/output"
IMAGE_DIRPATH = "/mnt/data1/user_cache/geoffrey.g.delhomme/data/2023_12/augmented/coco-1024x750/train/images"
DESTINATION_DIRPATH = "/mnt/data1/user_cache/geoffrey.g.delhomme/data/2023_12/augmented/coco-1024x750/train/images"
# IMAGE_DIRPATH = "/mnt/data1/user_cache/geoffrey.g.delhomme/data/2023_12/augmented-extended/coco-1024x750/train/images"
# DESTINATION_DIRPATH = "/mnt/data1/user_cache/geoffrey.g.delhomme/data/2023_12/augmented-extended/coco-1024x750/train/images"
if IMAGE_DIRPATH is not None:
    IMAGE_FILEPATHS = [
        f.as_posix()
        for f in Path(IMAGE_DIRPATH).iterdir()
        if f.suffix in [".png", ".jpg", ".jpeg"] and f.name[0] != "r"
    ]
MODEL_FILEPATH = "/home/geoffrey.g.delhomme/projects/joliGEN/checkpoints/synthetic2real_mask_online/1_net_G_A.pth"
# MODEL_FILEPATH = "/home/geoffrey.g.delhomme/projects/joliGEN/checkpoints/synthetic2real_mask_online/20_net_G_A.pth"
CONFIG_FILEPATH = "/home/geoffrey.g.delhomme/projects/joliGEN/examples/example_gan_synthetic2real_mask_online.json"
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 752
GPUID = 0  # None: cpu else gpu id
CONCAT = False


def get_z_random(batch_size=1, nz=8, random_type="gauss"):
    if random_type == "uni":
        z = torch.rand(batch_size, nz) * 2.0 - 1.0
    elif random_type == "gauss":
        z = torch.randn(batch_size, nz)
    return z.detach()


def worker_init():
    global model, device, tran, opt
    gpuid = int(mp.current_process().name.split("-")[1]) - 1
    print("Initialize worker with id:", gpuid)
    # load model
    with open(CONFIG_FILEPATH, "r") as f:
        config = json.load(f)
    opt = TrainOptions().parse_json(config, set_device=False)
    if opt.model_multimodal:
        opt.model_input_nc += opt.train_mm_nz
    opt.jg_dir = Path("..").resolve().as_posix()
    device = torch.device(gpuid)
    print(f"Device: {device}")
    model = gan_networks.define_G(**vars(opt))
    model.eval()
    model.load_state_dict(torch.load(MODEL_FILEPATH, map_location=device))
    model = model.to(device)


def worker_task(samples):
    global model, device, tran, opt
    for s in samples:
        # for image_filepath in image_filepaths:
        # image_filepath = Path(image_filepath)
        # # skip if name starts with 'r'
        # if image_filepath.name[0] == "r":
        #     continue
        # # copy image with prefix 'raw-'
        # raw_image_filepath = image_filepath.parent / ("raw-" + image_filepath.name)
        # if not raw_image_filepath.exists():
        #     shutil.copy(image_filepath.as_posix(), raw_image_filepath.as_posix())
        # # load image
        # img_orig = cv2.imread(raw_image_filepath.as_posix())
        # img = cv2.cvtColor(img_orig.copy(), cv2.COLOR_BGR2RGB)
        # orig_height, orig_width, _ = img.shape
        # img = cv2.resize(
        #     img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC
        # )
        # # preprocess
        # img_tensor = tran(img).to(device)

        img_tensor, orig_height, orig_width, image_filepath = s
        image_filepath = Path(image_filepath)
        img_tensor = img_tensor.to(device)

        if opt.model_multimodal:
            z_random = get_z_random(batch_size=1, nz=opt.train_mm_nz)
            z_random = z_random.to(device)
            # print('z_random shape=', self.z_random.shape)
            z_real = z_random.view(z_random.size(0), z_random.size(1), 1, 1).expand(
                z_random.size(0),
                z_random.size(1),
                img_tensor.size(1),
                img_tensor.size(2),
            )
            img_tensor = torch.cat([img_tensor.unsqueeze(0), z_real], 1)
        else:
            img_tensor = img_tensor.unsqueeze(0)

        # run through model
        out_tensor = model(img_tensor)[0].detach()

        # post-processing
        out_img = out_tensor.data.cpu().float().numpy()
        out_img = (np.transpose(out_img, (1, 2, 0)) + 1) / 2.0 * 255.0
        # print(out_img)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        out_img = cv2.resize(
            out_img, (orig_width, orig_height), interpolation=cv2.INTER_CUBIC
        )
        # if CONCAT:
        #     out_img = np.concatenate((img_orig, out_img), axis=1)
        out_image_filepath = Path(DESTINATION_DIRPATH) / image_filepath.name
        out_image_filepath.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_image_filepath.as_posix(), out_img)


import torch


class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        # preprocess
        tranlist = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        self.tran = transforms.Compose(tranlist)

    def __len__(self):
        return len(IMAGE_FILEPATHS)

    def __getitem__(self, index):
        image_filepath = IMAGE_FILEPATHS[index]
        image_filepath = Path(image_filepath)
        # copy image with prefix 'raw-'
        raw_image_filepath = image_filepath.parent / ("raw-" + image_filepath.name)
        if not raw_image_filepath.exists():
            shutil.copy(image_filepath.as_posix(), raw_image_filepath.as_posix())
        # load image
        img_orig = cv2.imread(raw_image_filepath.as_posix())
        img = cv2.cvtColor(img_orig.copy(), cv2.COLOR_BGR2RGB)
        orig_height, orig_width, _ = img.shape
        img = cv2.resize(
            img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC
        )
        # preprocess
        # img_tensor = self.tran(img).to(device)
        img_tensor = self.tran(img)
        return img_tensor, orig_height, orig_width, image_filepath.as_posix()


def collate_fn(batch):
    return batch


if __name__ == "__main__":
    dataset = MyDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=8, collate_fn=collate_fn
    )
    with mp.get_context("spawn").Pool(
        torch.cuda.device_count(), initializer=worker_init
    ) as p:
        for _ in p.imap_unordered(worker_task, tqdm(dataloader, total=len(dataset))):
            pass
