import importlib
from argparse import ArgumentParser
from omegaconf import OmegaConf
import torch
import imageio.v3 as imageio
from PIL import Image
import os
import numpy as np
import albumentations

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_sample_video(num_frames=300, size=32):
    data_txt = "data/test.txt"
    frames = []
    preprocessor = albumentations.Compose([
        albumentations.SmallestMaxSize(max_size=size),
        albumentations.CenterCrop(height=size, width=size)
    ])

    with open(data_txt, "r") as f:
        for _ in range(num_frames):
            frames.append(Image.open(f.readline().strip()))
        f.close()

    os.makedirs("results", exist_ok=True)
    imageio.imwrite("results/input.gif", frames, quality=10, duration=0.1)

    frames = [preprocess_image(frame, preprocessor) for frame in frames]
    video = torch.stack(frames, dim=0)

    return video

def preprocess_image(frame, preprocessor):
    
    if not frame.mode == "RGB":
        frame = frame.convert("RGB")
    frame = np.array(frame).astype(np.uint8)
    frame = preprocessor(image=frame)["image"]
    frame = (frame/127.5-1.0).astype(np.float32)

    frame = torch.tensor(frame)#.permute(2, 0, 1)

    return frame

if __name__ == "__main__":
    video = get_sample_video(num_frames=1000, size=32)
    parser = ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/calvin_vqgan_f8.yaml")
    parser.add_argument("--ckpt_path", type=str, default="logs/2024-09-12T17-26-00_calvin-vqgan-f8/checkpoints/last.ckpt")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model from config & checkpoint
    model = instantiate_from_config(config.model)
    model.init_from_ckpt(args.ckpt_path)
    model.to(device)
    model.eval()

    batch = {"image": video}

    with torch.no_grad():
        log = model.log_images(batch)

    rec_video = torch.clamp(log["reconstructions"].cpu(), -1., 1.)
    rec_video = (rec_video.permute(0, 2, 3, 1).numpy() * 127.5 + 127.5).astype(np.uint8)

    imageio.imwrite("results/reconstructions.gif", rec_video, quality=10, duration=0.1)