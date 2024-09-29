import importlib
from argparse import ArgumentParser
from omegaconf import OmegaConf
import torch
import imageio.v3 as imageio
from PIL import Image
import os
import numpy as np
import albumentations
from glob import glob
from tqdm import tqdm

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

def get_sample_video(pngs_dir, size=32):
    frames = []
    preprocessor = albumentations.Compose([
        albumentations.SmallestMaxSize(max_size=size),
        albumentations.CenterCrop(height=size, width=size)
    ])

    png_list = sorted(glob(os.path.join(pngs_dir, "*.png")))

    for png in png_list:
        frame = Image.open(png)
        frames.append(frame)
    
    # with open(data_txt, "r") as f:
    #     for _ in range(num_frames):
    #         frames.append(Image.open(f.readline().strip()))
    #     f.close()

    # imageio.imwrite(os.path.join(output_dir, "input.gif"), frames, quality=10, fps=30)

    frames_pps = [preprocess_image(frame, preprocessor) for frame in frames]
    video_pt = torch.stack(frames_pps, dim=0)

    return video_pt, frames

def preprocess_image(frame, preprocessor):
    
    if not frame.mode == "RGB":
        frame = frame.convert("RGB")
    frame = np.array(frame).astype(np.uint8)
    frame = preprocessor(image=frame)["image"]
    frame = (frame/127.5-1.0).astype(np.float32)

    frame = torch.tensor(frame)#.permute(2, 0, 1)

    return frame

if __name__ == "__main__":
    parser = ArgumentParser()

    # parser.add_argument("--config", type=str, default="configs/calvin_rvqgan_11224_lmdb.yaml")
    parser.add_argument("--ckpt_path", type=str, default="logs/2024-09-28T18-29-48_calvin-D-rvqgan_ch11224_cs256_cd128_nq4/checkpoints/epoch=000029.ckpt")
    parser.add_argument("--data_dir", type=str, default="/cvdata1/jihwan/calvin/dataset/D_gen_img32")
    args = parser.parse_args()

    ch_mult = args.ckpt_path.split("ch")[1].split("_")[0]
    config_path = f"configs/calvin_rvqgan_{ch_mult}_lmdb.yaml"

    config = OmegaConf.load(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # num_epoch = args.ckpt_path.split("epoch=")[1].split(".")[0]
    # output_dir = os.path.dirname(args.ckpt_path).replace("checkpoints", f"test_epoch={num_epoch}")
    # os.makedirs(output_dir, exist_ok=True)

    output_root = args.data_dir.replace("img", "qtz")
    input_dir = os.path.join(output_root, "input_mp4")
    output_dir = os.path.join(output_root, "output_mp4")
    codes_dir = os.path.join(output_root, "codes")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(codes_dir, exist_ok=True)

    codebook_size = int(args.ckpt_path.split("cs")[1].split("_")[0])
    codebook_dim = int(args.ckpt_path.split("cd")[1].split("_")[0])
    num_quantizers = int(args.ckpt_path.split("nq")[1].split("/")[0])

    config.model["params"]["n_embed"] = codebook_size
    config.model["params"]["embed_dim"] = codebook_dim
    config.model["params"]["num_quantizers"] = num_quantizers

    img_size = int(args.data_dir.split("img")[1])
    pngs_dir_list = sorted(os.listdir(args.data_dir))

    # load model from config & checkpoint
    model = instantiate_from_config(config.model)
    model.init_from_ckpt(args.ckpt_path)
    model.to(device)
    model.eval()

    for pngs_basename in tqdm(pngs_dir_list):
        pngs_dir = os.path.join(args.data_dir, pngs_basename)
        video_pt, frames = get_sample_video(pngs_dir, size=img_size)

        batch = {"image": video_pt}

        with torch.no_grad():
            log = model.log_images_and_codes(batch)

        rec_video = torch.clamp(log["reconstructions"].cpu(), -1., 1.)
        rec_video = (rec_video.permute(0, 2, 3, 1).numpy() * 127.5 + 127.5).astype(np.uint8)

        codes = log["codes"].cpu().numpy()

        input_path = os.path.join(input_dir, pngs_basename+".mp4")
        output_path = os.path.join(output_dir, pngs_basename+".mp4")
        codes_path = os.path.join(codes_dir, pngs_basename+".npy")

        imageio.imwrite(input_path, frames, quality=10, fps=30)
        imageio.imwrite(output_path, rec_video, quality=10, fps=30)
        np.save(codes_path, codes)