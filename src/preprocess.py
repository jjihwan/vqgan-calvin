from glob import glob
import os
from PIL import Image
import imageio.v3 as imageio
from tqdm import tqdm
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--resolution", "-r", type=int, default=32)

    args = parser.parse_args()
    res = args.resolution
    
    root_dir = "/cvdata1/jihwan/calvin/dataset/ABCD"
    files = sorted(glob(os.path.join(root_dir, "*.mp4")))
    print(f"Total files: {len(files)}")

    for file in tqdm(files):
        video = imageio.imread(file)
        output_dir = file.replace(".mp4", "").replace(root_dir, root_dir + f"_img{res}")
        os.makedirs(output_dir, exist_ok=True)

        print(f"Processing {file} to {output_dir}")

        for i, frame in enumerate(video):
            image = Image.fromarray(frame)
            image = image.resize((res, res), resample=Image.LANCZOS)
            image.save(os.path.join(output_dir, f"{i:06d}.png"))
            