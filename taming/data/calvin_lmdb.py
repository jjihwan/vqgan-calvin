import os, shutil, six
import numpy as np
import albumentations
from torch.utils.data import Dataset, DataLoader
from glob import glob
from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
import lmdb
from PIL import Image
from tqdm import tqdm
import torch
import pytorch_lightning as pl

class CalvinLMDBDataset(Dataset):
    def __init__(self, images_list_file_path: str, lmdb_path: str, size: int = 32):
        super().__init__()

        with open(images_list_file_path, "r") as f:
            self.img_paths = f.read().splitlines()
            f.close()
        
        self.length = len(self.img_paths)
        self.lmdb = None
        self.lmdb_path = lmdb_path
        self.preprocessor = albumentations.Compose([
            albumentations.SmallestMaxSize(max_size=size),
            albumentations.CenterCrop(height=size, width=size)
        ])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.lmdb is None:
            self._init_lmdb()
        
        key = self.img_paths[idx]

        img_buf = self.lmdb.begin(write=False).get(key.encode())
        buf = six.BytesIO()
        buf.write(img_buf)
        buf.seek(0)
        img = Image.open(buf)
        if not img.mode == "RGB":
            img = img.convert("RGB")
        img = np.array(img).astype(np.uint8)
        img = self.preprocessor(image=img)["image"]
        img = (img/127.5-1.0).astype(np.float32)
        img = torch.tensor(img) # B, H, W, C; will be permuted in the model

        example = dict()
        example["image"] = img
        return example

    def _init_lmdb(self):
        self.lmdb = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    def create_lmdb_database(self, lmdb_path):
        if os.path.exists(lmdb_path):
            shutil.rmtree(lmdb_path)
            print("File already exists. Removed the database at", lmdb_path)

        os.makedirs(lmdb_path, exist_ok=True)
        
        env = lmdb.open(lmdb_path, map_size=int(1e12))
        txn = env.begin(write=True)


        print("Creating LMDB database at", lmdb_path)
        for img_path in tqdm(self.img_paths):
            raw_img_bytes = open(img_path, "rb").read()
            txn.put(img_path.encode(), raw_img_bytes)

        txn.commit()
        env.sync()
        env.close()


        print(f"Wrote {len(self.img_paths)} images to {lmdb_path}")


class CalvinLMDBDataModule(pl.LightningDataModule):
    def __init__(self, path_train, path_val, path_train_lmdb, path_val_lmdb, batch_size, num_workers, size):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.ds_train = CalvinLMDBDataset(path_train, path_train_lmdb, size)
        self.ds_val = CalvinLMDBDataset(path_val, path_val_lmdb, size)
    
    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)


if __name__ == "__main__":
    # ABCD dataset
    # ds_train = CalvinLMDBDataset("data_txt/train.txt", "/cvdata1/jihwan/calvin_lmdb/lmdb/train")
    # ds_train.create_lmdb_database("/cvdata1/jihwan/calvin_lmdb/lmdb/train")

    # ds_val = CalvinLMDBDataset("data_txt/test.txt", "/cvdata1/jihwan/calvin_lmdb/lmdb/test")
    # ds_val.create_lmdb_database("/cvdata1/jihwan/calvin_lmdb/lmdb/test")

    # D_generated dataset
    ds_train = CalvinLMDBDataset("data_txt/train_D_gen.txt", "/cvdata1/jihwan/calvin_lmdb/lmdb/train_D_gen")
    ds_train.create_lmdb_database("/cvdata1/jihwan/calvin_lmdb/lmdb/train_D_gen")

    # ds_val = CalvinLMDBDataset("data_txt/test_D_gen.txt", "/cvdata1/jihwan/calvin_lmdb/lmdb/test_D_gen")
    # ds_val.create_lmdb_database("/cvdata1/jihwan/calvin_lmdb/lmdb/test_D_gen")