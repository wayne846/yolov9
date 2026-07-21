import numpy as np
import os
from tqdm import tqdm
import pyexr
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def Padding(img, w):
    return np.pad(img, ((w, w), (w, w), (0, 0)))

class DataBase:
    def __init__(self, crop_size=128):
        folder_name = os.path.join("dataset")
        # scene_names = ["classroom", "living-room", "san-miguel", "sponza-glossy", "sponza"]
        scene_names = ["bistro"]
            
        img_num_per_scene = 40
        tensor_file_names = [os.path.join(folder_name, scene_name, "tensors", "frame_"+str(i)+".pt") for scene_name in scene_names for i in range(img_num_per_scene)]

        self.train_files, self.test_files = [], []
        for i in range(len(tensor_file_names)):
            if i < (img_num_per_scene * 0.8) - 1:
                self.train_files.append(tensor_file_names[i])
            else:
                self.test_files.append(tensor_file_names[i])
                
        sample_data = torch.load(self.test_files[0])
        H, W, _ = sample_data['targets'].shape
        self.img_h, self.img_w = H - crop_size, W - crop_size



class BMFRFullResAlDataset(Dataset):
    def __init__(self, database, use_train=False, use_val=False, use_test=False, train_crops_every_frame=77, val_crops_every_frame=20, crop_size=128): # BMFR
        self.database = database
        self.use_train = use_train
        self.use_val = use_val
        self.use_test = use_test
        self.train_crops_every_frame = train_crops_every_frame
        self.val_crops_every_frame = val_crops_every_frame
        self.crop_size = crop_size

        def rotate90(inputs):
            inputs = torch.rot90(inputs, 1, (1, 2))
            return inputs
        def rotate270(inputs):
            inputs = torch.rot90(inputs, -1, (1, 2))
            return inputs
        self.transforms = [TF.hflip, TF.vflip, rotate90, rotate270]
        
            
    def _apply_transform(self, input_img, target_img):
        if self.use_train or self.use_val:
            # Random crop and convert ndarray to tensor
            i, j = np.random.randint(self.database.img_h - self.crop_size), np.random.randint(self.database.img_w-self.crop_size)
            input_crop = TF.to_tensor(input_img[i:i+self.crop_size, j:j+self.crop_size].astype(np.float32))
            target_crop = TF.to_tensor(target_img[i:i+self.crop_size, j:j+self.crop_size].astype(np.float32))
            
            if np.random.rand() > 0.5:
                transform = np.random.choice(self.transforms)
                input_crop = transform(input_crop)
                target_crop = transform(target_crop)
        elif self.use_test:
            input_crop = TF.to_tensor(input_img.astype(np.float32))
            target_crop = TF.to_tensor(target_img.astype(np.float32))
            
        return input_crop, target_crop
        
    def __getitem__(self, idx):
        if self.use_test:
            file_path = self.database.test_files[idx]
        elif self.use_train:
            file_path = self.database.train_files[idx // self.train_crops_every_frame]
        elif self.use_val:
            file_path = self.database.train_files[idx // self.val_crops_every_frame]
            
        data = torch.load(file_path)
        inputs = torch.nan_to_num(data['inputs'], nan=0.0, posinf=1.0, neginf=0.0).numpy()
        targets = torch.nan_to_num(data['targets'], nan=0.0, posinf=1.0, neginf=0.0).numpy()
        
        if self.use_train or self.use_val:
            inputs = Padding(inputs, self.crop_size)
            targets = Padding(targets, self.crop_size)
            
        return self._apply_transform(inputs, targets)
    
    def __len__(self):
        if self.use_train:
            return len(self.database.train_files) * self.train_crops_every_frame
        elif self.use_val:
            return len(self.database.train_files) * self.val_crops_every_frame
        elif self.use_test:
            return len(self.database.test_files)