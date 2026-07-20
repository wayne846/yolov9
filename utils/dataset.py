import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from tqdm import tqdm # 用來顯示載入進度

class DataBase:
    def __init__(self, crop_size=128):
        folder_name = "dataset"
        scene_names = ["arcade", "bistro", "living_room", "sibenik", "sponza", "subway"]
        img_num_per_scene = 55
        
        # 直接在記憶體中存放資料的 List
        self.train_data = []
        self.test_data = []
        
        print("Loading all tensors into RAM... (This may take a minute)")
        
        for scene_name in scene_names:
            tensor_dir = os.path.join(folder_name, scene_name, "tensors")
            
            # 使用 tqdm 顯示載入進度
            for i in tqdm(range(img_num_per_scene), desc=f"Loading {scene_name}"):
                file_path = os.path.join(tensor_dir, f"frame_{i}.pt")
                if not os.path.exists(file_path):
                    continue

                # 一次性讀取 Tensor 並常駐記憶體
                data = torch.load(file_path)

                split_idx = int(img_num_per_scene * 0.8)
                if i < split_idx:
                    self.train_data.append(data)
                else:
                    self.test_data.append(data)

        # 設定影像邊界與裁切基準
        H, W, _ = self.test_data[0]['targets'].shape
        self.img_h, self.img_w = H - crop_size, W - crop_size
        print(f"Loaded {len(self.train_data)} train images and {len(self.test_data)} test images into RAM.")

class BMFRFullResAlDataset(Dataset):
    def __init__(self, database, use_train=False, use_val=False, use_test=False, train_crops_every_frame=77, val_crops_every_frame=20, crop_size=128):
        self.database = database
        self.use_train, self.use_val, self.use_test = use_train, use_val, use_test
        self.train_crops_every_frame = train_crops_every_frame
        self.val_crops_every_frame = val_crops_every_frame
        self.crop_size = crop_size

        def rotate90(inputs): return torch.rot90(inputs, 1, (1, 2))
        def rotate270(inputs): return torch.rot90(inputs, -1, (1, 2))
        self.transforms = [TF.hflip, TF.vflip, rotate90, rotate270]

    def _apply_transform(self, input_img, target_img):
        if self.use_train or self.use_val:
            i, j = np.random.randint(self.database.img_h - self.crop_size), np.random.randint(self.database.img_w - self.crop_size)
            # 切割並將維度從 (H, W, C) 轉為 (C, H, W)
            input_crop = input_img[i:i+self.crop_size, j:j+self.crop_size].permute(2, 0, 1)
            target_crop = target_img[i:i+self.crop_size, j:j+self.crop_size].permute(2, 0, 1)
            
            if np.random.rand() > 0.5:
                transform = np.random.choice(self.transforms)
                input_crop = transform(input_crop)
                target_crop = transform(target_crop)
        else:
            input_crop = input_img.permute(2, 0, 1)
            target_crop = target_img.permute(2, 0, 1)
            
        return input_crop, target_crop
        
    def __getitem__(self, idx):
        # 這裡直接從記憶體拿資料，速度極快，不再有任何硬碟 I/O
        if self.use_test:
            data = self.database.test_data[idx]
        elif self.use_train:
            data = self.database.train_data[idx // self.train_crops_every_frame]
        elif self.use_val:
            data = self.database.train_data[idx // self.val_crops_every_frame]
            
        inputs = data['inputs']
        targets = data['targets']

        # 清洗資料，將 NaN 轉為 0.0，將無限大限制為合法數值
        inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1.0, neginf=0.0)
        targets = torch.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=0.0)
            
        return self._apply_transform(inputs, targets)
    
    def __len__(self):
        if self.use_train: return len(self.database.train_data) * self.train_crops_every_frame
        elif self.use_val: return len(self.database.train_data) * self.val_crops_every_frame
        elif self.use_test: return len(self.database.test_data)