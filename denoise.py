import os
import sys
import torch
import pyexr
import time
import numpy as np
import cv2
import torch.nn.functional as F
from pathlib import Path

# 加入 YOLO 路徑以確保能找到自定義的 WSKPNHead
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import utils.dataset as wskpn_dataset
from models.experimental import attempt_load

def run_denoise():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 1. 載入訓練好的 WSKPN 權重
    # 請確認這裡的 exp 資料夾編號與你實際跑完的最後一次一致（例如 exp25）
    weights = 'runs/train/exp25/weights/best.pt' 
    print(f"正在載入模型權重: {weights}")
    model = attempt_load(weights, device=device)
    model.eval()

    # 2. 準備測試集資料
    database = wskpn_dataset.DataBase()
    # 使用 use_test=True 來讀取完整解析度且不裁切的測試圖片
    test_dataset = wskpn_dataset.BMFRFullResAlDataset(database, use_test=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 3. 建立儲存結果的資料夾
    save_dir = 'results/denoised_output'
    os.makedirs(save_dir, exist_ok=True)
    print(f"影像將儲存至: {save_dir}")

    # 4. 開始推論
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(test_loader):
            imgs = imgs.to(device).float()
            
            # 取得原始長寬，並計算所需的 Padding
            _, _, H, W = imgs.shape
            pad_h = (32 - H % 32) % 32
            pad_w = (32 - W % 32) % 32

            # 使用邊緣複製 (replicate) 的方式補齊圖片，確保長寬都是 32 的倍數
            if pad_h > 0 or pad_w > 0:
                imgs = F.pad(imgs, (0, pad_w, 0, pad_h), mode='replicate')

            # 確保 GPU 前置作業完成
            torch.cuda.synchronize()
            start_time = time.time()

            # 模型前向傳播 (降噪)
            preds = model(imgs)

            # 等待 GPU 運算完成
            torch.cuda.synchronize()
            end_time = time.time()
            
            # 計算並印出毫秒 (ms)
            inference_time_ms = (end_time - start_time) * 1000
            print(f"第 {i} 張圖片推論時間: {inference_time_ms:.2f} ms")

            # 將剛才補齊的邊緣裁切掉，恢復原始影像大小
            if pad_h > 0 or pad_w > 0:
                preds = preds[:, :, :H, :W]
            
            # 取得輸出張量，並從 (1, C, H, W) 轉換為 NumPy 的 (H, W, C)
            output_img = preds[0].cpu().numpy().transpose(1, 2, 0)
            
            # --- 儲存 1: EXR 檔案 (保留原始物理精度的浮點數) ---
            exr_path = os.path.join(save_dir, f'image_{i}.exr')
            pyexr.write(exr_path, output_img)
            
            # --- 儲存 2: PNG 檔案 (方便人類肉眼檢視) ---
            # 限制數值範圍在 0.0 ~ 1.0 之間
            png_img = np.clip(output_img, 0.0, 1.0)
            # 套用與訓練時相同的 Gamma 校正
            png_img = np.power(png_img, 0.454545)
            # 轉換為 8-bit 整數 (0~255)
            png_img = (png_img * 255).astype(np.uint8)
            # OpenCV 使用 BGR 色彩空間，因此需要把 RGB 反轉為 BGR
            png_img = png_img[:, :, ::-1] 
            
            png_path = os.path.join(save_dir, f'image_{i}.png')
            cv2.imwrite(png_path, png_img)
            
            print(f"已完成處理影像 {i} -> {png_path}")

if __name__ == '__main__':
    run_denoise()