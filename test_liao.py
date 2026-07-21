"""
廖學長的 wskpn 推論腳本，我修改以符合yolo wskpn
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pyexr
# from thop import profile
import matplotlib.pyplot as plt
import matplotlib.image as img

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

import sys
from pathlib import Path
import torch.nn.functional as F

# 加入 YOLO 路徑以確保能找到自定義的 WSKPNHead
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.experimental import attempt_load
import utils.dataset as dataset  
# import net
import torch.onnx
import time
# import flip_evaluator as flip

def BMFRGammaCorrection(img):
    if isinstance(img, np.ndarray):
        return np.clip(np.power(np.maximum(img, 0.0), 0.454545), 0.0, 1.0)
    elif isinstance(img, torch.Tensor):
        return torch.pow(torch.clamp(img, min=0.0, max=1.0), 0.454545)

def ComputeMetrics(truth_img, test_img):    
    RMSE = np.sqrt(np.mean((truth_img - test_img) ** 2))
    truth_img = BMFRGammaCorrection(truth_img)
    test_img  = BMFRGammaCorrection(test_img)
    
    SSIM = structural_similarity(truth_img, test_img, channel_axis=2, data_range=1.0)
    PSNR = peak_signal_noise_ratio(truth_img, test_img)
    return SSIM, PSNR, RMSE

def Inference(model, device, dataloader, saving_root=""):
    model.eval()
    SSIMs = []
    PSNRs = []
    RMSEs = []
    # Flips = []
    runtime = []
    # GPU_runtime = [] # only work on GPU
    total_detection_time = 0.0
    with torch.no_grad():
        for img_idx, (inputs_crops, targets_crops) in enumerate(dataloader):
            inputs = inputs_crops.to(device, non_blocking=True)
            targets = targets_crops.to(device, non_blocking=True)

            # 取得原始長寬，並計算所需的 Padding
            _, _, H, W = inputs.shape
            pad_h = (32 - H % 32) % 32
            pad_w = (32 - W % 32) % 32

            if pad_h > 0 or pad_w > 0:
                inputs = F.pad(inputs, (0, pad_w, 0, pad_h), mode='replicate')

            # starter = torch.cuda.Event(enable_timing=True) # only work on GPU
            # ender = torch.cuda.Event(enable_timing=True) # only work on GPU

            start_time = time.time()
            # starter.record() # only work on GPU
            outputs = model(inputs).detach()
            # ender.record() # only work on GPU
            # torch.cuda.synchronize() # only work on GPU
            end_time = time.time()
            detection_time = end_time - start_time
            total_detection_time += detection_time
            if len(runtime) < 11:
                runtime.append(detection_time)
            # if len(GPU_runtime) < 11:
            #     GPU_runtime.append(starter.elapsed_time(ender))  # only work on GPU
            print(f"Processing image {img_idx} took {end_time - start_time:.4f} seconds")

            # 將剛才補齊的邊緣裁切掉
            if pad_h > 0 or pad_w > 0:
                outputs = outputs[:, :, :H, :W]
            
            output = outputs.cpu().numpy()[0].transpose((1, 2, 0)) # BMFR
            target = targets.cpu().numpy()[0].transpose((1, 2, 0))
            SSIM, PSNR, RMSE = ComputeMetrics(target, output)
            # flipErrorMap, meanFLIPError, parameters = flip.evaluate(target, output, "HDR")
            SSIMs.append(SSIM)
            PSNRs.append(PSNR)
            RMSEs.append(RMSE)
            # Flips.append(meanFLIPError)

            pyexr.write(os.path.join(saving_root, str(img_idx)+".exr"), output)

    print(f"total detection time =  {total_detection_time / len(dataloader):.4f} seconds")
    print(f"runtime max: {max(runtime):.4f} seconds, min: {min(runtime):.4f} seconds")
    # for i, t in enumerate(GPU_runtime): # only work on GPU
    #         print(f"[{i}] {t:.4f} ms")    
    print("Test:")
    SSIM_mean = np.mean(SSIMs)
    PSNR_mean = np.mean(PSNRs)
    RMSE_mean = np.mean(RMSEs)
    # Flip_mean = np.mean(Flips)
    print("mean SSIM:", SSIM_mean)
    print("mean PSNR:", PSNR_mean)
    print("mean RMSE:", RMSE_mean)
    # print("mean Flips:", Flip_mean)
    SSIMs.append("mean: "+str(SSIM_mean))
    PSNRs.append("mean: "+str(PSNR_mean))
    RMSEs.append("mean: "+str(RMSE_mean))
    # Flips.append("mean: "+str(Flip_mean))
    np.savetxt(os.path.join(saving_root, "ssim.txt"), SSIMs, fmt="%s")
    np.savetxt(os.path.join(saving_root, "psnr.txt"), PSNRs, fmt="%s")
    np.savetxt(os.path.join(saving_root, "rmse.txt"), RMSEs, fmt="%s")
    # np.savetxt(os.path.join(saving_root, "flips.txt"), Flips, fmt="%s")

    return SSIM_mean, PSNR_mean, max(runtime), min(runtime), total_detection_time / len(dataloader), runtime

def PlotResults(saving_root, amount_global, amount_local, ssim_global, ssim_local, runtime_global, runtime_local, avg_runtime_global, max_runtime_global, min_runtime_global, avg_runtime_local, max_runtime_local, min_runtime_local):
    plt.figure(figsize=(10, 6))
    plt.plot(amount_global, ssim_global, 'o-', label='Global Unstructured', 
            color='blue', linewidth=2, markersize=6)
    plt.plot(amount_local, ssim_local, 's-', label='Local Structured', 
            color='red', linewidth=2, markersize=6)
    plt.xlabel('Pruning Amount')
    plt.ylabel('SSIM')
    plt.title('SSIM vs Pruning Amount')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 1.0)

    plt.tight_layout()
    plt.savefig(os.path.join(saving_root, 'ssim_vs_pruning.png'), dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))

    plt.plot(amount_global, runtime_global, 'o-',
             label='Global Unstructured Runtime', color='blue', linewidth=2, markersize=6)
    plt.plot(amount_local, runtime_local, 'o-',
             label='Local Structured Runtime', color='red', linewidth=2, markersize=6)

    runtime_text = f'Runtime info:\nglobal: Avg: {avg_runtime_global:.2f}s\nMax: {max_runtime_global:.2f}s\nMin: {min_runtime_global:.2f}s\nlocal: Avg: {avg_runtime_local:.2f}s\nMax: {max_runtime_local:.2f}s\nMin: {min_runtime_local:.2f}s'
    plt.text(1.0, 1.0, runtime_text, 
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    plt.xlabel('Pruning Amount')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs Pruning Amount')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(saving_root, 'runtime_vs_pruning.png'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = True  # same result for cpu and gpu
    torch.backends.cudnn.benchmark = False # key in here: Should be False. Ture will make the training process unstable
    device = torch.device("cuda")
    # device = torch.device("cpu")

    database = dataset.DataBase()
    dataset_test = dataset.BMFRFullResAlDataset(database, use_test=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    timestamp = "open-source-test"
    episode_name = "general"
    test_saving_root = os.path.join("results", timestamp, episode_name)
    os.makedirs(test_saving_root, exist_ok=True)

    # 載入 YOLO 改版後的權重
    weights = 'runs/train/exp46/weights/best.pt' 
    model_deployment = attempt_load(weights, device=device)
    model_deployment.eval()
    
    Inference(model_deployment, device, dataloader_test, test_saving_root)