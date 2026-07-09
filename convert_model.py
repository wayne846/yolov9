import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

def export_to_mobile(weights_path, output_path):
    print("正在載入模型...")
    # 1. 載入模型與權重 (強制載入至 CPU)
    ckpt = torch.load(weights_path, map_location='cpu')
    
    # 取出模型實例，轉為單精度浮點數並設定為評估模式
    model = ckpt['model'].float().eval()

    # 2. 準備一個符合 WSKPN 輸入維度的假張量 (Dummy Input)
    # WSKPN 的輸入是 10 通道，長寬請設定為你在手機端預期送入的大小 (例如 128x128)
    dummy_input = torch.rand(1, 10, 128, 128)

    print("正在將模型轉換為 TorchScript (Tracing)...")
    # 3. 使用 JIT Trace 追蹤模型執行路徑
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)

    # print("正在針對手機端進行最佳化...")
    # # 4. 針對 PyTorch Mobile Lite Interpreter 進行最佳化
    # optimized_model = optimize_for_mobile(traced_model)

    # 5. 匯出為手機專用格式
    traced_model._save_for_lite_interpreter(output_path)
    print(f"模型轉換成功！已儲存為: {output_path}")

if __name__ == "__main__":
    # 替換成你剛剛訓練出來的 best.pt 路徑
    WEIGHTS = "runs/20260707_train/exp3/weights/best.pt" 
    OUTPUT = "yolo_wskpn_mobile.ptl"
    
    export_to_mobile(WEIGHTS, OUTPUT)