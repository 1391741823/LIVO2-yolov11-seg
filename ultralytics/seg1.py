from ultralytics import YOLO
import cv2
import numpy as np

# --------------------- 1. 初始化模型与视频读写器 ---------------------
# 加载实例分割模型
model = YOLO("/home/ao/yolov11/model/yolo11x-seg.pt")  # 替换为实际模型路径

# 输入视频路径
input_video = "/home/ao/yolov11/ultralytics/source/2.mp4"
# 输出视频路径（二值化掩码）
output_binary = "output_binary.mp4"
# 输出视频路径（彩色实例掩码）
output_color = "output_color.mp4"

# 读取输入视频信息
cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建视频写入器（H.264编码）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_binary = cv2.VideoWriter(output_binary, fourcc, fps, (width, height), isColor=False)
out_color = cv2.VideoWriter(output_color, fourcc, fps, (width, height))

# --------------------- 2. 逐帧处理视频 ---------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # --------------------- 3. 模型推理与掩码提取 ---------------------
    # 执行推理（禁用非掩码输出）
    results = model.predict(
        source=frame,
        conf=0.5,
        show_boxes=False,
        show_labels=False,
        verbose=False  # 禁用控制台日志
    )
    
    # 获取当前帧掩码
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
    else:
        masks = []  # 无检测时使用空掩码
    
    # --------------------- 4. 生成掩码帧 ---------------------
    # 选项A：二值化掩码帧
    binary_mask = np.zeros((height, width), dtype=np.uint8)
    for mask in masks:
        mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        _, instance_mask = cv2.threshold(mask_resized, 0.5, 255, cv2.THRESH_BINARY)
        binary_mask = cv2.bitwise_or(binary_mask, instance_mask.astype(np.uint8))
    out_binary.write(binary_mask)
    
    # 选项B：彩色实例掩码帧
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for i, mask in enumerate(masks):
        color = (np.random.randint(0, 255), 
                 np.random.randint(0, 255), 
                 np.random.randint(0, 255))
        mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        color_mask[mask_resized > 0.5] = color
    out_color.write(color_mask)

# --------------------- 5. 释放资源 ---------------------
cap.release()
out_binary.release()
out_color.release()
print("视频处理完成！输出文件：", output_binary, "和", output_color)