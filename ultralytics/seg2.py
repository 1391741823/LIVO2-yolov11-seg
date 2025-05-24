from ultralytics import YOLO
import cv2
import numpy as np

# --------------------- 1. 定义固定颜色表 ---------------------
# 预定义颜色列表（BGR格式）
COLORS = [
    (255, 0, 0),    # 红色
    (0, 255, 0),    # 绿色
    (0, 0, 255),    # 蓝色
    (255, 255, 0),  # 青色
    (255, 0, 255),  # 品红
    (0, 255, 255),  # 黄色
    (128, 0, 128),  # 紫色
    (0, 128, 128),  # 橄榄色
    (128, 128, 0),  # 深灰色
    (192, 192, 192),# 浅灰色
    (255, 165, 0),  # 橙色
    (128, 128, 128),# 灰色
    (0, 0, 0)       # 黑色
    # 可以根据需要添加更多颜色
    # (255, 20, 147),  # 深粉色 
]

# --------------------- 2. 初始化模型与视频读写器 ---------------------
model = YOLO("/home/ao/yolov11/model/yolo11x-seg.pt")
input_video = "/home/ao/yolov11/ultralytics/source/3.mp4"
output_color = "output_color_fixed.mp4"

cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_color = cv2.VideoWriter(output_color, fourcc, fps, (width, height))

# --------------------- 3. 逐帧处理（固定颜色按实例顺序循环） ---------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.predict(
        source=frame,
        conf=0.8,
        show_boxes=False,
        show_labels=False,
        verbose=False
    )
    
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        # 遍历实例并按顺序取颜色
        for i, mask in enumerate(masks):
            color = COLORS[i % len(COLORS)]  # 循环使用预定义颜色
            mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            color_mask[mask_resized > 0.5] = color
    
    out_color.write(color_mask)

cap.release()
out_color.release()
print("输出视频已保存：", output_color)