from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np

# --------------------- 1. 初始化模型与跟踪器 ---------------------
# YOLO实例分割模型
model = YOLO("/home/ao/yolov11/model/yolo11n-seg.pt")
# DeepSORT跟踪器（max_age=30表示丢失30帧后删除ID）
tracker = DeepSort(max_age=30, embedder="mobilenet")

# --------------------- 2. 颜色生成函数 ---------------------
def get_color_from_id(track_id):
    """根据跟踪ID生成固定颜色（兼容字符串或整数ID）"""
    try:
        seed = int(track_id)
    except (ValueError, TypeError):
        seed = hash(track_id) % (2**32)
    np.random.seed(seed)
    hue = np.random.randint(0, 179)# 色调范围0-179（OpenCV标准）
    saturation = np.random.randint(150, 255)# 高饱和度
    value = np.random.randint(150, 255)# 高亮度
    hsv_color = np.uint8([[[hue, saturation, value]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))# 返回BGR格式颜色

# --------------------- 3. 视频处理主循环 ---------------------
input_video = "/home/ao/yolov11/ultralytics/source/3.mp4"
output_video = "output_tracked.mp4"

cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 视频写入器（H.264编码）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --------------------- A. YOLO检测 ---------------------
    results = model.predict(
        source=frame,
        conf=0.8,
        show_boxes=False,
        show_labels=False,
        verbose=False
    )
    
    # 提取检测结果：边界框、置信度、类别、掩码
    detections = results[0].boxes.data.cpu().numpy()  # [x1,y1,x2,y2,conf,class]
    masks = results[0].masks.data.cpu().numpy() if results[0].masks else []

    # --------------------- B. DeepSORT跟踪 ---------------------
    # 准备跟踪输入数据：[(bbox, conf, class)]
    tracks_data = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        bbox = [x1, y1, x2 - x1, y2 - y1]  # 转换为[x,y,w,h]格式
        tracks_data.append((bbox, conf, int(cls)))
    
    # 执行跟踪（获取带ID的轨迹）
    tracks = tracker.update_tracks(tracks_data, frame=frame)

    # --------------------- C. 绘制掩码与颜色分配 ---------------------
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for track in tracks:
        if not track.is_confirmed():
            continue  # 跳过未确认的轨迹
        
        track_id = track.track_id
        ltrb = track.to_ltrb()  # 获取跟踪框[x1,y1,x2,y2]
        
        # 查找当前track对应的原始检测索引（通过IOU匹配）
        matched_idx = None
        max_iou = 0
        for i, det in enumerate(detections):
            det_box = det[:4]
            # 计算IOU
            x1_det, y1_det, x2_det, y2_det = det_box
            x1_trk, y1_trk, x2_trk, y2_trk = ltrb
            # 计算交集面积
            xx1 = max(x1_det, x1_trk)
            yy1 = max(y1_det, y1_trk)
            xx2 = min(x2_det, x2_trk)
            yy2 = min(y2_det, y2_trk)
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            # 计算并集面积
            area_det = (x2_det - x1_det) * (y2_det - y1_det)
            area_trk = (x2_trk - x1_trk) * (y2_trk - y1_trk)
            union = area_det + area_trk - inter
            iou = inter / union if union > 0 else 0
            if iou > max_iou:
                max_iou = iou
                matched_idx = i
        
        # 如果找到匹配的检测，绘制对应掩码
        if matched_idx is not None and matched_idx < len(masks):
            mask = masks[matched_idx]
            mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            color = get_color_from_id(track_id)
            color_mask[mask_resized > 0.5] = color

    # --------------------- D. 融合原图与掩码（可选） ---------------------
    # 将掩码叠加到原帧（alpha=0.3表示透明度）
    overlay = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
    out.write(overlay)

cap.release()
out.release()
print("处理完成！输出视频：", output_video)