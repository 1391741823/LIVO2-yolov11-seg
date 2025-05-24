from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import xxhash

# --------------------- 1. 全局配置 ---------------------
INPUT_VIDEO = "/home/ao/yolov11/ultralytics/source/3.mp4"
OUTPUT_MASK_VIDEO = "mask_output.mp4"
MODEL_PATH = "/home/ao/yolov11/model/yolo11n-seg.pt"
COLOR_PALETTE = [                     # 预定义高对比度颜色表（排除黑色）
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 128), (0, 128, 128),
    (255, 165, 0), (0, 0, 128), (0, 128, 0), (128, 0, 0),
    (64, 224, 208), (147, 20, 255), (255, 192, 203), (0, 128, 255),
    (240, 230, 140), (255, 105, 180), (139, 69, 19), (30, 144, 255)
]

# --------------------- 2. 双哈希颜色索引生成器 ---------------------
class DualHashColor:
    def __init__(self):
        self.cache = {}  # {track_id: color}
    
    def get_color(self, track_id):
        """双哈希混合 -> 颜色索引 -> 固定颜色分配"""
        if track_id not in self.cache:
            # Step1: 双哈希混合
            track_str = str(track_id).encode('utf-8')
            hash1 = xxhash.xxh64(track_str).intdigest()   # 64位哈希
            hash2 = hash(track_str) & 0xFFFFFFFF          # 32位哈希
            mixed_hash = (hash1 ^ (hash2 << 32))          # 混合高低位
            
            # Step2: 映射到颜色表
            color_idx = mixed_hash % len(COLOR_PALETTE)
            self.cache[track_id] = COLOR_PALETTE[color_idx]
        return self.cache[track_id]

# --------------------- 3. 视频处理主逻辑 ---------------------
def generate_mask_video():
    # 初始化模型与跟踪器
    model = YOLO(MODEL_PATH).cuda()  # GPU加速
    tracker = DeepSort(max_age=30)
    color_allocator = DualHashColor()
    
    # 视频输入输出设置
    cap = cv2.VideoCapture(INPUT_VIDEO)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(OUTPUT_MASK_VIDEO, 
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, 
                         (width, height))
    
    # 逐帧处理
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # A. YOLO检测
        results = model.predict(frame, conf=0.8, show_boxes=False,show_labels=False,verbose=False)
        detections = results[0].boxes.data.cpu().numpy()
        masks = results[0].masks.data.cpu().numpy() if results[0].masks else []
        
        # B. DeepSORT跟踪
        tracks_data = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            tracks_data.append(([x1, y1, x2-x1, y2-y1], conf, int(cls)))
        tracks = tracker.update_tracks(tracks_data, frame=frame)
        
        # C. 生成掩码帧
        mask_frame = np.zeros((height, width, 3), dtype=np.uint8)  # 背景黑
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()  # [x1,y1,x2,y2]
            
            # D. IOU匹配检测框（完整计算）
            matched_idx = None
            max_iou = 0.0
            for i, det in enumerate(detections):
                det_box = det[:4]
                # 计算交集坐标
                xx1 = max(det_box[0], ltrb[0])
                yy1 = max(det_box[1], ltrb[1])
                xx2 = min(det_box[2], ltrb[2])
                yy2 = min(det_box[3], ltrb[3])
                inter_area = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                det_area = (det_box[2]-det_box[0])*(det_box[3]-det_box[1])
                track_area = (ltrb[2]-ltrb[0])*(ltrb[3]-ltrb[1])
                union_area = det_area + track_area - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                if iou > max_iou:
                    max_iou = iou
                    matched_idx = i
            
            # E. 填充掩码颜色
            if matched_idx is not None and matched_idx < len(masks):
                mask = cv2.resize(masks[matched_idx], (width, height), 
                                interpolation=cv2.INTER_NEAREST)
                color = color_allocator.get_color(track_id)
                mask_frame[mask > 0.5] = color
        
        # F. 写入帧
        out.write(mask_frame)
    
    # 释放资源
    cap.release()
    out.release()
    print(f"纯掩码视频已生成: {OUTPUT_MASK_VIDEO}")

if __name__ == "__main__":
    generate_mask_video()