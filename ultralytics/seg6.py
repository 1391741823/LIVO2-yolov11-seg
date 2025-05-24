from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import xxhash

# --------------------- 1. 全局配置 ---------------------
INPUT_VIDEO = "/home/ao/yolov11/ultralytics/source/3.mp4"
OUTPUT_MASK_VIDEO = "class_name_mask.mp4"
MODEL_PATH = "/home/ao/yolov11/model/yolo11x-seg.pt"

# --------------------- 2. 类别颜色映射（按名称） ---------------------
CLASS_COLORS = {
    # ---------- 基础家居类 ---------- 
    "chair":         (0, 0, 255),       # 红色
    "sofa":          (0, 165, 255),     # 橙色
    "table":         (204, 204, 0),     # 金色
    "bed":           (147, 20, 255),    # 深粉色
    "refrigerator":  (255, 0, 0),       # 蓝色
    "microwave":     (255, 255, 0),     # 青色
    "oven":          (0, 255, 255),     # 黄色
    "sink":          (128, 0, 128),     # 紫色
    
    # ---------- 电子设备类 ----------
    "tv":            (0, 255, 0),       # 绿色
    "laptop":        (192, 192, 192),   # 银色
    "cell phone":    (255, 0, 255),     # 品红
    "keyboard":      (30, 144, 255),    # 道奇蓝
    
    # ---------- 厨房用品类 ----------
    "bottle":        (0, 128, 128),     # 橄榄绿
    "cup":           (64, 224, 208),    # 青绿色
    "kettle":        (0, 255, 127),     # 春绿色
    "knife":         (255, 105, 180),   # 热粉色
    
    # ---------- 交通工具类 ----------
    "car":           (255, 215, 0),     # 金橙色
    "bicycle":       (34, 139, 34),     # 森林绿
    "motorcycle":    (130, 0, 75),      # 深紫色
    "bus":           (0, 69, 255),      # 深橙色
    
    # ---------- 动植物类 ----------
    "person":        (240, 230, 140),   # 卡其色
    "dog":           (139, 69, 19),     # 马鞍棕
    "cat":           (255, 192, 203),   # 粉红色
    "plant":         (34, 139, 34),     # 森林绿
    "tree":          (0, 100, 0),       # 深绿色
    
    # ---------- 其他常用类 ----------
    "book":          (210, 105, 30),    # 巧克力色
    "clock":         (240, 128, 128),   # 亮珊瑚色
    "vase":          (138, 43, 226),    # 蓝紫色
    "umbrella":      (255, 140, 0),     # 深橙色
    "handbag":       (75, 0, 130)       # 靛蓝色
}
    # 其他类别自动分配哈希颜

# --------------------- 3. 双哈希颜色生成器（其他类别） ---------------------
class DualHashColor:
    def __init__(self):
        self.cache = {}
    
    def get_color(self, track_id):
        """双哈希生成颜色（排除与预设类别颜色冲突）"""
        if track_id not in self.cache:
            track_str = str(track_id).encode('utf-8')
            hash1 = xxhash.xxh64(track_str).intdigest()
            hash2 = hash(track_str) & 0xFFFFFFFF
            mixed_hash = (hash1 ^ (hash2 << 32)) % 0xFFFFFF  # 24位颜色
            # 转换为BGR并确保不与预设颜色重复
            b, g, r = (mixed_hash >> 16) & 0xFF, (mixed_hash >> 8) & 0xFF, mixed_hash & 0xFF
            self.cache[track_id] = (b, g, r)
        return self.cache[track_id]

# --------------------- 4. 视频处理主逻辑 ---------------------
def generate_class_name_mask():
    model = YOLO(MODEL_PATH).cuda()
    tracker = DeepSort(max_age=30)
    color_allocator = DualHashColor()
    
    cap = cv2.VideoCapture(INPUT_VIDEO)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(OUTPUT_MASK_VIDEO, 
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, (w, h))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # A. YOLO检测并获取类别名称
        results = model.predict(frame, conf=0.8, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()
        masks = results[0].masks.data.cpu().numpy() if results[0].masks else []
        cls_names = [results[0].names[int(cls_id)] for cls_id in detections[:, 5]]  # 类别名称列表
        
        # B. 准备跟踪数据（传递类别名称）
        tracks_data = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls_id = det
            tracks_data.append(([x1, y1, x2-x1, y2-y1], conf, cls_names[i]))  # 传入类别名称
        
        # C. DeepSORT跟踪（需修改库以支持自定义属性，此处假设跟踪结果可携带cls_name）
        tracks = tracker.update_tracks(tracks_data, frame=frame)
        
        # D. 生成掩码帧
        mask_frame = np.zeros((h, w, 3), dtype=np.uint8)
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            cls_name = track.det_class if hasattr(track, 'det_class') else "unknown"
        
            
            ltrb = track.to_ltrb()  # [x1,y1,x2,y2]
            

            # E. IOU匹配检测框（同前，省略）
            matched_idx = None
            max_iou = 0.8
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
            
            # F. 颜色分配
            if matched_idx is not None and matched_idx < len(masks):
                mask = cv2.resize(masks[matched_idx], (w, h), cv2.INTER_NEAREST)
                # 根据类别名称选择颜色
                color = CLASS_COLORS.get(cls_name.lower(), color_allocator.get_color(track_id))
                mask_frame[mask > 0.5] = color
        
        out.write(mask_frame)
    
    cap.release()
    out.release()
    print(f"按类别名称的掩码视频已生成: {OUTPUT_MASK_VIDEO}")

if __name__ == "__main__":
    generate_class_name_mask()