from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import xxhash

# --------------------- 1. 全局配置 ---------------------
INPUT_VIDEO = "/home/ao/yolov11/ultralytics/source/3.mp4"
OUTPUT_VIDEO = "output_final.mp4"
MODEL_PATH = "/home/ao/yolov11/model/yolo11n-seg.pt"  # 实例分割模型路径
COLOR_PALETTE = [               # 20种高对比度颜色(BGR)
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 128), (0, 128, 128),
    (255, 165, 0), (0, 0, 128), (0, 128, 0), (128, 0, 0),
    (64, 224, 208), (147, 20, 255), (255, 192, 203), (0, 128, 255),
    (240, 230, 140), (255, 105, 180), (139, 69, 19), (30, 144, 255)
]

# --------------------- 2. 颜色管理模块 ---------------------
class ColorManager:
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls.cache = {}  # {track_id: color}
        return cls._instance
    
    @staticmethod
    def get_color_index(track_id):
        """双哈希混合索引生成"""
        track_str = str(track_id).encode('utf-8')
        hash1 = xxhash.xxh64(track_str).intdigest()
        hash2 = hash(track_str) & 0xFFFFFFFFFFFFFFFF
        mixed_hash = (hash1 << 17) ^ (hash2 >> 23)
        return mixed_hash % len(COLOR_PALETTE)
    
    @staticmethod
    def adjust_color(base_color, bg_gray):
        """基于背景亮度的颜色自适应"""
        # 计算背景平均亮度
        avg_lum = cv2.mean(bg_gray)[0]
        
        # 转换到YUV空间
        yuv = cv2.cvtColor(np.array([[base_color]], dtype=np.uint8), 
                         cv2.COLOR_BGR2YUV)[0][0]
        
        # 动态调整亮度(Y通道)
        if avg_lum > 160:   y_new = max(30, yuv[0] * 0.6)
        elif avg_lum < 80:  y_new = min(220, yuv[0] * 1.4)
        else:               y_new = yuv[0]
        
        # 重构BGR颜色
        adjusted = cv2.cvtColor(
            np.array([[[y_new, yuv[1], yuv[2]]]], dtype=np.uint8),
            cv2.COLOR_YUV2BGR
        )[0][0]
        return tuple(map(int, adjusted))
    
    def get_color(self, track_id, bg_gray):
        """获取或生成颜色(带缓存)"""
        if track_id not in self.cache:
            idx = self.get_color_index(track_id)
            base_color = COLOR_PALETTE[idx]
            self.cache[track_id] = self.adjust_color(base_color, bg_gray)
        return self.cache[track_id]

# --------------------- 3. 视频处理主程序 ---------------------
def main():
    # 初始化模型与跟踪器
    model = YOLO(MODEL_PATH).cuda()  # 启用GPU加速
    tracker = DeepSort(max_age=30, embedder="mobilenet")
    
    # 视频输入输出设置
    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    
    # 颜色管理器实例
    color_mgr = ColorManager()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # --------------------- A. 目标检测 ---------------------
        results = model.predict(
            source=frame,
            conf=0.8,
            imgsz=640,
            show_boxes=False,
            show_labels=False,
            verbose=False
        )
        
        # 提取检测结果
        detections = results[0].boxes.data.cpu().numpy()  # [x1,y1,x2,y2,conf,cls]
        masks = results[0].masks.data.cpu().numpy() if results[0].masks else []
        
        # --------------------- B. 目标跟踪 ---------------------
        # 准备跟踪数据
        tracks_data = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            tracks_data.append(([x1, y1, x2-x1, y2-y1], conf, int(cls)))
        
        # 执行跟踪
        tracks = tracker.update_tracks(tracks_data, frame=frame)
        
        # --------------------- C. 掩码生成与颜色分配 ---------------------
        bg_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        color_mask = np.zeros_like(frame)
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()  # 获取跟踪框
            
            # 查找最佳匹配的检测结果
            matched_idx = None
            max_iou = 0
            for i, det in enumerate(detections):
                det_box = det[:4]
                # 计算IOU
                xx1 = max(det_box[0], ltrb[0])
                yy1 = max(det_box[1], ltrb[1])
                xx2 = min(det_box[2], ltrb[2])
                yy2 = min(det_box[3], ltrb[3])
                inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                area_det = (det_box[2]-det_box[0])*(det_box[3]-det_box[1])
                area_trk = (ltrb[2]-ltrb[0])*(ltrb[3]-ltrb[1])
                union = area_det + area_trk - inter
                iou = inter / union if union > 0 else 0
                if iou > max_iou:
                    max_iou = iou
                    matched_idx = i
            
            # 关联掩码与颜色
            if matched_idx is not None and matched_idx < len(masks):
                mask = masks[matched_idx]
                mask = cv2.resize(mask, (width, height), 
                                interpolation=cv2.INTER_NEAREST)
                color = color_mgr.get_color(track_id, bg_gray)
                color_mask[mask > 0.5] = color
        
        # --------------------- D. 输出合成 ---------------------
        overlay = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
        out.write(overlay)
    
    # 释放资源
    cap.release()
    out.release()
    print(f"处理完成! 输出视频已保存至: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()