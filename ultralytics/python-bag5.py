#!/usr/bin/env python3
import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import xxhash
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image
from concurrent.futures import ThreadPoolExecutor

# --------------------- 类别颜色映射 ---------------------
CLASS_COLORS = {
    "chair": (0, 0, 255),       # 红色
    "sofa": (0, 165, 255),      # 橙色
    "table": (204, 204, 0),     # 金色
    "bed": (147, 20, 255),      # 深粉色
    "toilet": (255, 0, 0),      # 蓝色
    "microwave": (255, 255, 0), # 青色
    "oven": (0, 255, 255),      # 黄色
    "sink": (128, 0, 128),      # 紫色
    "clock": (0, 255, 0),       # 绿色
    "laptop": (192, 192, 192),  # 银色
    "cell phone": (255, 0, 255),# 品红
    "keyboard": (30, 144, 255), # 道奇蓝
    "bottle": (0, 128, 128),    # 橄榄绿
    "cup": (64, 224, 208),      # 青绿色
    "kettle": (0, 255, 127),    # 春绿色
    "knife": (255, 105, 180),   # 热粉色
    "car": (255, 215, 0),       # 金橙色
    "bicycle": (34, 139, 34),   # 森林绿
    "motorcycle": (130, 0, 75), # 深紫色
    "bus": (0, 69, 255),        # 深橙色
    "person": (240, 230, 140),  # 卡其色
    "dog": (139, 69, 19),       # 马鞍棕
    "cat": (255, 192, 203),     # 粉红色
    "plant": (34, 139, 34),     # 森林绿
    "sheep": (0, 100, 0),       # 深绿色
    "bench": (210, 105, 30),    # 巧克力色
    "dining table": (240, 128, 128), # 亮珊瑚色
    "potted plant": (138, 43, 226),  # 蓝紫色
    "stop sign": (255, 140, 0),      # 深橙色
    "sandwich": (75, 0, 130),       # 靛蓝色
    "lamp": (43, 54, 130),          # 深蓝色
    "tree": (123, 255, 130)         # 浅绿
}

class DualHashColor:
    def __init__(self):
        self.cache = {}

    def get_color(self, track_id):
        if track_id not in self.cache:
            track_str = str(track_id).encode('utf-8')
            hash1 = xxhash.xxh64(track_str).intdigest()
            hash2 = hash(track_id) & 0xFFFFFFFF
            mixed_hash = (hash1 ^ (hash2 << 32)) % 0xFFFFFF
            b, g, r = (mixed_hash >> 16) & 0xFF, (mixed_hash >> 8) & 0xFF, mixed_hash & 0xFF
            self.cache[track_id] = (b, g, r)
        return self.cache[track_id]

class ImageProcessorNode:
    def __init__(self, topic, model_path):
        # 初始化ROS订阅者和发布者
        self.image_sub = rospy.Subscriber(topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.image_pub = rospy.Publisher("/setic/image_raw", Image, queue_size=10)

        # 初始化模型和其他组件
        self.model = YOLO(model_path).cuda()
        self.tracker = DeepSort(max_age=50, n_init=3)  # 调整跟踪参数
        self.color_allocator = DualHashColor()
        self.bridge = CvBridge()

        # 跟踪 ID 和类别的映射表
        self.track_id_to_class = {}
        self.track_id_to_color = {}

        # 使用线程池加速处理
        self.executor = ThreadPoolExecutor(max_workers=2)

        # 统计处理时间
        self.total_frames = 0
        self.total_time = 0.0

    def calculate_iou(self, box1, box2):
        """计算两个矩形框的 IOU"""
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        xi1 = max(x1, x1g)
        yi1 = max(y1, y1g)
        xi2 = min(x2, x2g)
        yi2 = min(y2, y2g)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2g - x1g) * (y2g - y1g)

        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def process_frame(self, frame):
        """处理单帧图像"""
        start_time = time.time()

        # YOLO检测
        results = self.model.predict(frame, conf=0.4, iou=0.5, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()
        masks = results[0].masks.data.cpu().numpy() if results[0].masks else []
        cls_names = [results[0].names[int(cls_id)] for cls_id in detections[:, 5]]

        # DeepSORT跟踪
        tracks_data = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls_id = det
            if conf < 0.4:  # 过滤低置信度的检测
                continue
            tracks_data.append(([x1, y1, x2-x1, y2-y1], conf, cls_names[i]))

        tracks = self.tracker.update_tracks(tracks_data, frame=frame)

        # 生成掩码帧
        mask_frame = np.zeros_like(frame)
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            cls_name = track.det_class if hasattr(track, 'det_class') else "unknown"

            # 检查并更新类别一致性
            if track_id in self.track_id_to_class:
                if cls_name != self.track_id_to_class[track_id]:
                    rospy.logwarn(f"Track ID {track_id} class changed from {self.track_id_to_class[track_id]} to {cls_name}")
                cls_name = self.track_id_to_class[track_id]  # 使用历史类别
            else:
                self.track_id_to_class[track_id] = cls_name

            # 为每个 track_id 分配固定颜色
            if track_id not in self.track_id_to_color:
                self.track_id_to_color[track_id] = self.color_allocator.get_color(track_id)
            color = self.track_id_to_color[track_id]

            ltrb = track.to_ltrb()

            # IOU匹配
            matched_idx = None
            max_iou = 0.0
            for i, det in enumerate(detections):
                det_box = det[:4]
                iou = self.calculate_iou(ltrb, det_box)
                if iou > max_iou and iou > 0.5:
                    max_iou = iou
                    matched_idx = i

            # 颜色分配
            if matched_idx is not None and matched_idx < len(masks):
                mask = masks[matched_idx]
                if mask is not None and mask.any():
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    mask_frame[mask > 0.5] = color

        # 检查处理时间
        processing_time = time.time() - start_time
        self.total_frames += 1
        self.total_time += processing_time

        rospy.loginfo(f"Processed frame in {processing_time:.2f}s")
        return mask_frame, processing_time

    def image_callback(self, msg):
        """ROS图像回调函数"""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting image: {str(e)}")
            return

        # 使用线程池处理图像
        future = self.executor.submit(self.process_frame, frame)
        mask_frame, processing_time = future.result()

        # 丢弃处理时间超过 0.1 秒的帧
        if processing_time > 0.1:
            rospy.logwarn(f"Frame processing time {processing_time:.2f}s exceeds 0.1s. Dropping frame.")
            return

        # 发布处理后的图像
        try:
            ros_image = self.bridge.cv2_to_imgmsg(mask_frame, encoding="bgr8")
            ros_image.header.stamp = msg.header.stamp
            self.image_pub.publish(ros_image)
        except Exception as e:
            rospy.logerr(f"Error publishing image: {str(e)}")

    def print_statistics(self):
        """打印统计信息"""
        if self.total_frames > 0:
            avg_time = self.total_time / self.total_frames
            rospy.loginfo(f"Processed {self.total_frames} frames. Average processing time: {avg_time:.2f}s")

if __name__ == "__main__":
    rospy.init_node('semantic_image_processor')

    # 配置参数（根据实际情况修改）
    TOPIC = "/left_camera/image"
    MODEL_PATH = "/home/ao/yolov11/models/yolo11x-seg.pt"

    processor_node = ImageProcessorNode(TOPIC, MODEL_PATH)

    rospy.loginfo("Semantic Image Processor Node is running...")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        processor_node.print_statistics()