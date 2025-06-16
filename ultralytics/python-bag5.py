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
    "tree": (123, 255, 130),        # 浅绿
    "default": (128, 128, 128)      # 默认灰色
}

class DualHashColor:
    def __init__(self):
        self.cache = {}

    def get_color_by_class(self, class_name):
        """根据类别名称获取颜色，确保同一类别显示相同颜色"""
        if class_name in CLASS_COLORS:
            return CLASS_COLORS[class_name]
        else:
            # 如果类别不在预定义中，返回默认颜色并记录警告
            rospy.logwarn(f"未定义的类别: {class_name}，使用默认颜色")
            return CLASS_COLORS["default"]

    def get_color(self, track_id):
        """保留原有功能用于兼容性"""
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
        self.tracker = DeepSort(max_age=30, n_init=2)  # 优化跟踪参数：减少最大年龄，减少初始化帧数
        self.color_allocator = DualHashColor()
        self.bridge = CvBridge()

        # 跟踪 ID 和类别的映射表
        self.track_id_to_class = {}
        
        # 类别统计
        self.class_detection_count = {}

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
        mask_frame = np.zeros_like(frame)  # 预先创建空白帧

        try:
            # YOLO检测 - 提高置信度阈值以获得更可靠的检测
            results = self.model.predict(frame, conf=0.75, iou=0.4, verbose=False)  # 提高置信度阈值到0.6
            
            # 检查是否超时
            if time.time() - start_time > 0.1:
                raise TimeoutError("Processing timeout")

            detections = results[0].boxes.data.cpu().numpy()
            masks = results[0].masks.data.cpu().numpy() if results[0].masks else []
            cls_names = [results[0].names[int(cls_id)] for cls_id in detections[:, 5]]

            # 过滤只保留CLASS_COLORS中定义的类别，并按类别分组保留最高置信度
            class_best_detections = {}  # 每个类别的最佳检测结果
            
            for i, (det, cls_name) in enumerate(zip(detections, cls_names)):
                if cls_name in CLASS_COLORS and det[4] >= 0.6:  # 只保留预定义类别且置信度>=0.6
                    if cls_name not in class_best_detections:
                        # 第一次遇到这个类别
                        mask = masks[i] if i < len(masks) else None
                        class_best_detections[cls_name] = {
                            'detection': det,
                            'mask': mask,
                            'confidence': det[4]
                        }
                    else:
                        # 如果当前检测的置信度更高，则替换
                        if det[4] > class_best_detections[cls_name]['confidence']:
                            mask = masks[i] if i < len(masks) else None
                            class_best_detections[cls_name] = {
                                'detection': det,
                                'mask': mask,
                                'confidence': det[4]
                            }
            
            # 按置信度排序，选择前3个不同类别的物体
            sorted_classes = sorted(class_best_detections.items(), 
                                  key=lambda x: x[1]['confidence'], reverse=True)
            
            # 只保留前3个不同类别
            top_3_classes = sorted_classes[:3]
            
            # 重新组织数据
            if len(top_3_classes) > 0:
                detections = np.array([item[1]['detection'] for item in top_3_classes])
                cls_names = [item[0] for item in top_3_classes]
                masks = [item[1]['mask'] for item in top_3_classes if item[1]['mask'] is not None]
            else:
                detections = np.array([])
                cls_names = []
                masks = []

            rospy.loginfo(f"过滤后保留 {len(detections)} 个不同类别的最佳检测结果（高置信度≥0.6）")
            
            # 输出检测到的物体详细信息
            if len(detections) > 0:
                rospy.loginfo("检测到的不同类别物体详情（每类最佳）：")
                for i, (det, cls_name) in enumerate(zip(detections, cls_names)):
                    conf = det[4]
                    color = self.color_allocator.get_color_by_class(cls_name)
                    rospy.loginfo(f"  {i+1}. {cls_name} (最高置信度: {conf:.3f}, 颜色: RGB{color})")
            else:
                rospy.loginfo("未检测到任何符合条件的物体（置信度≥0.6的预定义类别）")

            # DeepSORT跟踪 - 调整参数
            tracks_data = []
            for i, det in enumerate(detections):
                # 检查是否超时
                if time.time() - start_time > 0.1:
                    raise TimeoutError("Processing timeout")

                x1, y1, x2, y2, conf, cls_id = det
                if conf < 0.6:  # 提高过滤阈值与检测阈值一致
                    continue
                tracks_data.append(([x1, y1, x2-x1, y2-y1], conf, cls_names[i]))

            tracks = self.tracker.update_tracks(tracks_data, frame=frame)

            # 生成掩码帧
            for track in tracks:
                # 检查是否超时
                if time.time() - start_time > 0.1:
                    raise TimeoutError("Processing timeout")

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

                # 更新类别统计
                if cls_name not in self.class_detection_count:
                    self.class_detection_count[cls_name] = 0
                self.class_detection_count[cls_name] += 1

                color = self.color_allocator.get_color_by_class(cls_name)

                ltrb = track.to_ltrb()

                # IOU匹配 - 降低阈值提高匹配率
                matched_idx = None
                max_iou = 0.0
                for i, det in enumerate(detections):
                    det_box = det[:4]
                    iou = self.calculate_iou(ltrb, det_box)
                    if iou > max_iou and iou > 0.3:  # 降低IOU阈值从0.5到0.3
                        max_iou = iou
                        matched_idx = i

                # 颜色分配 - 确保同类别物体使用相同颜色
                if matched_idx is not None and matched_idx < len(masks):
                    mask = masks[matched_idx]
                    if mask is not None and mask.any():
                        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                        mask_frame[mask > 0.5] = color
                        conf = detections[matched_idx][4] if matched_idx < len(detections) else 0.0
                        rospy.logdebug(f"应用 {cls_name} 掩码，颜色: RGB{color}，跟踪ID: {track_id}，置信度: {conf:.2f}")  # 详细调试信息

        except TimeoutError as e:
            rospy.logwarn(f"Processing interrupted: {str(e)}")
            mask_frame = np.zeros_like(frame)  # 超时时返回空白帧
        except Exception as e:
            rospy.logerr(f"Error processing frame: {str(e)}")
            mask_frame = np.zeros_like(frame)  # 发生错误时返回空白帧

        # 检查处理时间
        processing_time = time.time() - start_time
        self.total_frames += 1
        self.total_time += processing_time

        if processing_time > 0.1:
            rospy.logwarn(f"帧处理时间 {processing_time:.2f}秒 超过 0.1秒")
            mask_frame = np.zeros_like(frame)  # 处理时间超过阈值时返回空白帧
        else:
            # 显示检测到的类别数量
            detected_classes = set(self.track_id_to_class.values()) if hasattr(self, 'track_id_to_class') else set()
            rospy.loginfo(f"处理帧耗时 {processing_time:.2f}秒，检测到 {len(detected_classes)} 种类别")

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

        # 如果处理时间超过 0.1 秒，输出空白帧
        if processing_time > 0.09999999:
            rospy.logwarn(f"帧处理时间 {processing_time:.2f}秒 超过 0.1秒。输出空白帧。")
            mask_frame = np.zeros_like(frame)

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
            rospy.loginfo(f"处理了 {self.total_frames} 帧。平均处理时间：{avg_time:.2f}秒")
            
            # 打印类别检测统计
            rospy.loginfo("类别检测统计：")
            for cls_name, count in sorted(self.class_detection_count.items(), key=lambda x: x[1], reverse=True):
                color = self.color_allocator.get_color_by_class(cls_name)
                rospy.loginfo(f"  {cls_name}: {count} 次检测，颜色: RGB{color}")
        else:
            rospy.loginfo("未处理任何帧")

if __name__ == "__main__":
    rospy.init_node('semantic_image_processor')

    # 显示支持的类别列表
    rospy.loginfo("支持的物体类别及其颜色映射：")
    for cls_name, color in CLASS_COLORS.items():
        if cls_name != "default":
            rospy.loginfo(f"  {cls_name}: RGB{color}")
    rospy.loginfo(f"总共支持 {len(CLASS_COLORS)-1} 种类别 (不包括默认颜色)")
    rospy.loginfo("检测规则：置信度≥0.6，每个类别只保留最高置信度的单个物体")
    rospy.loginfo("每次最多显示3个不同种类的物体")

    # 配置参数（根据实际情况修改）
    TOPIC = "/left_camera/image"
    MODEL_PATH = "/home/ao/yolov11/models/yolo11x-seg.pt"

    processor_node = ImageProcessorNode(TOPIC, MODEL_PATH)

    rospy.loginfo("Semantic Image Processor Node is running...")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        processor_node.print_statistics()
