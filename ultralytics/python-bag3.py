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

# --------------------- 类别颜色映射 ---------------------
CLASS_COLORS = {
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
    "sheep":          (0, 100, 0),       # 深绿色
    # ---------- 其他常用类 ----------
    "bench":         (210, 105, 30),    # 巧克力色
    "dining table":  (240, 128, 128),   # 亮珊瑚色
    "potted plant":  (138, 43, 226),    # 蓝紫色
    "stop sign":      (255, 140, 0),     # 深橙色
    "sandwich":       (75, 0, 130),       # 靛蓝色
    # ---------- 室外物体识别 ----------
    "lamp":       (43, 54, 130),
    "tree":       (123, 255, 130)
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
        self.image_sub = rospy.Subscriber(topic, Image, self.image_callback)
        self.image_pub = rospy.Publisher("/setic/image_raw", Image, queue_size=10)

        # 初始化模型和其他组件
        self.model = YOLO(model_path).cuda()
        self.tracker = DeepSort(max_age=30)
        self.color_allocator = DualHashColor()
        self.bridge = CvBridge()

    def image_callback(self, msg):
        try:
            # 将ROS图像消息转换为OpenCV图像
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting image: {str(e)}")
            return

        if frame is None:
            rospy.logwarn("Received an empty frame")
            return

        start_time = time.time()

        # YOLO检测
        results = self.model.predict(frame, conf=0.25, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()
        masks = results[0].masks.data.cpu().numpy() if results[0].masks else []
        cls_names = [results[0].names[int(cls_id)] for cls_id in detections[:, 5]]

        # DeepSORT跟踪
        tracks_data = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls_id = det
            tracks_data.append(([x1, y1, x2-x1, y2-y1], conf, cls_names[i]))

        tracks = self.tracker.update_tracks(tracks_data, frame=frame)

        # 生成掩码帧
        mask_frame = np.zeros_like(frame)
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            cls_name = track.det_class if hasattr(track, 'det_class') else "unknown"
            ltrb = track.to_ltrb()

            # IOU匹配
            matched_idx = None
            max_iou = 0.5
            for i, det in enumerate(detections):
                det_box = det[:4]
                # IOU计算逻辑保持不变...

            # 颜色分配
            if matched_idx is not None and matched_idx < len(masks):
                mask = masks[matched_idx]
                if mask is not None and mask.any():  # 确保掩码不为空且有值
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    color = CLASS_COLORS.get(cls_name.lower(), self.color_allocator.get_color(track_id))
                    mask_frame[mask > 0.5] = color
                else:
                    rospy.logwarn(f"Empty mask for track ID {track_id} and class {cls_name}")
            else:
                rospy.logwarn(f"No matched mask for track ID {track_id} and class {cls_name}")

        # 检查生成的掩码帧是否为空
        if not mask_frame.any():
            rospy.logwarn("Generated mask frame is empty. Check detection and mask logic.")

        # 发布处理后的图像
        try:
            ros_image = self.bridge.cv2_to_imgmsg(mask_frame, encoding="bgr8")
            ros_image.header.stamp = msg.header.stamp
            self.image_pub.publish(ros_image)
        except Exception as e:
            rospy.logerr(f"Error publishing image: {str(e)}")

        rospy.loginfo(f"Processed frame in {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    rospy.init_node('semantic_image_processor')

    # 配置参数（根据实际情况修改）
    TOPIC = "/left_camera/image"
    MODEL_PATH = "/home/ao/yolov11/models/yolo11n-seg.pt"

    processor_node = ImageProcessorNode(TOPIC, MODEL_PATH)

    rospy.loginfo("Semantic Image Processor Node is running...")
    rospy.spin()