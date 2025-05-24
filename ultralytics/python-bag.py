import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import xxhash
import rosbag
from cv_bridge import CvBridge
# 新增ROS依赖
import rospy
from sensor_msgs.msg import Image

# --------------------- 1. 初始化ROS节点 ---------------------
rospy.init_node('python_bag_processor', anonymous=True)
# --------------------- 2. 类别颜色映射（按名称） ---------------------
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


class BagProcessor:
    def __init__(self, bag_path, topic, output_folder, model_path):
        """
        初始化 BagProcessor 类
        :param bag_path: .bag 文件路径
        :param topic: 图像数据的 topic
        :param output_folder: 输出图片文件夹路径
        :param model_path: YOLO 模型路径
        """
        self.setic_pub = rospy.Publisher("/setic/image_raw", Image, queue_size=10)

        self.bag_path = bag_path
        self.topic = topic
        self.output_folder = output_folder
        self.model_path = model_path
        self.model = YOLO(model_path).cuda()
        self.tracker = DeepSort(max_age=30)
        self.color_allocator = DualHashColor()
        self.bridge = CvBridge()

        # 确保输出文件夹存在
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def process_bag(self):
        """
        处理 .bag 文件中的图像数据，并将结果保存到输出文件夹
        """
        print(f"正在处理 .bag 文件: {self.bag_path}")
        bag = rosbag.Bag(self.bag_path, 'r')
        frame_count = 0

        for topic, msg, t in bag.read_messages(topics=[self.topic]):
            start_time = time.time()  # 开始计时

            # 将 ROS 图像消息转换为 OpenCV 图像
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if frame is None:
                print(f"无法读取帧: {t}")
                continue

            h, w, _ = frame.shape
            output_image_path = os.path.join(self.output_folder, f"frame_{frame_count:06d}.jpg")

            # A. YOLO检测并获取类别名称
            results = self.model.predict(frame, conf=0.25, verbose=False)  # 设置置信度
            detections = results[0].boxes.data.cpu().numpy()
            masks = results[0].masks.data.cpu().numpy() if results[0].masks else []
            cls_names = [results[0].names[int(cls_id)] for cls_id in detections[:, 5]]  # 类别名称列表

            # B. 准备跟踪数据（传递类别名称）
            tracks_data = []
            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf, cls_id = det
                tracks_data.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_names[i]))  # 传入类别名称

            # C. DeepSORT跟踪
            tracks = self.tracker.update_tracks(tracks_data, frame=frame)

            # D. 生成掩码帧
            mask_frame = np.zeros((h, w, 3), dtype=np.uint8)
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                cls_name = track.det_class if hasattr(track, 'det_class') else "unknown"

                ltrb = track.to_ltrb()  # [x1, y1, x2, y2]

                # E. IOU匹配检测框
                matched_idx = None
                max_iou = 0.5
                for i, det in enumerate(detections):
                    det_box = det[:4]
                    xx1 = max(det_box[0], ltrb[0])
                    yy1 = max(det_box[1], ltrb[1])
                    xx2 = min(det_box[2], ltrb[2])
                    yy2 = min(det_box[3], ltrb[3])
                    inter_area = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                    det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
                    track_area = (ltrb[2] - ltrb[0]) * (ltrb[3] - ltrb[1])
                    union_area = det_area + track_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    if iou > max_iou:
                        max_iou = iou
                        matched_idx = i

                # F. 颜色分配
                if matched_idx is not None and matched_idx < len(masks):
                    mask = cv2.resize(masks[matched_idx], (w, h), cv2.INTER_NEAREST)
                    color = CLASS_COLORS.get(cls_name.lower(), self.color_allocator.get_color(track_id))
                    mask_frame[mask > 0.5] = color

            # 保存处理后的图片
            # cv2.imwrite(output_image_path, mask_frame)
                        # 转换为ROS图像消息
            ros_image = self.bridge.cv2_to_imgmsg(mask_frame, encoding="bgr8")
            # 添加时间戳 (重要！)
            ros_image.header.stamp = rospy.Time.from_sec(t.to_sec())
            # 发布到ROS话题
            self.setic_pub.publish(ros_image)

            end_time = time.time()  # 结束计时
            elapsed_time = end_time - start_time  # 计算处理时间
            print(f"处理完成: {output_image_path}，耗时: {elapsed_time:.2f} 秒")
            frame_count += 1

        bag.close()


class DualHashColor:
    def __init__(self):
        self.cache = {}

    def get_color(self, track_id):
        """双哈希生成颜色（排除与预设类别颜色冲突）"""
        if track_id not in self.cache:
            track_str = str(track_id).encode('utf-8')
            hash1 = xxhash.xxh64(track_str).intdigest()
            hash2 = hash(track_id) & 0xFFFFFFFF
            mixed_hash = (hash1 ^ (hash2 << 32)) % 0xFFFFFF  # 24位颜色
            b, g, r = (mixed_hash >> 16) & 0xFF, (mixed_hash >> 8) & 0xFF, mixed_hash & 0xFF
            self.cache[track_id] = (b, g, r)
        return self.cache[track_id]


# 示例运行
if __name__ == "__main__":
    BAG_PATH = "/media/ao/jiansheng/datasets/LIVO2_dataset/Retail_Street.bag"  # .bag 文件路径
    TOPIC = "/left_camera/image"  # 图像数据的 topic
    OUTPUT_IMAGE_FOLDER = "/home/ao/yolov11/ultralytics/1_out"  # 输出图片文件夹
    MODEL_PATH = "/home/ao/yolov11/model/yolo11x-seg.pt"  # 模型路径

    bag_processor = BagProcessor(BAG_PATH, TOPIC, OUTPUT_IMAGE_FOLDER, MODEL_PATH)
    bag_processor.process_bag()


  