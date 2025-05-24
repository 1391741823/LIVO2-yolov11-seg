from ultralytics import YOLO
import os
"""
这里是yolo11n-seg的预测代码

"""
# Load a model
#model = YOLO("yolo11n-seg.yaml")  # build a new model from YAML
model = YOLO("/home/ao/yolov11/model/yolo11x.pt",  
             task='segment'
             #conf=  0.8
             )  # load a pretrained model (recommended for training)
#model = YOLO("yolo11n-seg.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
#results = model.train(data="coco8-seg.yaml", epochs=100, imgsz=640)
#model.predict(source='',save=True,show=True)
results = model.predict(source='/home/ao/yolov11/ultralytics/source/2.mp4',
                        save=True,
                        show=True,
                        stream=True,

                        ##修改相关参数
                        conf=0.6,
                        iou=0.6
                        #
                        )
for r in results:
     boxes = r.boxes  # Boxes object for bbox outputs
     masks = r.masks  # Masks object for segment masks outputs
     probs = r.probs  # Class probabilities for classification outputs
     print(boxes)

# 确保保存为 .mp4 格式
output_dir = '/home/ao/yolov11/ultralytics/runs/predict'  # 默认保存目录
output_file = os.path.join(output_dir, 'predictions.mp4')  # 指定保存文件名
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # 如果目录不存在，则创建

