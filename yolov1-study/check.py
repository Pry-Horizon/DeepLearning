import os
root = "E:/DeepLearning/yolov1-study/data"
img_path = os.path.join(root, "VOC2007", "JPEGImages", "000005.jpg")
xml_path = os.path.join(root, "VOC2007", "Annotations", "000005.xml")
print("图片是否存在：", os.path.exists(img_path))
print("标注是否存在：", os.path.exists(xml_path))
