# Environment
```bash
pytorch >= 1.7.0    # export.py  need
opencv-python >= 4.5.2.52
depthai   2.13.3.0
```

# RUN
```bash
1.修改Yolo-FastestV2/model/detector.py 模型输出或者替换本项目的detector.py文件
2.转换模型到onnx，官方自带转换脚本pytorch2onnx.py（opset 11 ）。
3.使用在线转换工具 http://blobconverter.luxonis.com/ 
```

# Reference
```bash
https://github.com/dog-qiuqiu/Yolo-FastestV2
https://github.com/hpc203/yolo-fastestv2-opencv
``
