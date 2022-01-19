from pathlib import Path
import sys
import cv2
import argparse
import depthai as dai
import numpy as np
import time
from label_classes import COCO_CLASSES


class yolo_fastestv2():
    def __init__(self,input_shape=352, obj_threshold=0.3,prob_threshold=0.4, iou_threshold=0.3):
        self.num_classes = len(COCO_CLASSES)
        self.input_shape = (input_shape, input_shape)
        self.stride = [16, 32]
        self.anchor_num = 3
        self.anchors = np.array([12.64, 19.39, 37.88, 51.48, 55.71, 138.31, 126.91, 78.23, 131.57, 214.55, 279.92, 258.87],
                           dtype=np.float32).reshape(len(self.stride), self.anchor_num, 2)
        self.inpWidth = 352
        self.inpHeight = 352
        self.confThreshold = prob_threshold
        self.nmsThreshold = iou_threshold
        self.objThreshold = obj_threshold

    def to_tensor_result(self,packet):
        return {
            name: np.array(packet.getLayerFp16(name))
            for name in [tensor.name for tensor in packet.getRaw().tensors]
        }

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.	
        cv2.rectangle(frame, (left, top), (right, bottom), (255,255, 151), thickness=2)

        label = '%.2f' % conf
        #label = '%s:%s' % (self.classes[classId], label)
        label = '%s' % (COCO_CLASSES[classId])
        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 192, 203), thickness=2)
        return frame

    def postprocess(self,frame,outs):
        frameHeight = self.inpHeight
        frameWidth = self.inpWidth
        ratioh, ratiow = frameHeight / self.inpHeight, frameWidth / self.inpWidth
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for detection in outs:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > self.confThreshold and detection[4] > self.objThreshold:
                center_x = int(detection[0] * ratiow)
                center_y = int(detection[1] * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                # confidences.append(float(confidence))
                confidences.append(float(confidence*detection[4]))
                boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
                    i = i[0]
                    box = boxes[i]
                    left = box[0]
                    top = box[1]
                    width = box[2]
                    height = box[3]
                    frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
        return frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--imgpath', type=str, default='imgs/bus.jpg', help="image path")
    parser.add_argument('--input_shape', default=352, type=int, help='input image shape')
    parser.add_argument('--modelpath', type=str, default='./model/yolofastest_v2_352x352_openvino_2021.4_6shave.blob',help="filepath")
    parser.add_argument('--objThreshold', default=0.2, type=float, help='object confidence')
    parser.add_argument('--confThreshold', default=0.2, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.4, type=float, help='nms iou thresh')
    args = parser.parse_args()
    model_path = args.modelpath 

    model = yolo_fastestv2(input_shape=args.input_shape,obj_threshold=args.objThreshold, prob_threshold=args.confThreshold, iou_threshold=args.nmsThreshold)

    # Get argument first
    nnPath = str((Path(__file__).parent / Path(model_path)).resolve().absolute())

    if not Path(nnPath).exists():
        import sys
        raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

    syncNN = True

    # 开始定义管道
    pipeline = dai.Pipeline()

    # 创建彩色相机流
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(args.input_shape,args.input_shape)
    cam_rgb.setInterleaved(False)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # 创建输出
    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("preview")
    # 特定于网络的设置
    detectionNetwork = pipeline.createNeuralNetwork()
    detectionNetwork.setBlobPath(nnPath)
    detectionNetwork.input.setBlocking(False)
    #将视频流导入模型流
    cam_rgb.preview.link(detectionNetwork.input)
    detectionNetwork.passthrough.link(xout_rgb.input)
    nn_xout = pipeline.createXLinkOut()
    nn_xout.setStreamName("detections")
    detectionNetwork.out.link(nn_xout.input)

    # 管道已定义，现在设备已连接到管道
    with dai.Device(pipeline) as device:
        # 启动管道
        device.startPipeline()
        # 输出队列将用于从上面定义的输出中获取rgb帧和nn数据
        q_rgb = device.getOutputQueue(name="preview", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="detections",maxSize=4, blocking=False)
        frame = None

        while True:
            s = time.time()
            if(syncNN):
                in_rgb = q_rgb.get()
                in_nn = q_nn.get()
            else:
                in_rgb = q_rgb.tryGet()
                in_nn = q_nn.tryGet()

            out =  model.to_tensor_result(in_nn)
            #801为模型输出节点
            outs =out['801'].reshape((int(((out['801']).shape)[0]/95),95))
            outputs = np.zeros((outs.shape[0]* model.anchor_num, 5 + model.num_classes))
            row_ind = 0
            for i in range(len( model.stride)):
                h, w = int(model.inpHeight /  model.stride[i]), int(model.inpWidth /  model.stride[i])
                length = int(h * w)
                grid =  model._make_grid(w, h)
                for j in range(model.anchor_num):
                    top = row_ind+j*length
                    left = 4*j
                    outputs[top:top + length, 0:2] = (outs[row_ind:row_ind + length, left:left+2] * 2. - 0.5 + grid) * int(model.stride[i])
                    outputs[top:top + length, 2:4] = (outs[row_ind:row_ind + length, left+2:left+4] * 2) ** 2 * np.repeat(model.anchors[i, j, :].reshape(1,-1), h * w, axis=0)
                    outputs[top:top + length, 4] = outs[row_ind:row_ind + length, 4* model.anchor_num+j]
                    outputs[top:top + length, 5:] = outs[row_ind:row_ind + length, 5* model.anchor_num:]
                row_ind += length

            if in_rgb is not None:
                # 如果来自RGB相机的数据不为空，则将1D数据转换为HxWxC帧
                shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
                frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
                frame = np.ascontiguousarray(frame)
            #后处理
            frame = model.postprocess(frame,outputs)

            fps = int(1/(time.time()-s))
            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))
            cv2.imshow("rgb", frame)

            if cv2.waitKey(1) == ord('q'):
                break
            