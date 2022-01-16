from pathlib import Path
import sys
import cv2
import argparse
import depthai as dai
import numpy as np
import time
from label_classes import COCO_CLASSES


class my_yolov5():
    def __init__(self,input_shape=416,  confThreshold=0.4,  nmsThreshold=0.3):
        self.num_classes = len(COCO_CLASSES)
        self.input_shape = (input_shape, input_shape)
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.stride = np.array([8., 16., 32.])
        self.no = self.num_classes + 5 
        self.colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(self.num_classes)]
        self.anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.inpWidth = input_shape
        self.inpHeight = input_shape
    
    def to_tensor_result(self,packet):
        
        return {'output':np.array(packet.getLayerFp16('output'))}

        # return {
        #     name:np.array(packet.getLayerFp16(name))
        #     for name in [tensor.name for tensor in packet.getRaw().tensors]
        # }

    def xywh2xyxy(self,x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
       # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def non_max_suppression(self,prediction, conf_thres=0.25,agnostic=False):
        xc = prediction[..., 4] > conf_thres  # candidates
        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
       
        output = [np.zeros((0, 6))] * prediction.shape[0]

        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence
            if not x.shape[0]:
                continue
            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])
            # Detections matrix nx6 (xyxy, conf, cls)
            conf = np.max(x[:, 5:], axis=1)
            j = np.argmax(x[:, 5:],axis=1)
            #转为array：  x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            re = np.array(conf.reshape(-1)> conf_thres)
            #转为维度
            conf =conf.reshape(-1,1)
            j = j.reshape(-1,1)
            #numpy的拼接
            x = np.concatenate((box,conf,j),axis=1)[re]
            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            #转为list 使用opencv自带nms
            boxes = boxes.tolist()
            scores = scores.tolist()

            i = cv2.dnn.NMSBoxes(boxes, scores, self.confThreshold, self.nmsThreshold)
            #i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            output[xi] = x[i]
        return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--imgpath', type=str, default='imgs/bus.jpg', help="image path")
    parser.add_argument('--input_shape', default=640, type=int, help='input image shape')
    parser.add_argument('--modelpath', type=str, default='./model/yolov5n_640x640_openvino_2021.4_6shave.blob',help="filepath")
    parser.add_argument('--confThreshold', default=0.4, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.4, type=float, help='nms iou thresh')
    args = parser.parse_args()
    model_path = args.modelpath 

    yolo = my_yolov5(input_shape=args.input_shape, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold)
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
    cam_rgb.setFps(30)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # 创建输出
    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("preview")
    # cam_rgb.preview.link(xout_rgb.input)
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
        q_rgb = device.getOutputQueue(name="preview", maxSize=1, blocking=False)
        q_nn = device.getOutputQueue(name="detections",maxSize=1, blocking=False)
        frame = None

        while True:
            s = time.time()
            if(syncNN):
                in_rgb = q_rgb.get()
                in_nn = q_nn.get()
            else:
                in_rgb = q_rgb.tryGet()
                in_nn = q_nn.tryGet()
            
            out = yolo.to_tensor_result(in_nn)
            a = (out['output']).shape
            out =out['output'].reshape((1,int(a[0]/yolo.no),yolo.no))
            if in_rgb is not None:
                # 如果来自RGB相机的数据不为空，则将1D数据转换为HxWxC帧
                shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
                frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
                frame = np.ascontiguousarray(frame)
            #NMS
            pred = yolo.non_max_suppression(out,yolo.confThreshold,agnostic=False)
            #draw box
            if frame is not None:
                for i in pred[0]:
                    left = int(i[0])
                    top = int(i[1])
                    width = int(i[2])
                    height = int(i[3])
                    conf = i[4]
                    classId = i[5]
                    #frame = self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)
                    cv2.rectangle(frame, (int(left), int(top)), (int(width),int(height)), yolo.colors[int(classId)], thickness=2)
                    label = '%.2f' % conf
                    label = '%s:%s' % (COCO_CLASSES[int(classId)], label)
                    # Display the label at the top of the bounding box
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    top = max(top, labelSize[1])
                    #cv2.rectangle(srcimg, (int(left), int(top - round(1.5 * labelSize[1]))), (int(left + round(1.5 * labelSize[0])), int(top + baseLine)), (255,255,255), cv2.FILLED)
                    cv2.putText(frame, label, (int(left-20),int(top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), thickness=2)
        
            fps = int(1/(time.time()-s))
            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))
            cv2.imshow("rgb", frame)

            if cv2.waitKey(1) == ord('q'):
                break
            