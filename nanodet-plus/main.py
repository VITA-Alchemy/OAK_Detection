from pathlib import Path
import sys
import cv2
import argparse
import depthai as dai
import numpy as np
import time
import math
from label_classes import COCO_CLASSES


class my_nanodet():
    def __init__(self,input_shape=416, prob_threshold=0.4, iou_threshold=0.3):
        self.num_classes = len(COCO_CLASSES)
        self.strides = (8, 16, 32, 64)
        self.input_shape = (input_shape, input_shape)
        self.reg_max = 7
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.project = np.arange(self.reg_max + 1)
        self.keep_ratio = False
        self.mlvl_anchors = []
        for i in range(len(self.strides)):
            anchors = self.make_grid(
                (math.ceil(self.input_shape[0] / self.strides[i]), math.ceil(self.input_shape[1] / self.strides[i])),
                self.strides[i])
            self.mlvl_anchors.append(anchors)


    def to_tensor_result(self,packet):
        return {'output':np.array(packet.getLayerFp16('output'))}
        return {
            name: np.array(packet.getLayerFp16(name))
            for name in [tensor.name for tensor in packet.getRaw().tensors]
        }

    def make_grid(self,featmap_size, stride):
            feat_h, feat_w = featmap_size
            shift_x = np.arange(0, feat_w) * stride
            shift_y = np.arange(0, feat_h) * stride
            xv, yv = np.meshgrid(shift_x, shift_y)
            xv = xv.flatten()
            yv = yv.flatten()
            return np.stack((xv, yv), axis=-1)

    def softmax(self,x, axis=1):
            x_exp = np.exp(x)
            # 如果是列向量，则axis=0
            x_sum = np.sum(x_exp, axis=axis, keepdims=True)
            s = x_exp / x_sum
            return s

    def distance2bbox(self,points, distance, max_shape=None):
            x1 = points[:, 0] - distance[:, 0]
            y1 = points[:, 1] - distance[:, 1]
            x2 = points[:, 0] + distance[:, 2]
            y2 = points[:, 1] + distance[:, 3]
            if max_shape is not None:
                x1 = np.clip(x1, 0, max_shape[1])
                y1 = np.clip(y1, 0, max_shape[0])
                x2 = np.clip(x2, 0, max_shape[1])
                y2 = np.clip(y2, 0, max_shape[0])
            return np.stack([x1, y1, x2, y2], axis=-1)

    def post_process(self,preds, scale_factor=1, rescale=False):
        mlvl_bboxes = []
        mlvl_scores = []
        ind = 0
        for i in range(len(self.strides)):
                anchors = self.make_grid(
                    (math.ceil(self.input_shape[0] / self.strides[i]), math.ceil(self.input_shape[1] / self.strides[i])),
                    self.strides[i])
                self.mlvl_anchors.append(anchors)
        for stride, anchors in zip(self.strides,self.mlvl_anchors):
            cls_score, bbox_pred = preds[ind:(ind + anchors.shape[0]), :self.num_classes], preds[ind:(ind + anchors.shape[0]),self.num_classes:]
            ind += anchors.shape[0]
            bbox_pred =self.softmax(bbox_pred.reshape(-1, self.reg_max + 1), axis=1)
            # bbox_pred = np.sum(bbox_pred * np.expand_dims(self.project, axis=0), axis=1).reshape((-1, 4))
            bbox_pred = np.dot(bbox_pred,self.project).reshape(-1, 4)
            bbox_pred *= stride
            
            # nms_pre = cfg.get('nms_pre', -1)
            nms_pre = 1000
            if nms_pre > 0 and cls_score.shape[0] > nms_pre:
                max_scores = cls_score.max(axis=1)
                topk_inds = max_scores.argsort()[::-1][0:nms_pre]
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                cls_score = cls_score[topk_inds, :]
            
            bboxes = self.distance2bbox(anchors, bbox_pred, max_shape=self.input_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(cls_score)
        
        mlvl_bboxes = np.concatenate(mlvl_bboxes, axis=0)
        if rescale:
            mlvl_bboxes /= scale_factor
        mlvl_scores = np.concatenate(mlvl_scores, axis=0)
        
        bboxes_wh = mlvl_bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes_wh[:, 2:4] - bboxes_wh[:, 0:2]  ####xywh
        classIds = np.argmax(mlvl_scores, axis=1)
        confidences = np.max(mlvl_scores, axis=1)  ####max_class_confidence
        
        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.prob_threshold,self.iou_threshold)
       
        if len(indices) > 0:
            mlvl_bboxes = mlvl_bboxes[indices]
            confidences = confidences[indices]
            classIds = classIds[indices]
            return mlvl_bboxes, confidences, classIds
        else:
            print('nothing detect')
            return np.array([]), np.array([]), np.array([])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--imgpath', type=str, default='imgs/bus.jpg', help="image path")
    parser.add_argument('--input_shape', default=416, type=int, help='input image shape')
    parser.add_argument('--modelpath', type=str, default='./model/nanodet-plus-m-1.5x_416_openvino_2021.4_6shave.blob',help="filepath")
    parser.add_argument('--confThreshold', default=0.4, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.6, type=float, help='nms iou thresh')
    args = parser.parse_args()
    model_path = args.modelpath 

    nanodet = my_nanodet(input_shape=args.input_shape, prob_threshold=args.confThreshold, iou_threshold=args.nmsThreshold)

    # Get argument first
    nnPath = str((Path(__file__).parent / Path(model_path)).resolve().absolute())

    if not Path(nnPath).exists():
        import sys
        raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

    #COCO
    labelMap = COCO_CLASSES
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
    cam_rgb.getFps()


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
        q_rgb = device.getOutputQueue(name="preview", maxSize=1, blocking=True)
        q_nn = device.getOutputQueue(name="detections",maxSize=1, blocking=True)
        frame = None
       
        while True:
            s = time.time()
            if(syncNN):
                in_rgb = q_rgb.get()
                in_nn = q_nn.get()
            else:
                in_rgb = q_rgb.tryGet()
                in_nn = q_nn.tryGet()

            out = nanodet.to_tensor_result(in_nn)
            a = (out['output']).shape
            out =out['output'].reshape((int(a[0]/112),112))

            #后处理
            det_bboxes, det_conf, det_classid = nanodet.post_process(out)
        
            if in_rgb is not None:
                # 如果来自RGB相机的数据不为空，则将1D数据转换为HxWxC帧
                shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
                frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
                frame = np.ascontiguousarray(frame)
           
            if frame is not None:
                top, left = 0,0
                # 如果图像不为空，请在其上绘制边框并显示图像
                height = frame.shape[0]
                width  = frame.shape[1]
                ratioh, ratiow = frame.shape[0] / height, frame.shape[1] / width 
                for i in range(det_bboxes.shape[0]):
                    xmin, ymin, xmax, ymax = max(int((det_bboxes[i, 0] - left) * ratiow), 0), max(
                        int((det_bboxes[i, 1] - top) * ratioh), 0), min(
                        int((det_bboxes[i, 2] - left) * ratiow), frame.shape[1]), min(int((det_bboxes[i, 3] - top) * ratioh),
                                                                                    frame.shape[0])
                    cv2.rectangle(frame, (xmin,ymin),(xmax,ymax), (0, 0, 255), thickness=2)
                    label = '%.2f' % det_conf[i]
                    label = '%s:%s' % (labelMap[det_classid[i]], label)
                    # Display the label at the top of the bounding box
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    top = max(top, labelSize[1])
                    # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
                    cv2.putText(frame, label, (xmin,ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), thickness=1)
        
            print(time.time()-s)
            fps = int(1/(time.time()-s))
            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))
            cv2.imshow("rgb", frame)

            if cv2.waitKey(1) == ord('q'):
                break
            