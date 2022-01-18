from pathlib import Path
import sys
import cv2
import argparse
import depthai as dai
import numpy as np
import time


class Ultra_net():
    def to_tensor_result(self,packet):
        return {
            name: np.array(packet.getLayerFp16(name))
            for name in [tensor.name for tensor in packet.getRaw().tensors]
        }
    
    def area_of(self,left_top, right_bottom):
        """Compute the areas of rectangles given two corners.

        Args:
            left_top (N, 2): left top corner.
            right_bottom (N, 2): right bottom corner.

        Returns:
            area (N): return the area.
        """
        hw = np.clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]

    def iou_of(self,boxes0, boxes1, eps=1e-5):
        """Return intersection-over-union (Jaccard index) of boxes.

        Args:
            boxes0 (N, 4): ground truth boxes.
            boxes1 (N or 1, 4): predicted boxes.
            eps: a small number to avoid 0 as denominator.
        Returns:
            iou (N): IoU values.
        """
        overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

        overlap_area = self.area_of(overlap_left_top, overlap_right_bottom)
        area0 =  self.area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 =  self.area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)


    def hard_nms(self,box_scores, iou_threshold, top_k=-1, candidate_size=200):
        """
        Args:
            box_scores (N, 5): boxes in corner-form and probabilities.
            iou_threshold: intersection over union threshold.
            top_k: keep top_k results. If k <= 0, keep all the results.
            candidate_size: only consider the candidates with the highest scores.
        Returns:
            picked: a list of indexes of the kept boxes
        """
        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        picked = []
        # _, indexes = scores.sort(descending=True)
        indexes = np.argsort(scores)
        # indexes = indexes[:candidate_size]
        indexes = indexes[-candidate_size:]
        while len(indexes) > 0:
            # current = indexes[0]
            current = indexes[-1]
            picked.append(current)
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            # indexes = indexes[1:]
            indexes = indexes[:-1]
            rest_boxes = boxes[indexes, :]
            iou = self.iou_of(
                rest_boxes,
                np.expand_dims(current_box, axis=0),
            )
            indexes = indexes[iou <= iou_threshold]

        return box_scores[picked, :]

    def predict(self,width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = self.hard_nms(box_probs,
                                        iou_threshold=iou_threshold,
                                        top_k=top_k,
                                        )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_shape', default=320, type=int, help='input image shape')
    parser.add_argument('--modelpath', type=str, default='./model/Mb_Tiny_RFB_FD_train_input_320.blob',help="filepath")
    parser.add_argument('--confThreshold', default=0.4, type=float, help='class confidence')
    args = parser.parse_args()
    model_path = args.modelpath 
    
    fd_net = Ultra_net()

    # Get argument first
    nnPath = str((Path(__file__).parent / Path(model_path)).resolve().absolute())

    if not Path(nnPath).exists():
        import sys
        raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')
    
    class_names = ['BACKGROUND','face']
    if args.input_shape==320:
        shape = (320,240)
        feature_shape = 4420
    elif args.input_shape==640:
        shape = (640,480)
        feature_shape = 17640
    #COCO
    syncNN = True

    # 开始定义管道
    pipeline = dai.Pipeline()

    # 创建彩色相机流
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(shape[0],shape[1])
    cam_rgb.setInterleaved(False)
    # cam_rgb.setFps(30)
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
    detectionNetwork.setNumPoolFrames(4)
    detectionNetwork.input.setBlocking(False)
    detectionNetwork.setNumInferenceThreads(2)
    
    #将视频流导入模型流
    cam_rgb.preview.link(detectionNetwork.input)
    detectionNetwork.passthrough.link(xout_rgb.input)
    nn_xout = pipeline.createXLinkOut()
    nn_xout.setStreamName("detections")
    detectionNetwork.out.link(nn_xout.input)
    cam_rgb.setFps(30)

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

            if in_rgb is not None:
                # 如果来自RGB相机的数据不为空，则将1D数据转换为HxWxC帧
                shapes = (3, in_rgb.getHeight(), in_rgb.getWidth())
                frame = in_rgb.getData().reshape(shapes).transpose(1, 2, 0).astype(np.uint8)
                frame = np.ascontiguousarray(frame)

            out = fd_net.to_tensor_result(in_nn)
            confidences = out['scores'].reshape(1,feature_shape,2)
            boxes = out['boxes'].reshape(1,feature_shape,4)
            #后处理
            boxes, labels, probs = fd_net.predict(shape[0],shape[1], confidences, boxes, args.confThreshold)
            
            for i in range(boxes.shape[0]):
                box = boxes[i, :]
                label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
                x = box[0]
                y = box[1]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 3)

            fps = int(1/(time.time()-s))
            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))
            cv2.imshow("rgb",frame)
            if cv2.waitKey(1) == ord('q'):
                break
            