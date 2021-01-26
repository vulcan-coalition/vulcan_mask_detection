'''
Wrapper module for face detection api
The module comprises of three main components
   - preprocessing
   - model inferencing
   - postprocessing
'''


import torch
import torchvision
import cv2
from PIL import Image
import numpy as np
import datetime
import sys, os
from modules import FaceMask_CNN as ClassificationModel

class FaceMaskDetectionAPI:


    def __init__(self,  mask_classification_model_path,
                        model_config,
                        facenet_prototxt_path, 
                        facenet_weight_path,
                        face_detection_confidence_threshold,
                        input_size):

        self.mask_classification_model = self._get_mask_classification_model(mask_classification_model_path, model_config)
        self.facenet_model = self._get_facenet_model(facenet_prototxt_path, facenet_weight_path)
        self.face_detection_confidence_threshold = face_detection_confidence_threshold
        self.input_size = input_size
        self.pad = 0


    @staticmethod
    def _get_mask_classification_model(model_path, model_config):
        model = ClassificationModel(**model_config)
        model.eval()
        model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])
        return model


    @staticmethod
    def _get_facenet_model(prototxt_path, weight_path):
        return cv2.dnn.readNet(prototxt_path, weight_path)


    def _extract_bounding_box(self, bbox, w, h):

        (x1, y1, x2, y2) = bbox.astype("int")

        # check if it only detect eye
        width, height = x2 - x1, y2 - y1
        if width > height:
            y2 = y2 + int(2*(width - height))

        (x1, y1) = (max(0, x1 - self.pad), max(0, y1 - self.pad))
        (x2, y2) = (min(w - 1, x2 + self.pad), min(h - 1, y2 + self.pad))

        return [x1, y1, x2, y2]

    
    @staticmethod
    def _crop_image(image, bboxes):
        return [image[y1 :y2 , x1 :x2 , :] for x1, y1, x2, y2 in bboxes]


    @staticmethod
    def _softmax(outs):
        e = np.exp(outs)
        return e / np.sum(e, axis=1, keepdims=True)


    def face_detector(self, image):
        (h, w) = image.shape[:2]
        confidence_threshold = self.face_detection_confidence_threshold
    
        self.facenet_model.setInput(cv2.dnn.blobFromImage(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 1.0, (300, 300), (104.0, 177.0, 123.0)))
        detections = self.facenet_model.forward()

        # if face not found above the threshold, reduce confidence threshold
        if np.max(detections[0, 0, :, 2]) < confidence_threshold:
            confidence_threshold = 0.25
        
        bboxes = [self._extract_bounding_box(detections[0, 0, i, 3:7] * np.array([w, h, w, h]), w, h) for i in range(0, detections.shape[2]) if detections[0, 0, i, 2] > confidence_threshold]
        return bboxes


    def image_preprocessing(self, images):

        transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize([self.input_size,self.input_size]),
                    torchvision.transforms.Grayscale(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.5], [0.5])
                    ])

        # process as batch
        img_tensor = torch.stack([transforms(Image.fromarray(np.uint8(img))) for img in images])
        return img_tensor


    def preprocessing(self, image):
        bboxes = self.face_detector(image)
        face_images = self._crop_image(image, bboxes)
        if bboxes:
            prep_images = self.image_preprocessing(face_images)
        else:
            prep_images = []
        return prep_images, bboxes


    def inferencing(self, features):
        outs = self.mask_classification_model(features)
        outs = outs.detach().numpy()
        return outs


    def postprocessing(self, bboxes, outs, face_found_flag):
        '''
        prepare output in json format
        '''
        output_dict = dict()
        output_dict['results'] = dict()
        if face_found_flag:
            prob = self._softmax(outs)
            # print(prob)
            classes = np.argmax(outs, axis=1)
            for idx, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                output_dict['results'][str(idx)] = {'bbox':{'x1':int(x1), 'y1':int(y1), 'x2':int(x2), 'y2':int(y2)}, 'probability': float(prob[idx][int(classes[idx])]), 'is_mask': str(classes[idx])}
        else:
            output_dict['results'] = None
        output_dict['face_detect'] = face_found_flag
        output_dict['timestamp'] = str(datetime.datetime.now())
        output_dict['error'] = None
        return output_dict


    def run(self, image):
        features, bboxes = self.preprocessing(image)
        if bboxes != []:
            face_found_flag = True
            outs = self.inferencing(features)
        else:
            face_found_flag = False
            outs =[]
        output_dict = self.postprocessing(bboxes, outs, face_found_flag)
        return output_dict
    

