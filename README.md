# Face Mask Detection API by Vulcan Coalition

## Introduction

A face mask detection prototype model

The current version of the mask detection model performed two consecutive tasks, face detection, and mask classification. The face detection used the model from the Open-CV DNN module: 

https://github.com/opencv/opencv/tree/master/samples/dnn 

The mask classification model was trained using the data scraped from the internet, then performed data cleansing and data annotation. The model originally deployed on the edge device, raising the limitation in terms of computational power and memory. Therefore, we used mobilenet v2 as a backbone for our classification model. The model was trained by the transfer learning method, transferring pretrained weight on the lower layer and trained the upper layer with the new data. The model performed binary classification, learned to classify whether the person faces wearing a mask or not. 

The model optimized on the cross-entropy loss function. The ADAM optimizer was applied.

The <a href="https://fastapi.tiangolo.com"> FastAPI </a> is used as an API framework. 

The mask classification model's weights and the Open-CV face detection model and its weight can be download from the <a href="https://drive.google.com/drive/u/2/folders/1vU8-pdGlSZko7JSrou7mfbGSubhGke5q">Google Drive</a>



## API Document

To use API using Python, install `requests` package.

`pip install requests`

The API can be request from https://api.lab.ai/mask-detection. In Python file: 

```python 

 import requests

 response = requests.request('POST', "https://api.lab.ai/mask-detection" , files={'file':open(image_file, 'rb')})
 content = json.loads(response.content.decode('utf-8'))

```

API supports only jpg and png file format. The content returns from the API is as followed:

```json

{
 "results": { "0":
 {
 "bbox": {"x1": 100, "y1": 100, "x2": 200, "y2":200},
 "probability": 0.75, 
 "is_mask": "1"
 },
 "1":
 {
 "bbox": {"x1": 100, "y1": 100, "x2": 200, "y2":200},
 "probability": 0.75, 
 "is_mask": "0"
 }
 }, 
 "face_detect": true,
 "timestamp": "2021-01-26 15:54:02.309667",
 "error": null
}

```

| | Description |
|------------ | -------------|
|results | results from the mask detection model. If the face was found, the API return bounding box 'bbox' of (x1,y1) and (x2, y2) which are top left and bottom right consecutively. The API also returns the probability of a predicted class and the class where 1 is a person worn mask and 0 is a person not worn a mask. If multiple persons were detected, the API returns multiple results.|
|face_detect | return the boolean true if the person's face were detected, false otherwise.|
|timestamp | datetime of the current API request.|
|error | return an error if found. |


## Licenses
|| Licenses|
------------ | -------------
|Mask Classification Model | <a href="https://creativecommons.org/licenses/by-sa/4.0/deed.ast">Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)</a>|

## Contact

If you are interested in collaboration with us or interested in Data Labelling Service, please contact us via vulcan@lab.ai or https://www.facebook.com/vulcancoalition 

For more information:

http://vulcancoalition.com
