'''
Mask Detection API Test case 
'''

import cv2
import numpy as np
from PIL import Image
import requests
import json
import numpy as np

ROOT_URL = ''


def test_home():
    url = ROOT_URL + '/'
    response = requests.request('GET',url)
    print(response)


def test_detector(image_file):

    url = ROOT_URL + '/mask-detection'
    files = {'file':open(image_file, 'rb')}
    response = requests.request('POST',url, files=files)
    print(response)
    print(response.content)
    content = json.loads(response.content.decode('utf-8'))
    print(content)
    if content['face_detect']:
        image = np.array(Image.open(image_file))
        show_feedback(image, content, image_file)



def show_feedback(image, content, image_file):

    print(image.shape)

    if content['face_detect']:
        for _, result in content['results'].items():
            if result['is_mask'] == '1':
                color = (124,252,0) # RGB Green
            else:
                color = (220,20,60) # RGB Red
            font_color = (240,240,240)
            message = f'probability: {round(result["probability"], 3)}'
            x1, y1, x2, y2 = result['bbox']['x1'], result['bbox']['y1'], result['bbox']['x2'], result['bbox']['y2']
            image = cv2.rectangle(image, (x2,y2), (x1,y1), color, 2)
            image = cv2.putText(image, message, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, font_color)

    
    cv2.imwrite("out_"+image_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":

    test_home()
    for idx in range(20):
        try:
            test_detector(f'test{idx}.jpg')
        except:
            print(f'file test{idx}.jpg not found')

    test_detector(f'test1.bmp')