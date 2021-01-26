'''
Face mask detection API using FastAPI framework
'''

from fastapi import FastAPI, File, HTTPException, UploadFile
import base64
import io
import json
from PIL import Image
import uvicorn
import numpy as np
import datetime
from maskdetector import FaceMaskDetectionAPI

def read_config(config_file: str):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


app = FastAPI()
config = read_config('configs/config.json')
face_mask_detection = FaceMaskDetectionAPI(**config)


@app.get("/")
def home():
    return  {"Available Public API":
                {
                    {"API_Name":"VulcanMaskDetectionAPI", "Version": "1.0"}
                }
            }


@app.post("/mask-detection")
async def mask_detection_api(file: UploadFile = File(...)):

    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return {'error': "Image must be jpg or png format", 'result': None, 'face_detect': None, 'timestamp': str(datetime.datetime.now())}

    image = Image.open(io.BytesIO(await file.read()))
    return face_mask_detection.run(np.array(image))


if __name__ == '__main__':
    uvicorn.run(app, port = 9000, host = '0.0.0.0')