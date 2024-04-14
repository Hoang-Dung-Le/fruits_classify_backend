from fastapi import FastAPI, UploadFile, File, Form, \
      HTTPException, WebSocket, WebSocketDisconnect, status, \
      Response, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from model import create_transfer_learning_model
import tensorflow as tf
from PIL import Image
import io
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

transfer_learning_model = create_transfer_learning_model((224, 224, 3), 6)
transfer_learning_model.load_weights('checkpoint/')


@app.post("/predict")
async def predict_durian(file: UploadFile):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents))
    image = image.convert("RGB")
    resized_image = image.resize((224, 224))
    image_array = np.array(resized_image) / 255.0
    image_array = np.expand_dims(image_array,axis=0)
    # print(image_array.shape)
    pred = transfer_learning_model.predict(image_array)
    predicted_class = np.argmax(pred)
    # print(predicted_class)
    confidence = np.max(pred)
    print(confidence)
    if predicted_class == 0:
        return {"class": "fresh_peach",
                "conf": float(confidence)}
    elif predicted_class == 1:
        return {"class": "fresh_pomegranate",
                "conf": float(confidence)}
    elif predicted_class == 2:
        return {"class": "fresh_strawberry",
                "conf": float(confidence)}
    elif predicted_class == 1:
        return {"class": "rotten_peach",
                "conf": float(confidence)}
    elif predicted_class == 1:
        return {"class": "rotten_pomegranate",
                "conf": float(confidence)}
    else:
        return {"class": "rotten_strawberry",
                "conf": float(confidence)}
    