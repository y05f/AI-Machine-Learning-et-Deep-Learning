from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow import expand_dims
from PIL import Image
import numpy as np
import uvicorn
import os

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)

model_dir = "./model/modelV-0.h5"
model = load_model(model_dir)

class_predictions = np.array([
'cat',
'dog'
])

@app.get("/")
async def root():
    return {"message": "Welcome to catVdog API!"}

@app.post("/catsVdogs/predict")
async def get_net_image_prediction(file: UploadFile = File(...)):
    img = Image.open(file.file).resize((150,150))
    img_array = np.array(img)
    img_array = expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    return {
        "model-prediction": class_predictions[1 if pred[0] > 0.5 else 0],
        "model-output": str(pred[0])
    }

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)