# assigning API KEY to initialize openai environment

import openai
import os
from pydantic import BaseModel
import json
from typing import List, Optional
import re
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import joblib
from io import BytesIO
from PIL import Image
from SVMclassifier import preprocess_image, invert_image
import tensorflow as tf
from jsonfetcher import Numeric_fetch, WordFetch, test_fetch


openai.api_key = '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'


# Load the SVM classifier
svm_classifier = joblib.load(r"C:\Users\TCS\Desktop\hackademia\AI\svm_classifier.pkl")
# Initialize FastAPI app
app = FastAPI()


# Define the structure for each question using Pydantic
class Question(BaseModel):
    question: str
    options: List[str]
    correct: Optional[str]  # Index of the correct option (optional)

@app.get("/getquestions")
async def get_questions():
    try:
        with open('testquestions.json', 'r') as file:
            data = json.load(file)
            return data
    except:
        data = test_fetch()
        return data

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}, Content-Type: {file.content_type}")
    # if not file.content_type.startswith('image/'):
    #     raise HTTPException(status_code=400, detail="File must be an image.")
    
    # Read the image file
    image_bytes = await file.read()
    # print(image_bytes)
    image = Image.open(file.file)

    image.save("image.jpg")

    #Convert image to tensor for preprocessing
    image = cv2.imread("image.jpg")

    
    print(image.shape)
    # Preprocess the image
    processed_image = preprocess_image(image)




    # Make a prediction with the SVM classifier
    prediction = svm_classifier.predict([processed_image])

    # Map the prediction to a human-readable label
    if prediction == 1:
        result = "Dysgraphic handwriting detected."
    else:
        result = "Non-dysgraphic handwriting detected."


    print(result)
    # Return the result as JSON
    return JSONResponse(content={"result": result})

    

    # except Exception as e:
    #     return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/WordBased")
async def get_questions():
    try:
        with open('wordquestions.json', 'r') as file:
            data = json.load(file)
            return data
    except:
        data = WordFetch()
        return data


@app.get("/NumberBased")
async def get_questions():
    try:
        with open('questions.json', 'r') as file:
            data = json.load(file)
            return data
    except:
        data = Numeric_fetch()
        return data


class LoginRequest(BaseModel):
    username: str
    password: str
    
@app.post("/login")
def login(request: LoginRequest):
    if request.username == "shlokkoirala19@gmail.com" and request.password == "admin":
        return {"success": True, "message": "Login successful"}
    else:
        return {"success": False, "message": "Invalid credentials"}