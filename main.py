""""
DL webapp that predicts what's best to plant in given soil conditions
"""

# FastAPI module
from fastapi import FastAPI
import uvicorn

# input validation
from pydantic import ValidationError

# data handling
import numpy as np
import pandas as pd

# data scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# model
import tensorflow as tf
from tensorflow.keras.models import load_model
model = load_model("crop_yield_model.h5")


# scaler
import pickle 
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file) # the Scaler

# data type specs
from fastapi import Depends

# HTML
from fastapi import Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

#CSS
from fastapi.staticfiles import StaticFiles

# app objects
app = FastAPI()
templates = Jinja2Templates(directory="templates")
# mount css to api
app.mount("/static", StaticFiles(directory="static"), name="static")


# on load
@app.get('/')
async def ok():
    return {'status code':'200'}


# model API
# get the html
@app.get('/predict', response_class=HTMLResponse)
async def make_predictions(request: Request):
    return templates.TemplateResponse("index.html", {'request':request})

# without pydantic

@app.post('/predict', response_class=HTMLResponse)
async def make_predictions(request: Request, Rainfall=Form(None), Nitrogen=Form(None), Humidity=Form(None), Phosphorus=Form(None), Temperature=Form(None), Potassium=Form(None), pH=Form(None),):

    try:
        entries = [Rainfall, Nitrogen, Humidity, Phosphorus, Temperature, Potassium, pH]
        for length in entries:
            if length is None:
                error_message = "You didn't enter enything"
                return templates.TemplateResponse("index.html", {'request':request, 'error_message':error_message})
        
        entry = [float(Rainfall), float(Nitrogen), float(Humidity), float(Phosphorus), float(Temperature), float(Potassium), float(pH)]
       
            
    except (ValueError, ValidationError):
        error_message = "Wrong Input, Try Again"
        return templates.TemplateResponse("index.html", {'request':request, 'error_message':error_message})

    # transform data
    entry = scaler.transform([entry])

    prediction = model.predict(entry)
    predicted_class = np.argmax(prediction, axis=1)[0]

    classes = {0: 'rice', 1: 'maize', 2: 'chickpea', 3: 'kidneybeans', 4: 'pigeonpeas', 5: 'mothbeans', 6: 'mungbean', 7: 'blackgram', 8: 'lentil', 9: 'pomegranate', 10: 'banana', 11: 'mango', 12: 'grapes', 13: 'watermelon', 14: 'muskmelon', 15: 'apple', 16: 'orange', 17: 'papaya', 18:'coconut', 19: 'cotton', 20: 'jute', 21: 'coffee'}


    
    predicted_crop = classes[predicted_class]
    prediction_message = f"The best crop to plant is: {predicted_crop}"


    return templates.TemplateResponse("index.html", {'request':request, 'prediction':prediction_message})








# run app
if __name__ == "__main__":
    uvicorn.run(app)

# http://127.0.0.1:8000