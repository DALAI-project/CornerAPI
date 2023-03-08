from fastapi import FastAPI, File, UploadFile, HTTPException
from torchvision import transforms
import onnxruntime
import numpy as np
import io
from PIL import Image
import syslog

# Syslog library for Python is used for logging: 
# https://docs.python.org/3/library/syslog.html
syslog.openlog(ident="Corner-API", logoption=syslog.LOG_PID, facility=syslog.LOG_LOCAL0)

# Path to pretrained model file
MODEL_PATH = './model/corner_model.onnx'
# Input image size
IMG_SIZE = 224

# Transformations used for input images
img_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

# Predicted class labels
classes = {0: 'ok', 1: 'folded_corner' }

try:
    # Initialize API Server
    app = FastAPI()
except Exception as e:
    syslog.syslog(syslog.LOG_ERR, 'Failed to start the API server: {}'.format(e))
    sys.exit(1)

# Function is run (only) before the application starts
@app.on_event("startup")
async def load_model():
    """
    Load the pretrained model on startup.
    """
    try:
        # Load the onnx model and the trained weights
        model = onnxruntime.InferenceSession(MODEL_PATH)
        # Add model to app state
        app.package = {"model": model}
    except Exception as e:
        syslog.syslog(syslog.LOG_ERR, 'Failed to load the model file: {}'.format(e))
        raise HTTPException(status_code=500, detail=f"Failed to load the model file: {e}")


def predict(image):
    """
    Perform prediction on input image.
    """
    # Get model from app state
    model = app.package["model"]
    image = img_transforms(image).unsqueeze(0)
    # Transform tensor to numpy array
    img = image.detach().cpu().numpy()
    input = {model.get_inputs()[0].name: img}
    # Run model prediction
    output = model.run(None, input)
    # Get predicted class
    pred = np.argmax(output[0], 1)
    pred_class = pred.item()
    # Get the confidence value for the prediction
    pred_confidences = 1/(1+np.exp(-output[0]))
    # Confidence of the prediction as %
    class_confidence = float(pred_confidences[0][pred_class])
    # Return predicted class and confidence in dictionary form
    predictions = {'prediction': classes[pred_class], 'confidence': class_confidence}

    return predictions


# Endpoint for corner prediction
@app.post("/corner/")
async def detect_corner(file: UploadFile = File(...)):
    try:
        # Loads the image sent with the POST request
        req_content = await file.read()
        image = Image.open(io.BytesIO(req_content))
        image.draft('RGB', (IMG_SIZE, IMG_SIZE))
    except Exception as e:
        syslog.syslog(syslog.LOG_ERR, 'Failed to load the input image file: {}'.format(e)) 
        raise HTTPException(status_code=400, detail=f"Failed to load the input image file: {e}")

    # Get predicted class and confidence
    try: 
        predictions = predict(image)
    except Exception as e:
        syslog.syslog(syslog.LOG_ERR, 'Failed to analyze the input image file. Error: {}'.format(e))
        raise HTTPException(status_code=500, detail=f"Failed to analyze the input image file: {e}")

    return predictions

@app.get("/cornerurl/")
async def read_item(url: str):
    try:
        # Loads the image from the path sent with the GET request
        image = Image.open(url)
        image.draft('RGB', (IMG_SIZE, IMG_SIZE))

    except Exception as e:
        syslog.syslog(syslog.LOG_ERR, 'Failed to recognize file {} as an image. Error: {}'.format(url, e))
        raise HTTPException(status_code=400, detail=f"Failed to load the input image file: {e}")

    # Get predicted class and confidence
    try: 
        predictions = predict(image)
    except Exception as e:
        syslog.syslog(syslog.LOG_ERR, 'Failed to analyze the input image file. Error: {}'.format(e))
        raise HTTPException(status_code=500, detail=f"Failed to analyze the input image file: {e}")

    return predictions

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level="info")
