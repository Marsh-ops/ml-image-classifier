from fastapi import FastAPI, File, UploadFile, HTTPException
from .predict import predict

app = FastAPI(title="CIFAR-10 Image Classifier API")

@app.get("/")
def root():
    return {"message": "Welcome to CIFAR-10 Classifier API"}

@app.post("/predict")
async def classify_image(file: UploadFile = File(...)):
    # Check file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        label = predict(file.file)
        return {"filename": file.filename, "prediction": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
print("Main module loaded")
print(f"App exists? {'app' in globals()}")