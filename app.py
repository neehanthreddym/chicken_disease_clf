import uvicorn
import io
from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from cnn_classifier.pipeline.predicton import PredictionPipeline
from cnn_classifier.utils.utilities import bytes_to_data_url

templates = Jinja2Templates(directory='templates')

class ClientApp:
    def __init__(self):
        self.classifier = PredictionPipeline()

clApp = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global clApp
    clApp = ClientApp()
    yield
    # Cleanup code can go here if needed

app = FastAPI(lifespan=lifespan)

@app.get('/')
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    # Basic validation
    if not file.content_type or not file.content_type.startswith("image/"):
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Please upload an image file."}
        )

    contents = await file.read()
    image_data_url = bytes_to_data_url(contents, file.content_type)
    
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    result = clApp.classifier.predict(image)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "image_data_url": image_data_url, "result": result}
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)