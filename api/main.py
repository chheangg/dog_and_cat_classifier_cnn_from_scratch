from fastapi import FastAPI, UploadFile, HTTPException, status
import torch
from torchvision import transforms
import cv2
import numpy as np
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
    
from dog_and_cat_classifier_cnn_from_scratch.model import ResNet50


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_connection = redis.from_url("redis://redis", encoding="utf-8", decode_responses=True)
    await FastAPILimiter.init(redis_connection)
    
    print("FastAPI-Limiter initialized.")
    yield
    
    print("FastAPI-Limiter shutting down.")
    
origins = [
    "*"
]
    
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet50(num_classes=2, lr=0.01, in_channels=3, dropout_rate=0.3)

model_path = "./models/final_model_20250915_092314.pth"
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

def preprocess_img(img_array):
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224), cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    
    preprocess = transforms.Compose([
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    return preprocess(image).unsqueeze(0)


@app.post('/model/infer')
async def infer(file: UploadFile, dependencies=[RateLimiter(times=10, seconds=60)]):
    ALLOWED_IMG = ["image/jpeg", "image/png", "image/jpg"]
    
    if file.content_type not in ALLOWED_IMG:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image type. Allowed types are: {', '.join(ALLOWED_IMG)}"
        )
    
    content = await file.read()
    image_binary = np.frombuffer(content, np.uint8)
    image_tensor = preprocess_img(image_binary)
    
    with torch.no_grad():
        result = model(image_tensor)
        
    return {"result": result[0].tolist()}