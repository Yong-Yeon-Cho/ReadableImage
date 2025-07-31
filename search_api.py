from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import torch, open_clip, faiss, pickle, io
import os

# Render에서 OpenMP 경고 방지
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI()

# ✅ CLIP 모델 로드
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# ✅ FAISS DB 로드
index = faiss.read_index("artwork.index")
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

# ✅ 임베딩 함수
def extract_embedding_from_imagefile(file):
    image = Image.open(io.BytesIO(file)).convert("RGB")
    img_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.encode_image(img_tensor)
    return vec.cpu().numpy().astype("float32")

# ✅ 제목 반환 API
@app.post("/title")
async def get_title(file: UploadFile = File(...)):
    contents = await file.read()
    query_vec = extract_embedding_from_imagefile(contents)
    distances, indices = index.search(query_vec, 1)
    artwork_title = labels[indices[0][0]]
    return {"title": artwork_title}
