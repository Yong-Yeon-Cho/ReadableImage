from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import torch, open_clip, faiss, pickle, io
import os

# ✅ Railway 메모리 경고 방지용 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI()

# ✅ Lazy Loading (처음부터 모델을 로드하지 않음)
model = None
preprocess = None
device = "cpu"   # Railway 무료플랜에서는 GPU 없음, CPU 강제 사용

# ✅ 모델 로드 함수 (요청 들어올 때만 실행)
def get_model():
    global model, preprocess
    if model is None:
        print("▶ RN50 모델 처음 로드 중...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            'RN50', pretrained='openai'
        )
        model = model.to(device)
        print("✅ RN50 모델 로드 완료")
    return model, preprocess

# ✅ FAISS DB 로드 (DB는 가볍기 때문에 서버 시작 시 바로 로드)
index = faiss.read_index("artwork.index")
with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

# ✅ 이미지 임베딩 함수
def extract_embedding_from_imagefile(file):
    model, preprocess = get_model()   # ✅ Lazy Loading 적용 (첫 요청 시만 모델 로드)
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
