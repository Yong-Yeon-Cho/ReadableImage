import os
import torch
import open_clip
import faiss
from PIL import Image
import numpy as np
import pickle

# ✅ 1. CLIP 모델 불러오기
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# ✅ 2. 이미지 임베딩 함수
def extract_embedding(img_path):
    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
    return features[0].cpu().numpy()

# ✅ 3. 벡터 DB(Faiss) 생성
index = faiss.IndexFlatL2(512)  # CLIP ViT-B-32는 512차원
labels = []
vectors = []

dataset_path = "../dataset"  # server 폴더 기준 상대경로

for artwork in os.listdir(dataset_path):
    artwork_path = os.path.join(dataset_path, artwork)
    if os.path.isdir(artwork_path):
        for img_file in os.listdir(artwork_path):
            img_path = os.path.join(artwork_path, img_file)
            vec = extract_embedding(img_path)
            vectors.append(vec)
            labels.append(artwork)

vectors_np = np.array(vectors).astype("float32")
index.add(vectors_np)

# ✅ 4. 라벨 저장 (어떤 벡터가 어떤 작품인지)
with open("labels.pkl", "wb") as f:
    pickle.dump(labels, f)

faiss.write_index(index, "artwork.index")
print("✅ 임베딩 추출 및 DB 저장 완료")
