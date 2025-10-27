# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from PIL import Image
import io, torch, clip
from torch import nn

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load("recycler_mlp.pth", map_location="cpu")
    app.state.classes = ckpt["classes"]

    app.state.clip_model, app.state.preprocess = clip.load(
        ckpt["clip_name"], device=app.state.device
    )  # CLIP + preprocess [web:2]
    app.state.clip_model.eval()  # inference mode [web:22]

    mlp = nn.Sequential(
        nn.Linear(ckpt["in_dim"], ckpt["mlp_hidden"]),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(ckpt["mlp_hidden"], len(app.state.classes)),
    ).to(app.state.device)
    mlp.load_state_dict(ckpt["mlp_state"])  # load weights [web:23]
    mlp.eval()  # inference mode [web:22]
    app.state.head = mlp

    yield  # resources live for app lifetime [web:35]

app = FastAPI(lifespan=lifespan)

@app.get("/")
def print_hello():
    print("welcome to this application")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    import torch
    with torch.no_grad():
        x = app.state.preprocess(image).unsqueeze(0).to(app.state.device)
        feat = app.state.clip_model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        logits = app.state.head(feat)
        idx = int(logits.argmax(dim=1).item())
    return {"class": app.state.classes[idx]}
