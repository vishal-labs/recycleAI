# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
async def landing_page():
    """Beautiful landing page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Recyclable Trash Predictor - AI Powered Classification</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            
            .container {
                max-width: 1200px;
                width: 100%;
                background: white;
                border-radius: 30px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                overflow: hidden;
                display: flex;
                flex-direction: row;
                align-items: center;
            }
            
            .left-section {
                flex: 1;
                padding: 60px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            .right-section {
                flex: 1;
                padding: 60px;
                background: white;
            }
            
            h1 {
                font-size: 3em;
                margin-bottom: 20px;
                font-weight: 800;
            }
            
            .tagline {
                font-size: 1.3em;
                margin-bottom: 30px;
                opacity: 0.95;
                line-height: 1.6;
            }
            
            .features {
                margin: 40px 0;
            }
            
            .feature {
                display: flex;
                align-items: center;
                margin: 20px 0;
                font-size: 1.1em;
            }
            
            .feature-icon {
                font-size: 2em;
                margin-right: 15px;
            }
            
            .cta-button {
                display: inline-block;
                background: white;
                color: #667eea;
                padding: 18px 50px;
                border-radius: 50px;
                font-size: 1.2em;
                font-weight: bold;
                text-decoration: none;
                transition: all 0.3s ease;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            }
            
            .cta-button:hover {
                transform: translateY(-3px);
                box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
            }
            
            .cta-button:active {
                transform: translateY(-1px);
            }
            
            .demo-preview {
                text-align: center;
            }
            
            .demo-box {
                background: linear-gradient(135deg, #f5f7fa 0%, #e1e8ed 100%);
                padding: 30px;
                border-radius: 20px;
                margin: 20px 0;
            }
            
            .demo-step {
                background: white;
                padding: 20px;
                margin: 15px 0;
                border-radius: 15px;
                border-left: 5px solid #667eea;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            }
            
            .demo-step h3 {
                color: #667eea;
                margin-bottom: 10px;
                font-size: 1.3em;
            }
            
            .demo-step p {
                color: #666;
                line-height: 1.6;
            }
            
            .badge {
                display: inline-block;
                background: rgba(255, 255, 255, 0.2);
                padding: 8px 20px;
                border-radius: 50px;
                font-size: 0.9em;
                margin-bottom: 20px;
            }
            
            @media (max-width: 768px) {
                .container {
                    flex-direction: column;
                }
                
                .left-section,
                .right-section {
                    padding: 40px;
                }
                
                h1 {
                    font-size: 2em;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="left-section">
                <div class="badge">🔄 AI-Powered Recycling Assistant</div>
                <h1>Smart Trash Classification</h1>
                <p class="tagline">
                    Upload any image and let our AI determine if it's recyclable or not. 
                    Help make the world a greener place, one prediction at a time!
                </p>
                
                <div class="features">
                    <div class="feature">
                        <span class="feature-icon">🎯</span>
                        <span>Instant Classification</span>
                    </div>
                    <div class="feature">
                        <span class="feature-icon">🤖</span>
                        <span>AI-Powered Detection</span>
                    </div>
                    <div class="feature">
                        <span class="feature-icon">⚡</span>
                        <span>Lightning Fast</span>
                    </div>
                    <div class="feature">
                        <span class="feature-icon">🆓</span>
                        <span>100% Free</span>
                    </div>
                </div>
                
                <a href="/predict" class="cta-button">🚀 Try It Now</a>
            </div>
            
            <div class="right-section">
                <div class="demo-preview">
                    <h2 style="color: #333; margin-bottom: 20px;">How It Works</h2>
                    
                    <div class="demo-box">
                        <div class="demo-step">
                            <h3>1️⃣ Upload Your Image</h3>
                            <p>Simply select any image from your device using our intuitive file uploader.</p>
                        </div>
                        
                        <div class="demo-step">
                            <h3>2️⃣ AI Processing</h3>
                            <p>Our advanced machine learning model analyzes your image in seconds.</p>
                        </div>
                        
                        <div class="demo-step">
                            <h3>3️⃣ Get Results</h3>
                            <p>Receive instant feedback on whether your item is recyclable or not.</p>
                        </div>
                    </div>
                    
                    <p style="color: #999; font-size: 0.9em; margin-top: 30px;">
                        Powered by CLIP and Custom ML Architecture
                    </p>
                </div>
            </div>
        </div>
        
        <script>
            // Add some subtle animations
            document.querySelector('.container').style.opacity = '0';
            document.querySelector('.container').style.transform = 'translateY(20px)';
            document.querySelector('.container').style.transition = 'all 0.8s ease';
            
            setTimeout(() => {
                document.querySelector('.container').style.opacity = '1';
                document.querySelector('.container').style.transform = 'translateY(0)';
            }, 100);
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/predict", response_class=HTMLResponse)
async def predict_form():
    """Display file upload form"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Recyclable Trash Predictor</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
            }
            .container {
                background: white;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                max-width: 500px;
                width: 100%;
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
                text-align: center;
            }
            .subtitle {
                color: #666;
                text-align: center;
                margin-bottom: 30px;
                font-size: 14px;
            }
            .upload-box {
                border: 3px dashed #667eea;
                border-radius: 10px;
                padding: 30px;
                text-align: center;
                background: #f8f9fa;
                transition: all 0.3s ease;
            }
            .upload-box:hover {
                background: #f0f2ff;
                border-color: #764ba2;
            }
            input[type="file"] {
                margin: 10px 0;
                padding: 10px;
                width: 100%;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 40px;
                border: none;
                border-radius: 25px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                width: 100%;
                transition: transform 0.2s ease;
            }
            button:hover {
                transform: translateY(-2px);
            }
            button:active {
                transform: translateY(0);
            }
            .preview {
                margin: 20px 0;
                text-align: center;
            }
            .preview img {
                max-width: 200px;
                max-height: 200px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔄 Recyclable Trash Predictor</h1>
            <p class="subtitle">Upload an image to classify it as recyclable or not</p>
            
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-box">
                    <p>📸 Select an image file</p>
                    <input type="file" name="file" accept="image/*" id="fileInput" required>
                </div>
                <div class="preview" id="preview"></div>
                <button type="submit">🔍 Predict</button>
            </form>
        </div>

        <script>
            const fileInput = document.getElementById('fileInput');
            const preview = document.getElementById('preview');
            const form = document.getElementById('uploadForm');

            fileInput.addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(event) {
                        preview.innerHTML = '<img src="' + event.target.result + '" alt="Preview">';
                    }
                    reader.readAsDataURL(file);
                }
            });

            form.addEventListener('submit', async function(e) {
                e.preventDefault();
                const formData = new FormData(form);
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        const data = await response.text();
                        document.body.innerHTML = data;
                    } else {
                        alert('Error: ' + response.statusText);
                    }
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        return HTMLResponse(
            content="""
            <html>
            <head><title>Error</title></head>
            <body style="font-family: sans-serif; text-align: center; padding: 50px;">
                <h1>❌ Error</h1>
                <p>Invalid image file</p>
                <a href="/predict">Try again</a>
            </body>
            </html>
            """,
            status_code=400
        )

    # Convert image to base64 for display
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    import base64
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Run prediction
    with torch.no_grad():
        x = app.state.preprocess(image).unsqueeze(0).to(app.state.device)
        feat = app.state.clip_model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        logits = app.state.head(feat)
        idx = int(logits.argmax(dim=1).item())
    
    predicted_class = app.state.classes[idx]
    
    # Determine if recyclable and calculate CO2 savings
    recyclable_keywords = ['recyclable', 'recycle', 'cardboard', 'paper', 'metal', 'glass', 'plastic', 'bottle', 'can']
    is_recyclable = any(keyword.lower() in predicted_class.lower() for keyword in recyclable_keywords)
    
    # CO2 savings in kg per item (estimated values)
    co2_savings = {
        'recyclable': 2.5,  # kg CO2 saved per recyclable item
        'non_recyclable': 0.1  # kg CO2 saved even for non-recyclable if properly disposed
    }
    
    emission_saved = co2_savings['recyclable'] if is_recyclable else co2_savings['non_recyclable']
    background_gradient = (
        "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)" if is_recyclable 
        else "linear-gradient(135deg, #ee0979 0%, #ff6a00 100%)"
    )
    
    result_color = "#11998e" if is_recyclable else "#ee0979"
    result_icon = "♻️" if is_recyclable else "❌"
    result_message = "Great job!" if is_recyclable else "Needs proper disposal"
    
    # Create result HTML
    result_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prediction Result</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: {background_gradient};
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
                transition: background 0.5s ease;
            }}
            .result-container {{
                background: white;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                max-width: 500px;
                width: 100%;
                text-align: center;
            }}
            h1 {{
                color: #333;
                margin-bottom: 30px;
            }}
            .image-preview {{
                margin: 20px 0;
            }}
            .image-preview img {{
                max-width: 250px;
                max-height: 250px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            }}
            .result-box {{
                background: {background_gradient};
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }}
            .result-box h2 {{
                margin: 0;
                font-size: 28px;
            }}
            .co2-box {{
                background: {result_color};
                color: white;
                padding: 25px;
                border-radius: 15px;
                margin: 20px 0;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            }}
            .co2-box h3 {{
                margin: 0 0 10px 0;
                font-size: 24px;
            }}
            .co2-box .big-number {{
                font-size: 42px;
                font-weight: bold;
                margin: 10px 0;
            }}
            .co2-box .unit {{
                font-size: 16px;
                opacity: 0.9;
            }}
            .message-box {{
                background: {result_color};
                color: white;
                padding: 15px;
                border-radius: 10px;
                margin: 15px 0;
                font-size: 18px;
                font-weight: bold;
            }}
            .back-button {{
                background: {result_color};
                color: white;
                padding: 15px 40px;
                border: none;
                border-radius: 25px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
                margin-top: 20px;
                transition: all 0.3s ease;
            }}
            .back-button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            }}
        </style>
    </head>
    <body>
        <div class="result-container">
            <h1>🎯 Prediction Result</h1>
            <div class="image-preview">
                <img src="data:image/jpeg;base64,{img_str}" alt="Uploaded image">
            </div>
            <div class="result-box">
                <h2>{result_icon} {predicted_class}</h2>
            </div>
            
            <div class="message-box">
                {result_message}
            </div>
            
            <div class="co2-box">
                <h3>🌍 Environmental Impact</h3>
                <div class="big-number">{emission_saved:.2f}</div>
                <div class="unit">kg CO₂ emissions reduced</div>
                <p style="margin: 15px 0 0 0; font-size: 14px; opacity: 0.95;">
                    {('Recycling this item helps reduce greenhouse gas emissions and supports a circular economy!' if is_recyclable else 'Proper disposal still helps minimize environmental impact.')}
                </p>
            </div>
            
            <a href="/predict" class="back-button">🔄 Predict Another</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=result_html)
