# 🤖 RecycleAI: AI-Powered Waste Classification Portal
recycleAI is a research-driven system for intelligent waste identification, sorting, and impact tracking. It combines multi-modal sensing, modern deep learning, and an adaptive learning loop to handle evolving waste streams in real-world conditions.

This repository hosts:
- Data pipelines for building and hybridizing multi-modal datasets
- Model training and evaluation scripts for image and hyperspectral inputs
- An adaptive active learning loop for rapid updates with minimal labels
- A lightweight dashboard for impact metrics and remote monitoring

Core capabilities:
- **Multi‑modal fusion**: Blend visual and other sensor modalities to improve robustness and accuracy across diverse waste types and environments.
- **Adaptive learning**: Actively query uncertain samples (e.g., new packaging) and update models with minimal human annotation.
- **Real‑time feedback**: Drive LEDs, lights, or audio prompts on a prototype bin while streaming events for remote tracking and fleet insights.
- **Granular material recognition**: Use hyperspectral imaging with advanced CNNs/Transformers (e.g., EfficientNet, ViT) to distinguish fine‑grained categories (PET, HDPE, PVC; paper grades; e‑waste components).
- **Carbon impact analytics**: Estimate avoided emissions from sorting decisions and surface real‑time metrics; suggest optimized collection schedules to reduce transport emissions.

Goals:
- Deliver a robust, field‑ready sorting stack that adapts as new materials emerge
- Provide transparent metrics that connect sorting accuracy to climate impact
- Enable researchers and practitioners to replicate, extend, and deploy at scale



## 🚀 Quick Start: Running the Application

This guide provides the essential steps to install dependencies and run the local development server.

### Prerequisites

You must have **Python 3.8+** and **Git** installed on your system.

### Step 1: Clone and Install Dependencies

Open your Command Prompt (CMD) and run the following commands:

```bash
# Clone the repository
git clone [https://github.com/vishal-labs/recycleAI](https://github.com/vishal-labs/recycleAI)

# Move into the project folder
cd recycleAI

# Install all required Python packages (FastAPI, Uvicorn, PyTorch, CLIP)
pip install -r requirements.txt
```
Step 2: Start the Server
Run the core application file using the Uvicorn ASGI server.
```bash
python -m uvicorn main:app --reload
```
🌐 Application Access and Testing
Step 3: Access and Test Functionality
The application runs on the local server at http://127.0.0.1:8000/.

Upload an image via the classification endpoint.

Execute the request to see the material classification and the CO2 savings result.
