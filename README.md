# 🤖 RecycleAI: AI-Powered Waste Classification Portal

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
