🤖 RecycleAI: AI-Powered Waste Classification Portal🚀 Quick Start: Running the ApplicationThis guide provides the essential steps to install dependencies and run the local development server.PrerequisitesYou must have Python 3.8+ and Git installed on your system.Step 1: Clone and Install DependenciesOpen your Command Prompt (CMD) and run the following commands:Bash# Clone the repository
git clone https://github.com/vishal-labs/recycleAI

# Move into the project folder
cd recycleAI

# Install all required Python packages (FastAPI, Uvicorn, PyTorch, CLIP)
pip install -r requirements.txt
Step 2: Start the ServerRun the core application file using the Uvicorn ASGI server.Bashpython -m uvicorn main:app --reload
🌐 Application Access and TestingStep 3: Access and Test FunctionalityThe application runs on the local server at http://127.0.0.1:8000/.View Landing Page: Navigate to the root URL:$$\text{[http://127.0.0.1:8000/](http://127.0.0.1:8000/)}$$(This displays the project's professional landing page.)Test Classification: To test the ViT model directly, navigate to the interactive API documentation (Swagger UI):$$\text{[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)}$$Upload an image via the classification endpoint.Execute the request to see the material classification and the $\text{CO}_2$ savings result.
