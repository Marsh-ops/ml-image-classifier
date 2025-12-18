CIFAR-10 Image Classifier API

This project is a dockerized FastAPI service that classifies images into CIFAR-10 categories using a PyTorch model. This project demonstrates packaging a machine learning model into a production-ready API, containerization with Docker, and handling cross-platform deployment.

Within this README:

FEATURES ||
 TECH STACK ||
 REQUIREMENTS ||
 BUILD AND RUN WITH DOCKER ||
 TESTING THE API ||
 KEY LEARNINGS AND SKILLS DEMONSTRATED || 
 NOTES ||

-------------------------------------------------------------------------------

FEATURES

Pre-trained CIFAR-10 image classifier using PyTorch.

REST API for image classification using FastAPI.

Dockerized for consistent cross-platform deployment.

Handles image uploads via POST requests.

Ready for extension or integration into larger ML systems.

-------------------------------------------------------------------------------

TECH STACK

Programming Language: Python 3.10/3.11

Frameworks & Libraries: PyTorch, FastAPI, Uvicorn, Pillow

Tools: Docker, Linux containerization

-------------------------------------------------------------------------------

REQUIREMENTS

1) Docker Desktop
2) Optional: Python 3.10+ (for local testing outside Docker)

-------------------------------------------------------------------------------

BUILD AND RUN WITH DOCKER:

1) Open a terminal in the project root directory.

2) Build the Docker image:
"docker build --no-cache -t cifar10-api -f docker/Dockerfile ."

3) Run the container:
"docker run -p 8000:8000 cifar10-api"

Open your browser and go to:
"http://localhost:8000/docs" - This opens the FastAPI Swagger UI where you can test the API.

-------------------------------------------------------------------------------

TESTING THE API

1. Using Swagger UI

Click POST /predict

Upload an image (e.g., test_image.jpg) *image must match one of the CIFAR-10 classes below:

1) airplane

2) automobile

3) bird

4) cat

5) deer

6) dog

7) frog

8) horse

9) ship

10) truck

Click Execute to see the predicted class.

-------------------------------------------------------------------------------

KEY LEARNINGS AND SKILLS DEMONSTRATED

Packaging ML models into production-ready APIs with FastAPI.

Containerization for reproducible deployment using Docker.

Cross-platform development: Running a Linux-based Docker container on Windows.

Dependency management: Handling Python version and library compatibility issues.

API design & testing: Accepting file uploads and returning structured predictions.

-------------------------------------------------------------------------------

NOTES

The CIFAR-10 dataset is not included due to GitHub file size limits. Download it from https://www.cs.toronto.edu/~kriz/cifar.html
 if you want to retrain the model.

The project is ready to deploy and extend for additional ML applications.

PyTorch may show a TypedStorage deprecation warning — this is harmless.

All dependencies are included in requirements.txt.

The API is CPU-only; no GPU is required.
