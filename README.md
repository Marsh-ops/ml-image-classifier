CIFAR-10 Image Classifier API

This project is a FastAPI-based API for classifying images using a CIFAR-10 trained PyTorch model. It is fully containerized with Docker, making it easy to run anywhere.

-------------------------------------------------------------------------------

Requirements

1) Docker Desktop
2) Optional: Python 3.10+ (for local testing outside Docker)

-------------------------------------------------------------------------------

Build and Run with Docker:

1) Open a terminal in the project root directory.

2) Build the Docker image:
"docker build --no-cache -t cifar10-api -f docker/Dockerfile ."

3) Run the container:
"docker run -p 8000:8000 cifar10-api"

Open your browser and go to:
"http://localhost:8000/docs" - This opens the FastAPI Swagger UI where you can test the API.

-------------------------------------------------------------------------------

*Testing the API*

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

Notes

PyTorch may show a TypedStorage deprecation warning — this is harmless.

All dependencies are included in requirements.txt.

The API is CPU-only; no GPU is required.
