Requirements for Deployment

Ensure the .env file contains all required environment variables (e.g., BUCKET_NAME, GAR_IMAGE, etc.).

Authenticate with Google Cloud CLI:
gcloud auth login
gcloud config set project <your-project-id>


Push the Docker image to Google Artifact Registry:
make push


Deploy the app to Google Cloud Run:
make deploy