# Variables
APP_NAME = brainmap_app
DOCKER_IMAGE = brainmap:latest
DOCKERFILE = Dockerfile
PORT = 8000
GCP_REGION = europe-west1

# Default target
.PHONY: all
all: build

# Build the Docker image
.PHONY: build
build:
	docker build -t $(DOCKER_IMAGE) -f $(DOCKERFILE) .

# Run the Docker container locally
.PHONY: run
run:
	docker run -it --rm -p $(PORT):8000 --env-file .env $(DOCKER_IMAGE)

# Push the Docker image to Google Artifact Registry
.PHONY: push
push:
	gcloud auth configure-docker
	gcloud builds submit --tag $(GAR_IMAGE)

# Deploy to Google Cloud Run
.PHONY: deploy
deploy:
	gcloud run deploy $(APP_NAME) \
		--image $(GAR_IMAGE) \
		--region $(GCP_REGION) \
		--platform managed \
		--allow-unauthenticated

# Clean up local Docker images
.PHONY: clean
clean:
	docker rmi $(DOCKER_IMAGE)
