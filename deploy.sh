#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Shimba LLM — One-command deploy to any cloud
# ═══════════════════════════════════════════════════════════════
# Usage:
#   ./deploy.sh colab          # Open Google Colab (free GPUs)
#   ./deploy.sh docker         # Build & run locally with Docker
#   ./deploy.sh gcp-run        # Deploy to Google Cloud Run
#   ./deploy.sh aws-ecs        # Deploy to AWS ECS
#   ./deploy.sh azure-aci      # Deploy to Azure Container Instances
#   ./deploy.sh huggingface    # Deploy to Hugging Face Spaces
#   ./deploy.sh all            # Print all options
# ═══════════════════════════════════════════════════════════════

set -e

case "${1:-help}" in

# ── Google Colab (FREE GPUs) ────────────────────────────────
colab)
  echo "Open in browser:"
  echo "  https://colab.research.google.com/github/Shimba-crypto/shimba-llm/blob/main/shimba_colab.ipynb"
  echo ""
  echo "Or upload shimba_colab.ipynb manually to:"
  echo "  https://colab.research.google.com"
  echo ""
  echo "Then click Runtime -> Run all"
  ;;

# ── Docker (any cloud) ──────────────────────────────────────
docker)
  echo "Building Docker image..."
  docker build -t shimba-llm .
  echo ""
  echo "Running locally on port 8000..."
  echo "  docker run -p 8000:8000 -v \$(pwd)/models:/app/models shimba-llm"
  echo ""
  echo "Push to any container registry:"
  echo "  docker tag shimba-llm gcr.io/YOUR_PROJECT/shimba-llm"
  echo "  docker push gcr.io/YOUR_PROJECT/shimba-llm"
  echo ""
  echo "Then deploy from that registry to Cloud Run / ECS / ACI / etc."
  ;;

# ── Google Cloud Run ────────────────────────────────────────
gcp-run)
  PROJECT="${2:-$(gcloud config get project)}"
  echo "Deploying to Google Cloud Run (project: $PROJECT)..."
  echo ""
  echo "  gcloud builds submit --config cloudbuild.yaml --project $PROJECT"
  echo ""
  echo "Or manually:"
  echo "  docker build -t gcr.io/$PROJECT/shimba-llm ."
  echo "  docker push gcr.io/$PROJECT/shimba-llm"
  echo "  gcloud run deploy shimba-llm \\"
  echo "    --image gcr.io/$PROJECT/shimba-llm \\"
  echo "    --platform managed --region us-central1 \\"
  echo "    --allow-unauthenticated --memory 4Gi --cpu 2 --port 8000"
  echo ""
  echo "URL will be: https://shimba-llm-xxxxx-uc.a.run.app"
  echo "Test: curl https://URL -d '{\"prompt\":\"Hello\"}'"
  ;;

# ── AWS ECS ─────────────────────────────────────────────────
aws-ecs)
  echo "Deploying to AWS ECS..."
  echo ""
  echo "  1. Create ECR repo:"
  echo "     aws ecr create-repository --repository-name shimba-llm"
  echo ""
  echo "  2. Build & push:"
  echo "     docker build -t shimba-llm ."
  echo "     aws ecr get-login-password | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com"
  echo "     docker tag shimba-llm:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/shimba-llm:latest"
  echo "     docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/shimba-llm:latest"
  echo ""
  echo "  3. Create ECS task definition + service (CPU: 2048, Memory: 4096)"
  echo ""
  echo "  4. Or use Fargate:"
  echo "     aws ecs run-task --cluster default --task-definition shimba-llm --launch-type FARGATE"
  ;;

# ── Azure Container Instances ───────────────────────────────
azure-aci)
  echo "Deploying to Azure Container Instances..."
  echo ""
  echo "  1. Build & push to Azure Container Registry:"
  echo "     docker build -t shimba-llm ."
  echo "     az acr login --name YOUR_REGISTRY"
  echo "     docker tag shimba-llm YOUR_REGISTRY.azurecr.io/shimba-llm"
  echo "     docker push YOUR_REGISTRY.azurecr.io/shimba-llm"
  echo ""
  echo "  2. Deploy:"
  echo "     az container create \\"
  echo "       --resource-group myGroup \\"
  echo "       --name shimba-llm \\"
  echo "       --image YOUR_REGISTRY.azurecr.io/shimba-llm \\"
  echo "       --cpu 2 --memory 4 \\"
  echo "       --ports 8000 \\"
  echo "       --dns-name-label shimba-llm"
  echo ""
  echo "  URL: http://shimba-llm.YOUR_REGION.azurecontainer.io:8000"
  ;;

# ── Hugging Face Spaces ─────────────────────────────────────
huggingface)
  echo "Deploy to Hugging Face Spaces (free CPU/GPU):"
  echo ""
  echo "  1. Go to https://huggingface.co/new-space"
  echo "  2. Name: shimba-llm"
  echo "  3. Space SDK: Docker"
  echo "  4. Hardware: CPU or GPU (free options available)"
  echo "  5. Connect your GitHub repo: Shimba-crypto/shimba-llm"
  echo ""
  echo "  The Dockerfile auto-builds on HF. Done."
  ;;

# ── All options ─────────────────────────────────────────────
help|all|*)
  echo "Shimba LLM — Deploy to any cloud"
  echo ""
  echo "Usage: ./deploy.sh <target>"
  echo ""
  echo "Targets:"
  echo "  colab         Google Colab (free GPUs, easiest)"
  echo "  docker        Build Docker image locally"
  echo "  gcp-run       Google Cloud Run (serverless)"
  echo "  aws-ecs       AWS ECS / Fargate"
  echo "  azure-aci     Azure Container Instances"
  echo "  huggingface   Hugging Face Spaces (free option)"
  echo ""
  echo "Quickest:  ./deploy.sh colab"
  echo "Universal: ./deploy.sh docker && docker push ..."
  ;;
esac
