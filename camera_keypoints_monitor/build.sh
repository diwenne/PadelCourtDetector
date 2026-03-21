#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Config (override via env)
# ----------------------------
PROJECT_ID="${PROJECT_ID:-main-455618}"
REGION="${REGION:-europe-west4}"          # Artifact Registry location
REPO="${REPO:-eu-west4}"                   # Artifact Registry repository name
IMAGE_NAME="${IMAGE_NAME:-camera-movement}"
TAG="latest"
PLATFORM="${PLATFORM:-linux/amd64}"        # set to linux/arm64 if needed

# ----------------------------
# Required env secrets/creds
# ----------------------------
required_vars=(
  "APP_SUPABASE_KEY"
  "APP_SUPABASE_URL"
  "AWS_ACCESS_KEY_ID"
  "AWS_DEFAULT_REGION"
  "AWS_SECRET_ACCESS_KEY"
)

missing=0
for v in "${required_vars[@]}"; do
  if [[ -z "${!v:-}" ]]; then
    echo "ERROR: Environment variable '$v' is not set." >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "Aborting. Please export all required environment variables." >&2
  exit 1
fi

# ----------------------------
# Build target image URI
# ----------------------------
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"
echo "Building image: ${IMAGE_URI}"

# ----------------------------
# Ensure docker can push to Artifact Registry
# ----------------------------
if ! command -v gcloud >/dev/null 2>&1; then
  echo "ERROR: gcloud not found. Install Google Cloud SDK and authenticate." >&2
  exit 1
fi

# Configure Docker credential helper for this region’s Artifact Registry
gcloud auth configure-docker "${REGION}-docker.pkg.dev" -q

# ----------------------------
# Build
# ----------------------------
docker build \
  --platform "${PLATFORM}" \
  --build-arg APP_SUPABASE_KEY="${APP_SUPABASE_KEY}" \
  --build-arg APP_SUPABASE_URL="${APP_SUPABASE_URL}" \
  --build-arg AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
  --build-arg AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION}" \
  --build-arg AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
  -t "${IMAGE_URI}" .

# ----------------------------
# Push
# ----------------------------
echo "Pushing ${IMAGE_URI} ..."
docker push "${IMAGE_URI}"

echo "Done. Pushed: ${IMAGE_URI}"
