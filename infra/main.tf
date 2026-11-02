###############################################################################
# CityPark RAG Chatbot — container deployment (Docker provider)
#
# This describes a LOCAL Docker deployment equivalent to docker-compose.yml,
# expressed as Terraform so it can be adapted to remote container platforms
# (AWS ECS, GCP Cloud Run, Azure ACI) by swapping the provider block.
#
# Prerequisites:
#   terraform init   # pulls kreuzwerker/docker provider
#   terraform apply  # creates network, volume, and containers
###############################################################################

terraform {
  required_version = ">= 1.5"

  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {}

# ── Build the shared image ────────────────────────────────────────────────────

resource "docker_image" "parking_rag" {
  name = "${var.image_name}:${var.image_tag}"

  build {
    context    = "${path.module}/.."
    dockerfile = "Dockerfile"
    target     = "runtime"
    tag        = ["${var.image_name}:${var.image_tag}"]
  }

  triggers = {
    # Rebuild when requirements change.
    requirements_hash = filemd5("${path.module}/../requirements.txt")
  }
}

# ── Shared network ────────────────────────────────────────────────────────────

resource "docker_network" "parking_net" {
  name = "${var.project_name}_network"
}

# ── Persistent data volume ────────────────────────────────────────────────────

resource "docker_volume" "parking_data" {
  name = "${var.project_name}_data"
}

# ── API container ─────────────────────────────────────────────────────────────

resource "docker_container" "api" {
  name    = "${var.project_name}_api"
  image   = docker_image.parking_rag.image_id
  restart = "unless-stopped"

  command = [
    "uvicorn", "app.api.main:app",
    "--host", "0.0.0.0",
    "--port", "8000",
  ]

  networks_advanced {
    name    = docker_network.parking_net.name
    aliases = ["api"]
  }

  ports {
    internal = 8000
    external = var.api_port
  }

  volumes {
    volume_name    = docker_volume.parking_data.name
    container_path = "/data"
  }

  env = [
    "CHROMA_COLLECTION=${var.chroma_collection}",
    "CHROMA_PERSIST_DIR=/data/chroma_db",
    "DATABASE_URL=sqlite:////data/parking.db",
    "OPENAI_API_KEY=${var.openai_api_key}",
    "OPENAI_MODEL=${var.openai_model}",
    "LOG_LEVEL=INFO",
  ]

  healthcheck {
    test         = ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval     = "30s"
    timeout      = "10s"
    retries      = 5
    start_period = "60s"
  }
}

# ── UI container ──────────────────────────────────────────────────────────────

resource "docker_container" "ui" {
  name    = "${var.project_name}_ui"
  image   = docker_image.parking_rag.image_id
  restart = "unless-stopped"

  command = [
    "streamlit", "run", "app/ui/streamlit_app.py",
    "--server.port=8501",
    "--server.address=0.0.0.0",
    "--server.headless=true",
  ]

  networks_advanced {
    name    = docker_network.parking_net.name
    aliases = ["ui"]
  }

  ports {
    internal = 8501
    external = var.ui_port
  }

  env = [
    "API_URL=http://api:8000",
  ]

  depends_on = [docker_container.api]
}
