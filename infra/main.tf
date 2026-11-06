###############################################################################
# CityPark RAG Chatbot — Stage 3 container deployment (Docker provider)
#
# Describes a LOCAL Docker deployment equivalent to docker-compose.yml.
# Adapt to remote platforms (AWS ECS, GCP Cloud Run, Azure ACI) by swapping
# the provider block and the resource types.
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

# ── Shared image ──────────────────────────────────────────────────────────────

resource "docker_image" "parking_rag" {
  name = "${var.image_name}:${var.image_tag}"

  build {
    context    = "${path.module}/.."
    dockerfile = "Dockerfile"
    target     = "runtime"
    tag        = ["${var.image_name}:${var.image_tag}"]
  }

  triggers = {
    requirements_hash = filemd5("${path.module}/../requirements.txt")
  }
}

# ── Shared network ────────────────────────────────────────────────────────────

resource "docker_network" "parking_net" {
  name = "${var.project_name}_network"
}

# ── Internal-only network for the MCP server (Stage 3) ───────────────────────
# internal = true: no route to the host or the internet. Only the API and the
# MCP server are attached, and the MCP server publishes no ports.

resource "docker_network" "mcp_net" {
  name     = "${var.project_name}_mcp_network"
  internal = true
}

# ── Persistent data volumes ───────────────────────────────────────────────────

resource "docker_volume" "parking_data" {
  name = "${var.project_name}_data"
}

# data/output of the MCP server (confirmed_reservations.txt + .ids/.lock files)
resource "docker_volume" "mcp_output" {
  name = "${var.project_name}_mcp_output"
}

# ── MCP reservations server (Stage 3) ─────────────────────────────────────────
# Appends admin-approved reservations to a text file. No `ports` block on
# purpose: it is reachable only from the API over the internal network.

resource "docker_container" "mcp_server" {
  name    = "${var.project_name}_mcp_server"
  image   = docker_image.parking_rag.image_id
  restart = "unless-stopped"

  entrypoint = ["python", "scripts/run_mcp_server.py"]

  networks_advanced {
    name    = docker_network.mcp_net.name
    aliases = ["mcp-server"]
  }

  volumes {
    volume_name    = docker_volume.mcp_output.name
    container_path = "/data/output"
  }

  env = [
    "MCP_API_KEY=${var.mcp_api_key}",
    "MCP_HOST=0.0.0.0", # inside the container only; the network is internal
    "MCP_PORT=${var.mcp_port}",
    "MCP_RATE_LIMIT_PER_MINUTE=${var.mcp_rate_limit_per_minute}",
    "RESERVATIONS_FILE=/data/output/confirmed_reservations.txt",
    "ANONYMIZED_TELEMETRY=false",
  ]

  healthcheck {
    test         = ["CMD", "curl", "-f", "http://localhost:${var.mcp_port}/health"]
    interval     = "15s"
    timeout      = "5s"
    retries      = 5
    start_period = "20s"
  }
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

  # Stage 3: second network so the API can reach the MCP server.
  networks_advanced {
    name    = docker_network.mcp_net.name
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
    # Stage 1
    "CHROMA_COLLECTION=${var.chroma_collection}",
    "CHROMA_PERSIST_DIR=/data/chroma_db",
    "DATABASE_URL=sqlite:////data/parking.db",
    "OPENAI_API_KEY=${var.openai_api_key}",
    "OPENAI_MODEL=${var.openai_model}",
    "LOG_LEVEL=INFO",
    # Stage 2 — Admin API
    "ADMIN_API_KEY=${var.admin_api_key}",
    "ADMIN_API_BASE_URL=${var.admin_api_base_url}",
    # Stage 2 — Email channel
    "ADMIN_EMAIL_ENABLED=${var.admin_email_enabled}",
    "SMTP_HOST=${var.smtp_host}",
    "SMTP_PORT=${var.smtp_port}",
    "SMTP_USER=${var.smtp_user}",
    "SMTP_PASSWORD=${var.smtp_password}",
    "ADMIN_EMAIL=${var.admin_email}",
    "EMAIL_FROM=${var.email_from}",
    "EMAIL_TOKEN_SECRET=${var.email_token_secret}",
    "EMAIL_TOKEN_MAX_AGE_SECONDS=${var.email_token_max_age_seconds}",
    # Stage 3 — MCP reservations export (same key as the MCP server)
    "MCP_SERVER_URL=http://mcp-server:${var.mcp_port}/mcp",
    "MCP_API_KEY=${var.mcp_api_key}",
  ]

  # No hard dependency on the MCP server: approvals must keep working during an
  # MCP outage (the retry worker catches up afterwards).

  healthcheck {
    test         = ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval     = "30s"
    timeout      = "10s"
    retries      = 5
    start_period = "90s"
  }
}

# ── User Streamlit chat UI ────────────────────────────────────────────────────

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

# ── Admin Streamlit dashboard ─────────────────────────────────────────────────

resource "docker_container" "admin_ui" {
  name    = "${var.project_name}_admin_ui"
  image   = docker_image.parking_rag.image_id
  restart = "unless-stopped"

  command = [
    "streamlit", "run", "app/ui/admin_app.py",
    "--server.port=8502",
    "--server.address=0.0.0.0",
    "--server.headless=true",
  ]

  networks_advanced {
    name    = docker_network.parking_net.name
    aliases = ["admin-ui"]
  }

  ports {
    internal = 8502
    external = var.admin_ui_port
  }

  env = [
    "API_URL=http://api:8000",
    # Pre-fills the login form in the Streamlit admin app.
    "ADMIN_API_KEY=${var.admin_api_key}",
  ]

  depends_on = [docker_container.api]
}

# ── Mailpit (optional — SMTP capture for local dev) ───────────────────────────
# Uncomment to deploy.  Set admin_email_enabled=true and smtp_host="mailpit"
# to route email notifications through Mailpit.
#
# resource "docker_container" "mailpit" {
#   name    = "${var.project_name}_mailpit"
#   image   = "axllent/mailpit:latest"
#   restart = "unless-stopped"
#
#   networks_advanced {
#     name    = docker_network.parking_net.name
#     aliases = ["mailpit"]
#   }
#
#   ports {
#     internal = 8025
#     external = var.mailpit_ui_port
#   }
#
#   ports {
#     internal = 1025
#     external = var.mailpit_smtp_port
#   }
# }
