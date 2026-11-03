###############################################################################
# Input variables — Stage 1 + Stage 2
###############################################################################

variable "project_name" {
  description = "Prefix applied to all Docker resources (network, volume, containers)."
  type        = string
  default     = "parking_rag"
}

variable "image_name" {
  description = "Docker image name."
  type        = string
  default     = "parking-rag-api"
}

variable "image_tag" {
  description = "Docker image tag."
  type        = string
  default     = "latest"
}

# ── Ports ─────────────────────────────────────────────────────────────────────

variable "api_port" {
  description = "Host port for the FastAPI backend."
  type        = number
  default     = 8000
}

variable "ui_port" {
  description = "Host port for the user Streamlit chat UI."
  type        = number
  default     = 8501
}

variable "admin_ui_port" {
  description = "Host port for the admin Streamlit dashboard."
  type        = number
  default     = 8502
}

variable "mailpit_ui_port" {
  description = "Host port for the Mailpit web UI (optional SMTP capture)."
  type        = number
  default     = 8025
}

variable "mailpit_smtp_port" {
  description = "Host port for the Mailpit SMTP listener."
  type        = number
  default     = 1025
}

# ── RAG / LLM ─────────────────────────────────────────────────────────────────

variable "chroma_collection" {
  description = "ChromaDB collection name."
  type        = string
  default     = "parking_static"
}

variable "openai_api_key" {
  description = "OpenAI (or compatible) API key.  Leave as placeholder for offline/MockLLM mode."
  type        = string
  default     = "sk-placeholder"
  sensitive   = true
}

variable "openai_model" {
  description = "LLM model identifier."
  type        = string
  default     = "gpt-4o-mini"
}

# ── Stage 2 — Admin API ───────────────────────────────────────────────────────

variable "admin_api_key" {
  description = "Secret key protecting every /admin/* endpoint.  CHANGE before production."
  type        = string
  default     = "change-me-in-production"
  sensitive   = true
}

variable "admin_api_base_url" {
  description = "Externally reachable base URL of the API (used in email token links)."
  type        = string
  default     = "http://localhost:8000"
}

# ── Stage 2 — Email channel ───────────────────────────────────────────────────

variable "admin_email_enabled" {
  description = "Enable the SMTP email notification channel for admin approvals."
  type        = bool
  default     = false
}

variable "smtp_host" {
  description = "SMTP server hostname.  Use 'mailpit' when running the mailpit container."
  type        = string
  default     = "localhost"
}

variable "smtp_port" {
  description = "SMTP server port."
  type        = number
  default     = 1025
}

variable "smtp_user" {
  description = "SMTP username (empty string = unauthenticated relay)."
  type        = string
  default     = ""
  sensitive   = true
}

variable "smtp_password" {
  description = "SMTP password."
  type        = string
  default     = ""
  sensitive   = true
}

variable "admin_email" {
  description = "Destination email address for admin approval-request notifications."
  type        = string
  default     = "admin@citypark.example.com"
}

variable "email_from" {
  description = "Sender address on admin notification emails."
  type        = string
  default     = "no-reply@citypark.example.com"
}

variable "email_token_secret" {
  description = "HMAC secret for signing single-use email action tokens.  CHANGE before production."
  type        = string
  default     = "change-me-in-production"
  sensitive   = true
}

variable "email_token_max_age_seconds" {
  description = "Expiry window (seconds) for email action tokens."
  type        = number
  default     = 86400
}
