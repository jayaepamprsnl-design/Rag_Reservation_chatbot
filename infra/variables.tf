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

variable "api_port" {
  description = "Host port for the FastAPI backend."
  type        = number
  default     = 8000
}

variable "ui_port" {
  description = "Host port for the Streamlit UI."
  type        = number
  default     = 8501
}

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
