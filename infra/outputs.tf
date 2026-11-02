output "api_url" {
  description = "FastAPI backend URL."
  value       = "http://localhost:${var.api_port}"
}

output "ui_url" {
  description = "Streamlit UI URL."
  value       = "http://localhost:${var.ui_port}"
}

output "docker_network" {
  description = "Internal Docker network name."
  value       = docker_network.parking_net.name
}

output "data_volume" {
  description = "Named Docker volume holding SQLite + ChromaDB."
  value       = docker_volume.parking_data.name
}
