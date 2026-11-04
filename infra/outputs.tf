output "api_url" {
  description = "FastAPI backend URL."
  value       = "http://localhost:${var.api_port}"
}

output "ui_url" {
  description = "User Streamlit chat UI URL."
  value       = "http://localhost:${var.ui_port}"
}

output "admin_ui_url" {
  description = "Admin Streamlit dashboard URL."
  value       = "http://localhost:${var.admin_ui_port}"
}

output "docker_network" {
  description = "Internal Docker network name."
  value       = docker_network.parking_net.name
}

output "data_volume" {
  description = "Named Docker volume holding SQLite DB + ChromaDB."
  value       = docker_volume.parking_data.name
}

output "mcp_internal_url" {
  description = "MCP endpoint as seen from the API container (internal network only; not reachable from the host)."
  value       = "http://mcp-server:${var.mcp_port}/mcp"
}

output "mcp_output_volume" {
  description = "Named Docker volume holding confirmed_reservations.txt."
  value       = docker_volume.mcp_output.name
}

output "checkpoint_db" {
  description = "LangGraph checkpoint DB path inside the API container (on the data volume)."
  value       = var.checkpoint_db
}
