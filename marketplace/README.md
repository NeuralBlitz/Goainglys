# MCP Apps & Agents Marketplace

A comprehensive marketplace for MCP (Model Context Protocol) apps and agents, built entirely in Go.

## Overview

This marketplace provides:
- A registry for MCP-compatible apps and agents
- REST API for app management
- Web UI for browsing and searching apps
- Built-in MCP tools for interacting with the marketplace
- Support for multiple agent frameworks

## Architecture

```
marketplace/
├── api/           - REST API server
├── apps/          - App registry and models
├── mcp/           - MCP protocol implementation
├── web/           - Web UI (HTML/CSS/JS)
└── main.go        - Entry point
```

## MCP Protocol

The MCP (Model Context Protocol) enables communication between AI agents and external tools. This marketplace implements:

- **Tools**: Executable functions that agents can call
- **Resources**: Data sources that agents can read
- **Prompts**: Pre-defined templates for common tasks

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/apps` | List all apps |
| GET | `/api/apps/featured` | Get featured apps |
| GET | `/api/apps/trending` | Get trending apps |
| GET | `/api/apps/search?q=` | Search apps |
| GET | `/api/apps/categories` | List categories |
| GET | `/api/apps/{id}` | Get app details |
| POST | `/api/apps` | Register new app |
| PUT | `/api/apps/{id}` | Update app |
| DELETE | `/api/apps/{id}` | Delete app |
| GET | `/api/stats` | Marketplace statistics |
| GET | `/health` | Health check |

## App Categories

- **Agents**: Autonomous agents for various tasks
- **Tools**: Utility tools and integrations
- **Services**: Backend services and APIs
- **Dashboards**: Monitoring and visualization
- **Utilities**: Helper applications

## Built-in MCP Tools

The marketplace provides the following MCP tools:

- `search_apps`: Search for apps in the marketplace
- `get_app_details`: Get detailed information about an app
- `list_categories`: List all app categories
- `get_featured_apps`: Get featured or trending apps
- `get_marketplace_stats`: Get marketplace statistics

## Running the Marketplace

```bash
go run main.go
```

The server starts on port 8080:
- API: http://localhost:8080/api/
- Web UI: http://localhost:8080/web/index.html

## Integrating with Agents

Connect to the marketplace MCP server from your agent:

```go
config := mcp.ClientConfig{
    ServerURL: "http://localhost:8080",
}

// Use with your agent framework
```

## Sample Apps

The marketplace includes sample apps:

1. **Code Assistant** - AI-powered code review and suggestions
2. **Data Analyst** - Analyze datasets and generate insights
3. **Research Agent** - Academic research assistant
4. **DevOps Copilot** - Infrastructure and deployment automation
5. **Customer Support Bot** - Intelligent customer service agent
6. **ML Training Monitor** - Monitor model training jobs
7. **API Tester** - Test and validate API endpoints
8. **Documentation Writer** - Generate and maintain documentation

## License

MIT
