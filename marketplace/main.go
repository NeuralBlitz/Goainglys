package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"marketplace/api"
	"marketplace/apps"
	"marketplace/mcp"
)

func main() {
	fmt.Println("╔═══════════════════════════════════════════════════════════════╗")
	fmt.Println("║          MCP Apps & Agents Marketplace                     ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Built with Go | Model Context Protocol | REST API        ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	ctx := context.Background()

	server := api.NewMarketplaceServer(":8080")
	registry := server.GetRegistry()

	fmt.Println("Initializing MCP servers...")
	initMCPServers(registry)

	fmt.Println("Starting marketplace server...")
	fmt.Println()

	go func() {
		if err := server.Start(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server error: %v", err)
		}
	}()

	go func() {
		http.Handle("/", staticFileServer())
		fs := http.FileServer(http.Dir("web"))
		http.Handle("/web/", http.StripPrefix("/web/", fs))
		fmt.Println("Web UI: http://localhost:8080/web/index.html")
	}()

	fmt.Println("API Endpoints:")
	fmt.Println("  GET  /api/apps              - List all apps")
	fmt.Println("  GET  /api/apps/featured    - Featured apps")
	fmt.Println("  GET  /api/apps/trending     - Trending apps")
	fmt.Println("  GET  /api/apps/search?q=    - Search apps")
	fmt.Println("  GET  /api/apps/categories   - List categories")
	fmt.Println("  GET  /api/apps/{id}        - Get app details")
	fmt.Println("  POST /api/apps             - Register new app")
	fmt.Println("  GET  /api/stats            - Marketplace stats")
	fmt.Println("  GET  /health               - Health check")
	fmt.Println()
	fmt.Println("Marketplace is ready! Press Ctrl+C to stop.")
	fmt.Println()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	fmt.Println("\nShutting down marketplace...")

	shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	if err := server.Stop(shutdownCtx); err != nil {
		log.Printf("Server shutdown error: %v", err)
	}

	fmt.Println("Marketplace stopped.")
}

func initMCPServers(registry *apps.AppRegistry) {
	server := mcp.NewMCPServer("marketplace-tools", "1.0.0")

	server.RegisterTool("search_apps", "Search for apps in the marketplace", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{
				"type":        "string",
				"description": "Search query",
			},
		},
	}, func(ctx context.Context, args map[string]any) (*mcp.ToolCallResult, error) {
		query, _ := args["query"].(string)
		results := registry.SearchApps(query)

		var sb strings.Builder
		sb.WriteString("Search results for '")
		sb.WriteString(query)
		sb.WriteString("':\n\n")

		for i, app := range results {
			sb.WriteString(fmt.Sprintf("%d. %s (v%s)\n", i+1, app.Name, app.Version))
			sb.WriteString("   ")
			sb.WriteString(app.Description)
			sb.WriteString("\n\n")
		}

		return &mcp.ToolCallResult{
			Content: []mcp.ContentBlock{
				{Type: "text", Text: sb.String()},
			},
		}, nil
	})

	server.RegisterTool("get_app_details", "Get detailed information about an app", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"app_id": map[string]any{
				"type":        "string",
				"description": "The app ID",
			},
		},
	}, func(ctx context.Context, args map[string]any) (*mcp.ToolCallResult, error) {
		appID, _ := args["app_id"].(string)
		app, ok := registry.GetApp(appID)

		if !ok {
			return &mcp.ToolCallResult{
				Content: []mcp.ContentBlock{
					{Type: "text", Text: "App not found: " + appID},
				},
				IsError: true,
			}, nil
		}

		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("# %s\n\n", app.Name))
		sb.WriteString(fmt.Sprintf("**Version:** %s\n", app.Version))
		sb.WriteString(fmt.Sprintf("**Author:** %s\n", app.Author))
		sb.WriteString(fmt.Sprintf("**Category:** %s\n", app.Category))
		sb.WriteString(fmt.Sprintf("**Rating:** %.1f ⭐\n", app.Rating))
		sb.WriteString(fmt.Sprintf("**Downloads:** %d\n\n", app.Downloads))
		sb.WriteString(fmt.Sprintf("## Description\n\n%s\n\n", app.Description))
		sb.WriteString("## Tags\n\n")
		for _, tag := range app.Tags {
			sb.WriteString(fmt.Sprintf("- %s\n", tag))
		}

		if app.MCPServer != nil {
			sb.WriteString("\n## MCP Tools\n\n")
			for _, tool := range app.MCPServer.Tools {
				sb.WriteString(fmt.Sprintf("- **%s:** %s\n", tool.Name, tool.Description))
			}
		}

		return &mcp.ToolCallResult{
			Content: []mcp.ContentBlock{
				{Type: "text", Text: sb.String()},
			},
		}, nil
	})

	server.RegisterTool("list_categories", "List all app categories", map[string]any{
		"type":       "object",
		"properties": map[string]any{},
	}, func(ctx context.Context, args map[string]any) (*mcp.ToolCallResult, error) {
		categories := registry.GetCategories()

		var sb strings.Builder
		sb.WriteString("# App Categories\n\n")
		for i, cat := range categories {
			sb.WriteString(fmt.Sprintf("%d. %s\n", i+1, cat))
		}

		return &mcp.ToolCallResult{
			Content: []mcp.ContentBlock{
				{Type: "text", Text: sb.String()},
			},
		}, nil
	})

	server.RegisterTool("get_featured_apps", "Get featured/trending apps", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"type": map[string]any{
				"type":        "string",
				"description": "Type: 'featured' or 'trending'",
			},
			"limit": map[string]any{
				"type":        "number",
				"description": "Maximum number of apps to return",
			},
		},
	}, func(ctx context.Context, args map[string]any) (*mcp.ToolCallResult, error) {
		appType, _ := args["type"].(string)
		limit, _ := args["limit"].(float64)

		var apps []*apps.App
		if appType == "trending" {
			apps = registry.GetTrendingApps(int(limit))
		} else {
			apps = registry.GetFeaturedApps(int(limit))
		}

		var sb strings.Builder
		sb.WriteString("# ")
		if appType == "trending" {
			sb.WriteString("Trending")
		} else {
			sb.WriteString("Featured")
		}
		sb.WriteString(" Apps\n\n")

		for i, app := range apps {
			sb.WriteString(fmt.Sprintf("## %d. %s\n", i+1, app.Name))
			sb.WriteString(fmt.Sprintf("**Rating:** %.1f | **Downloads:** %d\n\n", app.Rating, app.Downloads))
			sb.WriteString(app.Description)
			sb.WriteString("\n\n")
		}

		return &mcp.ToolCallResult{
			Content: []mcp.ContentBlock{
				{Type: "text", Text: sb.String()},
			},
		}, nil
	})

	server.RegisterTool("get_marketplace_stats", "Get marketplace statistics", map[string]any{
		"type":       "object",
		"properties": map[string]any{},
	}, func(ctx context.Context, args map[string]any) (*mcp.ToolCallResult, error) {
		stats := registry.GetStats()

		var sb strings.Builder
		sb.WriteString("# Marketplace Statistics\n\n")
		sb.WriteString(fmt.Sprintf("| Metric | Value |\n"))
		sb.WriteString(fmt.Sprintf("|--------|-------|\n"))
		sb.WriteString(fmt.Sprintf("| Total Apps | %d |\n", stats.TotalApps))
		sb.WriteString(fmt.Sprintf("| Total Downloads | %d |\n", stats.TotalDownloads))
		sb.WriteString(fmt.Sprintf("| Total Installed | %d |\n", stats.TotalInstalled))
		sb.WriteString(fmt.Sprintf("| Average Rating | %.2f |\n", stats.AverageRating))

		return &mcp.ToolCallResult{
			Content: []mcp.ContentBlock{
				{Type: "text", Text: sb.String()},
			},
		}, nil
	})

	fmt.Printf("  ✓ Registered marketplace tools\n")
}

func staticFileServer() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			http.Redirect(w, r, "/web/index.html", http.StatusFound)
			return
		}

		path := r.URL.Path
		if strings.HasPrefix(path, "/web/") {
			filePath := strings.TrimPrefix(path, "/web/")
			if filePath == "" || filePath == "index.html" {
				http.ServeFile(w, r, "web/index.html")
				return
			}

			if _, err := os.Stat("web/" + filePath); err == nil {
				http.ServeFile(w, r, "web/"+filePath)
				return
			}
		}

		http.NotFound(w, r)
	})
}
