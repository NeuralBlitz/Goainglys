package main

import (
	"fmt"
	"net/http"
	"os"
	"strings"
)

func main() {
	storagePath := "data"
	port := 8081

	if len(os.Args) > 1 {
		if os.Args[1] == "-h" || os.Args[1] == "--help" {
			fmt.Println("Model Registry Server")
			fmt.Println("Usage: go run main.go [port] [storage_path]")
			fmt.Println("  port: HTTP port (default: 8081)")
			fmt.Println("  storage_path: Data storage directory (default: data)")
			return
		}
		if _, err := fmt.Sscanf(os.Args[1], "%d", &port); err != nil {
			storagePath = os.Args[1]
		}
		if len(os.Args) > 2 {
			storagePath = os.Args[2]
		}
	}

	registry := NewModelRegistry(storagePath)
	registry.LoadFromDisk()

	server := NewAPIServer(registry)

	mux := http.NewServeMux()
	mux.Handle("/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasPrefix(r.URL.Path, "/api/") {
			server.ServeHTTP(w, r)
		} else if r.URL.Path == "" || r.URL.Path == "/" {
			http.ServeFile(w, r, "index.html")
		} else {
			http.ServeFile(w, r, r.URL.Path[1:])
		}
	}))

	fmt.Printf("Model Registry starting on http://localhost:%d\n", port)
	fmt.Println("API Endpoints:")
	fmt.Println("  GET    /api/models                    - List all models")
	fmt.Println("  POST   /api/models                    - Create model")
	fmt.Println("  GET    /api/models/:id                - Get model")
	fmt.Println("  DELETE /api/models/:id                - Delete model")
	fmt.Println("  GET    /api/models/:id/versions      - List versions")
	fmt.Println("  POST   /api/models/:id/versions      - Register version")
	fmt.Println("  GET    /api/models/:id/versions/:v   - Get version")
	fmt.Println("  PUT    /api/models/:id/versions/:v    - Set stage")
	fmt.Println("  GET    /api/models/:id/artifact/:v    - Get artifact")
	fmt.Println("  GET    /api/search?q=query            - Search models")
	fmt.Println("  GET    /api/stages?stage=PRODUCTION  - Get by stage")
	fmt.Println("  GET    /api/summary                   - Registry summary")
	fmt.Println("  GET    /                             - Web UI")

	if err := http.ListenAndServe(fmt.Sprintf(":%d", port), mux); err != nil {
		fmt.Printf("Server error: %v\n", err)
	}
}
