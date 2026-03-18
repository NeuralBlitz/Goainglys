package apps

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"marketplace/mcp"
)

type App struct {
	ID           string           `json:"id"`
	Name         string           `json:"name"`
	Description  string           `json:"description"`
	Version      string           `json:"version"`
	Author       string           `json:"author"`
	Category     string           `json:"category"`
	Tags         []string         `json:"tags"`
	Icon         string           `json:"icon,omitempty"`
	Homepage     string           `json:"homepage,omitempty"`
	Repository   string           `json:"repository,omitempty"`
	License      string           `json:"license"`
	Rating       float64          `json:"rating"`
	Downloads    int64            `json:"downloads"`
	Installed    int64            `json:"installed"`
	Favorite     bool             `json:"favorite"`
	MCPServer    *MCPServerConfig `json:"mcpServer,omitempty"`
	Dependencies []string         `json:"dependencies,omitempty"`
	CreatedAt    time.Time        `json:"createdAt"`
	UpdatedAt    time.Time        `json:"updatedAt"`
	PublishedAt  time.Time        `json:"publishedAt"`
}

type MCPServerConfig struct {
	Command string            `json:"command"`
	Args    []string          `json:"args,omitempty"`
	Env     map[string]string `json:"env,omitempty"`
	Tools   []mcp.Tool        `json:"tools,omitempty"`
}

type AppReview struct {
	ID        string    `json:"id"`
	AppID     string    `json:"appId"`
	Author    string    `json:"author"`
	Rating    float64   `json:"rating"`
	Title     string    `json:"title"`
	Content   string    `json:"content"`
	CreatedAt time.Time `json:"createdAt"`
}

type AppStats struct {
	TotalApps      int64   `json:"totalApps"`
	TotalDownloads int64   `json:"totalDownloads"`
	TotalInstalled int64   `json:"totalInstalled"`
	AverageRating  float64 `json:"averageRating"`
}

type AppRegistry struct {
	mu         sync.RWMutex
	apps       map[string]*App
	reviews    map[string][]AppReview
	indexes    map[string][]string
	categories []string
}

func NewAppRegistry() *AppRegistry {
	registry := &AppRegistry{
		apps:    make(map[string]*App),
		reviews: make(map[string][]AppReview),
		indexes: make(map[string][]string),
		categories: []string{
			"AI Agents",
			"Productivity",
			"Development",
			"Data Analysis",
			"Communication",
			"Automation",
			"Creative",
			"Research",
		},
	}

	registry.seedSampleApps()

	return registry
}

func (r *AppRegistry) seedSampleApps() {
	sampleApps := []App{
		{
			ID:          "react-code-agent",
			Name:        "ReAct Code Agent",
			Description: "An intelligent code generation agent using ReAct reasoning pattern",
			Version:     "1.2.0",
			Author:      "AI Studio",
			Category:    "Development",
			Tags:        []string{"coding", "agent", "react", "llm"},
			Rating:      4.8,
			Downloads:   15000,
			Installed:   3400,
			License:     "MIT",
			MCPServer: &MCPServerConfig{
				Command: "npx",
				Args:    []string{"@ai/coding-agent"},
				Tools: []mcp.Tool{
					{Name: "write_file", Description: "Write content to a file"},
					{Name: "read_file", Description: "Read content from a file"},
					{Name: "run_command", Description: "Execute a shell command"},
				},
			},
			CreatedAt:   time.Now().AddDate(0, -6, 0),
			UpdatedAt:   time.Now().AddDate(0, 0, -5),
			PublishedAt: time.Now().AddDate(0, -6, 0),
		},
		{
			ID:          "data-analyzer",
			Name:        "Data Analyzer Agent",
			Description: "Automated data analysis and visualization agent",
			Version:     "2.1.0",
			Author:      "DataLabs",
			Category:    "Data Analysis",
			Tags:        []string{"data", "analysis", "visualization", "pandas"},
			Rating:      4.6,
			Downloads:   8500,
			Installed:   1200,
			License:     "Apache-2.0",
			MCPServer: &MCPServerConfig{
				Command: "python",
				Args:    []string{"-m", "mcp_data_analyzer"},
				Tools: []mcp.Tool{
					{Name: "load_csv", Description: "Load CSV file into dataframe"},
					{Name: "analyze_stats", Description: "Calculate statistics"},
					{Name: "plot_chart", Description: "Generate visualization"},
				},
			},
			CreatedAt:   time.Now().AddDate(0, -4, 0),
			UpdatedAt:   time.Now().AddDate(0, -1, 0),
			PublishedAt: time.Now().AddDate(0, -4, 0),
		},
		{
			ID:          "research-assistant",
			Name:        "Research Assistant",
			Description: "AI-powered research assistant for academic papers and literature review",
			Version:     "1.5.0",
			Author:      "ScholarAI",
			Category:    "Research",
			Tags:        []string{"research", "papers", "academic", "literature"},
			Rating:      4.9,
			Downloads:   22000,
			Installed:   5600,
			License:     "MIT",
			MCPServer: &MCPServerConfig{
				Command: "python",
				Args:    []string{"-m", "mcp_research"},
				Tools: []mcp.Tool{
					{Name: "search_papers", Description: "Search academic papers"},
					{Name: "summarize", Description: "Summarize a paper"},
					{Name: "extract_citations", Description: "Extract citations"},
				},
			},
			CreatedAt:   time.Now().AddDate(0, -8, 0),
			UpdatedAt:   time.Now().AddDate(0, 0, -2),
			PublishedAt: time.Now().AddDate(0, -8, 0),
		},
		{
			ID:          "email-assistant",
			Name:        "Email Assistant",
			Description: "Smart email management and response agent",
			Version:     "1.0.0",
			Author:      "ProductivityPro",
			Category:    "Communication",
			Tags:        []string{"email", "communication", "automation", "productivity"},
			Rating:      4.4,
			Downloads:   12000,
			Installed:   2800,
			License:     "MIT",
			MCPServer: &MCPServerConfig{
				Command: "python",
				Args:    []string{"-m", "mcp_email"},
				Tools: []mcp.Tool{
					{Name: "send_email", Description: "Send an email"},
					{Name: "read_emails", Description: "Read inbox emails"},
					{Name: "draft_reply", Description: "Draft a reply"},
				},
			},
			CreatedAt:   time.Now().AddDate(0, -2, 0),
			UpdatedAt:   time.Now().AddDate(0, 0, -10),
			PublishedAt: time.Now().AddDate(0, -2, 0),
		},
		{
			ID:          "git-agent",
			Name:        "Git Agent",
			Description: "Automated Git operations and repository management",
			Version:     "1.3.0",
			Author:      "DevTools Inc",
			Category:    "Development",
			Tags:        []string{"git", "version-control", "automation", "devops"},
			Rating:      4.7,
			Downloads:   18000,
			Installed:   4100,
			License:     "MIT",
			MCPServer: &MCPServerConfig{
				Command: "python",
				Args:    []string{"-m", "mcp_git"},
				Tools: []mcp.Tool{
					{Name: "git_status", Description: "Check git status"},
					{Name: "git_commit", Description: "Create a commit"},
					{Name: "git_push", Description: "Push to remote"},
					{Name: "git_pull", Description: "Pull from remote"},
					{Name: "git_branch", Description: "Manage branches"},
				},
			},
			CreatedAt:   time.Now().AddDate(0, -5, 0),
			UpdatedAt:   time.Now().AddDate(0, 0, -3),
			PublishedAt: time.Now().AddDate(0, -5, 0),
		},
		{
			ID:          "creative-writer",
			Name:        "Creative Writer Agent",
			Description: "AI creative writing assistant for stories, articles, and content",
			Version:     "2.0.0",
			Author:      "CreativeAI",
			Category:    "Creative",
			Tags:        []string{"writing", "creative", "content", "stories"},
			Rating:      4.5,
			Downloads:   9500,
			Installed:   1900,
			License:     "MIT",
			MCPServer: &MCPServerConfig{
				Command: "python",
				Args:    []string{"-m", "mcp_creative"},
				Tools: []mcp.Tool{
					{Name: "write_story", Description: "Write a creative story"},
					{Name: "edit_text", Description: "Edit existing text"},
					{Name: "generate_idea", Description: "Generate story ideas"},
				},
			},
			CreatedAt:   time.Now().AddDate(0, -3, 0),
			UpdatedAt:   time.Now().AddDate(0, 0, -7),
			PublishedAt: time.Now().AddDate(0, -3, 0),
		},
	}

	for _, app := range sampleApps {
		r.apps[app.ID] = &app
		r.indexApp(&app)
	}
}

func (r *AppRegistry) indexApp(app *App) {
	for _, tag := range app.Tags {
		r.indexes["tag:"+tag] = append(r.indexes["tag:"+tag], app.ID)
	}
	r.indexes["category:"+app.Category] = append(r.indexes["category:"+app.Category], app.ID)
	r.indexes["author:"+app.Author] = append(r.indexes["author:"+app.Author], app.ID)
}

func (r *AppRegistry) GetApp(id string) (*App, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	app, ok := r.apps[id]
	return app, ok
}

func (r *AppRegistry) ListApps(filter *AppFilter) []*App {
	r.mu.RLock()
	defer r.mu.RUnlock()

	results := make([]*App, 0)

	for _, app := range r.apps {
		if filter != nil {
			if filter.Category != "" && app.Category != filter.Category {
				continue
			}
			if filter.Author != "" && app.Author != filter.Author {
				continue
			}
			if filter.Search != "" {
				if !containsIgnoreCase(app.Name, filter.Search) &&
					!containsIgnoreCase(app.Description, filter.Search) {
					continue
				}
			}
			if len(filter.Tags) > 0 {
				if !hasAnyTag(app.Tags, filter.Tags) {
					continue
				}
			}
		}
		results = append(results, app)
	}

	return results
}

func (r *AppRegistry) SearchApps(query string) []*App {
	r.mu.RLock()
	defer r.mu.RUnlock()

	query = toLower(query)
	results := make([]*App, 0)

	for _, app := range r.apps {
		if containsIgnoreCase(app.Name, query) ||
			containsIgnoreCase(app.Description, query) ||
			containsAny(app.Tags, query) {
			results = append(results, app)
		}
	}

	return results
}

func (r *AppRegistry) GetFeaturedApps(limit int) []*App {
	r.mu.RLock()
	defer r.mu.RUnlock()

	apps := make([]*App, 0, len(r.apps))
	for _, app := range r.apps {
		apps = append(apps, app)
	}

	sortByRating(apps)

	if limit > 0 && limit < len(apps) {
		apps = apps[:limit]
	}

	return apps
}

func (r *AppRegistry) GetTrendingApps(limit int) []*App {
	r.mu.RLock()
	defer r.mu.RUnlock()

	apps := make([]*App, 0, len(r.apps))
	for _, app := range r.apps {
		apps = append(apps, app)
	}

	sortByDownloads(apps)

	if limit > 0 && limit < len(apps) {
		apps = apps[:limit]
	}

	return apps
}

func (r *AppRegistry) GetAppsByCategory(category string) []*App {
	r.mu.RLock()
	defer r.mu.RUnlock()

	ids := r.indexes["category:"+category]
	apps := make([]*App, 0, len(ids))

	for _, id := range ids {
		if app, ok := r.apps[id]; ok {
			apps = append(apps, app)
		}
	}

	return apps
}

func (r *AppRegistry) GetCategories() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.categories
}

func (r *AppRegistry) RegisterApp(app *App) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.apps[app.ID]; exists {
		return fmt.Errorf("app already exists: %s", app.ID)
	}

	app.CreatedAt = time.Now()
	app.UpdatedAt = time.Now()
	app.PublishedAt = time.Now()

	r.apps[app.ID] = app
	r.indexApp(app)

	return nil
}

func (r *AppRegistry) UpdateApp(app *App) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	existing, ok := r.apps[app.ID]
	if !ok {
		return fmt.Errorf("app not found: %s", app.ID)
	}

	app.CreatedAt = existing.CreatedAt
	app.UpdatedAt = time.Now()

	r.apps[app.ID] = app

	return nil
}

func (r *AppRegistry) DeleteApp(id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, ok := r.apps[id]; !ok {
		return fmt.Errorf("app not found: %s", id)
	}

	delete(r.apps, id)

	return nil
}

func (r *AppRegistry) AddReview(review *AppReview) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, ok := r.apps[review.AppID]; !ok {
		return fmt.Errorf("app not found: %s", review.AppID)
	}

	review.ID = fmt.Sprintf("review-%d", len(r.reviews[review.AppID]))
	review.CreatedAt = time.Now()

	r.reviews[review.AppID] = append(r.reviews[review.AppID], *review)

	return nil
}

func (r *AppRegistry) GetReviews(appID string) []AppReview {
	r.mu.RLock()
	defer r.mu.RUnlock()

	reviews, ok := r.reviews[appID]
	if !ok {
		return []AppReview{}
	}

	return reviews
}

func (r *AppRegistry) GetStats() *AppStats {
	r.mu.RLock()
	defer r.mu.RUnlock()

	stats := &AppStats{
		TotalApps: int64(len(r.apps)),
	}

	var totalRating float64

	for _, app := range r.apps {
		stats.TotalDownloads += app.Downloads
		stats.TotalInstalled += app.Installed
		totalRating += app.Rating
	}

	if len(r.apps) > 0 {
		stats.AverageRating = totalRating / float64(len(r.apps))
	}

	return stats
}

type AppFilter struct {
	Category string
	Author   string
	Tags     []string
	Search   string
}

func containsIgnoreCase(s, substr string) bool {
	return toLower(s) == toLower(substr)
}

func containsAny(strs []string, substr string) bool {
	lower := toLower(substr)
	for _, s := range strs {
		if toLower(s) == lower {
			return true
		}
	}
	return false
}

func hasAnyTag(appTags, filterTags []string) bool {
	appTagSet := make(map[string]bool)
	for _, t := range appTags {
		appTagSet[toLower(t)] = true
	}
	for _, t := range filterTags {
		if appTagSet[toLower(t)] {
			return true
		}
	}
	return false
}

func toLower(s string) string {
	result := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		result[i] = c
	}
	return string(result)
}

func sortByRating(apps []*App) {
	n := len(apps)
	for i := 1; i < n; i++ {
		for j := i; j > 0 && apps[j].Rating > apps[j-1].Rating; j-- {
			apps[j], apps[j-1] = apps[j-1], apps[j]
		}
	}
}

func sortByDownloads(apps []*App) {
	n := len(apps)
	for i := 1; i < n; i++ {
		for j := i; j > 0 && apps[j].Downloads > apps[j-1].Downloads; j-- {
			apps[j], apps[j-1] = apps[j-1], apps[j]
		}
	}
}

func (a *App) ToJSON() ([]byte, error) {
	return json.MarshalIndent(a, "", "  ")
}

func AppFromJSON(data []byte) (*App, error) {
	var app App
	if err := json.Unmarshal(data, &app); err != nil {
		return nil, err
	}
	return &app, nil
}
