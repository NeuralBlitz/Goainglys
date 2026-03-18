package integrations

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"agents/framework/core"
)

type GitTool struct {
	repoPath string
}

func NewGitTool(repoPath string) *GitTool {
	return &GitTool{repoPath: repoPath}
}

func (t *GitTool) Name() string { return "git" }
func (t *GitTool) Description() string {
	return "Git operations - clone, commit, push, pull, branch, status, log, diff"
}
func (t *GitTool) Parameters() map[string]core.Parameter {
	return map[string]core.Parameter{
		"action": {
			Type:        "string",
			Description: "Action: 'clone', 'commit', 'push', 'pull', 'branch', 'status', 'log', 'diff', 'add', 'checkout'",
			Required:    true,
		},
		"repo": {
			Type:        "string",
			Description: "Repository URL (for clone)",
			Required:    false,
		},
		"message": {
			Type:        "string",
			Description: "Commit message (for commit)",
			Required:    false,
		},
		"path": {
			Type:        "string",
			Description: "File path (for add/checkout)",
			Required:    false,
		},
		"branch": {
			Type:        "string",
			Description: "Branch name (for branch/checkout)",
			Required:    false,
		},
	}
}

func (t *GitTool) Execute(ctx context.Context, args map[string]any) (*core.ToolResult, error) {
	action, _ := args["action"].(string)

	switch action {
	case "clone":
		return t.clone(args)
	case "commit":
		return t.commit(args)
	case "push":
		return t.push(args)
	case "pull":
		return t.pull(args)
	case "branch":
		return t.branch(args)
	case "status":
		return t.status()
	case "log":
		return t.log(args)
	case "diff":
		return t.diff(args)
	case "add":
		return t.add(args)
	case "checkout":
		return t.checkout(args)
	default:
		return &core.ToolResult{Content: "Unknown action: " + action, Success: false}, nil
	}
}

func (t *GitTool) runGit(args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = t.repoPath
	output, err := cmd.CombinedOutput()
	if err != nil {
		return string(output), fmt.Errorf("git error: %v\n%s", err, output)
	}
	return string(output), nil
}

func (t *GitTool) clone(args map[string]any) (*core.ToolResult, error) {
	repo, ok := args["repo"].(string)
	if !ok || repo == "" {
		return &core.ToolResult{Content: "Repository URL is required", Success: false}, nil
	}

	dest := ""
	if d, ok := args["path"].(string); ok {
		dest = d
	}

	cmd := exec.Command("git", "clone", repo)
	if dest != "" {
		cmd = exec.Command("git", "clone", repo, dest)
	}
	cmd.Dir = ""

	output, err := cmd.CombinedOutput()
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Clone failed: %v\n%s", err, output), Success: false}, nil
	}

	return &core.ToolResult{
		Content: fmt.Sprintf("Cloned successfully: %s", string(output)),
		Success: true,
	}, nil
}

func (t *GitTool) commit(args map[string]any) (*core.ToolResult, error) {
	msg, ok := args["message"].(string)
	if !ok || msg == "" {
		return &core.ToolResult{Content: "Commit message is required", Success: false}, nil
	}

	output, err := t.runGit("commit", "-m", msg)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Commit failed: %v", err), Success: false}, nil
	}

	return &core.ToolResult{
		Content: "Committed: " + output,
		Success: true,
	}, nil
}

func (t *GitTool) push(args map[string]any) (*core.ToolResult, error) {
	branch := ""
	if b, ok := args["branch"].(string); ok {
		branch = b
	}

	var err error
	if branch != "" {
		_, err = t.runGit("push", "-u", "origin", branch)
	} else {
		_, err = t.runGit("push")
	}

	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Push failed: %v", err), Success: false}, nil
	}

	return &core.ToolResult{
		Content: "Pushed successfully",
		Success: true,
	}, nil
}

func (t *GitTool) pull(args map[string]any) (*core.ToolResult, error) {
	output, err := t.runGit("pull")
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Pull failed: %v", err), Success: false}, nil
	}

	return &core.ToolResult{
		Content: "Pulled: " + output,
		Success: true,
	}, nil
}

func (t *GitTool) branch(args map[string]any) (*core.ToolResult, error) {
	list, ok := args["list"].(bool)
	if !ok {
		list = false
	}

	var output string
	var err error

	if list {
		output, err = t.runGit("branch", "-a")
	} else {
		name, ok := args["branch"].(string)
		if !ok || name == "" {
			output, err = t.runGit("branch")
		} else {
			output, err = t.runGit("branch", name)
		}
	}

	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Branch operation failed: %v", err), Success: false}, nil
	}

	return &core.ToolResult{
		Content: output,
		Success: true,
	}, nil
}

func (t *GitTool) status() (*core.ToolResult, error) {
	output, err := t.runGit("status", "--short")
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Status failed: %v", err), Success: false}, nil
	}

	if output == "" {
		return &core.ToolResult{
			Content: "Working tree is clean",
			Success: true,
		}, nil
	}

	return &core.ToolResult{
		Content: "Changes:\n" + output,
		Success: true,
	}, nil
}

func (t *GitTool) log(args map[string]any) (*core.ToolResult, error) {
	n := 10
	if limit, ok := args["limit"].(float64); ok {
		n = int(limit)
	}

	format := "%h|%an|%s|%ad"
	if verbose, ok := args["verbose"].(bool); ok && verbose {
		format = "%h|%an|%ae|%s|%ad|%D"
	}

	output, err := t.runGit("log", fmt.Sprintf("-%d", n), fmt.Sprintf("--format=%s", format), "--date=short")
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Log failed: %v", err), Success: false}, nil
	}

	lines := strings.Split(strings.TrimSpace(output), "\n")
	var formatted []string
	for _, line := range lines {
		parts := strings.Split(line, "|")
		if len(parts) >= 4 {
			formatted = append(formatted, fmt.Sprintf("[%s] %s: %s (%s)",
				parts[0], parts[1], parts[2], parts[3]))
		} else {
			formatted = append(formatted, line)
		}
	}

	return &core.ToolResult{
		Content: strings.Join(formatted, "\n"),
		Success: true,
	}, nil
}

func (t *GitTool) diff(args map[string]any) (*core.ToolResult, error) {
	path := ""
	if p, ok := args["path"].(string); ok {
		path = p
	}

	var output string
	var err error

	staged, _ := args["staged"].(bool)
	if staged {
		output, err = t.runGit("diff", "--cached", "--stat")
		if err == nil {
			details, _ := t.runGit("diff", "--cached")
			output += "\n\n" + details
		}
	} else if path != "" {
		output, err = t.runGit("diff", path)
	} else {
		output, err = t.runGit("diff", "--stat")
		if err == nil {
			details, _ := t.runGit("diff")
			output += "\n\n" + details
		}
	}

	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Diff failed: %v", err), Success: false}, nil
	}

	return &core.ToolResult{
		Content: output,
		Success: true,
	}, nil
}

func (t *GitTool) add(args map[string]any) (*core.ToolResult, error) {
	path := "."
	if p, ok := args["path"].(string); ok && p != "" {
		path = p
	}

	_, err := t.runGit("add", path)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Add failed: %v", err), Success: false}, nil
	}

	return &core.ToolResult{
		Content: "Added: " + path,
		Success: true,
	}, nil
}

func (t *GitTool) checkout(args map[string]any) (*core.ToolResult, error) {
	branch, ok := args["branch"].(string)
	if !ok || branch == "" {
		return &core.ToolResult{Content: "Branch name is required", Success: false}, nil
	}

	_, err := t.runGit("checkout", branch)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Checkout failed: %v", err), Success: false}, nil
	}

	return &core.ToolResult{
		Content: "Checked out: " + branch,
		Success: true,
	}, nil
}

type GitHubTool struct {
	token string
}

func NewGitHubTool(token string) *GitHubTool {
	return &GitHubTool{token: token}
}

func (t *GitHubTool) Name() string { return "github" }
func (t *GitHubTool) Description() string {
	return "GitHub API operations - issues, PRs, repos, releases"
}
func (t *GitHubTool) Parameters() map[string]core.Parameter {
	return map[string]core.Parameter{
		"action": {
			Type:        "string",
			Description: "Action: 'issues', 'prs', 'repos', 'releases', 'search'",
			Required:    true,
		},
		"owner": {
			Type:        "string",
			Description: "Repository owner",
			Required:    false,
		},
		"repo": {
			Type:        "string",
			Description: "Repository name",
			Required:    false,
		},
		"query": {
			Type:        "string",
			Description: "Search query",
			Required:    false,
		},
	}
}

func (t *GitHubTool) Execute(ctx context.Context, args map[string]any) (*core.ToolResult, error) {
	action, _ := args["action"].(string)

	switch action {
	case "issues":
		return t.issues(args)
	case "prs":
		return t.prs(args)
	case "repos":
		return t.repos(args)
	case "releases":
		return t.releases(args)
	case "search":
		return t.search(args)
	default:
		return &core.ToolResult{Content: "Unknown action: " + action, Success: false}, nil
	}
}

func (t *GitHubTool) doRequest(method, url string, body *strings.Reader) ([]byte, error) {
	var req *http.Request
	var err error

	if body != nil {
		req, err = http.NewRequest(method, url, body)
	} else {
		req, err = http.NewRequest(method, url, nil)
	}
	if err != nil {
		return nil, err
	}

	req.Header.Set("Accept", "application/vnd.github.v3+json")
	if t.token != "" {
		req.Header.Set("Authorization", "token "+t.token)
	}

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	data := make([]byte, 0, 1024)
	buf := make([]byte, 1024)
	for {
		n, _ := resp.Body.Read(buf)
		if n == 0 {
			break
		}
		data = append(data, buf[:n]...)
	}

	return data, nil
}

func (t *GitHubTool) issues(args map[string]any) (*core.ToolResult, error) {
	owner, _ := args["owner"].(string)
	repo, _ := args["repo"].(string)

	if owner == "" || repo == "" {
		return &core.ToolResult{Content: "Owner and repo are required", Success: false}, nil
	}

	url := fmt.Sprintf("https://api.github.com/repos/%s/%s/issues?state=open&per_page=10", owner, repo)
	data, err := t.doRequest("GET", url, nil)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Request failed: %v", err), Success: false}, nil
	}

	re := regexp.MustCompile(`"title":\s*"([^"]+)".*?"number":\s*(\d+).*?"state":\s*"([^"]+)"`)
	matches := re.FindAllStringSubmatch(string(data), -1)

	var output strings.Builder
	output.WriteString(fmt.Sprintf("Open issues for %s/%s:\n\n", owner, repo))

	for _, match := range matches {
		if len(match) >= 4 {
			output.WriteString(fmt.Sprintf("#%s: %s [%s]\n", match[2], match[1], match[3]))
		}
	}

	if output.Len() == 0 {
		output.WriteString("No open issues found.")
	}

	return &core.ToolResult{Content: output.String(), Success: true}, nil
}

func (t *GitHubTool) prs(args map[string]any) (*core.ToolResult, error) {
	owner, _ := args["owner"].(string)
	repo, _ := args["repo"].(string)

	if owner == "" || repo == "" {
		return &core.ToolResult{Content: "Owner and repo are required", Success: false}, nil
	}

	url := fmt.Sprintf("https://api.github.com/repos/%s/%s/pulls?state=open&per_page=10", owner, repo)
	data, err := t.doRequest("GET", url, nil)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Request failed: %v", err), Success: false}, nil
	}

	re := regexp.MustCompile(`"title":\s*"([^"]+)".*?"number":\s*(\d+)`)
	matches := re.FindAllStringSubmatch(string(data), -1)

	var output strings.Builder
	output.WriteString(fmt.Sprintf("Open PRs for %s/%s:\n\n", owner, repo))

	for _, match := range matches {
		if len(match) >= 3 {
			output.WriteString(fmt.Sprintf("#%s: %s\n", match[2], match[1]))
		}
	}

	if output.Len() == 0 {
		output.WriteString("No open PRs found.")
	}

	return &core.ToolResult{Content: output.String(), Success: true}, nil
}

func (t *GitHubTool) repos(args map[string]any) (*core.ToolResult, error) {
	owner, _ := args["owner"].(string)

	if owner == "" {
		return &core.ToolResult{Content: "Owner is required", Success: false}, nil
	}

	url := fmt.Sprintf("https://api.github.com/users/%s/repos?per_page=10&sort=updated", owner)
	data, err := t.doRequest("GET", url, nil)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Request failed: %v", err), Success: false}, nil
	}

	re := regexp.MustCompile(`"name":\s*"([^"]+)".*?"description":\s*"([^"]*)"`)
	matches := re.FindAllStringSubmatch(string(data), -1)

	var output strings.Builder
	output.WriteString(fmt.Sprintf("Repositories for %s:\n\n", owner))

	for _, match := range matches {
		if len(match) >= 3 {
			desc := match[2]
			if desc == "" {
				desc = "(no description)"
			}
			output.WriteString(fmt.Sprintf("- %s: %s\n", match[1], desc))
		}
	}

	if output.Len() == 0 {
		output.WriteString("No repositories found.")
	}

	return &core.ToolResult{Content: output.String(), Success: true}, nil
}

func (t *GitHubTool) releases(args map[string]any) (*core.ToolResult, error) {
	owner, _ := args["owner"].(string)
	repo, _ := args["repo"].(string)

	if owner == "" || repo == "" {
		return &core.ToolResult{Content: "Owner and repo are required", Success: false}, nil
	}

	url := fmt.Sprintf("https://api.github.com/repos/%s/%s/releases?per_page=5", owner, repo)
	data, err := t.doRequest("GET", url, nil)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Request failed: %v", err), Success: false}, nil
	}

	re := regexp.MustCompile(`"tag_name":\s*"([^"]+)".*?"name":\s*"([^"]*)"`)
	matches := re.FindAllStringSubmatch(string(data), -1)

	var output strings.Builder
	output.WriteString(fmt.Sprintf("Recent releases for %s/%s:\n\n", owner, repo))

	for _, match := range matches {
		if len(match) >= 3 {
			name := match[2]
			if name == "" {
				name = match[1]
			}
			output.WriteString(fmt.Sprintf("- %s: %s\n", match[1], name))
		}
	}

	if output.Len() == 0 {
		output.WriteString("No releases found.")
	}

	return &core.ToolResult{Content: output.String(), Success: true}, nil
}

func (t *GitHubTool) search(args map[string]any) (*core.ToolResult, error) {
	query, _ := args["query"].(string)
	if query == "" {
		return &core.ToolResult{Content: "Query is required", Success: false}, nil
	}

	url := fmt.Sprintf("https://api.github.com/search/repositories?q=%s&per_page=5", query)
	data, err := t.doRequest("GET", url, nil)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Search failed: %v", err), Success: false}, nil
	}

	re := regexp.MustCompile(`"full_name":\s*"([^"]+)".*?"description":\s*"([^"]*)"`)
	matches := re.FindAllStringSubmatch(string(data), -1)

	var output strings.Builder
	output.WriteString(fmt.Sprintf("Search results for '%s':\n\n", query))

	for _, match := range matches {
		if len(match) >= 3 {
			desc := match[2]
			if desc == "" {
				desc = "(no description)"
			}
			output.WriteString(fmt.Sprintf("- %s: %s\n", match[1], desc))
		}
	}

	if output.Len() == 0 {
		output.WriteString("No results found.")
	}

	return &core.ToolResult{Content: output.String(), Success: true}, nil
}

func GetGitUserInfo(repoPath string) (string, string, error) {
	cmd := exec.Command("git", "config", "user.name")
	cmd.Dir = repoPath
	name, _ := cmd.Output()

	cmd = exec.Command("git", "config", "user.email")
	cmd.Dir = repoPath
	email, _ := cmd.Output()

	return strings.TrimSpace(string(name)), strings.TrimSpace(string(email)), nil
}

func IsGitRepo(path string) bool {
	gitPath := filepath.Join(path, ".git")
	if _, err := os.Stat(gitPath); err == nil {
		return true
	}

	cmd := exec.Command("git", "rev-parse", "--git-dir")
	cmd.Dir = path
	err := cmd.Run()
	return err == nil
}

func GetCurrentBranch(repoPath string) (string, error) {
	cmd := exec.Command("git", "rev-parse", "--abbrev-ref", "HEAD")
	cmd.Dir = repoPath
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(output)), nil
}

func GetRepoURL(repoPath string) (string, error) {
	cmd := exec.Command("git", "config", "--get", "remote.origin.url")
	cmd.Dir = repoPath
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(output)), nil
}
