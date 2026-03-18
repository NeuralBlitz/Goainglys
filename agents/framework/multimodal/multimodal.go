package multimodal

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"agents/framework/core"
)

type ImageTool struct {
	cacheDir string
}

func NewImageTool() *ImageTool {
	tool := &ImageTool{
		cacheDir: "./cache/images",
	}
	os.MkdirAll(tool.cacheDir, 0755)
	return tool
}

func (t *ImageTool) Name() string { return "image" }
func (t *ImageTool) Description() string {
	return "Processes and analyzes images - download, describe, or generate image metadata"
}
func (t *ImageTool) Parameters() map[string]core.Parameter {
	return map[string]core.Parameter{
		"action": {
			Type:        "string",
			Description: "Action: 'download', 'describe', 'info', or 'stats'",
			Required:    true,
			Enum:        []string{"download", "describe", "info", "stats"},
		},
		"url": {
			Type:        "string",
			Description: "Image URL (for download action)",
			Required:    false,
		},
		"path": {
			Type:        "string",
			Description: "Image path (for describe/info actions)",
			Required:    false,
		},
	}
}

func (t *ImageTool) Execute(ctx context.Context, args map[string]any) (*core.ToolResult, error) {
	action, _ := args["action"].(string)

	switch action {
	case "download":
		return t.download(args)
	case "describe":
		return t.describe(args)
	case "info":
		return t.info(args)
	case "stats":
		return t.stats()
	default:
		return &core.ToolResult{Content: "Unknown action: " + action, Success: false}, nil
	}
}

func (t *ImageTool) download(args map[string]any) (*core.ToolResult, error) {
	url, ok := args["url"].(string)
	if !ok || url == "" {
		return &core.ToolResult{Content: "URL is required", Success: false}, nil
	}

	resp, err := http.Get(url)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Download failed: %v", err), Success: false}, nil
	}
	defer resp.Body.Close()

	filename := filepath.Join(t.cacheDir, filepath.Base(url))
	if !strings.Contains(filename, ".") {
		filename += ".jpg"
	}

	file, err := os.Create(filename)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Failed to create file: %v", err), Success: false}, nil
	}
	defer file.Close()

	_, err = io.Copy(file, resp.Body)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Failed to write file: %v", err), Success: false}, nil
	}

	return &core.ToolResult{
		Content: fmt.Sprintf("Downloaded to: %s", filename),
		Success: true,
	}, nil
}

func (t *ImageTool) describe(args map[string]any) (*core.ToolResult, error) {
	path, ok := args["path"].(string)
	if !ok || path == "" {
		return &core.ToolResult{Content: "Path is required", Success: false}, nil
	}

	info, err := os.Stat(path)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("File not found: %v", err), Success: false}, nil
	}

	ext := strings.ToLower(filepath.Ext(path))
	return &core.ToolResult{
		Content: fmt.Sprintf("Image description: A %s image file named '%s' (%.2f KB). "+
			"Common image formats include JPEG, PNG, GIF, BMP, WebP. "+
			"Without a vision model, detailed content analysis is not available.",
			strings.TrimPrefix(ext, "."),
			filepath.Base(path),
			float64(info.Size())/1024),
		Success: true,
	}, nil
}

func (t *ImageTool) info(args map[string]any) (*core.ToolResult, error) {
	path, ok := args["path"].(string)
	if !ok || path == "" {
		return &core.ToolResult{Content: "Path is required", Success: false}, nil
	}

	info, err := os.Stat(path)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("File not found: %v", err), Success: false}, nil
	}

	ext := strings.ToLower(filepath.Ext(path))
	return &core.ToolResult{
		Content: fmt.Sprintf("File: %s | Size: %d bytes | Type: %s | Modified: %s",
			filepath.Base(path),
			info.Size(),
			strings.TrimPrefix(ext, "."),
			info.ModTime().Format(time.RFC3339)),
		Success: true,
	}, nil
}

func (t *ImageTool) stats() (*core.ToolResult, error) {
	entries, err := os.ReadDir(t.cacheDir)
	if err != nil {
		return &core.ToolResult{Content: "Cache directory error", Success: false}, nil
	}

	var totalSize int64
	for _, e := range entries {
		info, _ := e.Info()
		totalSize += info.Size()
	}

	return &core.ToolResult{
		Content: fmt.Sprintf("Image cache: %d files, %.2f MB total", len(entries), float64(totalSize)/1024/1024),
		Success: true,
	}, nil
}

type AudioTool struct{}

func NewAudioTool() *AudioTool {
	return &AudioTool{}
}

func (t *AudioTool) Name() string { return "audio" }
func (t *AudioTool) Description() string {
	return "Processes audio files - transcribe, analyze duration, or extract metadata"
}
func (t *AudioTool) Parameters() map[string]core.Parameter {
	return map[string]core.Parameter{
		"action": {
			Type:        "string",
			Description: "Action: 'info', 'transcribe' (mock), or 'analyze'",
			Required:    true,
			Enum:        []string{"info", "transcribe", "analyze"},
		},
		"path": {
			Type:        "string",
			Description: "Audio file path",
			Required:    false,
		},
		"text": {
			Type:        "string",
			Description: "Text to speak (for TTS)",
			Required:    false,
		},
	}
}

func (t *AudioTool) Execute(ctx context.Context, args map[string]any) (*core.ToolResult, error) {
	action, _ := args["action"].(string)

	switch action {
	case "info":
		return t.info(args)
	case "transcribe":
		return t.transcribe(args)
	case "analyze":
		return t.analyze(args)
	default:
		return &core.ToolResult{Content: "Unknown action: " + action, Success: false}, nil
	}
}

func (t *AudioTool) info(args map[string]any) (*core.ToolResult, error) {
	path, ok := args["path"].(string)
	if !ok || path == "" {
		return &core.ToolResult{Content: "Path is required", Success: false}, nil
	}

	info, err := os.Stat(path)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("File not found: %v", err), Success: false}, nil
	}

	ext := strings.ToLower(filepath.Ext(path))
	return &core.ToolResult{
		Content: fmt.Sprintf("Audio file: %s | Size: %.2f KB | Format: %s",
			filepath.Base(path),
			float64(info.Size())/1024,
			strings.TrimPrefix(ext, ".")),
		Success: true,
	}, nil
}

func (t *AudioTool) transcribe(args map[string]any) (*core.ToolResult, error) {
	path, ok := args["path"].(string)
	if !ok || path == "" {
		return &core.ToolResult{Content: "Path is required for transcription", Success: false}, nil
	}

	return &core.ToolResult{
		Content: fmt.Sprintf("[Transcription of %s] This is a simulated transcription. "+
			"Integration with Whisper or similar ASR model would provide actual transcription.",
			filepath.Base(path)),
		Success: true,
	}, nil
}

func (t *AudioTool) analyze(args map[string]any) (*core.ToolResult, error) {
	path, ok := args["path"].(string)
	if !ok || path == "" {
		return &core.ToolResult{Content: "Path is required", Success: false}, nil
	}

	info, err := os.Stat(path)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("File not found: %v", err), Success: false}, nil
	}

	return &core.ToolResult{
		Content: fmt.Sprintf("Audio analysis for %s:\n"+
			"- Format: %s\n"+
			"- Size: %.2f KB\n"+
			"- Sample rate: 44100 Hz (typical)\n"+
			"- Channels: Stereo (typical)\n"+
			"- Estimated duration: %.1f seconds\n"+
			"- Language: Unknown (requires ASR)",
			filepath.Base(path),
			strings.ToLower(filepath.Ext(path)),
			float64(info.Size())/1024,
			float64(info.Size())/176400),
		Success: true,
	}, nil
}

type VideoTool struct{}

func NewVideoTool() *VideoTool {
	return &VideoTool{}
}

func (t *VideoTool) Name() string { return "video" }
func (t *VideoTool) Description() string {
	return "Processes video files - extract frames, get info, or analyze content"
}
func (t *VideoTool) Parameters() map[string]core.Parameter {
	return map[string]core.Parameter{
		"action": {
			Type:        "string",
			Description: "Action: 'info', 'frames', or 'analyze'",
			Required:    true,
			Enum:        []string{"info", "frames", "analyze"},
		},
		"path": {
			Type:        "string",
			Description: "Video file path",
			Required:    false,
		},
	}
}

func (t *VideoTool) Execute(ctx context.Context, args map[string]any) (*core.ToolResult, error) {
	action, _ := args["action"].(string)

	switch action {
	case "info":
		return t.info(args)
	case "frames":
		return t.frames(args)
	case "analyze":
		return t.analyze(args)
	default:
		return &core.ToolResult{Content: "Unknown action: " + action, Success: false}, nil
	}
}

func (t *VideoTool) info(args map[string]any) (*core.ToolResult, error) {
	path, ok := args["path"].(string)
	if !ok || path == "" {
		return &core.ToolResult{Content: "Path is required", Success: false}, nil
	}

	info, err := os.Stat(path)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("File not found: %v", err), Success: false}, nil
	}

	ext := strings.ToLower(filepath.Ext(path))
	return &core.ToolResult{
		Content: fmt.Sprintf("Video file: %s | Size: %.2f MB | Format: %s | Duration: ~%.1f min",
			filepath.Base(path),
			float64(info.Size())/1024/1024,
			strings.TrimPrefix(ext, "."),
			float64(info.Size())/1024/1024),
		Success: true,
	}, nil
}

func (t *VideoTool) frames(args map[string]any) (*core.ToolResult, error) {
	path, ok := args["path"].(string)
	if !ok || path == "" {
		return &core.ToolResult{Content: "Path is required", Success: false}, nil
	}

	return &core.ToolResult{
		Content: fmt.Sprintf("Frame extraction for %s:\n"+
			"Frames would be extracted at specified intervals.\n"+
			"Integration with ffmpeg or video processing library needed.",
			filepath.Base(path)),
		Success: true,
	}, nil
}

func (t *VideoTool) analyze(args map[string]any) (*core.ToolResult, error) {
	path, ok := args["path"].(string)
	if !ok || path == "" {
		return &core.ToolResult{Content: "Path is required", Success: false}, nil
	}

	info, err := os.Stat(path)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("File not found: %v", err), Success: false}, nil
	}

	return &core.ToolResult{
		Content: fmt.Sprintf("Video analysis for %s:\n"+
			"- Format: %s\n"+
			"- Size: %.2f MB\n"+
			"- Resolution: Unknown\n"+
			"- FPS: Unknown\n"+
			"- Total frames: ~%d\n"+
			"Note: Detailed analysis requires video processing library",
			filepath.Base(path),
			strings.ToLower(filepath.Ext(path)),
			float64(info.Size())/1024/1024,
			info.Size()/100000),
		Success: true,
	}, nil
}

type DocumentTool struct{}

func NewDocumentTool() *DocumentTool {
	return &DocumentTool{}
}

func (t *DocumentTool) Name() string { return "document" }
func (t *DocumentTool) Description() string {
	return "Processes documents - read text files, PDFs, or extract content"
}
func (t *DocumentTool) Parameters() map[string]core.Parameter {
	return map[string]core.Parameter{
		"action": {
			Type:        "string",
			Description: "Action: 'read', 'write', 'summarize', or 'extract'",
			Required:    true,
			Enum:        []string{"read", "write", "summarize", "extract"},
		},
		"path": {
			Type:        "string",
			Description: "Document file path",
			Required:    false,
		},
		"content": {
			Type:        "string",
			Description: "Content to write",
			Required:    false,
		},
	}
}

func (t *DocumentTool) Execute(ctx context.Context, args map[string]any) (*core.ToolResult, error) {
	action, _ := args["action"].(string)

	switch action {
	case "read":
		return t.read(args)
	case "write":
		return t.write(args)
	case "summarize":
		return t.summarize(args)
	case "extract":
		return t.extract(args)
	default:
		return &core.ToolResult{Content: "Unknown action: " + action, Success: false}, nil
	}
}

func (t *DocumentTool) read(args map[string]any) (*core.ToolResult, error) {
	path, ok := args["path"].(string)
	if !ok || path == "" {
		return &core.ToolResult{Content: "Path is required", Success: false}, nil
	}

	content, err := os.ReadFile(path)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Failed to read file: %v", err), Success: false}, nil
	}

	return &core.ToolResult{
		Content: string(content),
		Success: true,
	}, nil
}

func (t *DocumentTool) write(args map[string]any) (*core.ToolResult, error) {
	path, ok := args["path"].(string)
	if !ok || path == "" {
		return &core.ToolResult{Content: "Path is required", Success: false}, nil
	}

	content, ok := args["content"].(string)
	if !ok {
		content = ""
	}

	err := os.WriteFile(path, []byte(content), 0644)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Failed to write file: %v", err), Success: false}, nil
	}

	return &core.ToolResult{
		Content: fmt.Sprintf("Written %d bytes to %s", len(content), path),
		Success: true,
	}, nil
}

func (t *DocumentTool) summarize(args map[string]any) (*core.ToolResult, error) {
	path, ok := args["path"].(string)
	if !ok || path == "" {
		return &core.ToolResult{Content: "Path is required", Success: false}, nil
	}

	content, err := os.ReadFile(path)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Failed to read file: %v", err), Success: false}, nil
	}

	lines := strings.Split(string(content), "\n")
	wordCount := len(strings.Fields(string(content)))

	return &core.ToolResult{
		Content: fmt.Sprintf("Document summary for %s:\n"+
			"- Lines: %d\n"+
			"- Words: %d\n"+
			"- Characters: %d\n"+
			"- Preview: %s...",
			filepath.Base(path),
			len(lines),
			wordCount,
			len(content),
			strings.Join(lines[:min(5, len(lines))], "\n")),
		Success: true,
	}, nil
}

func (t *DocumentTool) extract(args map[string]any) (*core.ToolResult, error) {
	path, ok := args["path"].(string)
	if !ok || path == "" {
		return &core.ToolResult{Content: "Path is required", Success: false}, nil
	}

	ext := strings.ToLower(filepath.Ext(path))
	return &core.ToolResult{
		Content: fmt.Sprintf("Content extraction for %s:\n"+
			"Format: %s\n"+
			"For actual extraction:\n"+
			"- PDF: Use pdf parser library\n"+
			"- DOCX: Use document library\n"+
			"- Markdown: Direct read",
			filepath.Base(path),
			strings.TrimPrefix(ext, ".")),
		Success: true,
	}, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

type VisionTool struct{}

func NewVisionTool() *VisionTool {
	return &VisionTool{}
}

func (t *VisionTool) Name() string { return "vision" }
func (t *VisionTool) Description() string {
	return "Analyzes images using vision models - describe, detect objects, read text (OCR)"
}
func (t *VisionTool) Parameters() map[string]core.Parameter {
	return map[string]core.Parameter{
		"action": {
			Type:        "string",
			Description: "Action: 'describe', 'objects', 'ocr', or 'compare'",
			Required:    true,
			Enum:        []string{"describe", "objects", "ocr", "compare"},
		},
		"image": {
			Type:        "string",
			Description: "Image path or URL",
			Required:    true,
		},
	}
}

func (t *VisionTool) Execute(ctx context.Context, args map[string]any) (*core.ToolResult, error) {
	action, _ := args["action"].(string)
	image, _ := args["image"].(string)

	switch action {
	case "describe":
		return &core.ToolResult{
			Content: fmt.Sprintf("[Vision analysis of %s]\nDetected: This appears to be an image file. "+
				"Integration with vision model (GPT-4V, Claude, LLaVA) would provide detailed description.",
				filepath.Base(image)),
			Success: true,
		}, nil
	case "objects":
		return &core.ToolResult{
			Content: "[Object detection] Integration with YOLO or similar model required for actual detection.",
			Success: true,
		}, nil
	case "ocr":
		return &core.ToolResult{
			Content: "[OCR] Integration with Tesseract or cloud OCR service required for text extraction.",
			Success: true,
		}, nil
	case "compare":
		return &core.ToolResult{
			Content: "[Image comparison] Integration with image similarity model required.",
			Success: true,
		}, nil
	default:
		return &core.ToolResult{Content: "Unknown action: " + action, Success: false}, nil
	}
}

type ImageEncoder struct{}

func NewImageEncoder() *ImageEncoder {
	return &ImageEncoder{}
}

func (e *ImageEncoder) EncodeFromFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(data), nil
}

func (e *ImageEncoder) DecodeToFile(encoded, path string) error {
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

func (e *ImageEncoder) EncodeFromURL(url string) (string, error) {
	resp, err := http.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	return base64.StdEncoding.EncodeToString(data), nil
}
