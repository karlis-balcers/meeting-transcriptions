package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const DefaultBaseURL = "https://api.openai.com"

type Options struct {
	Model      string
	Language   string
	Keywords   []string
	Timeout    time.Duration
	MaxRetries int
	RetryBase  time.Duration
}

type Client struct {
	APIKey     string
	BaseURL    string
	HTTPClient *http.Client
}

func (c Client) Transcribe(ctx context.Context, filePath string, opts Options) (string, error) {
	if strings.TrimSpace(c.APIKey) == "" {
		return "", errors.New("OPENAI_API_KEY is required for transcription")
	}
	if opts.Model == "" {
		return "", errors.New("OpenAI transcription model is required")
	}
	baseURL := strings.TrimRight(c.BaseURL, "/")
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	httpClient := c.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: opts.Timeout}
	}
	if opts.RetryBase <= 0 {
		opts.RetryBase = time.Second
	}
	var lastErr error
	for attempt := 0; attempt <= opts.MaxRetries; attempt++ {
		text, retry, err := c.once(ctx, httpClient, baseURL+"/v1/audio/transcriptions", filePath, opts)
		if err == nil {
			return text, nil
		}
		lastErr = err
		if !retry || attempt == opts.MaxRetries {
			break
		}
		backoff := opts.RetryBase * time.Duration(1<<attempt)
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(backoff):
		}
	}
	return "", lastErr
}

func (c Client) once(ctx context.Context, httpClient *http.Client, url, filePath string, opts Options) (string, bool, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", false, fmt.Errorf("open audio chunk: %w", err)
	}
	defer file.Close()

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	if err := writer.WriteField("model", opts.Model); err != nil {
		return "", false, err
	}
	if strings.TrimSpace(opts.Language) != "" {
		if err := writer.WriteField("language", opts.Language); err != nil {
			return "", false, err
		}
	}
	if prompt := promptFromKeywords(opts.Keywords); prompt != "" {
		if err := writer.WriteField("prompt", prompt); err != nil {
			return "", false, err
		}
	}
	part, err := writer.CreateFormFile("file", filepath.Base(filePath))
	if err != nil {
		return "", false, err
	}
	if _, err := io.Copy(part, file); err != nil {
		return "", false, err
	}
	if err := writer.Close(); err != nil {
		return "", false, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, body)
	if err != nil {
		return "", false, err
	}
	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	resp, err := httpClient.Do(req)
	if err != nil {
		return "", true, fmt.Errorf("OpenAI transcription request failed: %w", err)
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		retry := resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500
		if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
			retry = false
		}
		return "", retry, fmt.Errorf("OpenAI transcription failed with status %d: %s", resp.StatusCode, strings.TrimSpace(string(respBody)))
	}
	var decoded struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(respBody, &decoded); err != nil {
		return "", false, fmt.Errorf("decode OpenAI transcription response: %w", err)
	}
	text := strings.TrimSpace(decoded.Text)
	if isPromptEcho(text, opts) {
		return "", false, nil
	}
	return text, false, nil
}

func promptFromKeywords(keywords []string) string {
	var cleaned []string
	seen := map[string]struct{}{}
	for _, keyword := range keywords {
		keyword = strings.TrimSpace(keyword)
		if keyword == "" {
			continue
		}
		key := strings.ToLower(keyword)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		cleaned = append(cleaned, keyword)
	}
	if len(cleaned) == 0 {
		return ""
	}
	return "Meeting vocabulary and names to preserve: " + strings.Join(cleaned, ", ")
}

func isPromptEcho(text string, opts Options) bool {
	prompt := promptFromKeywords(opts.Keywords)
	if prompt == "" {
		return false
	}
	return strings.TrimSpace(text) == prompt
}
