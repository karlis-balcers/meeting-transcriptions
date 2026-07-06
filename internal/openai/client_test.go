package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestClientTranscribeMultipartAndRetry(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/audio/transcriptions" {
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Fatalf("missing auth header")
		}
		if calls.Add(1) == 1 {
			http.Error(w, "temporary", http.StatusTooManyRequests)
			return
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		if r.FormValue("model") != "model" || r.FormValue("language") != "en" {
			t.Fatalf("bad form values: %v", r.Form)
		}
		if !strings.Contains(r.FormValue("prompt"), "Paymentology") {
			t.Fatalf("missing keyword prompt: %q", r.FormValue("prompt"))
		}
		_ = json.NewEncoder(w).Encode(map[string]string{"text": " hello "})
	}))
	defer server.Close()

	file := filepath.Join(t.TempDir(), "chunk.wav")
	if err := os.WriteFile(file, []byte("wav"), 0o600); err != nil {
		t.Fatal(err)
	}
	client := Client{APIKey: "test-key", BaseURL: server.URL, HTTPClient: server.Client()}
	text, err := client.Transcribe(context.Background(), file, Options{Model: "model", Language: "en", Keywords: []string{"Paymentology"}, MaxRetries: 1, RetryBase: time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	if text != "hello" {
		t.Fatalf("unexpected text %q", text)
	}
	if calls.Load() != 2 {
		t.Fatalf("expected retry, got %d calls", calls.Load())
	}
}

func TestClientDoesNotRetryAuthFailure(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		http.Error(w, "no", http.StatusUnauthorized)
	}))
	defer server.Close()
	file := filepath.Join(t.TempDir(), "chunk.wav")
	if err := os.WriteFile(file, []byte("wav"), 0o600); err != nil {
		t.Fatal(err)
	}
	client := Client{APIKey: "test-key", BaseURL: server.URL, HTTPClient: server.Client()}
	_, err := client.Transcribe(context.Background(), file, Options{Model: "model", MaxRetries: 3, RetryBase: time.Millisecond})
	if err == nil {
		t.Fatal("expected auth error")
	}
	if calls.Load() != 1 {
		t.Fatalf("auth failure should not retry, got %d calls", calls.Load())
	}
}

func TestClientSuppressesExactKeywordPromptEcho(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		prompt := r.FormValue("prompt")
		_ = json.NewEncoder(w).Encode(map[string]string{"text": "  " + prompt + "  "})
	}))
	defer server.Close()

	file := filepath.Join(t.TempDir(), "chunk.wav")
	if err := os.WriteFile(file, []byte("wav"), 0o600); err != nil {
		t.Fatal(err)
	}
	client := Client{APIKey: "test-key", BaseURL: server.URL, HTTPClient: server.Client()}
	text, err := client.Transcribe(context.Background(), file, Options{Model: "model", Keywords: []string{"Paymentology"}})
	if err != nil {
		t.Fatal(err)
	}
	if text != "" {
		t.Fatalf("expected exact prompt echo to be suppressed, got %q", text)
	}
}

func TestClientKeepsSpeechThatContainsPromptKeywords(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		prompt := r.FormValue("prompt")
		_ = json.NewEncoder(w).Encode(map[string]string{"text": prompt + " came up in the meeting agenda"})
	}))
	defer server.Close()

	file := filepath.Join(t.TempDir(), "chunk.wav")
	if err := os.WriteFile(file, []byte("wav"), 0o600); err != nil {
		t.Fatal(err)
	}
	client := Client{APIKey: "test-key", BaseURL: server.URL, HTTPClient: server.Client()}
	text, err := client.Transcribe(context.Background(), file, Options{Model: "model", Keywords: []string{"Paymentology"}})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(text, "Paymentology") {
		t.Fatalf("expected normal speech containing the keyword prompt to be preserved, got %q", text)
	}
}
