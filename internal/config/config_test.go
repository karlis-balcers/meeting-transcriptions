package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLoadPrecedenceConfigOverEnvironmentAndRuntimeOverrides(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	content := []byte(`user_name: Config User
language: fr
keywords: [ConfigTerm]
openai:
  model: gpt-4o-transcribe
audio:
  max_segment_duration: 12s
paths:
  output_dir: /from/config
`)
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, warnings, src, err := Load(path, true, map[string]string{
		"LANGUAGE":                    "de",
		"OUTPUT_DIR":                  "/from/env",
		"OPENAI_MODEL_FOR_TRANSCRIPT": "env-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(warnings) != 0 {
		t.Fatalf("expected no warnings, got %v", warnings)
	}
	if !src.Found || src.Path != path || !src.Explicit {
		t.Fatalf("config source should report the explicit file as found, got %+v", src)
	}
	if src.SummaryLine() != "Config loaded from "+path {
		t.Fatalf("unexpected source summary line: %q", src.SummaryLine())
	}
	if cfg.Language != "fr" {
		t.Fatalf("config language should win over env, got %q", cfg.Language)
	}
	if cfg.Paths.OutputDir != "/from/config" {
		t.Fatalf("config output should win over env, got %q", cfg.Paths.OutputDir)
	}
	if cfg.OpenAI.Model != "gpt-4o-transcribe" {
		t.Fatalf("config model should win over env, got %q", cfg.OpenAI.Model)
	}
	cfg.ApplyRuntimeOverrides(RuntimeOverrides{OutputDir: "/from/flag", Language: "es"})
	if cfg.Paths.OutputDir != "/from/flag" || cfg.Language != "es" {
		t.Fatalf("runtime overrides not applied: %+v", cfg)
	}
	if cfg.Audio.MaxSegmentDuration.Duration != 12*time.Second {
		t.Fatalf("duration not parsed: %s", cfg.Audio.MaxSegmentDuration.Duration)
	}
}

func TestLoadMissingDefaultUsesDefaults(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "missing.yaml")
	cfg, warnings, src, err := Load(missing, false, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(warnings) != 0 {
		t.Fatalf("expected no warnings, got %v", warnings)
	}
	if cfg.UserName != "You" || cfg.Language != "en" {
		t.Fatalf("unexpected defaults: %+v", cfg)
	}
	if src.Found || src.Path != missing || src.Explicit {
		t.Fatalf("default-path miss should not report found, got %+v", src)
	}
	if src.SummaryLine() != "Config not found; using defaults and environment variables" {
		t.Fatalf("unexpected missing-default source summary line: %q", src.SummaryLine())
	}
}

func TestInvalidEnvironmentFallsBackWithWarning(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "missing.yaml")
	cfg, warnings, src, err := Load(missing, false, map[string]string{
		"FRAME_DURATION_MS": "nope",
		"RECORD_SECONDS":    "-1",
	})
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Audio.FrameDurationMS != 100 {
		t.Fatalf("expected default frame duration, got %d", cfg.Audio.FrameDurationMS)
	}
	if len(warnings) != 2 {
		t.Fatalf("expected two warnings, got %v", warnings)
	}
	if src.Found {
		t.Fatalf("default-path miss with env fallback should not report found, got %+v", src)
	}
}

func TestLoadExplicitMissingReportsNotFound(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "explicit-missing.yaml")
	_, _, src, err := Load(missing, true, nil)
	if err == nil {
		t.Fatal("expected an error loading an explicitly-missing config")
	}
	if src.Found || !src.Explicit || src.Path != missing {
		t.Fatalf("explicit missing config should report not-found with the attempted path, got %+v", src)
	}
	want := "Config " + missing + " not found; using defaults and environment variables"
	if src.SummaryLine() != want {
		t.Fatalf("unexpected explicit-missing source summary line: got %q want %q", src.SummaryLine(), want)
	}
}
