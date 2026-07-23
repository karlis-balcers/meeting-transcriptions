package config

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

const (
	DefaultConfigSubdir = ".transcribe"
	DefaultConfigFile   = "config.yaml"
)

type Duration struct {
	time.Duration
}

func NewDuration(d time.Duration) Duration {
	return Duration{Duration: d}
}

func (d Duration) MarshalYAML() (any, error) {
	return d.Duration.String(), nil
}

func (d *Duration) UnmarshalYAML(value *yaml.Node) error {
	if value == nil || value.Value == "" {
		return nil
	}
	if value.Tag == "!!int" {
		seconds, err := strconv.Atoi(value.Value)
		if err != nil {
			return err
		}
		d.Duration = time.Duration(seconds) * time.Second
		return nil
	}
	parsed, err := time.ParseDuration(value.Value)
	if err == nil {
		d.Duration = parsed
		return nil
	}
	seconds, convErr := strconv.ParseFloat(value.Value, 64)
	if convErr == nil {
		d.Duration = time.Duration(seconds * float64(time.Second))
		return nil
	}
	return fmt.Errorf("invalid duration %q", value.Value)
}

type Config struct {
	UserName string       `yaml:"user_name"`
	Language string       `yaml:"language"`
	Keywords []string     `yaml:"keywords"`
	OpenAI   OpenAIConfig `yaml:"openai"`
	Audio    AudioConfig  `yaml:"audio"`
	Paths    PathsConfig  `yaml:"paths"`
	Teams    TeamsConfig  `yaml:"teams"`
	TUI      TUIConfig    `yaml:"tui"`
	Filter   FilterConfig `yaml:"filter"`
}

type OpenAIConfig struct {
	Model            string   `yaml:"model"`
	Timeout          Duration `yaml:"timeout"`
	MaxRetries       int      `yaml:"max_retries"`
	RetryBase        Duration `yaml:"retry_base"`
	RetryMaxInterval Duration `yaml:"retry_max_interval"`
}

type AudioConfig struct {
	MicDeviceID          string   `yaml:"mic_device_id"`
	MicDeviceName        string   `yaml:"mic_device_name"`
	OutputDeviceID       string   `yaml:"output_device_id"`
	OutputDeviceName     string   `yaml:"output_device_name"`
	CaptureChunkDuration Duration `yaml:"capture_chunk_duration"`
	FrameDurationMS      int      `yaml:"frame_duration_ms"`
	SilenceThreshold     float64  `yaml:"silence_threshold"`
	SilenceDuration      Duration `yaml:"silence_duration"`
	MaxSegmentDuration   Duration `yaml:"max_segment_duration"`
}

type PathsConfig struct {
	OutputDir string `yaml:"output_dir"`
	TempDir   string `yaml:"temp_dir"`
}

type TeamsConfig struct {
	Enabled bool `yaml:"enabled"`
}

type TUIConfig struct {
	ShowLoudnessMeters bool `yaml:"show_loudness_meters"`
}

type FilterConfig struct {
	MinChars int      `yaml:"min_chars"`
	Exact    []string `yaml:"exact"`
	Prefixes []string `yaml:"prefixes"`
	Contains []string `yaml:"contains"`
	Regex    []string `yaml:"regex"`
}

type RuntimeOverrides struct {
	OutputDir string
	TempDir   string
	Mic       string
	Output    string
	Language  string
	Model     string
}

// Source records where a loaded Config actually came from so callers (the CLI
// startup banner) can distinguish "loaded from a real file" from "no file
// present, defaults + environment used". Path is the resolved path Load
// attempted (default path when none was passed); Explicit reports whether the
// caller asked for a specific file; Found reports whether that file existed and
// was read successfully.
type Source struct {
	Path     string
	Explicit bool
	Found    bool
}

// SummaryLine renders a one-line, human-readable description of the config
// source for the startup banner, e.g. "Config loaded from /home/u/.transcribe/config.yaml".
func (s Source) SummaryLine() string {
	if s.Found {
		return fmt.Sprintf("Config loaded from %s", s.Path)
	}
	if s.Explicit {
		return fmt.Sprintf("Config %s not found; using defaults and environment variables", s.Path)
	}
	return "Config not found; using defaults and environment variables"
}

func Defaults() Config {
	return Config{
		UserName: "You",
		Language: "en",
		Keywords: nil,
		OpenAI: OpenAIConfig{
			Model:            "gpt-4o-mini-transcribe",
			Timeout:          NewDuration(60 * time.Second),
			MaxRetries:       3,
			RetryBase:        NewDuration(time.Second),
			RetryMaxInterval: NewDuration(8 * time.Second),
		},
		Audio: AudioConfig{
			CaptureChunkDuration: NewDuration(2 * time.Second),
			FrameDurationMS:      100,
			SilenceThreshold:     50,
			SilenceDuration:      NewDuration(2 * time.Second),
			MaxSegmentDuration:   NewDuration(15 * time.Minute),
		},
		Paths: PathsConfig{},
		Teams: TeamsConfig{Enabled: true},
		TUI:   TUIConfig{ShowLoudnessMeters: true},
		Filter: FilterConfig{
			MinChars: 2,
			Exact:    []string{"LAMPA", "MEMMEE"},
			Prefixes: []string{"[Music]", "(Music)", "♪"},
			Contains: []string{"thank you for watching"},
		},
	}
}

func DefaultPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil || strings.TrimSpace(home) == "" {
		return "", errors.New("cannot determine home directory for default config path; pass --config")
	}
	return filepath.Join(home, DefaultConfigSubdir, DefaultConfigFile), nil
}

func Load(path string, explicitPath bool, env map[string]string) (Config, []string, Source, error) {
	cfg := Defaults()
	warnings := applyEnv(&cfg, env)
	if strings.TrimSpace(path) == "" {
		defaultPath, err := DefaultPath()
		if err != nil {
			return cfg, warnings, Source{Explicit: explicitPath}, err
		}
		path = defaultPath
	}
	source := Source{Path: path, Explicit: explicitPath}

	content, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) && !explicitPath {
			return cfg, warnings, source, nil
		}
		return cfg, warnings, source, fmt.Errorf("read config %s: %w", path, err)
	}
	if len(strings.TrimSpace(string(content))) == 0 {
		warnings = append(warnings, fmt.Sprintf("config %s is empty; using defaults and environment", path))
		return cfg, warnings, source, cfg.Validate()
	}
	if err := yaml.Unmarshal(content, &cfg); err != nil {
		return cfg, warnings, source, fmt.Errorf("parse config %s: %w", path, err)
	}
	source.Found = true
	return cfg, warnings, source, cfg.Validate()
}

func Save(path string, cfg Config) error {
	if strings.TrimSpace(path) == "" {
		defaultPath, err := DefaultPath()
		if err != nil {
			return err
		}
		path = defaultPath
	}
	if err := cfg.Validate(); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("create config directory: %w", err)
	}
	encoded, err := yaml.Marshal(cfg)
	if err != nil {
		return err
	}
	return os.WriteFile(path, encoded, 0o600)
}

func (c *Config) ApplyRuntimeOverrides(o RuntimeOverrides) {
	if strings.TrimSpace(o.OutputDir) != "" {
		c.Paths.OutputDir = strings.TrimSpace(o.OutputDir)
	}
	if strings.TrimSpace(o.TempDir) != "" {
		c.Paths.TempDir = strings.TrimSpace(o.TempDir)
	}
	if strings.TrimSpace(o.Mic) != "" {
		c.Audio.MicDeviceID = strings.TrimSpace(o.Mic)
		c.Audio.MicDeviceName = ""
	}
	if strings.TrimSpace(o.Output) != "" {
		c.Audio.OutputDeviceID = strings.TrimSpace(o.Output)
		c.Audio.OutputDeviceName = ""
	}
	if strings.TrimSpace(o.Language) != "" {
		c.Language = strings.ToLower(strings.TrimSpace(o.Language))
	}
	if strings.TrimSpace(o.Model) != "" {
		c.OpenAI.Model = strings.TrimSpace(o.Model)
	}
}

func (c Config) EffectiveOutputDir(cwd string) string {
	if strings.TrimSpace(c.Paths.OutputDir) != "" {
		return c.Paths.OutputDir
	}
	return cwd
}

func (c Config) EffectiveTempDir() string {
	if strings.TrimSpace(c.Paths.TempDir) != "" {
		return c.Paths.TempDir
	}
	return filepath.Join(os.TempDir(), "transcribe")
}

func (c Config) Validate() error {
	if strings.TrimSpace(c.UserName) == "" {
		return errors.New("user_name cannot be empty")
	}
	if strings.TrimSpace(c.Language) == "" {
		return errors.New("language cannot be empty")
	}
	if strings.TrimSpace(c.OpenAI.Model) == "" {
		return errors.New("openai.model cannot be empty")
	}
	if c.OpenAI.Timeout.Duration <= 0 {
		return errors.New("openai.timeout must be positive")
	}
	if c.OpenAI.MaxRetries < 0 {
		return errors.New("openai.max_retries cannot be negative")
	}
	if c.OpenAI.RetryBase.Duration <= 0 {
		return errors.New("openai.retry_base must be positive")
	}
	if c.Audio.FrameDurationMS <= 0 {
		return errors.New("audio.frame_duration_ms must be positive")
	}
	if c.Audio.CaptureChunkDuration.Duration <= 0 {
		return errors.New("audio.capture_chunk_duration must be positive")
	}
	if c.Audio.SilenceThreshold < 0 {
		return errors.New("audio.silence_threshold cannot be negative")
	}
	if c.Audio.SilenceDuration.Duration <= 0 {
		return errors.New("audio.silence_duration must be positive")
	}
	if c.Audio.MaxSegmentDuration.Duration <= 0 {
		return errors.New("audio.max_segment_duration must be positive")
	}
	if c.Filter.MinChars < 0 {
		return errors.New("filter.min_chars cannot be negative")
	}
	return nil
}

func EnvMapFromOS() map[string]string {
	result := make(map[string]string)
	for _, pair := range os.Environ() {
		key, value, found := strings.Cut(pair, "=")
		if !found {
			continue
		}
		result[key] = value
	}
	return result
}

func applyEnv(cfg *Config, env map[string]string) []string {
	var warnings []string
	if value := strings.TrimSpace(env["YOUR_NAME"]); value != "" {
		cfg.UserName = value
	}
	if value := strings.TrimSpace(env["LANGUAGE"]); value != "" {
		cfg.Language = strings.ToLower(strings.Split(value, ",")[0])
	}
	if value := strings.TrimSpace(env["KEYWORDS"]); value != "" {
		cfg.Keywords = splitCSV(value)
	}
	if value := strings.TrimSpace(env["OPENAI_MODEL_FOR_TRANSCRIPT"]); value != "" {
		cfg.OpenAI.Model = value
	}
	if value := strings.TrimSpace(env["OUTPUT_DIR"]); value != "" {
		cfg.Paths.OutputDir = value
	}
	if value := strings.TrimSpace(env["TEMP_DIR"]); value != "" {
		cfg.Paths.TempDir = value
	}
	if value := strings.TrimSpace(env["AUDIO_INPUT_DEVICE_NAME"]); value != "" {
		cfg.Audio.MicDeviceName = value
	}
	if value := strings.TrimSpace(env["AUDIO_OUTPUT_DEVICE_NAME"]); value != "" {
		cfg.Audio.OutputDeviceName = value
	}
	if value := strings.TrimSpace(env["AUDIO_INPUT_DEVICE_INDEX"]); value != "" {
		cfg.Audio.MicDeviceID = value
	}
	if value := strings.TrimSpace(env["AUDIO_OUTPUT_DEVICE_INDEX"]); value != "" {
		cfg.Audio.OutputDeviceID = value
	}
	if value := strings.TrimSpace(env["FRAME_DURATION_MS"]); value != "" {
		if parsed, err := strconv.Atoi(value); err == nil && parsed > 0 {
			cfg.Audio.FrameDurationMS = parsed
		} else {
			warnings = append(warnings, "FRAME_DURATION_MS is invalid; using configured/default value")
		}
	}
	if value := strings.TrimSpace(env["CAPTURE_CHUNK_DURATION"]); value != "" {
		if parsed, err := parseDurationCompat(value); err == nil && parsed > 0 {
			cfg.Audio.CaptureChunkDuration = NewDuration(parsed)
		} else {
			warnings = append(warnings, "CAPTURE_CHUNK_DURATION is invalid; using configured/default value")
		}
	}
	if value := strings.TrimSpace(env["SILENCE_THRESHOLD"]); value != "" {
		if parsed, err := strconv.ParseFloat(value, 64); err == nil && parsed >= 0 {
			cfg.Audio.SilenceThreshold = parsed
		} else {
			warnings = append(warnings, "SILENCE_THRESHOLD is invalid; using configured/default value")
		}
	}
	if value := strings.TrimSpace(env["SILENCE_DURATION"]); value != "" {
		if parsed, err := parseDurationCompat(value); err == nil && parsed > 0 {
			cfg.Audio.SilenceDuration = NewDuration(parsed)
		} else {
			warnings = append(warnings, "SILENCE_DURATION is invalid; using configured/default value")
		}
	}
	if value := strings.TrimSpace(env["RECORD_SECONDS"]); value != "" {
		if parsed, err := strconv.ParseFloat(value, 64); err == nil && parsed > 0 {
			cfg.Audio.MaxSegmentDuration = NewDuration(time.Duration(parsed * float64(time.Second)))
		} else {
			warnings = append(warnings, "RECORD_SECONDS is invalid; using configured/default value")
		}
	}
	if value := strings.TrimSpace(env["TRANSCRIBE_API_TIMEOUT_SECONDS"]); value != "" {
		if parsed, err := strconv.ParseFloat(value, 64); err == nil && parsed > 0 {
			cfg.OpenAI.Timeout = NewDuration(time.Duration(parsed * float64(time.Second)))
		} else {
			warnings = append(warnings, "TRANSCRIBE_API_TIMEOUT_SECONDS is invalid; using configured/default value")
		}
	}
	if value := strings.TrimSpace(env["TRANSCRIBE_API_MAX_RETRIES"]); value != "" {
		if parsed, err := strconv.Atoi(value); err == nil && parsed >= 0 {
			cfg.OpenAI.MaxRetries = parsed
		} else {
			warnings = append(warnings, "TRANSCRIBE_API_MAX_RETRIES is invalid; using configured/default value")
		}
	}
	if value := strings.TrimSpace(env["TRANSCRIBE_API_RETRY_BASE_SECONDS"]); value != "" {
		if parsed, err := strconv.ParseFloat(value, 64); err == nil && parsed > 0 {
			cfg.OpenAI.RetryBase = NewDuration(time.Duration(parsed * float64(time.Second)))
		} else {
			warnings = append(warnings, "TRANSCRIBE_API_RETRY_BASE_SECONDS is invalid; using configured/default value")
		}
	}
	return warnings
}

func splitCSV(value string) []string {
	parts := strings.Split(value, ",")
	result := make([]string, 0, len(parts))
	seen := map[string]struct{}{}
	for _, part := range parts {
		trimmed := strings.TrimSpace(strings.Trim(part, `"'`))
		if trimmed == "" {
			continue
		}
		key := strings.ToLower(trimmed)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		result = append(result, trimmed)
	}
	return result
}

func parseDurationCompat(value string) (time.Duration, error) {
	if parsed, err := time.ParseDuration(value); err == nil {
		return parsed, nil
	}
	seconds, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return 0, err
	}
	return time.Duration(seconds * float64(time.Second)), nil
}
