package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/audio"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/config"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/filter"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/logging"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/openai"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/speaker"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/transcript"
)

type Transcriber interface {
	Transcribe(ctx context.Context, filePath string, opts openai.Options) (string, error)
}

type Dependencies struct {
	Discoverer  audio.Discoverer
	Recorder    audio.Recorder
	Speaker     speaker.Detector
	Transcriber Transcriber
	Clock       func() time.Time
}

type Prepared struct {
	Config       config.Config
	ConfigPath   string
	APIKey       string
	Devices      []audio.Device
	Selection    audio.Selection
	Warnings     []string
	OutputDir    string
	TempDir      string
	Dependencies Dependencies
}

func Prepare(ctx context.Context, cfg config.Config, configPath string, apiKey string, deps Dependencies) (*Prepared, error) {
	if strings.TrimSpace(apiKey) == "" {
		return nil, errors.New("OPENAI_API_KEY is required; set it in the environment before starting transcription")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if deps.Discoverer == nil {
		deps.Discoverer = audio.SystemDiscoverer{Preferences: audio.Preferences{
			MicDeviceID:      cfg.Audio.MicDeviceID,
			MicDeviceName:    cfg.Audio.MicDeviceName,
			OutputDeviceID:   cfg.Audio.OutputDeviceID,
			OutputDeviceName: cfg.Audio.OutputDeviceName,
		}}
	}
	if deps.Recorder == nil {
		deps.Recorder = audio.ExternalRecorder{}
	}
	if deps.Speaker == nil {
		deps.Speaker = speaker.New(cfg.Teams.Enabled)
	}
	if deps.Transcriber == nil {
		deps.Transcriber = openai.Client{APIKey: apiKey}
	}
	if deps.Clock == nil {
		deps.Clock = time.Now
	}

	devices, warnings, err := deps.Discoverer.ListDevices(ctx)
	if err != nil {
		return nil, fmt.Errorf("discover audio devices: %w", err)
	}
	logging.Printf("discovered %d audio devices", len(devices))
	for _, warning := range warnings {
		logging.Printf("device discovery warning: %s", warning)
	}
	selection, selectionWarnings, err := audio.SelectDevices(devices, audio.Preferences{
		MicDeviceID:      cfg.Audio.MicDeviceID,
		MicDeviceName:    cfg.Audio.MicDeviceName,
		OutputDeviceID:   cfg.Audio.OutputDeviceID,
		OutputDeviceName: cfg.Audio.OutputDeviceName,
	})
	warnings = append(warnings, selectionWarnings...)
	for _, warning := range selectionWarnings {
		logging.Printf("device selection warning: %s", warning)
	}
	if err != nil {
		return nil, err
	}
	if err := deps.Recorder.Validate(ctx, selection); err != nil {
		return nil, err
	}
	cwd, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	outputDir := cfg.EffectiveOutputDir(cwd)
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return nil, fmt.Errorf("create transcript output directory %s: %w", outputDir, err)
	}
	tempDir := cfg.EffectiveTempDir()
	if err := os.MkdirAll(tempDir, 0o700); err != nil {
		return nil, fmt.Errorf("create temp directory %s: %w", tempDir, err)
	}
	absOutput, _ := filepath.Abs(outputDir)
	absTemp, _ := filepath.Abs(tempDir)
	logging.Printf("selected microphone=%s output=%s output_dir=%s temp_dir=%s", selection.Mic.DisplayName(), selection.Output.DisplayName(), absOutput, absTemp)
	return &Prepared{
		Config:       cfg,
		ConfigPath:   configPath,
		APIKey:       apiKey,
		Devices:      devices,
		Selection:    selection,
		Warnings:     warnings,
		OutputDir:    absOutput,
		TempDir:      absTemp,
		Dependencies: deps,
	}, nil
}

func RunSilent(ctx context.Context, prepared *Prepared, stdout io.Writer, stderr io.Writer) error {
	logging.Printf("running silent session: transcript_dir=%s", prepared.OutputDir)
	for _, warning := range prepared.Warnings {
		fmt.Fprintln(stderr, "warning:", warning)
	}
	fmt.Fprintf(stderr, "microphone: %s\n", prepared.Selection.Mic.DisplayName())
	fmt.Fprintf(stderr, "output capture: %s\n", prepared.Selection.Output.DisplayName())
	fmt.Fprintf(stderr, "transcript file: %s\n", prepared.OutputDir)

	session, err := NewSession(prepared)
	if err != nil {
		return err
	}
	if err := session.Start(ctx); err != nil {
		logging.Printf("silent session start failed: %v", err)
		return err
	}
	<-ctx.Done()
	if err := session.Stop(context.Background()); err != nil {
		logging.Printf("silent session stop failed: %v", err)
		return err
	}
	logging.Printf("silent session completed")
	_, err = io.WriteString(stdout, transcript.RenderPlain(session.Snapshot().Transcript))
	return err
}

func openAIOptions(cfg config.Config) openai.Options {
	return openai.Options{
		Model:      cfg.OpenAI.Model,
		Language:   cfg.Language,
		Keywords:   cfg.Keywords,
		Timeout:    cfg.OpenAI.Timeout.Duration,
		MaxRetries: cfg.OpenAI.MaxRetries,
		RetryBase:  cfg.OpenAI.RetryBase.Duration,
	}
}

func filterConfig(cfg config.Config) filter.Config {
	return filter.Config{
		MinChars: cfg.Filter.MinChars,
		Exact:    cfg.Filter.Exact,
		Prefixes: cfg.Filter.Prefixes,
		Contains: cfg.Filter.Contains,
		Regex:    cfg.Filter.Regex,
		Keywords: cfg.Keywords,
	}
}
