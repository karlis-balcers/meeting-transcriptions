package cli

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"runtime"
	"syscall"

	"github.com/spf13/cobra"

	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/app"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/audio"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/config"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/logging"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/openai"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/speaker"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/tui"
)

type IO struct {
	Stdout io.Writer
	Stderr io.Writer
}

type options struct {
	silent     bool
	configPath string
	outputDir  string
	tempDir    string
	mic        string
	output     string
	language   string
	model      string
	logging    int
	list       bool
	version    bool
}

func NewRootCommand(ioStreams IO, version string) *cobra.Command {
	if ioStreams.Stdout == nil {
		ioStreams.Stdout = io.Discard
	}
	if ioStreams.Stderr == nil {
		ioStreams.Stderr = io.Discard
	}
	var opts options
	cmd := &cobra.Command{
		Use:          "transcribe",
		Short:        "Record microphone and system output, then transcribe with OpenAI",
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			if opts.version {
				fmt.Fprintln(ioStreams.Stdout, version)
				return nil
			}
			return run(cmd.Context(), ioStreams, opts)
		},
	}
	cmd.SetOut(ioStreams.Stdout)
	cmd.SetErr(ioStreams.Stderr)
	cmd.Flags().BoolVarP(&opts.silent, "silent", "s", false, "disable TUI; write final transcript to stdout only after interrupt")
	cmd.Flags().StringVar(&opts.configPath, "config", "", "config file path (default ~/.transcribe/config.yaml)")
	cmd.Flags().StringVarP(&opts.outputDir, "output-dir", "o", "", "transcript output directory for this run")
	cmd.Flags().StringVar(&opts.tempDir, "temp-dir", "", "temporary WAV chunk directory for this run")
	cmd.Flags().StringVar(&opts.mic, "mic", "", "microphone device ID or name for this run")
	cmd.Flags().StringVar(&opts.output, "output", "", "system-output capture device ID or name for this run")
	cmd.Flags().StringVar(&opts.language, "language", "", "OpenAI transcription language code for this run")
	cmd.Flags().StringVar(&opts.model, "model", "", "OpenAI transcription model for this run")
	cmd.Flags().IntVar(&opts.logging, "logging", 0, "write runtime logs to transcribe.log in the current directory when set to 1")
	cmd.Flags().BoolVar(&opts.list, "list-devices", false, "list detected audio devices and exit")
	cmd.Flags().BoolVar(&opts.version, "version", false, "print version and exit")
	return cmd
}

func run(ctx context.Context, ioStreams IO, opts options) error {
	cleanupLogging, err := setupLogging(opts.logging)
	if err != nil {
		return err
	}
	if cleanupLogging != nil {
		defer cleanupLogging()
	}
	explicitConfig := opts.configPath != ""
	cfg, warnings, source, err := config.Load(opts.configPath, explicitConfig, config.EnvMapFromOS())
	if err != nil {
		logging.Printf("config load failed: %v", err)
		return err
	}
	// Load resolves the effective config path (default path when none was passed);
	// seed opts.configPath with it so downstream code (Prepare, saved preferences)
	// points at the file we actually evaluated.
	if opts.configPath == "" {
		opts.configPath = source.Path
	}
	cfg.ApplyRuntimeOverrides(config.RuntimeOverrides{
		OutputDir: opts.outputDir,
		TempDir:   opts.tempDir,
		Mic:       opts.mic,
		Output:    opts.output,
		Language:  opts.language,
		Model:     opts.model,
	})
	discoverer := audio.SystemDiscoverer{Preferences: audio.Preferences{
		MicDeviceID:      cfg.Audio.MicDeviceID,
		MicDeviceName:    cfg.Audio.MicDeviceName,
		OutputDeviceID:   cfg.Audio.OutputDeviceID,
		OutputDeviceName: cfg.Audio.OutputDeviceName,
	}}
	if opts.list {
		devices, deviceWarnings, err := discoverer.ListDevices(ctx)
		if err != nil {
			logging.Printf("list-devices failed: %v", err)
			return err
		}
		for _, warning := range append(warnings, deviceWarnings...) {
			fmt.Fprintln(ioStreams.Stderr, "warning:", warning)
			logging.Printf("warning: %s", warning)
		}
		fmt.Fprint(ioStreams.Stdout, audio.FormatDevices(devices))
		return nil
	}

	apiKey := os.Getenv("OPENAI_API_KEY")
	// Print the resolved config source once at startup so the user can see at a
	// glance whether config came from the default file, an explicit --config file,
	// or nowhere (defaults + environment). Suppressed for --version (handled above
	// before run()) and for --list-devices (to keep device output clean); --silent
	// and TUI runs both print it on stdout so the line prefixes the session.
	configLine := source.SummaryLine()
	fmt.Fprintln(ioStreams.Stdout, configLine)
	logging.Printf("%s", configLine)
	prepared, err := app.Prepare(ctx, cfg, opts.configPath, apiKey, app.Dependencies{
		Discoverer:  discoverer,
		Recorder: audio.ExternalRecorder{
			Platform: runtime.GOOS,
		},
		Speaker:     speaker.New(cfg.Teams.Enabled),
		Transcriber: openai.Client{APIKey: apiKey},
	})
	if err != nil {
		logging.Printf("prepare failed: %v", err)
		return err
	}
	prepared.Warnings = append(warnings, prepared.Warnings...)
	for _, warning := range prepared.Warnings {
		logging.Printf("warning: %s", warning)
	}
	logging.Printf("selected devices: mic=%q output=%q transcript=%q temp=%q", prepared.Selection.Mic.DisplayName(), prepared.Selection.Output.DisplayName(), prepared.OutputDir, prepared.TempDir)
	ctx, stopSignals := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	defer stopSignals()
	if opts.silent {
		return app.RunSilent(ctx, prepared, ioStreams.Stdout, ioStreams.Stderr)
	}
	session, err := app.NewSession(prepared)
	if err != nil {
		return err
	}
	if err := session.Start(ctx); err != nil {
		return err
	}
	if err := tui.Run(ctx, session, prepared.Devices, prepared.Warnings); err != nil {
		return err
	}
	return session.Stop(context.Background())
}
