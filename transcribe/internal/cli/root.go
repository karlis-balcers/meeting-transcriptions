package cli

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"

	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/app"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/audio"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/config"
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
	cmd.Flags().BoolVar(&opts.list, "list-devices", false, "list detected audio devices and exit")
	cmd.Flags().BoolVar(&opts.version, "version", false, "print version and exit")
	return cmd
}

func run(ctx context.Context, ioStreams IO, opts options) error {
	explicitConfig := opts.configPath != ""
	cfg, warnings, err := config.Load(opts.configPath, explicitConfig, config.EnvMapFromOS())
	if err != nil {
		return err
	}
	if opts.configPath == "" {
		if defaultPath, err := config.DefaultPath(); err == nil {
			opts.configPath = defaultPath
		}
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
			return err
		}
		for _, warning := range append(warnings, deviceWarnings...) {
			fmt.Fprintln(ioStreams.Stderr, "warning:", warning)
		}
		fmt.Fprint(ioStreams.Stdout, audio.FormatDevices(devices))
		return nil
	}

	apiKey := os.Getenv("OPENAI_API_KEY")
	prepared, err := app.Prepare(ctx, cfg, opts.configPath, apiKey, app.Dependencies{
		Discoverer:  discoverer,
		Recorder:    audio.ExternalRecorder{},
		Speaker:     speaker.New(cfg.Teams.Enabled),
		Transcriber: openai.Client{APIKey: apiKey},
	})
	if err != nil {
		return err
	}
	prepared.Warnings = append(warnings, prepared.Warnings...)
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
