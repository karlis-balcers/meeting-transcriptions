package cli

import "github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/logging"

func setupLogging(level int) (func(), error) {
	if level <= 0 {
		return nil, nil
	}
	path, err := logging.EnableCurrentDir()
	if err != nil {
		return nil, err
	}
	logging.Printf("logging enabled: %s", path)
	return func() {
		_ = logging.Close()
	}, nil
}