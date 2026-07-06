//go:build !windows

package speaker

func New(enabled bool) Detector {
	if !enabled {
		return &noopDetector{current: "Person_?"}
	}
	return &noopDetector{warning: "MS Teams speaker recognition is only available on Windows in this build; using Person_? for output audio", current: "Person_?"}
}
