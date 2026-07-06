package speaker

import (
	"regexp"
	"strings"
)

var speakerTitlePatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)^(.+?)\s+(?:is\s+)?speaking(?:\s+now)?(?:\s*[-|].*)?$`),
	regexp.MustCompile(`(?i)^(.+?)\s+(?:is\s+)?presenting(?:\s*[-|].*)?$`),
	regexp.MustCompile(`(?i)^(.+?)\s+started\s+speaking(?:\s*[-|].*)?$`),
	regexp.MustCompile(`(?i)^(.+?)\s*\|\s*Microsoft Teams$`),
	regexp.MustCompile(`(?i)^Microsoft Teams\s*[-–]\s*(.+?)$`),
}

var genericTitleWords = map[string]struct{}{
	"activity":        {},
	"calendar":        {},
	"chat":            {},
	"calls":           {},
	"files":           {},
	"meeting":         {},
	"microsoft teams": {},
	"teams":           {},
}

// DetectSpeakerFromTitles extracts a best-effort active speaker from Teams window titles.
func DetectSpeakerFromTitles(titles []string) string {
	for _, title := range titles {
		candidate := speakerFromTitle(title)
		if candidate != "" {
			return candidate
		}
	}
	return ""
}

func speakerFromTitle(title string) string {
	title = strings.TrimSpace(title)
	if title == "" {
		return ""
	}
	for _, pattern := range speakerTitlePatterns {
		matches := pattern.FindStringSubmatch(title)
		if len(matches) < 2 {
			continue
		}
		if candidate := normalizeSpeakerName(matches[1]); candidate != "" {
			return candidate
		}
	}
	return ""
}

func normalizeSpeakerName(value string) string {
	value = strings.TrimSpace(value)
	value = strings.Trim(value, `"'`)
	value = strings.TrimSpace(strings.TrimSuffix(value, "(Guest)"))
	value = strings.Join(strings.Fields(value), " ")
	if value == "" {
		return ""
	}
	key := strings.ToLower(value)
	if _, generic := genericTitleWords[key]; generic {
		return ""
	}
	lower := strings.ToLower(value)
	if strings.Contains(lower, "microsoft teams") || strings.Contains(lower, "meeting compact view") || strings.Contains(lower, "teams meeting") {
		return ""
	}
	return value
}
