package filter

import (
	"regexp"
	"strings"
)

type Config struct {
	MinChars int
	Exact    []string
	Prefixes []string
	Contains []string
	Regex    []string
	Keywords []string
}

type Filter struct {
	cfg     Config
	regexes []*regexp.Regexp
}

func New(cfg Config) (*Filter, []string) {
	var warnings []string
	f := &Filter{cfg: cfg}
	for _, pattern := range cfg.Regex {
		compiled, err := regexp.Compile(pattern)
		if err != nil {
			warnings = append(warnings, "ignored invalid transcript filter regex: "+pattern)
			continue
		}
		f.regexes = append(f.regexes, compiled)
	}
	return f, warnings
}

func (f *Filter) Keep(text string) bool {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return false
	}
	if f.containsKeyword(trimmed) {
		return true
	}
	if f.cfg.MinChars > 0 && len([]rune(trimmed)) < f.cfg.MinChars {
		return false
	}
	lower := strings.ToLower(trimmed)
	for _, exact := range f.cfg.Exact {
		if strings.ToLower(strings.TrimSpace(exact)) == lower {
			return false
		}
	}
	for _, prefix := range f.cfg.Prefixes {
		if strings.HasPrefix(lower, strings.ToLower(strings.TrimSpace(prefix))) {
			return false
		}
	}
	for _, contains := range f.cfg.Contains {
		if strings.Contains(lower, strings.ToLower(strings.TrimSpace(contains))) {
			return false
		}
	}
	for _, regex := range f.regexes {
		if regex.MatchString(trimmed) {
			return false
		}
	}
	return true
}

func (f *Filter) containsKeyword(text string) bool {
	lower := strings.ToLower(text)
	for _, keyword := range f.cfg.Keywords {
		keyword = strings.ToLower(strings.TrimSpace(keyword))
		if keyword != "" && strings.Contains(lower, keyword) {
			return true
		}
	}
	return false
}
