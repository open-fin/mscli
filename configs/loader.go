package configs

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// LoadWithEnv loads built-in defaults and applies environment variable overrides.
func LoadWithEnv() (*Config, error) {
	cfg := DefaultConfig()
	defaultContextWindow := cfg.Context.Window

	if fileCfg, err := LoadUserConfig(); err == nil {
		cfg.Merge(fileCfg)
	} else if !os.IsNotExist(err) {
		return nil, err
	}

	// model-derived context defaults > env overrides > built-in defaults
	applyModelTokenDefaults(cfg, defaultContextWindow)
	previousModel := cfg.Model.Model
	ApplyEnvOverrides(cfg)
	RefreshModelTokenDefaults(cfg, previousModel)
	cfg.normalize()

	// Validate
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("validate config: %w", err)
	}

	return cfg, nil
}

// ApplyEnvOverrides applies environment variable overrides to the config.
// Unified MSCLI_* overrides are applied on top of built-in defaults.
func ApplyEnvOverrides(cfg *Config) {
	previousContextWindow := cfg.Context.Window
	contextWindowSet := false
	contextReserveSet := false

	// Model settings
	if v := strings.TrimSpace(os.Getenv("MSCLI_MODEL")); v != "" {
		cfg.Model.Model = v
	}
	if v := strings.TrimSpace(os.Getenv("MSCLI_API_KEY")); v != "" {
		cfg.Model.Key = v
	}
	if v := strings.TrimSpace(os.Getenv("MSCLI_BASE_URL")); v != "" {
		cfg.Model.URL = v
	}
	if v := strings.TrimSpace(os.Getenv("MSCLI_PROVIDER")); v != "" {
		cfg.Model.Provider = v
	}
	if v := os.Getenv("MSCLI_TEMPERATURE"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			cfg.Request.Temperature = &f
		}
	}
	if v := os.Getenv("MSCLI_MAX_TOKENS"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.Request.MaxTokens = &i
		}
	}
	if v := os.Getenv("MSCLI_MAX_ITERATIONS"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.Request.MaxIterations = &i
		}
	}
	if v := strings.TrimSpace(os.Getenv("MSCLI_EFFORT")); v != "" {
		cfg.Request.Effort = v
	}
	if v := os.Getenv("MSCLI_TIMEOUT"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.Model.TimeoutSec = i
		}
	}

	// UI settings
	if v := os.Getenv("MSCLI_UI_ENABLED"); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			cfg.UI.Enabled = b
		}
	}
	if v := os.Getenv("MSCLI_THEME"); v != "" {
		cfg.UI.Theme = v
	}

	// Permissions
	if v := os.Getenv("MSCLI_PERMISSIONS_SKIP"); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			cfg.Permissions.SkipRequests = b
		}
	}
	if v := os.Getenv("MSCLI_PERMISSIONS_DEFAULT"); v != "" {
		cfg.Permissions.DefaultLevel = v
	}

	// Filesystem
	if roots := pathListEnv("MSCLI_EXTERNAL_READ_ROOTS"); len(roots) > 0 {
		cfg.Filesystem.ExternalReadRoots = roots
	}
	if roots := pathListEnv("MSCLI_EXTERNAL_WRITE_ROOTS"); len(roots) > 0 {
		cfg.Filesystem.ExternalWriteRoots = roots
	}

	// Context settings
	if v := os.Getenv("MSCLI_CONTEXT_WINDOW"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.Context.Window = i
			contextWindowSet = true
		}
	}
	if v := os.Getenv("MSCLI_CONTEXT_RESERVE"); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			cfg.Context.ReserveTokens = i
			contextReserveSet = true
		}
	}
	if contextWindowSet && !contextReserveSet {
		refreshContextReserveDefaults(cfg, previousContextWindow)
	}

}

func UserConfigPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolve home directory: %w", err)
	}
	return filepath.Join(home, ".mscli", "config.yaml"), nil
}

func LoadUserConfig() (*Config, error) {
	path, err := UserConfigPath()
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse user config %q: %w", path, err)
	}
	return &cfg, nil
}

func saveUserFilesystemRoots(readRoots, writeRoots []string) error {
	path, err := UserConfigPath()
	if err != nil {
		return err
	}
	cfg := DefaultConfig()
	if existing, err := LoadUserConfig(); err == nil {
		cfg.Merge(existing)
	} else if !os.IsNotExist(err) {
		return err
	}
	if readRoots != nil {
		cfg.Filesystem.ExternalReadRoots = append([]string(nil), readRoots...)
	}
	if writeRoots != nil {
		cfg.Filesystem.ExternalWriteRoots = append([]string(nil), writeRoots...)
	}
	data, err := yaml.Marshal(cfg)
	if err != nil {
		return fmt.Errorf("marshal user config: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create user config directory: %w", err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		return fmt.Errorf("write user config %q: %w", path, err)
	}
	return nil
}

func SaveUserExternalReadRoots(roots []string) error {
	return saveUserFilesystemRoots(roots, nil)
}

func SaveUserExternalWriteRoots(roots []string) error {
	return saveUserFilesystemRoots(nil, roots)
}

func pathListEnv(key string) []string {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return nil
	}
	sep := string(os.PathListSeparator)
	if !strings.Contains(v, sep) && strings.Contains(v, ",") {
		sep = ","
	}
	parts := strings.Split(v, sep)
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		if trimmed := strings.TrimSpace(part); trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

// StringSliceEnv splits an environment variable by comma.
func StringSliceEnv(key string) []string {
	v := os.Getenv(key)
	if v == "" {
		return nil
	}
	parts := strings.Split(v, ",")
	for i, p := range parts {
		parts[i] = strings.TrimSpace(p)
	}
	return parts
}
