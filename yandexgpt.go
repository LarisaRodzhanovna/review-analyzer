package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"time"
)

// YandexGPT через OpenAI-совместимый эндпоинт Yandex AI Studio.
// Авторизация — API-ключ сервисного аккаунта с ролью ai.languageModels.user.
// Обычный публичный TLS: никаких особых сертификатов не нужно.

type YandexConfig struct {
	APIKey     string
	FolderID   string
	BaseURL    string
	Model      string // имя модели без префикса: yandexgpt-lite/latest, yandexgpt/latest
	Timeout    time.Duration
	MaxRetries int
}

func YandexConfigFromEnv() (YandexConfig, error) {
	c := YandexConfig{
		APIKey:     os.Getenv("YANDEX_API_KEY"),
		FolderID:   os.Getenv("YANDEX_FOLDER_ID"),
		BaseURL:    envOr("YANDEX_BASE_URL", "https://ai.api.cloud.yandex.net/v1"),
		Model:      envOr("YANDEX_MODEL", "yandexgpt-lite/latest"),
		Timeout:    60 * time.Second,
		MaxRetries: 3,
	}
	if c.APIKey == "" || c.FolderID == "" {
		return c, errors.New("нужны YANDEX_API_KEY и YANDEX_FOLDER_ID (см. README)")
	}
	return c, nil
}

type YandexClient struct {
	cfg  YandexConfig
	http *http.Client
	log  *slog.Logger
}

func NewYandexClient(cfg YandexConfig, log *slog.Logger) *YandexClient {
	return &YandexClient{
		cfg:  cfg,
		log:  log,
		http: &http.Client{Timeout: cfg.Timeout, Transport: &http.Transport{Proxy: http.ProxyFromEnvironment}},
	}
}

func (c *YandexClient) modelURI() string {
	return fmt.Sprintf("gpt://%s/%s", c.cfg.FolderID, c.cfg.Model)
}

func (c *YandexClient) Chat(ctx context.Context, msgs []Message) (ChatResult, error) {
	payload, _ := json.Marshal(map[string]any{
		"model":       c.modelURI(),
		"messages":    msgs,
		"temperature": 0.1,
		"max_tokens":  1000,
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.BaseURL+"/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return ChatResult{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Api-Key "+c.cfg.APIKey)
	req.Header.Set("OpenAI-Project", c.cfg.FolderID)

	body, status, err := doWithRetry(c.http, c.log, c.cfg.MaxRetries, req, payload, "yandex.chat")
	if err != nil {
		return ChatResult{}, err
	}
	switch {
	case status == http.StatusUnauthorized || status == http.StatusForbidden:
		return ChatResult{}, fmt.Errorf("доступ запрещён (%d): проверь API-ключ, folder ID и роль ai.languageModels.user у сервисного аккаунта: %s", status, trim(body))
	case status != http.StatusOK:
		return ChatResult{}, fmt.Errorf("yandex: статус %d: %s", status, trim(body))
	}
	var r struct {
		Choices []struct {
			Message Message `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &r); err != nil || len(r.Choices) == 0 {
		return ChatResult{}, fmt.Errorf("yandex: пустой или битый ответ: %s", trim(body))
	}
	return ChatResult{r.Choices[0].Message.Content, r.Usage.PromptTokens, r.Usage.CompletionTokens}, nil
}
