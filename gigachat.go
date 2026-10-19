package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

// Config — всё, что клиенту нужно знать снаружи. Секреты берём только из окружения.
type Config struct {
	AuthKey    string // ключ авторизации из личного кабинета (Base64 client_id:secret)
	Scope      string // GIGACHAT_API_PERS для физлиц
	OAuthURL   string
	APIURL     string
	Model      string
	CAFile     string // сертификат НУЦ Минцифры (у GigaChat свой корневой сертификат)
	Insecure   bool   // только для локальной отладки!
	Timeout    time.Duration
	MaxRetries int
}

func ConfigFromEnv() (Config, error) {
	c := Config{
		AuthKey:    os.Getenv("GIGACHAT_AUTH_KEY"),
		Scope:      envOr("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
		OAuthURL:   envOr("GIGACHAT_OAUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"),
		APIURL:     envOr("GIGACHAT_API_URL", "https://gigachat.devices.sberbank.ru/api/v1"),
		Model:      envOr("GIGACHAT_MODEL", "GigaChat-2"),
		CAFile:     os.Getenv("GIGACHAT_CA_FILE"),
		Insecure:   os.Getenv("GIGACHAT_INSECURE") == "1",
		Timeout:    60 * time.Second,
		MaxRetries: 3,
	}
	if c.AuthKey == "" {
		return c, errors.New("не задан GIGACHAT_AUTH_KEY (ключ авторизации из личного кабинета GigaChat API)")
	}
	return c, nil
}

type Client struct {
	cfg  Config
	http *http.Client
	log  *slog.Logger

	mu       sync.Mutex
	token    string
	tokenExp time.Time
}

func NewClient(cfg Config, log *slog.Logger) (*Client, error) {
	tlsCfg := &tls.Config{}
	if cfg.CAFile != "" {
		pem, err := os.ReadFile(cfg.CAFile)
		if err != nil {
			return nil, fmt.Errorf("чтение CA-файла: %w", err)
		}
		pool, _ := x509.SystemCertPool()
		if pool == nil {
			pool = x509.NewCertPool()
		}
		if !pool.AppendCertsFromPEM(pem) {
			// .cer с госуслуг приходит в DER — парсим напрямую
			cert, err := x509.ParseCertificate(pem)
			if err != nil {
				return nil, errors.New("CA-файл не похож ни на PEM, ни на DER-сертификат")
			}
			pool.AddCert(cert)
		}
		tlsCfg.RootCAs = pool
	}
	if cfg.Insecure {
		log.Warn("проверка TLS-сертификата ОТКЛЮЧЕНА — только для отладки")
		tlsCfg.InsecureSkipVerify = true
	}
	return &Client{
		cfg: cfg,
		log: log,
		http: &http.Client{
			Timeout:   cfg.Timeout,
			Transport: &http.Transport{TLSClientConfig: tlsCfg, Proxy: http.ProxyFromEnvironment},
		},
	}, nil
}

// ---------- OAuth: токен живёт 30 минут, кешируем и обновляем заранее ----------

func (c *Client) getToken(ctx context.Context, force bool) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !force && c.token != "" && time.Until(c.tokenExp) > time.Minute {
		return c.token, nil
	}

	form := url.Values{"scope": {c.cfg.Scope}}.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.OAuthURL, strings.NewReader(form))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("RqUID", newUUID())
	req.Header.Set("Authorization", "Basic "+c.cfg.AuthKey)

	body, status, err := doWithRetry(c.http, c.log, c.cfg.MaxRetries, req, []byte(form), "oauth")
	if err != nil {
		return "", err
	}
	if status != http.StatusOK {
		return "", fmt.Errorf("oauth: статус %d: %s", status, trim(body))
	}
	var r struct {
		AccessToken string `json:"access_token"`
		ExpiresAt   int64  `json:"expires_at"` // миллисекунды
	}
	if err := json.Unmarshal(body, &r); err != nil || r.AccessToken == "" {
		return "", fmt.Errorf("oauth: не разобрали ответ: %v", err)
	}
	c.token = r.AccessToken
	c.tokenExp = time.UnixMilli(r.ExpiresAt)
	c.log.Info("получен access token", "expires_at", c.tokenExp.Format(time.TimeOnly))
	return c.token, nil
}

// ---------- Chat completions ----------

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatResult struct {
	Content      string
	PromptTokens int
	OutTokens    int
}

func (c *Client) Chat(ctx context.Context, msgs []Message) (ChatResult, error) {
	payload, _ := json.Marshal(map[string]any{
		"model":       c.cfg.Model,
		"messages":    msgs,
		"temperature": 0.1, // для классификации нужна стабильность, а не креатив
	})

	for authAttempt := 0; authAttempt < 2; authAttempt++ {
		tok, err := c.getToken(ctx, authAttempt > 0)
		if err != nil {
			return ChatResult{}, err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.APIURL+"/chat/completions", bytes.NewReader(payload))
		if err != nil {
			return ChatResult{}, err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json")
		req.Header.Set("Authorization", "Bearer "+tok)

		body, status, err := doWithRetry(c.http, c.log, c.cfg.MaxRetries, req, payload, "chat")
		if err != nil {
			return ChatResult{}, err
		}
		if status == http.StatusUnauthorized {
			c.log.Warn("401 — токен протух, обновляю и повторяю")
			continue
		}
		if status != http.StatusOK {
			return ChatResult{}, fmt.Errorf("chat: статус %d: %s", status, trim(body))
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
			return ChatResult{}, fmt.Errorf("chat: пустой или битый ответ: %s", trim(body))
		}
		return ChatResult{r.Choices[0].Message.Content, r.Usage.PromptTokens, r.Usage.CompletionTokens}, nil
	}
	return ChatResult{}, errors.New("chat: не удалось авторизоваться после обновления токена")
}

// ---------- Ретраи: сеть, 429 и 5xx — повторяем с экспоненциальным backoff ----------

func doWithRetry(hc *http.Client, log *slog.Logger, maxRetries int, req *http.Request, payload []byte, op string) ([]byte, int, error) {
	var lastErr error
	for attempt := 1; attempt <= maxRetries; attempt++ {
		r := req.Clone(req.Context())
		r.Body = io.NopCloser(bytes.NewReader(payload))
		start := time.Now()

		resp, err := hc.Do(r)
		if err == nil {
			body, rerr := io.ReadAll(resp.Body)
			resp.Body.Close()
			log.Info("http", "op", op, "status", resp.StatusCode, "attempt", attempt, "ms", time.Since(start).Milliseconds())
			if rerr == nil && !retryable(resp.StatusCode) {
				return body, resp.StatusCode, nil
			}
			lastErr = fmt.Errorf("статус %d", resp.StatusCode)
			if rerr != nil {
				lastErr = rerr
			}
		} else {
			lastErr = err
			log.Warn("http error", "op", op, "attempt", attempt, "err", err)
		}
		if attempt == maxRetries || req.Context().Err() != nil {
			break
		}
		// 1s, 2s, 4s… + случайный джиттер, чтобы не долбить сервер синхронно
		wait := time.Duration(1<<(attempt-1))*time.Second + time.Duration(rand.Intn(300))*time.Millisecond
		log.Info("retry", "op", op, "wait", wait.String())
		select {
		case <-time.After(wait):
		case <-req.Context().Done():
			return nil, 0, req.Context().Err()
		}
	}
	return nil, 0, fmt.Errorf("%s: исчерпаны попытки: %w", op, lastErr)
}

func retryable(status int) bool {
	return status == http.StatusTooManyRequests || status >= 500
}

func newUUID() string {
	b := make([]byte, 16)
	rand.Read(b)
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
}

func trim(b []byte) string {
	s := string(b)
	if len(s) > 300 {
		return s[:300] + "…"
	}
	return s
}
