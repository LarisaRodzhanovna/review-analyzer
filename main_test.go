package main

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// Фейковый GigaChat: проверяем ретраи, обновление токена и самоисправление модели.
func newFakeServer(t *testing.T, replies []string, failFirst int) (*httptest.Server, *int32, *int32) {
	var chatCalls, tokenCalls int32
	mux := http.NewServeMux()
	mux.HandleFunc("/oauth", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("RqUID") == "" || r.Header.Get("Authorization") != "Basic test-key" {
			t.Errorf("oauth: нет RqUID или неверный Basic")
		}
		n := atomic.AddInt32(&tokenCalls, 1)
		json.NewEncoder(w).Encode(map[string]any{
			"access_token": "tok" + string(rune('0'+n)),
			"expires_at":   time.Now().Add(30 * time.Minute).UnixMilli(),
		})
	})
	mux.HandleFunc("/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		n := int(atomic.AddInt32(&chatCalls, 1))
		if n <= failFirst {
			w.WriteHeader(http.StatusServiceUnavailable) // имитируем сбой
			return
		}
		if r.Header.Get("Authorization") == "Bearer tok1" && failFirst < 0 {
			w.WriteHeader(http.StatusUnauthorized) // первый токен «протух»
			return
		}
		io.ReadAll(r.Body)
		idx := min(n-1-max(failFirst, 0), len(replies)-1)
		if failFirst < 0 {
			idx = min(n-2, len(replies)-1)
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{"message": map[string]string{"role": "assistant", "content": replies[idx]}}},
			"usage":   map[string]int{"prompt_tokens": 100, "completion_tokens": 20},
		})
	})
	return httptest.NewServer(mux), &chatCalls, &tokenCalls
}

func testClient(srv *httptest.Server) *Client {
	c, _ := NewClient(Config{
		AuthKey: "test-key", Scope: "GIGACHAT_API_PERS",
		OAuthURL: srv.URL + "/oauth", APIURL: srv.URL, Model: "GigaChat-2",
		Timeout: 5 * time.Second, MaxRetries: 3,
	}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	return c
}

const review = "Через месяц отвалилась ручка, пластик очень хлипкий."
const good = `{"sentiment":"negative","category":"качество","pain":"ломается ручка","quote":"отвалилась ручка"}`

func TestHappyPathWithMarkdownWrapper(t *testing.T) {
	srv, _, _ := newFakeServer(t, []string{"Вот ответ:\n```json\n" + good + "\n```"}, 0)
	defer srv.Close()
	res := AnalyzeReview(context.Background(), testClient(srv), review, 2)
	if res.Analysis == nil || res.Analysis.Category != "качество" || res.Fixes != 0 {
		t.Fatalf("ожидали успешный разбор без исправлений, получили %+v", res)
	}
}

func TestRetryOn503(t *testing.T) {
	srv, calls, _ := newFakeServer(t, []string{good}, 2)
	defer srv.Close()
	res := AnalyzeReview(context.Background(), testClient(srv), review, 0)
	if res.Analysis == nil || *calls != 3 {
		t.Fatalf("ожидали успех на 3-й попытке, calls=%d res=%+v", *calls, res)
	}
}

func TestTokenRefreshOn401(t *testing.T) {
	srv, _, tokens := newFakeServer(t, []string{good}, -1)
	defer srv.Close()
	res := AnalyzeReview(context.Background(), testClient(srv), review, 0)
	if res.Analysis == nil || *tokens != 2 {
		t.Fatalf("ожидали обновление токена, tokens=%d res=%+v", *tokens, res)
	}
}

func TestSelfCorrectionOnHallucinatedQuote(t *testing.T) {
	bad := `{"sentiment":"negative","category":"качество","pain":"ломается","quote":"ужасный товар, не покупайте"}`
	srv, _, _ := newFakeServer(t, []string{bad, good}, 0)
	defer srv.Close()
	res := AnalyzeReview(context.Background(), testClient(srv), review, 2)
	if res.Analysis == nil || res.Fixes != 1 {
		t.Fatalf("ожидали 1 самоисправление, получили %+v", res)
	}
}

func TestGivesUpAfterMaxFixes(t *testing.T) {
	srv, _, _ := newFakeServer(t, []string{`{"sentiment":"angry"}`}, 0)
	defer srv.Close()
	res := AnalyzeReview(context.Background(), testClient(srv), review, 2)
	if res.Analysis != nil || res.Error == "" || res.Fixes != 2 {
		t.Fatalf("ожидали отказ после 2 исправлений, получили %+v", res)
	}
}

func TestYandexClient(t *testing.T) {
	var gotAuth, gotProject, gotModel string
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&calls, 1) == 1 {
			w.WriteHeader(http.StatusTooManyRequests) // первый раз — лимит, проверяем ретрай
			return
		}
		gotAuth, gotProject = r.Header.Get("Authorization"), r.Header.Get("OpenAI-Project")
		var body struct {
			Model string `json:"model"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		gotModel = body.Model
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{"message": map[string]string{"role": "assistant", "content": good}}},
		})
	}))
	defer srv.Close()
	c := NewYandexClient(YandexConfig{APIKey: "k", FolderID: "b1gfolder", BaseURL: srv.URL,
		Model: "yandexgpt-lite/latest", Timeout: 5 * time.Second, MaxRetries: 3},
		slog.New(slog.NewTextHandler(io.Discard, nil)))
	res := AnalyzeReview(context.Background(), c, review, 0)
	if res.Analysis == nil {
		t.Fatalf("ожидали успех: %+v", res)
	}
	if gotAuth != "Api-Key k" || gotProject != "b1gfolder" || gotModel != "gpt://b1gfolder/yandexgpt-lite/latest" {
		t.Fatalf("неверные заголовки/модель: %q %q %q", gotAuth, gotProject, gotModel)
	}
}
