package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"sort"
	"strings"
	"time"
)

func main() {
	in := flag.String("in", "reviews.txt", "файл с отзывами, по одному на строку")
	out := flag.String("out", "results.json", "куда сохранить результат")
	fixes := flag.Int("fixes", 2, "сколько раз просить модель исправить невалидный ответ")
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))

	client, model, err := newLLM(envOr("LLM_PROVIDER", "yandex"), log)
	if err != nil {
		fail(err)
	}
	reviews, err := readLines(*in)
	if err != nil {
		fail(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	log.Info("старт", "model", model, "reviews", len(reviews))
	results := make([]Result, 0, len(reviews))
	for i, r := range reviews {
		res := AnalyzeReview(ctx, client, r, *fixes)
		log.Info("отзыв обработан", "n", i+1, "ok", res.Analysis != nil, "fixes", res.Fixes, "tokens", res.Tokens)
		results = append(results, res)
	}

	data, _ := json.MarshalIndent(results, "", "  ")
	if err := os.WriteFile(*out, data, 0o644); err != nil {
		fail(err)
	}
	printSummary(results)
	fmt.Printf("\nПодробности: %s\n", *out)
}

// newLLM — выбор провайдера. Бизнес-логика (analyze.go) знает только интерфейс LLM.
func newLLM(provider string, log *slog.Logger) (LLM, string, error) {
	switch provider {
	case "yandex":
		cfg, err := YandexConfigFromEnv()
		if err != nil {
			return nil, "", err
		}
		c := NewYandexClient(cfg, log)
		return c, c.modelURI(), nil
	case "gigachat":
		cfg, err := ConfigFromEnv()
		if err != nil {
			return nil, "", err
		}
		c, err := NewClient(cfg, log)
		return c, cfg.Model, err
	default:
		return nil, "", fmt.Errorf("неизвестный LLM_PROVIDER=%q (yandex | gigachat)", provider)
	}
}

func printSummary(results []Result) {
	pains := map[string]int{}
	sent := map[string]int{}
	var failed, fixed, tokens int
	for _, r := range results {
		tokens += r.Tokens
		if r.Fixes > 0 {
			fixed++
		}
		if r.Analysis == nil {
			failed++
			continue
		}
		sent[r.Analysis.Sentiment]++
		if r.Analysis.Sentiment == "negative" {
			pains[r.Analysis.Category]++
		}
	}
	fmt.Printf("\n=== Итого: %d отзывов ===\n", len(results))
	fmt.Printf("Тональность: 👍 %d  😐 %d  👎 %d\n", sent["positive"], sent["neutral"], sent["negative"])
	fmt.Println("Боли клиентов (негатив по категориям):")
	type kv struct {
		k string
		v int
	}
	var list []kv
	for k, v := range pains {
		list = append(list, kv{k, v})
	}
	sort.Slice(list, func(i, j int) bool { return list[i].v > list[j].v })
	for _, p := range list {
		fmt.Printf("  %-22s %s %d\n", p.k, strings.Repeat("█", p.v), p.v)
	}
	fmt.Printf("Качество: самоисправлений %d, не удалось разобрать %d, токенов всего %d\n", fixed, failed, tokens)
}

func readLines(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var lines []string
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	for sc.Scan() {
		if s := strings.TrimSpace(sc.Text()); s != "" && !strings.HasPrefix(s, "#") {
			lines = append(lines, s)
		}
	}
	return lines, sc.Err()
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "ошибка:", err)
	os.Exit(1)
}
