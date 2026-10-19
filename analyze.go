package main

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
)

// Analysis — контракт ответа модели. Всё, что не проходит Validate, назад в модель на исправление.
type Analysis struct {
	Sentiment string `json:"sentiment"` // positive | negative | neutral
	Category  string `json:"category"`  // одна из Categories
	Pain      string `json:"pain"`      // боль клиента одной фразой, "" если её нет
	Quote     string `json:"quote"`     // дословный фрагмент отзыва, подтверждающий вывод
}

var Sentiments = []string{"positive", "negative", "neutral"}
var Categories = []string{"качество", "доставка", "цена", "сервис", "описание_не_совпадает", "другое"}

const systemPrompt = `Ты аналитик клиентских отзывов. На вход — один отзыв.
Верни ТОЛЬКО JSON-объект без пояснений и без markdown:
{"sentiment": "...", "category": "...", "pain": "...", "quote": "..."}
- sentiment: одно из positive, negative, neutral
- category: одно из %s
- pain: главная проблема клиента одной короткой фразой; пустая строка, если проблемы нет
- quote: ДОСЛОВНЫЙ фрагмент из отзыва (до 15 слов), на котором основан вывод`

// Validate — проверяем не только формат, но и «честность»: цитата должна реально быть в отзыве.
func (a Analysis) Validate(review string) []string {
	var errs []string
	if !slices.Contains(Sentiments, a.Sentiment) {
		errs = append(errs, fmt.Sprintf("sentiment=%q не из списка %v", a.Sentiment, Sentiments))
	}
	if !slices.Contains(Categories, a.Category) {
		errs = append(errs, fmt.Sprintf("category=%q не из списка %v", a.Category, Categories))
	}
	if a.Sentiment == "negative" && strings.TrimSpace(a.Pain) == "" {
		errs = append(errs, "у негативного отзыва пустое поле pain")
	}
	if a.Quote == "" || !strings.Contains(normalize(review), normalize(a.Quote)) {
		errs = append(errs, "quote не является дословным фрагментом отзыва (галлюцинация?)")
	}
	return errs
}

func normalize(s string) string {
	s = strings.ToLower(s)
	s = strings.NewReplacer("ё", "е", "«", "\"", "»", "\"", " ", " ").Replace(s)
	return strings.Join(strings.Fields(s), " ")
}

// extractJSON — модели любят оборачивать ответ в ```json … ``` или добавлять «Вот ответ:».
func extractJSON(s string) string {
	start, end := strings.Index(s, "{"), strings.LastIndex(s, "}")
	if start < 0 || end <= start {
		return s
	}
	return s[start : end+1]
}

type Result struct {
	Review   string    `json:"review"`
	Analysis *Analysis `json:"analysis,omitempty"`
	Fixes    int       `json:"fixes"` // сколько раз пришлось просить модель исправиться
	Error    string    `json:"error,omitempty"`
	Tokens   int       `json:"tokens"`
}

type LLM interface {
	Chat(ctx context.Context, msgs []Message) (ChatResult, error)
}

// AnalyzeReview: запрос → парсинг → валидация → при ошибке возвращаем модели её же ответ
// с перечнем нарушений (self-correction), максимум maxFixes раз.
func AnalyzeReview(ctx context.Context, llm LLM, review string, maxFixes int) Result {
	res := Result{Review: review}
	msgs := []Message{
		{Role: "system", Content: fmt.Sprintf(systemPrompt, strings.Join(Categories, ", "))},
		{Role: "user", Content: review},
	}
	for try := 0; try <= maxFixes; try++ {
		out, err := llm.Chat(ctx, msgs)
		res.Tokens += out.PromptTokens + out.OutTokens
		if err != nil {
			res.Error = err.Error()
			return res
		}
		var a Analysis
		var problems []string
		if err := json.Unmarshal([]byte(extractJSON(out.Content)), &a); err != nil {
			problems = []string{"ответ не является валидным JSON: " + err.Error()}
		} else {
			problems = a.Validate(review)
		}
		if len(problems) == 0 {
			res.Analysis = &a
			res.Error = ""
			return res
		}
		res.Fixes++
		msgs = append(msgs,
			Message{Role: "assistant", Content: out.Content},
			Message{Role: "user", Content: "Ответ не прошёл проверку:\n- " + strings.Join(problems, "\n- ") +
				"\nИсправь и верни только JSON."},
		)
		res.Error = strings.Join(problems, "; ")
	}
	res.Fixes-- // последняя «попытка исправления» не состоялась
	return res
}
