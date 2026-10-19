package main

import "os"

// envOr - читает переменную среды; если её нет, возвращает значение по умолчанию.
func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
