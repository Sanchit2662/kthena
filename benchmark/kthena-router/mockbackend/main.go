/*
Copyright The Volcano Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// mockbackend simulates a vLLM inference backend for router benchmarking.
// It exposes /v1/chat/completions (streaming and non-streaming) and /metrics
// with the exact vllm:* Prometheus metric names that kthena-router scrapes.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

var (
	port      = flag.Int("port", 8000, "TCP port to listen on")
	ttftMs    = flag.Int("ttft-ms", 50, "simulated TTFT in milliseconds")
	tpotMs    = flag.Int("tpot-ms", 20, "simulated time-per-output-token in milliseconds")
	numTokens = flag.Int("tokens", 64, "output tokens per response")
	kvUsage   = flag.Float64("kv-usage", 0.3, "simulated KV cache usage fraction [0,1]")
	jitterPct = flag.Float64("jitter", 0.3, "latency jitter as a fraction of the base value")
)

// rollingStats accumulates sum and count for histogram approximation.
type rollingStats struct {
	mu    sync.Mutex
	sum   float64
	count uint64
}

func (s *rollingStats) record(v float64) {
	s.mu.Lock()
	s.sum += v
	s.count++
	s.mu.Unlock()
}

func (s *rollingStats) snapshot() (sum float64, count uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sum, s.count
}

var (
	runningReqs int64
	waitingReqs int64
	ttftStats   rollingStats
	itlStats    rollingStats
)

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model    string        `json:"model"`
	Stream   bool          `json:"stream"`
	Messages []chatMessage `json:"messages"`
}

type chatChoice struct {
	Index        int          `json:"index"`
	Message      *chatMessage `json:"message,omitempty"`
	Delta        *chatMessage `json:"delta,omitempty"`
	FinishReason *string      `json:"finish_reason"`
}

type usageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type chatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   *usageInfo   `json:"usage,omitempty"`
}

// jittered returns base ± jitterPct%, minimum 1 ms.
func jittered(baseMs int) time.Duration {
	delta := float64(baseMs) * *jitterPct
	v := float64(baseMs) + (rand.Float64()*2-1)*delta
	if v < 1 {
		v = 1
	}
	return time.Duration(v) * time.Millisecond
}

func handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	atomic.AddInt64(&waitingReqs, 1)
	var req chatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		atomic.AddInt64(&waitingReqs, -1)
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	atomic.AddInt64(&waitingReqs, -1)
	atomic.AddInt64(&runningReqs, 1)
	defer atomic.AddInt64(&runningReqs, -1)

	model := req.Model
	if model == "" {
		model = "mock-model"
	}

	if req.Stream {
		serveStreaming(w, model)
		return
	}
	serveNonStreaming(w, model)
}

func serveStreaming(w http.ResponseWriter, model string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	ttft := jittered(*ttftMs)
	time.Sleep(ttft)
	ttftStats.record(ttft.Seconds())

	tokens := *numTokens
	for i := 0; i < tokens; i++ {
		tpot := jittered(*tpotMs)
		time.Sleep(tpot)
		itlStats.record(tpot.Seconds())

		var finishReason *string
		if i == tokens-1 {
			s := "stop"
			finishReason = &s
		}
		chunk := chatResponse{
			ID:      "mock-stream-1",
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   model,
			Choices: []chatChoice{{
				Index:        0,
				Delta:        &chatMessage{Content: "t"},
				FinishReason: finishReason,
			}},
		}
		b, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", b)
		flusher.Flush()
	}
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func serveNonStreaming(w http.ResponseWriter, model string) {
	ttft := jittered(*ttftMs)
	time.Sleep(ttft)
	ttftStats.record(ttft.Seconds())

	tokens := *numTokens
	for i := 0; i < tokens; i++ {
		tpot := jittered(*tpotMs)
		time.Sleep(tpot)
		itlStats.record(tpot.Seconds())
	}

	done := "stop"
	resp := chatResponse{
		ID:      "mock-1",
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []chatChoice{{
			Index:        0,
			Message:      &chatMessage{Role: "assistant", Content: "mock response"},
			FinishReason: &done,
		}},
		Usage: &usageInfo{
			PromptTokens:     16,
			CompletionTokens: tokens,
			TotalTokens:      16 + tokens,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleMetrics writes vllm:* Prometheus exposition text that kthena-router scrapes.
func handleMetrics(w http.ResponseWriter, r *http.Request) {
	running := atomic.LoadInt64(&runningReqs)
	waiting := atomic.LoadInt64(&waitingReqs)
	ttftSum, ttftCount := ttftStats.snapshot()
	itlSum, itlCount := itlStats.snapshot()

	// Approximate histogram: all observed samples fall in the +Inf bucket.
	// kthena-router calls metrics.LastPeriodAvg which only needs _sum/_count.
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprintf(w, `# HELP vllm:kv_cache_usage_perc KV cache usage fraction
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc %f
# HELP vllm:num_requests_running Running requests
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running %d
# HELP vllm:num_requests_waiting Waiting requests
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting %d
# HELP vllm:time_to_first_token_seconds Time to first token
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.05"} 0
vllm:time_to_first_token_seconds_bucket{le="0.1"} 0
vllm:time_to_first_token_seconds_bucket{le="0.5"} %d
vllm:time_to_first_token_seconds_bucket{le="+Inf"} %d
vllm:time_to_first_token_seconds_sum %f
vllm:time_to_first_token_seconds_count %d
# HELP vllm:inter_token_latency_seconds Inter-token latency
# TYPE vllm:inter_token_latency_seconds histogram
vllm:inter_token_latency_seconds_bucket{le="0.01"} 0
vllm:inter_token_latency_seconds_bucket{le="0.05"} 0
vllm:inter_token_latency_seconds_bucket{le="0.1"} %d
vllm:inter_token_latency_seconds_bucket{le="+Inf"} %d
vllm:inter_token_latency_seconds_sum %f
vllm:inter_token_latency_seconds_count %d
`,
		*kvUsage,
		running,
		waiting,
		ttftCount, ttftCount, ttftSum, ttftCount,
		itlCount, itlCount, itlSum, itlCount,
	)
}

func NewMux() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", handleChat)
	mux.HandleFunc("/metrics", handleMetrics)
	return mux
}

func main() {
	flag.Parse()
	addr := fmt.Sprintf(":%d", *port)
	log.Printf("mock vLLM backend listening on %s  ttft=%dms tpot=%dms tokens=%d kv=%.2f",
		addr, *ttftMs, *tpotMs, *numTokens, *kvUsage)
	if err := http.ListenAndServe(addr, NewMux()); err != nil {
		log.Fatal(err)
	}
}
