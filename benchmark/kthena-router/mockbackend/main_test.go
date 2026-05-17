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

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func resetState() {
	atomic.StoreInt64(&runningReqs, 0)
	atomic.StoreInt64(&waitingReqs, 0)
	ttftStats = rollingStats{}
	itlStats = rollingStats{}
}

func TestNonStreamingResponse(t *testing.T) {
	resetState()
	*ttftMs = 1
	*tpotMs = 1
	*numTokens = 3
	*jitterPct = 0

	body := `{"model":"test-model","stream":false,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp chatResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Object != "chat.completion" {
		t.Errorf("expected chat.completion, got %q", resp.Object)
	}
	if len(resp.Choices) == 0 {
		t.Fatal("no choices in response")
	}
	if resp.Choices[0].Message == nil {
		t.Fatal("nil message in choice")
	}
	if resp.Usage == nil {
		t.Fatal("nil usage in non-streaming response")
	}
	if resp.Usage.CompletionTokens != *numTokens {
		t.Errorf("expected %d completion tokens, got %d", *numTokens, resp.Usage.CompletionTokens)
	}
}

func TestStreamingResponse(t *testing.T) {
	resetState()
	*ttftMs = 1
	*tpotMs = 1
	*numTokens = 4
	*jitterPct = 0

	body := `{"model":"test-model","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	ct := w.Header().Get("Content-Type")
	if !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("expected text/event-stream content-type, got %q", ct)
	}

	lines := bufio.NewScanner(w.Body)
	tokenChunks := 0
	sawDone := false
	for lines.Scan() {
		line := lines.Text()
		if line == "data: [DONE]" {
			sawDone = true
			continue
		}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		var chunk chatResponse
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			t.Fatalf("unmarshal chunk: %v", err)
		}
		if chunk.Object != "chat.completion.chunk" {
			t.Errorf("expected chat.completion.chunk, got %q", chunk.Object)
		}
		tokenChunks++
	}
	if !sawDone {
		t.Error("missing [DONE] terminator")
	}
	if tokenChunks != *numTokens {
		t.Errorf("expected %d token chunks, got %d", *numTokens, tokenChunks)
	}
}

func TestMetricsEndpointFormat(t *testing.T) {
	resetState()

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	w := httptest.NewRecorder()

	handleMetrics(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	body := w.Body.String()
	required := []string{
		"vllm:kv_cache_usage_perc",
		"vllm:num_requests_running",
		"vllm:num_requests_waiting",
		"vllm:time_to_first_token_seconds_sum",
		"vllm:time_to_first_token_seconds_count",
		"vllm:inter_token_latency_seconds_sum",
		"vllm:inter_token_latency_seconds_count",
	}
	for _, m := range required {
		if !strings.Contains(body, m) {
			t.Errorf("metrics output missing %q", m)
		}
	}
}

func TestMetricsReflectRunningCount(t *testing.T) {
	resetState()
	atomic.StoreInt64(&runningReqs, 3)
	atomic.StoreInt64(&waitingReqs, 1)

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	w := httptest.NewRecorder()
	handleMetrics(w, req)

	body := w.Body.String()
	if !strings.Contains(body, "vllm:num_requests_running 3") {
		t.Errorf("expected running=3 in metrics, got:\n%s", body)
	}
	if !strings.Contains(body, "vllm:num_requests_waiting 1") {
		t.Errorf("expected waiting=1 in metrics, got:\n%s", body)
	}
}

func TestInvalidMethod(t *testing.T) {
	resetState()
	req := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	handleChat(w, req)
	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", w.Code)
	}
}

func TestMalformedBody(t *testing.T) {
	resetState()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString("not json"))
	w := httptest.NewRecorder()
	handleChat(w, req)
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}
