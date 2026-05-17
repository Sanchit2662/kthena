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
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestPercentile(t *testing.T) {
	cases := []struct {
		sorted []float64
		p      float64
		want   float64
	}{
		{[]float64{1, 2, 3, 4, 5}, 50, 3},
		{[]float64{1, 2, 3, 4, 5}, 0, 1},
		{[]float64{1, 2, 3, 4, 5}, 100, 5},
		{[]float64{10}, 99, 10},
		{nil, 50, 0},
	}
	for _, tc := range cases {
		got := percentile(tc.sorted, tc.p)
		if math.Abs(got-tc.want) > 0.01 {
			t.Errorf("percentile(%v, %.0f) = %.2f, want %.2f", tc.sorted, tc.p, got, tc.want)
		}
	}
}

func TestFireNonStreaming(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	r := fire(srv.Client(), srv.URL, "m", "hi", false)
	if r.err != nil {
		t.Fatalf("unexpected error: %v", r.err)
	}
	if r.e2e <= 0 {
		t.Error("expected positive e2e latency")
	}
	if r.ttft != 0 {
		t.Error("expected zero ttft for non-streaming request")
	}
}

func TestFireStreaming(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		chunks := []string{
			`{"id":"1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hel"}}]}`,
			`{"id":"1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}`,
		}
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
			flusher.Flush()
		}
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	r := fire(srv.Client(), srv.URL, "m", "hi", true)
	if r.err != nil {
		t.Fatalf("unexpected error: %v", r.err)
	}
	if r.ttft <= 0 {
		t.Error("expected positive ttft for streaming request")
	}
	if r.e2e < r.ttft {
		t.Error("e2e should be >= ttft")
	}
}

func TestFireHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "overloaded", http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	r := fire(srv.Client(), srv.URL, "m", "hi", false)
	if r.err == nil {
		t.Error("expected error on non-2xx response")
	}
	if !strings.Contains(r.err.Error(), "503") {
		t.Errorf("expected 503 in error, got %v", r.err)
	}
}

func TestRunOpenLoop(t *testing.T) {
	var count int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&count, 1)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	// 10 QPS for 500ms => expect roughly 5 requests (allow wide tolerance)
	results := run(srv.Client(), srv.URL, "m", "hi", 10, 500*time.Millisecond, false)
	if len(results) == 0 {
		t.Fatal("expected at least one result")
	}
	for _, r := range results {
		if r.err != nil {
			t.Errorf("unexpected error: %v", r.err)
		}
	}
}

func TestRunReturnsArrivedTimestamp(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"1","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	before := time.Now()
	results := run(srv.Client(), srv.URL, "m", "hi", 5, 300*time.Millisecond, false)
	after := time.Now()

	for i, r := range results {
		if r.arrived.Before(before) || r.arrived.After(after) {
			t.Errorf("result[%d].arrived %v out of [%v, %v]", i, r.arrived, before, after)
		}
	}
}
