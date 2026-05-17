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

// loadgen is an open-loop Poisson load generator for kthena-router.
//
// Open-loop means arrival times are drawn from an exponential distribution
// independently of whether prior requests have completed. This avoids the
// coordinated-omission bias present in closed-loop tools (e.g. wrk, hey)
// where a slow backend causes the generator to slow down, hiding tail latency.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

var (
	target   = flag.String("target", "http://localhost:8080", "kthena-router base URL")
	model    = flag.String("model", "mock-model", "model name sent in each request")
	qps      = flag.Float64("qps", 10, "target requests per second (Poisson mean)")
	duration = flag.Duration("duration", 30*time.Second, "load generation duration")
	stream   = flag.Bool("stream", true, "use streaming (measures TTFT); false measures E2E only")
	prompt   = flag.String("prompt", "Tell me a short story.", "prompt text sent in each request")
)

type result struct {
	err      error
	e2e      time.Duration
	ttft     time.Duration // zero for non-streaming or on error
	arrived  time.Time     // when the request was fired (for CO accounting)
}

// run fires requests at the target for the configured duration using
// exponential inter-arrival times (Poisson process at rate=qps).
// Each request is fired in its own goroutine; the caller collects results.
func run(client *http.Client, targetURL, modelName, promptText string, targetQPS float64, dur time.Duration, useStream bool) []result {
	deadline := time.Now().Add(dur)
	results := make([]result, 0, int(targetQPS*dur.Seconds())+16)
	var mu sync.Mutex
	var wg sync.WaitGroup

	for {
		now := time.Now()
		if now.After(deadline) {
			break
		}
		// exponential inter-arrival: mean = 1/qps seconds
		gap := time.Duration(rand.ExpFloat64() / targetQPS * float64(time.Second))
		fireAt := now.Add(gap)
		if fireAt.After(deadline) {
			break
		}
		time.Sleep(time.Until(fireAt))

		wg.Add(1)
		arrived := time.Now()
		go func(arrivedAt time.Time) {
			defer wg.Done()
			r := fire(client, targetURL, modelName, promptText, useStream)
			r.arrived = arrivedAt
			mu.Lock()
			results = append(results, r)
			mu.Unlock()
		}(arrived)
	}
	wg.Wait()
	return results
}

// fire sends a single chat completion request and records timing.
func fire(client *http.Client, targetURL, modelName, promptText string, useStream bool) result {
	payload := map[string]interface{}{
		"model":  modelName,
		"stream": useStream,
		"messages": []map[string]string{
			{"role": "user", "content": promptText},
		},
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return result{err: err}
	}

	req, err := http.NewRequest(http.MethodPost, targetURL+"/v1/chat/completions", bytes.NewReader(b))
	if err != nil {
		return result{err: err}
	}
	req.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return result{err: err, e2e: time.Since(start)}
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return result{err: fmt.Errorf("HTTP %d", resp.StatusCode), e2e: time.Since(start)}
	}

	if !useStream {
		// drain body so the connection is reusable
		buf := make([]byte, 4096)
		for {
			_, err := resp.Body.Read(buf)
			if err != nil {
				break
			}
		}
		return result{e2e: time.Since(start)}
	}

	// streaming: scan SSE lines, record TTFT on first data chunk
	var ttft time.Duration
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			break
		}
		if ttft == 0 {
			ttft = time.Since(start)
		}
	}
	return result{e2e: time.Since(start), ttft: ttft}
}

// percentile returns the p-th percentile (0–100) of a sorted slice.
func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := p / 100 * float64(len(sorted)-1)
	lo := int(math.Floor(idx))
	hi := int(math.Ceil(idx))
	if lo == hi {
		return sorted[lo]
	}
	frac := idx - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}

func printReport(results []result, dur time.Duration) {
	var errors, successes int
	e2eMs := make([]float64, 0, len(results))
	ttftMs := make([]float64, 0, len(results))

	for _, r := range results {
		if r.err != nil {
			errors++
			continue
		}
		successes++
		e2eMs = append(e2eMs, float64(r.e2e.Milliseconds()))
		if r.ttft > 0 {
			ttftMs = append(ttftMs, float64(r.ttft.Milliseconds()))
		}
	}

	sort.Float64s(e2eMs)
	sort.Float64s(ttftMs)

	actualQPS := float64(len(results)) / dur.Seconds()
	fmt.Printf("\n=== kthena-router load test results ===\n")
	fmt.Printf("duration:      %s\n", dur)
	fmt.Printf("total:         %d  success: %d  errors: %d\n", len(results), successes, errors)
	fmt.Printf("actual QPS:    %.2f\n", actualQPS)
	fmt.Println()
	if len(e2eMs) > 0 {
		fmt.Printf("E2E latency (ms)\n")
		fmt.Printf("  p50:  %.1f\n", percentile(e2eMs, 50))
		fmt.Printf("  p95:  %.1f\n", percentile(e2eMs, 95))
		fmt.Printf("  p99:  %.1f\n", percentile(e2eMs, 99))
		fmt.Printf("  max:  %.1f\n", e2eMs[len(e2eMs)-1])
	}
	if len(ttftMs) > 0 {
		fmt.Println()
		fmt.Printf("TTFT (ms)  — streaming only\n")
		fmt.Printf("  p50:  %.1f\n", percentile(ttftMs, 50))
		fmt.Printf("  p95:  %.1f\n", percentile(ttftMs, 95))
		fmt.Printf("  p99:  %.1f\n", percentile(ttftMs, 99))
		fmt.Printf("  max:  %.1f\n", ttftMs[len(ttftMs)-1])
	}
}

func main() {
	flag.Parse()

	client := &http.Client{
		Transport: &http.Transport{
			MaxIdleConnsPerHost: 256,
		},
	}

	log.Printf("load test: target=%s model=%s qps=%.1f duration=%s stream=%v",
		*target, *model, *qps, *duration, *stream)

	results := run(client, *target, *model, *prompt, *qps, *duration, *stream)
	printReport(results, *duration)
}
