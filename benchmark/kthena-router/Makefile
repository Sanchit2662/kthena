MOCKBACKEND_PORT ?= 8000
ROUTER_URL       ?= http://localhost:8080
QPS              ?= 10
DURATION         ?= 30s
STREAM           ?= true
MODEL            ?= mock-model

.PHONY: build test run-mockbackend run-loadgen

build:
	cd mockbackend && go build -o ../bin/mockbackend .
	cd loadgen     && go build -o ../bin/loadgen .

test:
	cd mockbackend && go test -v -race ./...
	cd loadgen     && go test -v -race ./...

run-mockbackend:
	./bin/mockbackend -port=$(MOCKBACKEND_PORT) -ttft-ms=50 -tpot-ms=20 -tokens=64

run-loadgen:
	./bin/loadgen \
		-target=$(ROUTER_URL) \
		-model=$(MODEL) \
		-qps=$(QPS) \
		-duration=$(DURATION) \
		-stream=$(STREAM)
