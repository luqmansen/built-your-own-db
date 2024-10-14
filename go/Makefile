
coverage:
	go test -coverprofile=coverage.out ./... ; go tool cover -html=coverage.out


test:
	rm -rf ./testdata/*
	rm -f coverage.out coverage.out.tmp
	go clean -testcache
	go test ./... -v -race -failfast -coverprofile coverage.out -covermode=atomic
	go tool cover -func coverage.out


lint:
	docker run --rm -v $(pwd):/app -w /app golangci/golangci-lint:v1.46.2 golangci-lint run -v
