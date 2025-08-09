
init:
	@docker compose up -d

app:
	@docker compose up app

build:
	@docker compose build app

logs:
	@docker compose logs -f

stop:
	@docker compose down

list-models:
	@docker exec ollama_service ollama list

pull-models:
	@docker exec ollama_service ollama pull llama3
	@docker exec ollama_service ollama pull mxbai-embed-large

restart:
	@make stop
	@make init

seed:
	docker compose run app python seed.py

drop-db:
	@docker compose down db -v
