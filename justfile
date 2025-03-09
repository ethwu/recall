
set shell := ['mise', 'exec', '--', 'sh', '-c']

main := 'recall'

run:
	ollama serve > /dev/null 2>&1 & uv run -m {{main}}

