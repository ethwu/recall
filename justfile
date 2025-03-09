
set shell := ['mise', 'exec', '--', 'sh', '-c']

main := 'recall'

run:
	uv run -m {{main}}

