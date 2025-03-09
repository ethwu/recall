
set shell := ['mise', 'exec', '--', 'sh', '-c']

main := 'recall'
sources := './content/'

run:
	ollama serve > /dev/null 2>&1 & uv run -m {{main}} {{sources}}

build:
	mise run build

clean:
	rm -rf *.{chroma,log,pickle}
alias c := clean

