.PHONY: setup run clean

VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

setup:
	@echo "Criando ambiente virtual e instalando dependencias..."
	python3 -m venv $(VENV)
	$(PIP) install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install --no-cache-dir -r requirements.txt
	@echo "Configuração concluída! Execute 'make run' para iniciar a aplicação."

run:
	@echo "Iniciando a aplicação..."
	PATH="$$HOME/.local/bin:$$PATH" $(PYTHON) app.py

clean:
	@echo "Limpando o ambiente..."
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
