# Transcritor de Áudio (Whisper)

Uma interface amigável para transcrição automática de áudios locais utilizando o modelo [Whisper](https://github.com/openai/whisper) da OpenAI e o [Gradio](https://gradio.app/). 

Com esta aplicação, você pode transcrever facilmente áudios do WhatsApp (.ogg, .mp3) ou de qualquer gravador através de uma interface de **Drag & Drop** direto no seu navegador.

## 🚀 Pré-requisitos
- **Python 3.10+** instalado no sistema.
- **FFmpeg** instalado e disponível no seu `PATH` (ex: em `~/.local/bin/ffmpeg` ou no sistema através do apt/brew).

## 📦 Como Instalar

Este projeto utiliza um `Makefile` para automatizar a criação do ambiente virtual e a instalação correta de bibliotecas de Inteligência Artificial usando apenas a CPU (para economizar espaço e funcionar em qualquer máquina).

No terminal, acesse a pasta do projeto e execute:
```bash
make setup
```

## 🎮 Como Usar

Para iniciar a interface web, digite o seguinte comando:
```bash
make run
```
Após executar, o terminal mostrará um endereço local (ex: `http://localhost:7860`). 
1. Acesse o endereço pelo seu navegador.
2. Arraste e solte (Drag & Drop) seus arquivos de áudio para dentro do painel indicado.
3. Aguarde o fim do processamento e copie o texto gerado!

## 🧹 Limpeza

Caso deseje apagar o ambiente virtual e os arquivos de cache para liberar espaço:
```bash
make clean
```
