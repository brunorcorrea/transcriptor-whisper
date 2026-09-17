import gradio as gr
import whisper
import warnings
import tempfile

# Ignora avisos inofensivos de FP16 quando executado na CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

print("Carregando o modelo Whisper (Base)...")
model = whisper.load_model("base")

def transcrever(audio_path):
    if not audio_path:
        return "Nenhum áudio fornecido.", None
    try:
        # Transcreve o áudio processado
        result = model.transcribe(audio_path, language="pt")
        texto = result["text"].strip()
        
        # Salva a transcrição inicial em um arquivo .txt temporário para download
        temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt", encoding="utf-8")
        temp_file.write(texto)
        temp_file.close()
        
        return texto, temp_file.name
    except Exception as e:
        return f"Erro ao processar áudio: {str(e)}", None

def preparar_download(texto_editado):
    """Atualiza o arquivo de download com o texto que o usuário editou na tela."""
    if not texto_editado:
        return None
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt", encoding="utf-8")
    temp_file.write(texto_editado)
    temp_file.close()
    return temp_file.name

# Construção da Interface com Gradio Blocks (permite um layout mais customizado)
with gr.Blocks(title="Transcritor Inteligente", theme=gr.themes.Default()) as interface:
    gr.Markdown("# 🎙️ Transcritor de Áudio (OpenAI Whisper)")
    gr.Markdown("""
    **Instruções:**
    1. Arraste e solte o seu arquivo de áudio ou clique para enviar.
    2. *Opcional:* Você pode ouvir e **cortar o áudio** usando as ferramentas da própria caixa de áudio antes de transcrever (caso queira transcrever só um pedaço).
    3. Clique em **"Transcrever Áudio"**.
    4. O texto aparecerá ao lado. Você pode **editar o texto livremente** na caixa.
    5. Se fizer edições no texto e quiser baixar, clique em **"Confirmar Edições"** para gerar um novo arquivo para download.
    """)
    
    with gr.Row():
        with gr.Column():
            # A propriedade interactive=True já habilita o editor de áudio (cortar) nativo do Gradio
            audio_input = gr.Audio(type="filepath", label="Áudio (Faça Drag & Drop ou grave)", interactive=True)
            transcrever_btn = gr.Button("Transcrever Áudio", variant="primary")
            
        with gr.Column():
            texto_output = gr.Textbox(label="Transcrição (Você pode editar este texto!)", lines=12, interactive=True)
            
            with gr.Row():
                atualizar_btn = gr.Button("💾 Confirmar Edições de Texto", variant="secondary")
                arquivo_output = gr.File(label="Arquivo para Download (.txt)", interactive=False)

    # Evento de clique para transcrever o áudio
    transcrever_btn.click(
        fn=transcrever,
        inputs=audio_input,
        outputs=[texto_output, arquivo_output]
    )
    
    # Evento de clique para atualizar o arquivo de download com possíveis edições feitas no texto
    atualizar_btn.click(
        fn=preparar_download,
        inputs=texto_output,
        outputs=arquivo_output
    )

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
