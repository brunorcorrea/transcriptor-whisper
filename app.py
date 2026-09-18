import gradio as gr
import whisper
import warnings
import tempfile

# Ignora avisos inofensivos de FP16 quando executado na CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Cache para não carregar o mesmo modelo duas vezes
modelos_carregados = {}

def obter_modelo(tamanho):
    if tamanho not in modelos_carregados:
        print(f"Carregando o modelo Whisper ({tamanho})... Pode demorar um pouco na primeira vez.")
        modelos_carregados[tamanho] = whisper.load_model(tamanho)
    return modelos_carregados[tamanho]

# Pré-carrega o modelo base para a interface abrir rápido
obter_modelo("base")

def transcrever(audio_path, contexto, tamanho_modelo):
    if not audio_path:
        return "Nenhum áudio fornecido.", None
    try:
        modelo = obter_modelo(tamanho_modelo)
        
        # Parâmetros da transcrição
        opcoes = {"language": "pt"}
        
        # O prompt inicial "ensina" palavras específicas à IA
        if contexto and contexto.strip() != "":
            opcoes["initial_prompt"] = contexto.strip()
            
        result = modelo.transcribe(audio_path, **opcoes)
        texto = result["text"].strip()
        
        # Salva a transcrição em um arquivo temporário
        temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt", encoding="utf-8")
        temp_file.write(texto)
        temp_file.close()
        
        return texto, temp_file.name
    except Exception as e:
        return f"Erro ao processar áudio: {str(e)}", None

def preparar_download(texto_editado):
    if not texto_editado:
        return None
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt", encoding="utf-8")
    temp_file.write(texto_editado)
    temp_file.close()
    return temp_file.name

with gr.Blocks(title="Transcritor Inteligente", theme=gr.themes.Default()) as interface:
    gr.Markdown("# 🎙️ Transcritor de Áudio Avançado (OpenAI Whisper)")
    gr.Markdown("""
    **Dicas para melhorar a precisão:**
    - **Palavras-chave (Contexto):** Se o áudio possui termos técnicos em inglês (ex: *fairness, machine learning, dataset*), digite-os na caixa de contexto. Isso guia a IA a não "alucinar" palavras parecidas em português.
    - **Modelo maior:** O modelo 'base' é muito rápido, mas o modelo 'small' é bem mais inteligente (vai baixar ~240MB na primeira vez que for usado).
    """)
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Áudio (Drag & Drop ou gravar)", interactive=True)
            
            with gr.Row():
                modelo_dropdown = gr.Dropdown(choices=["base", "small"], value="base", label="Tamanho da IA (Precisão vs Velocidade)")
            
            contexto_input = gr.Textbox(
                label="Palavras-chave / Contexto (Opcional)", 
                placeholder="Ex: fairness, machine learning, dataset, viés, trade-off", 
                lines=2
            )
            
            transcrever_btn = gr.Button("Transcrever Áudio", variant="primary")
            
        with gr.Column():
            texto_output = gr.Textbox(label="Transcrição (Você pode editar este texto!)", lines=12, interactive=True)
            
            with gr.Row():
                atualizar_btn = gr.Button("💾 Confirmar Edições de Texto", variant="secondary")
                arquivo_output = gr.File(label="Arquivo para Download (.txt)", interactive=False)

    transcrever_btn.click(
        fn=transcrever,
        inputs=[audio_input, contexto_input, modelo_dropdown],
        outputs=[texto_output, arquivo_output]
    )
    
    atualizar_btn.click(
        fn=preparar_download,
        inputs=texto_output,
        outputs=arquivo_output
    )

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
