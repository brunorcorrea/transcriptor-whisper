import gradio as gr
import whisper
import warnings
import tempfile

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

modelos_carregados = {}

def obter_modelo(tamanho):
    if tamanho not in modelos_carregados:
        print(f"Carregando o modelo Whisper ({tamanho})...")
        modelos_carregados[tamanho] = whisper.load_model(tamanho)
    return modelos_carregados[tamanho]

obter_modelo("base")

def transcrever(audio_path, contexto, tamanho_modelo):
    if not audio_path:
        return "Nenhum áudio fornecido.", None
    try:
        modelo = obter_modelo(tamanho_modelo)
        opcoes = {"language": "pt"}
        
        if contexto and contexto.strip() != "":
            opcoes["initial_prompt"] = contexto.strip()
            
        result = modelo.transcribe(audio_path, **opcoes)
        texto = result["text"].strip()
        
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

# ---------------- Tema Futurista (Liquid Glass / Cyberpunk) ----------------
tema_cyberglass = gr.themes.Base(
    primary_hue="cyan",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="linear-gradient(135deg, #0f172a 0%, #020617 100%)",
    body_text_color="#e2e8f0",
    block_background_fill="rgba(30, 41, 59, 0.4)",
    block_border_width="1px",
    block_border_color="rgba(255, 255, 255, 0.1)",
    block_radius="16px",
    block_shadow="0 4px 30px rgba(0, 0, 0, 0.5)",
    button_primary_background_fill="linear-gradient(90deg, #06b6d4 0%, #3b82f6 100%)",
    button_primary_text_color="white",
    slider_color="#06b6d4",
)

css_personalizado = """
/* Glassmorphism e desfoque para os painéis */
.gradio-container .gr-block {
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
}

/* Efeito de neon pulsante nos botões principais */
.gradio-container button.primary {
    box-shadow: 0 0 15px rgba(6, 182, 212, 0.4) !important;
    transition: all 0.3s ease !important;
    border: none !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
}
.gradio-container button.primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 25px rgba(6, 182, 212, 0.8) !important;
}

/* Inputs focados com brilho cyber */
.gradio-container textarea:focus, .gradio-container input:focus {
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.4) !important;
    border-color: #06b6d4 !important;
}


/* Cores dos inputs */
.gradio-container textarea, .gradio-container input {
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(6, 182, 212, 0.3) !important;
    color: #38bdf8 !important;
}
/* Tipografia de destaque para títulos */
.gradio-container h1, .gradio-container h2 {
    text-shadow: 0 0 10px rgba(56, 189, 248, 0.6) !important;
    color: #e0f2fe !important;
    font-family: 'Courier New', Courier, monospace;
}

/* Caixas de texto monospaced para a transcrição parecer código/terminal */
.gradio-container textarea {
    font-family: 'Consolas', 'Courier New', monospace !important;
}
"""
# ---------------------------------------------------------------------------

with gr.Blocks(title="Transcritor Inteligente") as interface:
    gr.Markdown("# 🎙️ SYS.TRANSCRIBE // WHISPER_AI")
    gr.Markdown("""
    **[ STATUS: ONLINE ]**  
    Interface neural de processamento de áudio ativada. Envie dados sonoros para extração de texto estruturado.
    
    *Dicas de calibração:* Insira parâmetros de **Contexto** para orientar a rede neural sobre jargões técnicos. Faça o upgrade de nó para o modelo **'small'** se desejar precisão máxima (download de 240MB no primeiro boot).
    """)
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Input de Áudio (Drag & Drop | Rec)", interactive=True)
            
            with gr.Row():
                modelo_dropdown = gr.Dropdown(choices=["base", "small"], value="base", label="Tamanho da Rede Neural")
            
            contexto_input = gr.Textbox(
                label="Parâmetros de Contexto / Jargões", 
                placeholder="Ex: fairness, dataset, machine learning, trade-off", 
                lines=2
            )
            
            transcrever_btn = gr.Button("INICIAR EXTRAÇÃO", variant="primary")
            
        with gr.Column():
            texto_output = gr.Textbox(label="Saída de Dados (Console de Edição)", lines=12, interactive=True)
            
            with gr.Row():
                atualizar_btn = gr.Button("💾 COMPILAR ALTERAÇÕES", variant="secondary")
                arquivo_output = gr.File(label="Arquivo de Exportação (.txt)", interactive=False)

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
    interface.launch(server_name="0.0.0.0", server_port=7860, theme=tema_cyberglass, css=css_personalizado)
