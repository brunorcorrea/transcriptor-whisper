import gradio as gr
import whisper
import warnings
import tempfile
import os

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

modelos_carregados = {}

INFO_MODELOS = {
    "tiny": {"peso": "~75MB", "classificacao": "Muito Leve"},
    "base": {"peso": "~140MB", "classificacao": "Leve"},
    "small": {"peso": "~460MB", "classificacao": "Mediano"},
    "medium": {"peso": "~1.5GB", "classificacao": "Pesado"},
    "turbo": {"peso": "~1.6GB", "classificacao": "Pesado (Rápido)"},
    "large": {"peso": "~2.9GB", "classificacao": "Muito Pesado"}
}

def obter_opcoes_modelos():
    cache_dir = os.path.expanduser("~/.cache/whisper")
    opcoes = []
    
    for modelo, info in INFO_MODELOS.items():
        caminho_arquivo = os.path.join(cache_dir, f"{modelo}.pt")
        # Removido os emojis, utilizando símbolos limpos (unicode)
        status = "✓ PRONTO" if os.path.exists(caminho_arquivo) else "↓ REQUER DOWNLOAD"
        
        texto_exibicao = f"{modelo.upper()} • {info['classificacao']} ({info['peso']}) • {status}"
        opcoes.append((texto_exibicao, modelo))
        
    return opcoes

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

# ---------------- Tema Web Moderno (Estilo App Nativo) ----------------
tema_moderno = gr.themes.Base(
    primary_hue="zinc",
    secondary_hue="zinc",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Poppins"), gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_background_fill="#09090B",
    body_text_color="#FAFAFA",
    block_background_fill="#18181B",
    block_border_width="1px",
    block_border_color="#27272A",
    block_radius="16px",
    block_shadow="0 10px 30px -10px rgba(0,0,0,0.5)",
    button_primary_background_fill="#FAFAFA",
    button_primary_text_color="#09090B",
    button_secondary_background_fill="#27272A",
    button_secondary_text_color="#FAFAFA",
    input_background_fill="#09090B",
    input_border_color="#27272A",
    input_radius="12px",
    slider_color="#FAFAFA",
)

css_moderno = """
/* Títulos e Tipografia */
.gradio-container h1 {
    font-weight: 600 !important;
    letter-spacing: -0.03em !important;
    margin-bottom: 0.2rem !important;
}
.gradio-container h3 {
    font-weight: 500 !important;
    color: #A1A1AA !important;
    margin-bottom: 1rem !important;
    font-size: 1rem !important;
}
.subtitulo {
    color: #71717A;
    font-size: 1.1rem;
    margin-top: 0;
}

/* Campos totalmente arredondados e suaves */
.gradio-container textarea, .gradio-container input {
    border-radius: 12px !important;
    padding: 12px !important;
    border: 1px solid #27272A !important;
    transition: all 0.2s ease !important;
}
.gradio-container textarea:focus, .gradio-container input:focus {
    border-color: #71717A !important;
    box-shadow: 0 0 0 2px rgba(250, 250, 250, 0.1) !important;
}

/* Botões Modernos (Pill shape e hover states) */
.gradio-container button.primary {
    border-radius: 9999px !important; /* Totalmente redondo */
    font-weight: 600 !important;
    padding: 14px 28px !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.gradio-container button.primary:hover {
    background: #D4D4D8 !important;
    transform: scale(0.99) !important;
}
.gradio-container button.secondary {
    border-radius: 9999px !important;
    font-weight: 500 !important;
    border: 1px solid #3F3F46 !important;
}
.gradio-container button.secondary:hover {
    background: #3F3F46 !important;
}

/* Áudio box - cantos arredondados */
.gr-audio {
    border-radius: 16px !important;
    overflow: hidden !important;
}
"""
# ---------------------------------------------------------------------------

with gr.Blocks(title="AudioScribe") as interface:
    
    # Cabeçalho usando HTML inline para incluir SVGs clean no lugar dos Emojis
    gr.HTML("""
    <div style="text-align: center; max-width: 600px; margin: 2rem auto;">
        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="margin: 0 auto 1rem auto; color: #FAFAFA;">
            <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
            <line x1="12" x2="12" y1="19" y2="22"></line>
        </svg>
        <h1 style="font-size: 2.5rem; color: #FAFAFA;">AudioScribe</h1>
        <p class="subtitulo">Transcrição avançada e edição de áudio inteligente</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Parâmetros da Transcrição")
            with gr.Group():
                modelo_dropdown = gr.Dropdown(
                    choices=obter_opcoes_modelos(), 
                    value="base", 
                    label="Modelo de Inteligência Artificial",
                    info="Selecione o motor. Modelos mais pesados são mais precisos."
                )
                
                contexto_input = gr.Textbox(
                    label="Dicionário de Contexto (Opcional)", 
                    placeholder="Ex: fairness, machine learning, trade-off", 
                    lines=2,
                    info="Forneça termos técnicos específicos presentes no áudio."
                )
                
        with gr.Column(scale=1):
            gr.Markdown("### 2. Mídia de Entrada")
            with gr.Group():
                audio_input = gr.Audio(type="filepath", label="Upload de Arquivo (Ou Grave Agora)", interactive=True)
                
    with gr.Row():
        # Botão centralizado de grande destaque
        transcrever_btn = gr.Button("Iniciar Transcrição", variant="primary", size="lg")
        
    gr.Markdown("---")
    
    with gr.Column():
        gr.Markdown("### 3. Resultado e Edição")
        with gr.Group():
            texto_output = gr.Textbox(
                label="Texto Transcrito (Editável)", 
                lines=10, 
                interactive=True,
            )
            
            with gr.Row():
                atualizar_btn = gr.Button("Confirmar Edições (Salvar)", variant="secondary")
                # Substituí o arquivo grande por um botão de download elegante
                download_btn = gr.DownloadButton("Exportar Arquivo .txt", variant="secondary", interactive=True)

    # Conexões de Eventos
    transcrever_btn.click(
        fn=transcrever,
        inputs=[audio_input, contexto_input, modelo_dropdown],
        outputs=[texto_output, download_btn]
    )
    
    atualizar_btn.click(
        fn=preparar_download,
        inputs=texto_output,
        outputs=download_btn
    )

if __name__ == "__main__":
    # Injetar dependências visuais na função launch (Gradio 6 compatibilidade)
    interface.launch(server_name="0.0.0.0", server_port=7860, theme=tema_moderno, css=css_moderno)
