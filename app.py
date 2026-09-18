import gradio as gr
import whisper
import warnings
import tempfile
import os

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

modelos_carregados = {}

# Informações sobre os modelos
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
        # Whisper salva os modelos como .pt no cache
        caminho_arquivo = os.path.join(cache_dir, f"{modelo}.pt")
        status = "✅ BAIXADO" if os.path.exists(caminho_arquivo) else "⬇️ Baixar na hora"
        
        texto_exibicao = f"{modelo.upper()} | {info['classificacao']} | {info['peso']} | {status}"
        opcoes.append((texto_exibicao, modelo))
        
    return opcoes

def obter_modelo(tamanho):
    if tamanho not in modelos_carregados:
        print(f"Carregando o modelo Whisper ({tamanho})... Pode demorar um pouco se não estiver baixado.")
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

# ---------------- Tema Minimalista (Estilo PERSUA / Clean Dark Mode) ----------------
tema_persua = gr.themes.Base(
    primary_hue="zinc",
    secondary_hue="zinc",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#0E0E10",
    body_text_color="#EDEDED",
    block_background_fill="#18181B",
    block_border_width="1px",
    block_border_color="#27272A",
    block_radius="8px",
    block_shadow="none",
    button_primary_background_fill="#EDEDED",
    button_primary_text_color="#09090B",
    button_secondary_background_fill="#27272A",
    button_secondary_text_color="#EDEDED",
    input_background_fill="#09090B",
    input_border_color="#27272A",
    slider_color="#EDEDED",
)

css_minimalista = """
/* Remove efeitos e ajusta bordas para um visual mais clean */
.gradio-container .gr-block {
    backdrop-filter: none !important;
}

/* Botões primários com estilo sólido e contraste alto */
.gradio-container button.primary {
    border-radius: 6px !important;
    font-weight: 600 !important;
    border: 1px solid #EDEDED !important;
    box-shadow: none !important;
    transition: background 0.15s ease !important;
}
.gradio-container button.primary:hover {
    background: #D4D4D8 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Botões secundários */
.gradio-container button.secondary {
    border-radius: 6px !important;
    border: 1px solid #3F3F46 !important;
}
.gradio-container button.secondary:hover {
    background: #3F3F46 !important;
}

/* Inputs discretos */
.gradio-container textarea, .gradio-container input {
    border-radius: 6px !important;
    color: #EDEDED !important;
    box-shadow: none !important;
}
.gradio-container textarea:focus, .gradio-container input:focus {
    border-color: #52525B !important;
    box-shadow: none !important;
}

/* Tipografia limpa */
.gradio-container h1, .gradio-container h2 {
    color: #FAFAFA !important;
    font-weight: 700 !important;
    text-shadow: none !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: -0.02em;
}

/* Remove a borda azul padrão do Gradio nos áudios focados */
.gr-audio {
    border: 1px solid #27272A !important;
}
"""
# ---------------------------------------------------------------------------

with gr.Blocks(title="Persua-like Transcritor") as interface:
    gr.Markdown("# Transcritor de Áudio")
    gr.Markdown("Adicione um arquivo de áudio abaixo para gerar a transcrição automática utilizando Whisper.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload de Áudio", interactive=True)
            
            with gr.Row():
                # O Dropdown carrega as opções dinamicamente usando a função
                modelo_dropdown = gr.Dropdown(
                    choices=obter_opcoes_modelos(), 
                    value="base", 
                    label="Modelo de Transcrição",
                    info="Modelos pesados requerem download."
                )
            
            contexto_input = gr.Textbox(
                label="Contexto ou Palavras-chave (Opcional)", 
                placeholder="Ex: fairness, machine learning, viés algorítmico", 
                lines=2
            )
            
            transcrever_btn = gr.Button("Transcrever Áudio", variant="primary")
            
        with gr.Column():
            texto_output = gr.Textbox(label="Texto Transcrito", lines=12, interactive=True)
            
            with gr.Row():
                atualizar_btn = gr.Button("Salvar Edições", variant="secondary")
                arquivo_output = gr.File(label="Baixar Arquivo", interactive=False)

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
    interface.launch(server_name="0.0.0.0", server_port=7860, theme=tema_persua, css=css_minimalista)
