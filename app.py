import gradio as gr
import whisper
import warnings

# Ignora avisos do FP16 na CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

print("Carregando o modelo Whisper (Base)...")
model = whisper.load_model("base")

def transcrever(audio_path):
    if not audio_path:
        return "Nenhum áudio fornecido."
    try:
        result = model.transcribe(audio_path, language="pt")
        return result["text"].strip()
    except Exception as e:
        return f"Erro ao processar áudio: {str(e)}"

interface = gr.Interface(
    fn=transcrever,
    inputs=gr.Audio(type="filepath", label="Envie seu áudio (Drag & Drop ou Clique)"),
    outputs=gr.Textbox(label="Transcrição Gerada", lines=10),
    title="Transcritor de Áudio Automático",
    description="Arraste seus arquivos de áudio (do WhatsApp, gravador, etc.) aqui para gerar o texto automaticamente usando Inteligência Artificial (OpenAI Whisper).",
    theme="default"
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
