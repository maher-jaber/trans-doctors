from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
import os

app = FastAPI()
# ✅ Autorise les requêtes depuis le front (localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tu peux remplacer "*" par ["http://localhost"] pour plus de sécurité
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Charger le modèle Whisper local une seule fois au démarrage
# Choix possibles : "tiny", "base", "small", "medium", "large"
# → Plus gros = plus précis mais plus lent.
model = whisper.load_model("large")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Sauvegarde temporaire du fichier audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Transcription locale
    result = model.transcribe(tmp_path, language="fr")

    # Nettoyage du fichier temporaire
    os.remove(tmp_path)

    return JSONResponse(content={"text": result["text"]})
