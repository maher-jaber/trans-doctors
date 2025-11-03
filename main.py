from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
import os

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou ["http://localhost"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Charger le modèle Whisper (large pour meilleure précision médicale)
model = whisper.load_model("large")

# ✅ Charger le dictionnaire médical
medical_terms = {}
with open("dictionnaire_medical.txt", "r", encoding="utf-8") as f:
    for line in f:
        term = line.strip()
        if term:
            medical_terms[term.lower()] = term  # clé en minuscule, valeur originale


def post_process_with_medical_dict(text: str) -> str:
    """
    Parcourt la transcription et corrige uniquement les mots
    présents dans le dictionnaire médical (sans correcteur orthographique).
    """
    words = text.split()
    corrected_words = []

    for word in words:
        clean_word = word.strip(",.;:!?").lower()
        if clean_word in medical_terms:
            # Remplace par la version correcte (ex : majuscules ou accents)
            corrected_words.append(medical_terms[clean_word])
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Sauvegarde temporaire du fichier audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Transcription Whisper locale
    result = model.transcribe(tmp_path, language="fr")
    transcription = result["text"]

    # Suppression du fichier temporaire
    os.remove(tmp_path)

    # ✅ Post-traitement dictionnaire médical
    final_text = post_process_with_medical_dict(transcription)

    return JSONResponse(content={"text": final_text})
