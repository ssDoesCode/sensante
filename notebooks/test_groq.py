# notebooks/test_groq.py
# Test de l'API Groq avec Llama 3

import os
from dotenv import load_dotenv
from groq import Groq

# Charger la cle depuis .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERREUR : GROQ_API_KEY non trouvee dans .env")
    exit()

# Creer le client Groq
client = Groq(api_key=api_key)

# --- Test 1 : question simple ---
print("=== Test 1 : question simple ===")
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": (
                "Tu es un assistant medical senegalais. "
                "Reponds en francais simple. "
                "Maximum 3 phrases."
            )
        },
        {
            "role": "user",
            "content": "Quels sont les symptomes du paludisme ?"
        }
    ],
    max_tokens=200,
    temperature=0.3
)

print(response.choices[0].message.content)
print(f"\nTokens utilises : {response.usage.total_tokens}")

# --- Test 2 : format SenSante ---
print("\n=== Test 2 : explication SenSante ===")
response2 = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "system",
            "content": (
                "Tu es un assistant medical senegalais. "
                "Tu recois un diagnostic et des donnees patient. "
                "Explique le resultat en francais simple, "
                "comme un medecin parlerait a son patient. "
                "Sois rassurant mais recommande une consultation. "
                "Maximum 3 phrases. "
                "Ne fais JAMAIS de diagnostic toi-meme."
            )
        },
        {
            "role": "user",
            "content": (
                "Patient : Femme, 28 ans, region Dakar\n"
                "Symptomes : temperature 39.5, toux, fatigue, maux de tete\n"
                "Diagnostic du modele : paludisme (probabilite 72%)\n"
                "Explique ce resultat au patient."
            )
        }
    ],
    max_tokens=200,
    temperature=0.3
)

print(response2.choices[0].message.content)
# --- Exercice 2 : tester differentes temperatures ---
print("\n=== Exercice 2 : impact de la temperature ===")

user_test = (
    "Patient : F, 28 ans, region Dakar\n"
    "Temperature : 39.5 C\n"
    "Diagnostic du modele : paludisme (probabilite 72%)\n"
    "Explique ce resultat au patient."
)

system_test = (
    "Tu es un assistant medical senegalais. "
    "Explique le diagnostic en francais simple. "
    "Maximum 3 phrases. "
    "Ne fais JAMAIS de diagnostic toi-meme."
)

for temp in [0.0, 0.5, 1.0]:
    r = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_test},
            {"role": "user", "content": user_test}
        ],
        max_tokens=200,
        temperature=temp
    )
    print(f"\n--- temperature={temp} ---")
    print(r.choices[0].message.content)