"""
Exercice 2: Tester le modele avec 3 patients differents
Test the trained model with 3 new patients with different profiles
"""

import pandas as pd
import numpy as np
import joblib

# Charger le modele entrainé et les encodeurs
print("=" * 60)
print("Chargement du modèle et des encodeurs...")
print("=" * 60)

model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

print(f"✓ Modèle chargé avec succès")
print(f"  Classes: {list(model.classes_)}")
print(f"  Features: {feature_cols}")

# Créer 3 patients de test avec des profils différents
print("\n" + "=" * 60)
print("CRÉATION DE 3 PATIENTS DE TEST")
print("=" * 60)

# Patient 1: Jeune sans symptômes
patient1 = {
    'nom': 'Patient 1: Jeune sans symptômes',
    'age': 22,
    'sexe': 'M',
    'temperature': 37.0,  # Normale
    'tension_sys': 11,
    'toux': 0,            # Pas de toux
    'fatigue': 0,         # Pas de fatigue
    'maux_tete': 0,       # Pas de maux de tête
    'region': 'Dakar'
}

# Patient 2: Adulte avec forte fièvre
patient2 = {
    'nom': 'Patient 2: Adulte avec forte fièvre',
    'age': 45,
    'sexe': 'F',
    'temperature': 40.2,  # Fièvre élevée
    'tension_sys': 12,
    'toux': 0,
    'fatigue': 1,         # Fatigue
    'maux_tete': 1,       # Maux de tête
    'region': 'Thiès'
}

# Patient 3: Patient âgé avec toux
patient3 = {
    'nom': 'Patient 3: Patient âgé avec toux',
    'age': 72,
    'sexe': 'M',
    'temperature': 38.9,
    'tension_sys': 13,
    'toux': 1,            # Toux présente
    'fatigue': 1,
    'maux_tete': 0,
    'region': 'Fatick'
}

patients = [patient1, patient2, patient3]

# Afficher les profils des patients
for i, patient in enumerate(patients, 1):
    print(f"\nPatient {i}: {patient['nom']}")
    print(f"  Âge: {patient['age']} ans")
    print(f"  Sexe: {patient['sexe']}")
    print(f"  Température: {patient['temperature']}°C")
    print(f"  Tension systolique: {patient['tension_sys']}")
    print(f"  Toux: {'Oui' if patient['toux'] else 'Non'}")
    print(f"  Fatigue: {'Oui' if patient['fatigue'] else 'Non'}")
    print(f"  Maux de tête: {'Oui' if patient['maux_tete'] else 'Non'}")
    print(f"  Région: {patient['region']}")

# Préparer les données pour la prédiction
print("\n" + "=" * 60)
print("PRÉPARATION DES DONNÉES")
print("=" * 60)

test_data = []

for patient in patients:
    # Encoder les variables catégoriques
    sexe_encoded = le_sexe.transform([patient['sexe']])[0]
    region_encoded = le_region.transform([patient['region']])[0]
    
    # Créer la ligne de données avec les features dans le bon ordre
    row = {
        'age': patient['age'],
        'sexe_encoded': sexe_encoded,
        'temperature': patient['temperature'],
        'tension_sys': patient['tension_sys'],
        'toux': patient['toux'],
        'fatigue': patient['fatigue'],
        'maux_tete': patient['maux_tete'],
        'region_encoded': region_encoded
    }
    test_data.append(row)

# Créer un DataFrame avec les données de test
X_test = pd.DataFrame(test_data)
print(f"\nDonnées encodées:")
print(X_test)

# Faire les prédictions
print("\n" + "=" * 60)
print("PRÉDICTIONS")
print("=" * 60)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Afficher les résultats
results = pd.DataFrame({
    'Patient': [f"Patient {i+1}: {patients[i]['nom'].split(': ')[1]}" for i in range(len(patients))],
    'Diagnostic Prédit': predictions,
    'Confiance (%)': [f"{max(prob)*100:.1f}%" for prob in probabilities]
})

print("\nRÉSULTATS DES PRÉDICTIONS:")
print(results.to_string(index=False))

# Afficher les probabilités pour chaque classe
print("\n" + "=" * 60)
print("DÉTAILS DES PROBABILITÉS PAR DIAGNOSTIC")
print("=" * 60)

for i, patient in enumerate(patients):
    print(f"\n{patient['nom']}:")
    print(f"  Diagnostic prédit: {predictions[i]}")
    
    for j, diagnostic in enumerate(model.classes_):
        confidence = probabilities[i][j] * 100
        bar = "█" * int(confidence / 5) + "░" * (20 - int(confidence / 5))
        print(f"  {diagnostic:12} {bar} {confidence:5.1f}%")

# Analyse de cohérence
print("\n" + "=" * 60)
print("ANALYSE DE COHÉRENCE DES RÉSULTATS")
print("=" * 60)

analyses = []

# Analyse du Patient 1 (jeune sans symptômes)
print(f"\nPatient 1 - {patient1['nom']}:")
print(f"  Prédiction: {predictions[0]}")
expected_p1 = "sain (ou symptômes mineurs)"
print(f"  Attendu: {expected_p1}")
if predictions[0] == "sain":
    print(f"  ✓ COHÉRENT: Un jeune sans symptômes devrait être en bonne santé")
else:
    print(f"  ⚠ À VÉRIFIER: Résultat différent de l'attendu")

# Analyse du Patient 2 (adulte avec forte fièvre)
print(f"\nPatient 2 - {patient2['nom']}:")
print(f"  Prédiction: {predictions[1]}")
expected_p2 = "paludisme, typhoïde ou grippe (forte fièvre + symptômes)"
print(f"  Attendu: {expected_p2}")
fever_related = ["paludisme", "typhoide", "grippe"]
if any(predictions[1].lower().startswith(d.lower()) for d in fever_related):
    print(f"  ✓ COHÉRENT: La forte fièvre + fatigue + maux de tête sont des symptômes classiques")
else:
    print(f"  ⚠ À VÉRIFIER: Résultat différent de l'attendu")

# Analyse du Patient 3 (patient âgé avec toux)
print(f"\nPatient 3 - {patient3['nom']}:")
print(f"  Prédiction: {predictions[2]}")
expected_p3 = "grippe ou paludisme (toux + âge avancé)"
print(f"  Attendu: {expected_p3}")
respiratory = ["grippe", "paludisme"]
if any(predictions[2].lower().startswith(d.lower()) for d in respiratory):
    print(f"  ✓ COHÉRENT: La toux chez un patient âgé avec fièvre suggère une maladie infectieuse")
else:
    print(f"  ⚠ À VÉRIFIER: Résultat différent de l'attendu")

# Résumé statistique
print("\n" + "=" * 60)
print("RÉSUMÉ STATISTIQUE")
print("=" * 60)

print(f"\nNombre de prédictions: {len(predictions)}")
print(f"Diagnostics prédits uniques: {len(set(predictions))}")
print(f"Distribution des prédictions:")
for diagnostic in model.classes_:
    count = sum(predictions == diagnostic)
    if count > 0:
        print(f"  {diagnostic}: {count}")

# Confiance moyenne
avg_confidence = np.mean(np.max(probabilities, axis=1)) * 100
print(f"\nConfiance moyenne du modèle: {avg_confidence:.1f}%")

if avg_confidence > 80:
    print("✓ Le modèle a une confiance ÉLEVÉE dans ses prédictions")
elif avg_confidence > 60:
    print("⚠ Le modèle a une confiance MODÉRÉE dans ses prédictions")
else:
    print("⚠ Le modèle a une confiance FAIBLE - à prendre avec prudence")

print("\n" + "=" * 60)
