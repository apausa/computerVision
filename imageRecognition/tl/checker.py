import json
from sklearn.metrics import accuracy_score, precision_score, recall_score
import re

# === CONFIGURACIÓN ===
json_path = "prediction.json"  # Cambia por la ruta a tu archivo JSON

# === CARGAR JSON ===
with open(json_path, "r") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]

# === EXTRAER LABELS Y PREDICCIONES ===
true_labels = []
pred_labels = []

for ann in annotations.values():
    image_id = ann["image_id"]
    predicted_class = ann["category_id"]

    # Buscar el entry de esa imagen
    image_entry = next(
        (img for img in images.values() if img["image_id"] == image_id),
        None
    )
    if not image_entry:
        continue

    # Clase real a partir del nombre de carpeta
    match = re.search(r"_test/([^/]+)/", image_entry["filename"])
    if match:
        true_class = match.group(1)
    else:
        continue

    true_labels.append(true_class)
    pred_labels.append(predicted_class)

# === COMPROBAR ===
print(f"Número de muestras: {len(true_labels)}")

# === CÁLCULO DE MÉTRICAS ===
accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
recall = recall_score(true_labels, pred_labels, average="macro", zero_division=0)

print(f"Mean Accuracy:  {accuracy:.4f}")
print(f"Mean Precision: {precision:.4f}")
print(f"Mean Recall:    {recall:.4f}")
