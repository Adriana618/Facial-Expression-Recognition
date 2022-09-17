from OuluCasia import OuluCasia

# Mapeo de clases
classes = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

# Rutas de las bases de datos
OuluCasia_path = "Datasets/OULUCASIAMIXED"
MUG_path = "Datasets/MUGEXTRACT"

# Parámetros para creación de conjuntos de entrenamiento
training_ratio = 0.7  # que porcentaje se usa para el conjunto de entrenamiento
