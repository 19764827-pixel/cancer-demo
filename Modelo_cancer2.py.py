import torch
from torchvision import transforms, models
from PIL import Image
import os

# ==========================================
# CONFIGURACIÓN
# ==========================================
MODEL_PATH = r"C:\bootcamp-ia\scr\modelo_cancer_mobilenet.pth"
IMAGE_PATH = r"C:\bootcamp-ia\scr\test_img\m02.png"  # Puede ser carpeta también

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASES = ["Benigno", "Maligno"]  # 0 y 1

# Transformaciones idénticas al entrenamiento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==========================================
# LISTA DE HOSPITALES
# ==========================================
HOSPITALES = [
    {
        "nombre": "Instituto del Cáncer de El Salvador (ICES) - Dr. Narciso Díaz Bazán (Liga contra el Cáncer)",
        "servicios": "Detección, diagnóstico y tratamiento (incluye Radioterapia). Atiende referencias de hospitales nacionales y público en general.",
        "ubicacion": "1a Calle Poniente y 33 Avenida Norte, Colonia Escalón, San Salvador"
    },
    {
        "nombre": "Hospital Nacional Rosales",
        "servicios": "Cirugía Oncológica y referencia para Quimioterapia/Radioterapia.",
        "ubicacion": "San Salvador"
    },
    {
        "nombre": "Hospital de la Mujer",
        "servicios": "Detección y referencia para el manejo del cáncer de mama en mujeres.",
        "ubicacion": "San Salvador"
    },
    {
        "nombre": "Hospital Oncológico del ISSS",
        "servicios": "Ofrece tratamiento integral (Diagnóstico, Cirugía, Quimio y Radioterapia) a pacientes asegurados.",
        "ubicacion": "San Salvador (parte de la red del ISSS)"
    },
    {
        "nombre": "Hospitales Nacionales Regionales",
        "servicios": "Detección, diagnóstico inicial (mamografías, biopsias) y procedimientos quirúrgicos primarios, con posterior referencia a los centros especializados para terapias complementarias (Quimio/Radio).",
        "ubicacion": "Santa Ana, San Miguel, etc."
    }
]

# ==========================================
# FUNCIONES
# ==========================================
def cargar_modelo(path_modelo):
    """Cargar el modelo entrenado con MobileNetV2."""
    if not os.path.exists(path_modelo):
        raise FileNotFoundError(f"No se encontró el modelo en {path_modelo}")

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(1280, len(CLASES))

    state_dict = torch.load(path_modelo, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def cargar_imagen(ruta):
    """Abrir imagen y aplicar transformaciones."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró la imagen en {ruta}")
    img = Image.open(ruta).convert("RGB")
    return transform(img).unsqueeze(0).to(DEVICE)

def predecir(modelo, img_tensor):
    """Realizar predicción y devolver clase + probabilidad."""
    with torch.no_grad():
        outputs = modelo(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        pred_idx = outputs.argmax(dim=1).item()
        resultado = CLASES[pred_idx]
        confianza = probs[pred_idx].item() * 100
    return resultado, confianza

def recomendaciones(resultado):
    """Recomendaciones según el diagnóstico."""
    if resultado == "Benigno":
        return "🔹 Control periódico recomendado. Mantener hábitos saludables."
    elif resultado == "Maligno":
        texto = "⚠️ Se sugiere consultar a un especialista inmediatamente.\n\n🏥 Puedes visitar uno de los siguientes hospitales especializados:\n"
        for h in HOSPITALES:
            texto += f"\n• {h['nombre']}\n  Servicios: {h['servicios']}\n  Ubicación: {h['ubicacion']}\n"
        return texto
    else:
        return "❓ Resultado desconocido."

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================
def main():
    print("🚀 Iniciando sistema de diagnóstico...")
    print(f"⚙️  Usando dispositivo: {DEVICE}\n")

    try:
        model = cargar_modelo(MODEL_PATH)
        print("✅ Modelo cargado correctamente.\n")
    except Exception as e:
        print(f"❌ Error cargando el modelo: {e}")
        return

    # Soporte para analizar carpeta o imagen individual
    rutas = []
    if os.path.isdir(IMAGE_PATH):
        for f in os.listdir(IMAGE_PATH):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                rutas.append(os.path.join(IMAGE_PATH, f))
    else:
        rutas.append(IMAGE_PATH)

    for ruta in rutas:
        try:
            img_tensor = cargar_imagen(ruta)
            resultado, probabilidad = predecir(model, img_tensor)
            
            print("\n" + "="*50)
            print(f"🩺 Diagnóstico de: {os.path.basename(ruta)}")
            print("="*50)
            print(f"🦠 Predicción: {resultado.upper()}")
            print(f"📊 Confianza: {probabilidad:.2f}%")
            print(f"💡 Recomendación:\n{recomendaciones(resultado)}")
            print("="*50 + "\n")

        except Exception as e:
            print(f"❌ Error procesando {ruta}: {e}")

if __name__ == "__main__":
    main()
