# ===== CELDA DE PREDICCIÓN EN GOOGLE COLAB =====

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import json
import os

# 🔧 CONFIGURACIÓN PARA COLAB
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Usando dispositivo: {device}")

# 📂 RUTAS PARA GOOGLE COLAB (ajusta según tu estructura)
MODEL_DIR = "/content/drive/MyDrive/DonutModel/models/modelo-final-v0"
IMAGES_DIR = "/content/drive/MyDrive/DonutModel/train_images"
TEST_JSON = "/content/drive/MyDrive/DonutModel/test.json"

# ✅ VERIFICAR QUE EL MODELO EXISTE
if not os.path.exists(MODEL_DIR):
    print(f"❌ ERROR: No se encuentra el modelo en {MODEL_DIR}")
    print("💡 Asegúrate de que index.py haya terminado correctamente")
else:
    print(f"✅ Modelo encontrado en: {MODEL_DIR}")

# 🚀 CARGAR MODELO Y PROCESSOR
try:
    print("📥 Cargando processor...")
    processor = DonutProcessor.from_pretrained(MODEL_DIR)
    
    print("📥 Cargando modelo...")
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    
    print(f"✅ ¡Modelo cargado exitosamente!")
    print(f"📊 Vocabulario size: {len(processor.tokenizer)}")
    print(f"🔧 Max length: {model.config.decoder.max_length}")
    
except Exception as e:
    print(f"❌ ERROR cargando modelo: {e}")
    raise

def predict_ticket(image_filename, ground_truth=None, verbose=True):
    """
    Predice los datos de un ticket Z con debug completo
    
    Args:
        image_filename: nombre del archivo (ej: "archivo_148.jpg")
        ground_truth: diccionario con valores reales (opcional)
        verbose: mostrar información detallada
    """
    image_path = os.path.join(IMAGES_DIR, image_filename)
    
    if verbose:
        print(f"\n🔍 PREDICIENDO: {image_filename}")
        print("=" * 60)
    
    # ✅ 1. CARGAR Y PROCESAR IMAGEN
    try:
        image = Image.open(image_path).convert("RGB")
        if verbose:
            print(f"📷 Imagen original: {image.size} (ancho x alto)")
    except Exception as e:
        print(f"❌ ERROR cargando imagen: {e}")
        return {}

    # ✅ 2. PROCESAMIENTO (igual que en entrenamiento)
    processed = processor(image, return_tensors="pt")
    pixel_values = processed.pixel_values.to(device)
    
    if verbose:
        print(f"🖼️ Tensor procesado: {pixel_values.shape}")

    # ✅ 3. GENERACIÓN DE PREDICCIÓN
    with torch.no_grad():
        decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s>")
        
        outputs = model.generate(
            pixel_values=pixel_values,
            decoder_start_token_id=decoder_start_token_id,
            max_length=300,  # ✅ Mismo MAX_LENGTH del entrenamiento
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            early_stopping=True,
            do_sample=False,      # Generación determinística
            num_beams=1,         # Consistente con entrenamiento
            # SIN no_repeat_ngram_size para permitir patrones repetitivos
        )

    # ✅ 4. DECODIFICACIÓN Y ANÁLISIS
    raw_seq = processor.tokenizer.decode(outputs[0], skip_special_tokens=False)
    clean_seq = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if verbose:
        print(f"\n📝 SECUENCIA RAW:")
        print(raw_seq[:500] + "..." if len(raw_seq) > 500 else raw_seq)
        print(f"\n🧹 SECUENCIA LIMPIA:")
        print(clean_seq[:200] + "..." if len(clean_seq) > 200 else clean_seq)
    
    # ✅ 5. EXTRAER JSON
    try:
        json_out = processor.token2json(raw_seq)
        if verbose:
            print(f"\n✅ JSON EXTRAÍDO:")
            print(json.dumps(json_out, indent=2, ensure_ascii=False))
    except Exception as e:
        if verbose:
            print(f"\n❌ ERROR en token2json: {e}")
        # Fallback: extraer campos manualmente
        json_out = extract_fields_manual(raw_seq)
    
    # ✅ 6. COMPARACIÓN CON GROUND TRUTH
    if ground_truth and verbose:
        print(f"\n📊 COMPARACIÓN CON GROUND TRUTH:")
        print("-" * 40)
        accuracy = 0
        total_fields = len(ground_truth)
        
        for key, true_val in ground_truth.items():
            pred_val = json_out.get(key, "❌ FALTANTE")
            match = str(pred_val).strip() == str(true_val).strip()
            if match:
                accuracy += 1
            
            status = "✅" if match else "❌"
            print(f"{key}: {status}")
            print(f"  Predicho: '{pred_val}'")
            print(f"  Real:     '{true_val}'")
        
        acc_percent = (accuracy / total_fields) * 100
        print(f"\n📈 ACCURACY: {acc_percent:.1f}% ({accuracy}/{total_fields})")
    
    return json_out

def extract_fields_manual(raw_sequence):
    """
    Extrae campos manualmente si token2json falla
    """
    import re
    
    fields = {}
    field_names = [
        "CUIT_EMISOR", "REGIMEN", "FECHA", "NUMERO_DOCUMENTO",
        "PRIMER_COMPROBANTE", "ULTIMO_COMPROBANTE",
        "GRAVADO", "NO_GRAVADO", "EXENTO", "DESCUENTOS",
        "COMP_GENERADOS", "COMP_CANCELADOS", "IVA", "TOTAL"
    ]
    
    for field in field_names:
        pattern = f"<s_{field}>(.*?)</s_{field}>"
        match = re.search(pattern, raw_sequence)
        if match:
            fields[field] = match.group(1).strip()
    
    return fields

def test_multiple_samples(num_samples=5):
    """
    Prueba con múltiples muestras del conjunto de test
    """
    print("🧪 PRUEBA CON MÚLTIPLES MUESTRAS")
    print("=" * 60)
    
    # Cargar datos de test
    try:
        with open(TEST_JSON, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except:
        print("❌ No se pudo cargar test.json")
        return
    
    total_accuracy = 0
    sample_count = min(num_samples, len(test_data))
    
    for i in range(sample_count):
        sample = test_data[i]
        filename = sample["file_name"]
        ground_truth = sample["ground_truth"]
        
        print(f"\n🔍 MUESTRA {i+1}/{sample_count}: {filename}")
        print("-" * 30)
        
        prediction = predict_ticket(filename, ground_truth, verbose=False)
        
        # Calcular accuracy para esta muestra
        correct = 0
        for key, true_val in ground_truth.items():
            if str(prediction.get(key, "")).strip() == str(true_val).strip():
                correct += 1
        
        sample_acc = (correct / len(ground_truth)) * 100
        total_accuracy += sample_acc
        
        print(f"✅ Accuracy muestra: {sample_acc:.1f}%")
    
    avg_accuracy = total_accuracy / sample_count
    print(f"\n🎯 ACCURACY PROMEDIO: {avg_accuracy:.1f}%")
    
    return avg_accuracy

# ===== EJEMPLOS DE USO =====

# 🔍 1. PREDICCIÓN INDIVIDUAL CON GROUND TRUTH
print("🚀 EJECUTANDO PREDICCIÓN DE EJEMPLO...")

# Datos de ejemplo (ajusta según tus archivos)
sample_filename = "archivo_148.jpg"
sample_ground_truth = {
    "CUIT_EMISOR": "20213102827",
    "REGIMEN": "IVA Responsable Inscripto",
    "NUMERO_DOCUMENTO": "00000326",
    "FECHA": "30/05/2023",
    "PRIMER_COMPROBANTE": "00007911",
    "ULTIMO_COMPROBANTE": "00007925",
    "GRAVADO": "222231.39",
    "NO_GRAVADO": "0.00",
    "EXENTO": "0.00",
    "DESCUENTOS": "0.00",
    "COMP_GENERADOS": "00000015",
    "COMP_CANCELADOS": "00000000",
    "IVA": "46668.61",
    "TOTAL": "268900.00"
}

# Ejecutar predicción
prediction = predict_ticket(sample_filename, sample_ground_truth)

# 🧪 2. PRUEBA CON MÚLTIPLES MUESTRAS
print("\n" + "="*60)
test_multiple_samples(3)