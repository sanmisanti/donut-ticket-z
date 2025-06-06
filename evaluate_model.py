import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import json
import re

# Configuraciones
REQUIRED_FIELDS = [
    "CUIT_EMISOR", "REGIMEN", "FECHA", "NUMERO_DOCUMENTO",
    "PRIMER_COMPROBANTE", "ULTIMO_COMPROBANTE",
    "GRAVADO", "NO_GRAVADO", "EXENTO", "DESCUENTOS",
    "COMP_GENERADOS", "COMP_CANCELADOS", "IVA", "TOTAL"
]

def extract_json_from_sequence(sequence):
    """
    Extrae campos estructurados de una secuencia tokenizada.
    Maneja tokens malformados y repetidos.
    """
    extracted = {}
    
    # Patrones mejorados para extraer información
    patterns = {
        field: re.compile(rf"<s_{field}>(.*?)</s_{field}>", re.DOTALL)
        for field in REQUIRED_FIELDS
    }
    
    print(f"🔍 Analizando secuencia: {sequence[:100]}...")
    
    for field, pattern in patterns.items():
        matches = pattern.findall(sequence)
        if matches:
            # Tomar la primera coincidencia válida (no vacía)
            value = next((match.strip() for match in matches if match.strip()), "")
            if value:
                extracted[field] = value
                print(f"✅ {field}: '{value}'")
            else:
                print(f"❌ {field}: encontrado pero vacío")
        else:
            print(f"❌ {field}: no encontrado")
    
    return extracted

def load_model_and_processor(model_path):
    """Carga el modelo y processor con verificaciones"""
    print(f"📥 Cargando desde: {model_path}")
    
    try:
        processor = DonutProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        
        print(f"✅ Modelo cargado exitosamente!")
        print(f"📊 Vocabulario size: {len(processor.tokenizer)}")
        
        # Verificar configuración crítica
        print(f"⚙️ Decoder start token ID: {model.config.decoder_start_token_id}")
        print(f"⚙️ PAD token ID: {model.config.pad_token_id}")
        print(f"⚙️ EOS token ID: {model.config.eos_token_id}")
        
        return model, processor
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None, None

def predict_with_better_params(model, processor, image_path, debug=True):
    """
    Realiza predicción con parámetros de generación mejorados
    """
    # Cargar y procesar imagen
    image = Image.open(image_path).convert("RGB")
    if debug:
        print(f"📷 Imagen: {image.size} (ancho x alto)")
    
    # Procesar imagen
    pixel_values = processor(image, return_tensors="pt").pixel_values
    if debug:
        print(f"🖼️ Tensor procesado: {pixel_values.shape}")
    
    # Configurar device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    pixel_values = pixel_values.to(device)
    
    # PARÁMETROS DE GENERACIÓN MEJORADOS
    generation_kwargs = {
        "pixel_values": pixel_values,
        "decoder_start_token_id": processor.tokenizer.convert_tokens_to_ids("<s>"),
        "max_new_tokens": 300,  # Cambiado de max_length a max_new_tokens
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "do_sample": False,  # Determinístico para debugging
        "num_beams": 1,      # Sin beam search para empezar
        "repetition_penalty": 1.2,  # Penalizar repeticiones
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 3,  # Evitar repetir n-gramas
    }
    
    if debug:
        print(f"🎯 Parámetros de generación:")
        for k, v in generation_kwargs.items():
            if k != "pixel_values":
                print(f"   {k}: {v}")
    
    # Generar
    with torch.no_grad():
        generated_ids = model.generate(**generation_kwargs)
    
    # Decodificar
    generated_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    if debug:
        print(f"\n📝 SECUENCIA GENERADA COMPLETA:")
        print(f"{generated_text}")
        print(f"\n📏 Longitud: {len(generated_text)} caracteres")
    
    return generated_text

def evaluate_model(model_path, test_json_path, images_dir, num_samples=5):
    """
    Evalúa el modelo con parámetros mejorados
    """
    print("🚀 INICIANDO EVALUACIÓN MEJORADA")
    print("="*60)
    
    # Cargar modelo
    model, processor = load_model_and_processor(model_path)
    if not model:
        return
    
    # Cargar datos de prueba
    with open(test_json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"📊 Datos de prueba cargados: {len(test_data)} muestras")
    
    # Evaluar muestras
    total_accuracy = 0
    
    for i, sample in enumerate(test_data[:num_samples]):
        print(f"\n{'='*60}")
        print(f"🔍 MUESTRA {i+1}/{num_samples}: {sample['file_name']}")
        print("="*60)
        
        image_path = f"{images_dir}/{sample['file_name']}"
        
        try:
            # Predicción
            predicted_sequence = predict_with_better_params(
                model, processor, image_path, debug=True
            )
            
            # Extraer JSON
            predicted_json = extract_json_from_sequence(predicted_sequence)
            ground_truth = sample['ground_truth']
            
            # Calcular accuracy
            correct = 0
            total_fields = len(REQUIRED_FIELDS)
            
            print(f"\n📊 COMPARACIÓN:")
            print("-" * 40)
            
            for field in REQUIRED_FIELDS:
                pred_val = predicted_json.get(field, "❌ FALTANTE")
                true_val = ground_truth.get(field, "")
                
                is_correct = pred_val == true_val
                if is_correct:
                    correct += 1
                
                status = "✅" if is_correct else "❌"
                print(f"{status} {field}:")
                print(f"  Predicho: '{pred_val}'")
                print(f"  Real:     '{true_val}'")
            
            sample_accuracy = correct / total_fields
            total_accuracy += sample_accuracy
            
            print(f"\n📈 ACCURACY MUESTRA: {sample_accuracy:.1%} ({correct}/{total_fields})")
            
        except Exception as e:
            print(f"❌ Error procesando {sample['file_name']}: {e}")
    
    # Resultado final
    avg_accuracy = total_accuracy / num_samples
    print(f"\n🎯 ACCURACY PROMEDIO: {avg_accuracy:.1%}")
    
    return avg_accuracy

if __name__ == "__main__":
    # Configurar rutas (ajustar según tu entorno)
    model_path = "/content/drive/MyDrive/donut_project/models/modelo-final-v3"  # ACTUALIZADO
    test_json = "/content/test.json"  # Ajustar ruta
    images_dir = "/content/train_images"  # Ajustar ruta
    
    # Ejecutar evaluación
    evaluate_model(model_path, test_json, images_dir, num_samples=3) 