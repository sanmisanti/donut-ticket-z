import json
from transformers import DonutProcessor

# Configuraciones
REQUIRED_FIELDS = [
    "CUIT_EMISOR", "REGIMEN", "FECHA", "NUMERO_DOCUMENTO",
    "PRIMER_COMPROBANTE", "ULTIMO_COMPROBANTE",
    "GRAVADO", "NO_GRAVADO", "EXENTO", "DESCUENTOS",
    "COMP_GENERADOS", "COMP_CANCELADOS", "IVA", "TOTAL"
]

NEW_SPECIAL_TOKENS = []

def sanitize_ground_truth(gt_dict):
    """
    Asegura que todos los campos requeridos estén presentes y únicos.
    Rellena los faltantes con '0.00' o '' según corresponda.
    """
    sanitized = {}
    for key in REQUIRED_FIELDS:
        value = gt_dict.get(key, "0.00" if "TOTAL" in key or "GRAVADO" in key or "IVA" in key else "")
        sanitized[key] = str(value).strip()
    return sanitized

def json2token(obj, new_special_tokens):
    if isinstance(obj, dict):
        output = ""
        # USAR ORDEN FIJO EN LUGAR DE ALFABÉTICO
        for k in REQUIRED_FIELDS:  # Orden consistente y lógico
            if k in obj:
                start_token = f"<s_{k}>"
                end_token = f"</s_{k}>"
                if start_token not in new_special_tokens:
                    new_special_tokens.extend([start_token, end_token])
                output += start_token + json2token(obj[k], new_special_tokens) + end_token
        return output
    else:
        return str(obj)

def verify_training_data(json_path, num_samples=5):
    """
    Verifica que los datos de entrenamiento estén bien formateados
    """
    print("🔍 VERIFICANDO DATOS DE ENTRENAMIENTO")
    print("="*50)
    
    # Cargar datos
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Total de muestras: {len(data)}")
    
    # Procesar algunas muestras
    processed_samples = []
    local_tokens = []
    
    for i, entry in enumerate(data[:num_samples]):
        print(f"\n{'='*30}")
        print(f"🔍 MUESTRA {i+1}/{num_samples}: {entry['file_name']}")
        print("="*30)
        
        # Mostrar ground truth original
        gt_original = entry["ground_truth"]
        print(f"📋 Ground Truth original:")
        for field in REQUIRED_FIELDS:
            value = gt_original.get(field, "❌ FALTANTE")
            print(f"  {field}: '{value}'")
        
        # Sanitizar
        gt_sanitized = sanitize_ground_truth(gt_original)
        print(f"\n🧹 Ground Truth sanitizado:")
        for field in REQUIRED_FIELDS:
            value = gt_sanitized.get(field, "❌ FALTANTE")
            print(f"  {field}: '{value}'")
        
        # Convertir a secuencia
        sequence = "<s>" + json2token(gt_sanitized, local_tokens) + "</s>"
        print(f"\n📝 Secuencia generada:")
        print(f"{sequence}")
        print(f"📏 Longitud: {len(sequence)} caracteres")
        
        processed_samples.append({
            "file_name": entry["file_name"],
            "sequence": sequence,
            "original_gt": gt_original,
            "sanitized_gt": gt_sanitized
        })
    
    # Mostrar tokens especiales generados
    print(f"\n🎯 TOKENS ESPECIALES GENERADOS:")
    print(f"📊 Total: {len(local_tokens)}")
    print(f"📋 Lista completa:")
    for i, token in enumerate(local_tokens):
        print(f"  {i+1:2d}. {token}")
    
    # Verificar que todos los campos tienen sus tokens
    expected_tokens = []
    for field in REQUIRED_FIELDS:
        expected_tokens.extend([f"<s_{field}>", f"</s_{field}>"])
    
    missing_tokens = set(expected_tokens) - set(local_tokens)
    if missing_tokens:
        print(f"\n⚠️ TOKENS FALTANTES:")
        for token in missing_tokens:
            print(f"  ❌ {token}")
    else:
        print(f"\n✅ TODOS LOS TOKENS NECESARIOS ESTÁN PRESENTES")
    
    return processed_samples, local_tokens

def test_tokenization(sequences, num_test=3):
    """
    Prueba la tokenización con DonutProcessor
    """
    print(f"\n🧪 PROBANDO TOKENIZACIÓN")
    print("="*40)
    
    # Cargar processor base
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    
    for i, seq in enumerate(sequences[:num_test]):
        print(f"\n🔬 PRUEBA {i+1}/{num_test}")
        print("-" * 20)
        
        # Tokenizar
        tokens = processor.tokenizer.tokenize(seq)
        token_ids = processor.tokenizer.convert_tokens_to_ids(tokens)
        
        print(f"📝 Secuencia: {seq[:100]}...")
        print(f"🔤 Tokens (primeros 20): {tokens[:20]}")
        print(f"🔢 IDs (primeros 20): {token_ids[:20]}")
        print(f"📊 Total tokens: {len(tokens)}")
        
        # Verificar tokens UNK
        unk_count = tokens.count('[UNK]')
        if unk_count > 0:
            print(f"⚠️ Tokens UNK encontrados: {unk_count}")
        else:
            print(f"✅ Sin tokens UNK")

def main():
    # Rutas de archivos (ajustar según tu entorno)
    train_json = "/content/train.json"  # Cambiar por tu ruta
    
    print("🚀 INICIANDO VERIFICACIÓN DE DATOS")
    
    try:
        # Verificar datos
        processed_samples, tokens = verify_training_data(train_json, num_samples=3)
        
        # Probar tokenización
        sequences = [sample["sequence"] for sample in processed_samples]
        test_tokenization(sequences, num_test=3)
        
        print(f"\n✅ VERIFICACIÓN COMPLETADA")
        
    except Exception as e:
        print(f"❌ Error durante la verificación: {e}")

if __name__ == "__main__":
    main() 