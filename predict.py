import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import json

# 0) Configura tu dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Carga processor + modelo desde la misma carpeta
save_dir = "/content/drive/MyDrive/donut_project/dataset/modelo-final-v0"  # ⚠️ Usar el nombre correcto
processor = DonutProcessor.from_pretrained(save_dir)
model = VisionEncoderDecoderModel.from_pretrained(save_dir)
model.to(device)
model.eval()  # ✅ Modo evaluación

print(f"✅ Modelo cargado desde: {save_dir}")
print(f"📊 Vocabulario size: {len(processor.tokenizer)}")
print(f"🔧 Max length configurado: {model.config.decoder.max_length}")

def predict_debug(image_path, ground_truth=None):
    """
    Predice con debug completo y comparación opcional con ground truth
    """
    print(f"\n🔍 PREDICIENDO: {image_path}")
    print("=" * 50)
    
    # 1) Carga y preprocesa la imagen
    image = Image.open(image_path).convert("RGB")
    print(f"📷 Imagen original: {image.size} (ancho x alto)")

    # Procesar (igual que en entrenamiento)
    processed = processor(image, return_tensors="pt")
    pixel_values = processed.pixel_values.to(device)
    print(f"🖼️ Tensor procesado: {pixel_values.shape}")

    # 2) Generación con parámetros optimizados
    with torch.no_grad():
        decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s>")
        
        outputs = model.generate(
            pixel_values=pixel_values,
            decoder_start_token_id=decoder_start_token_id,
            max_length=300,  # ✅ Usar el mismo MAX_LENGTH del entrenamiento
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            # ⚠️ QUITAR no_repeat_ngram_size para permitir patrones repetitivos válidos
            early_stopping=True,
            do_sample=False,  # ✅ Generación determinística
            num_beams=1       # ✅ Consistente con entrenamiento
        )

    # 3) Decodifica y analiza
    raw_seq = processor.tokenizer.decode(outputs[0], skip_special_tokens=False)
    clean_seq = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n📝 SECUENCIA RAW:")
    print(raw_seq)
    print(f"\n🧹 SECUENCIA LIMPIA:")
    print(clean_seq)
    
    # 4) Aplicar token2json
    try:
        json_out = processor.token2json(raw_seq)
        print(f"\n✅ JSON EXTRAÍDO:")
        print(json.dumps(json_out, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"\n❌ ERROR en token2json: {e}")
        json_out = {}
    
    # 5) Comparación con ground truth (opcional)
    if ground_truth:
        print(f"\n📊 COMPARACIÓN CON GROUND TRUTH:")
        print("-" * 30)
        for key in ground_truth.keys():
            pred_val = json_out.get(key, "❌ FALTANTE")
            true_val = ground_truth[key]
            match = "✅" if str(pred_val) == str(true_val) else "❌"
            print(f"{key}: {match}")
            print(f"  Predicho: '{pred_val}'")
            print(f"  Real:     '{true_val}'")
    
    return json_out

def batch_predict(image_list, ground_truth_list=None):
    """
    Predice múltiples imágenes y calcula métricas
    """
    results = []
    correct_fields = 0
    total_fields = 0
    
    for i, img_path in enumerate(image_list):
        gt = ground_truth_list[i] if ground_truth_list else None
        pred = predict_debug(img_path, gt)
        results.append(pred)
        
        if gt:
            for key, true_val in gt.items():
                total_fields += 1
                if str(pred.get(key, "")) == str(true_val):
                    correct_fields += 1
    
    if ground_truth_list:
        accuracy = correct_fields / total_fields * 100
        print(f"\n📈 ACCURACY TOTAL: {accuracy:.1f}% ({correct_fields}/{total_fields})")
    
    return results

# 3) Prueba con ejemplos
if __name__ == "__main__":
    # Ejemplo single
    sample_path = "/content/drive/MyDrive/donut_project/dataset/train_images/archivo_148.jpg"
    
    # Si tienes el ground truth correspondiente
    sample_gt = {
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
    
    # Predicción con comparación
    pred = predict_debug(sample_path, sample_gt)