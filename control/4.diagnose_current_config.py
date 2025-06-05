from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import json

# Configuración actual
CONST_TARGET_SIZE = {"height": 1536, "width": 384}

print("=== DIAGNÓSTICO DE CONFIGURACIÓN ACTUAL ===")

# 1. Cargar processor y modelo
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# 2. Tu configuración actual
processor.feature_extractor.size = {"height": CONST_TARGET_SIZE["height"], "width": CONST_TARGET_SIZE["width"]}
processor.feature_extractor.do_resize = True
processor.feature_extractor.do_align_long_axis = False

print(f"✅ Feature extractor size: {processor.feature_extractor.size}")
print(f"❌ do_align_long_axis: {processor.feature_extractor.do_align_long_axis}")
print(f"✅ do_resize: {processor.feature_extractor.do_resize}")

# 3. Configuración del modelo
model.config.encoder.image_size = [CONST_TARGET_SIZE["height"], CONST_TARGET_SIZE["width"]]
model.config.decoder.max_length = 86

print(f"✅ Model encoder image size: {model.config.encoder.image_size}")
print(f"❓ Model decoder max_length: {model.config.decoder.max_length}")
print(f"✅ Model patch_size: {model.config.encoder.patch_size}")
print(f"✅ Model window_size: {model.config.encoder.window_size}")

# 4. Probar con una imagen real
image_path = "C:\\Users\\sanmi\\Documents\\Proyectos\\DonutModel\\train_images\\archivo_203.jpg"
image = Image.open(image_path).convert("RGB")

print(f"\n=== PROCESAMIENTO DE IMAGEN ===")
print(f"Original: {image.size} (ancho x alto)")

# Con tu configuración actual
processed = processor(image, return_tensors="pt")
print(f"Procesado: {processed.pixel_values.shape} -> [batch, canales, alto, ancho]")

# Verificar distorsión
original_ratio = image.size[0] / image.size[1]
processed_ratio = processed.pixel_values.shape[3] / processed.pixel_values.shape[2]
print(f"Relación original: {original_ratio:.3f}")
print(f"Relación procesada: {processed_ratio:.3f}")
print(f"Distorsión: {abs(original_ratio - processed_ratio):.3f}")

if abs(original_ratio - processed_ratio) > 0.1:
    print("❌ ¡IMAGEN DISTORSIONADA! do_align_long_axis=False causa esto")
else:
    print("✅ Imagen bien proporcionada")

# 5. Verificar longitud de secuencias
with open("C:\\Users\\sanmi\\Documents\\Proyectos\\DonutModel\\train.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# Tomar muestra de secuencias
sample_lengths = []
for i in range(min(10, len(data))):
    # Simular tu proceso actual
    gt = data[i]["ground_truth"]
    
    # Tu json2token actual (simplificado)
    seq = "<s>"
    for k in sorted(gt.keys(), reverse=True):
        seq += f"<s_{k}>{gt[k]}</s_{k}>"
    seq += "</s>"
    
    tokens = processor.tokenizer(seq, add_special_tokens=False)
    sample_lengths.append(len(tokens['input_ids']))

print(f"\n=== ANÁLISIS DE LONGITUDES ===")
print(f"Longitudes de muestra: {sample_lengths}")
print(f"Longitud promedio: {sum(sample_lengths)/len(sample_lengths):.1f}")
print(f"Longitud máxima: {max(sample_lengths)}")
print(f"Tu max_length actual: 86")

if max(sample_lengths) > 86:
    print(f"❌ ¡SECUENCIAS TRUNCADAS! {sum(1 for x in sample_lengths if x > 86)} de {len(sample_lengths)} se truncan")
else:
    print("✅ Todas las secuencias caben en max_length=86")

print(f"\n=== RECOMENDACIONES ===")
print(f"1. Cambiar do_align_long_axis = True")
print(f"2. Aumentar max_length a al menos {max(sample_lengths) + 10}")
print(f"3. Usar orden lógico en lugar de alfabético reverso")
print(f"4. Considerar dimensiones de imagen más grandes si hay memoria")