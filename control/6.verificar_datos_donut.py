import torch
from transformers import DonutProcessor
from PIL import Image, ImageDraw, ImageFont
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datasets import Dataset, Features, Image as HFImage, Value
import numpy as np

# Configuración igual a tu index.py
NEW_SPECIAL_TOKENS = []
CONST_TARGET_SIZE = {"height": 1536, "width": 384}
REQUIRED_FIELDS = [
    "CUIT_EMISOR", "REGIMEN", "FECHA", "NUMERO_DOCUMENTO",
    "PRIMER_COMPROBANTE", "ULTIMO_COMPROBANTE",
    "GRAVADO", "NO_GRAVADO", "EXENTO", "DESCUENTOS",
    "COMP_GENERADOS", "COMP_CANCELADOS", "IVA", "TOTAL"
]
MAX_LENGTH = 300

def sanitize_ground_truth(gt_dict):
    sanitized = {}
    for key in REQUIRED_FIELDS:
        value = gt_dict.get(key, "0.00" if "TOTAL" in key or "GRAVADO" in key or "IVA" in key else "")
        sanitized[key] = str(value).strip()
    return sanitized

def json2token(obj, new_special_tokens):
    if isinstance(obj, dict):
        output = ""
        for k in REQUIRED_FIELDS:
            if k in obj:
                start_token = f"<s_{k}>"
                end_token = f"</s_{k}>"
                if start_token not in new_special_tokens:
                    new_special_tokens.extend([start_token, end_token])
                output += start_token + json2token(obj[k], new_special_tokens) + end_token
        return output
    else:
        return str(obj)

def verificar_correlacion_datos(num_muestras=3):
    """Verifica que imágenes y anotaciones se correlacionen correctamente"""
    
    print("🔍 VERIFICACIÓN DE CORRELACIÓN IMAGEN-ANOTACIÓN")
    print("=" * 60)
    
    # 1. Cargar datos RAW (igual que tu código)
    print("📂 Cargando train.json...")
    ds_train_raw = json.loads(open("train.json", encoding='utf-8').read())
    
    # 2. Configurar processor (igual que tu código)
    print("🔧 Configurando processor...")
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    
    task_start = "<s>"
    eos_token = "</s>"
    # Procesar algunas muestras para obtener todos los tokens especiales
    temp_tokens = []
    for i in range(min(5, len(ds_train_raw))):
        gt = ds_train_raw[i]["ground_truth"]
        sanitized_gt = sanitize_ground_truth(gt)
        json2token(sanitized_gt, temp_tokens)
    
    all_special = temp_tokens + [task_start, eos_token]
    processor.tokenizer.add_special_tokens({"additional_special_tokens": all_special})
    processor.feature_extractor.size = {"height": CONST_TARGET_SIZE["height"], "width": CONST_TARGET_SIZE["width"]}
    processor.feature_extractor.do_resize = True
    processor.feature_extractor.do_align_long_axis = True
    
    # 3. Verificar muestras específicas
    for i in range(min(num_muestras, len(ds_train_raw))):
        print(f"\n🔍 MUESTRA {i+1}/{num_muestras}")
        print("-" * 40)
        
        # Datos RAW
        entry = ds_train_raw[i]
        file_name = entry["file_name"]
        ground_truth = entry["ground_truth"]
        
        print(f"📁 Archivo: {file_name}")
        
        # Verificar que la imagen existe
        image_path = f"train_images\\{file_name}"
        try:
            image = Image.open(image_path).convert("RGB")
            print(f"✅ Imagen cargada: {image.size} (ancho x alto)")
        except Exception as e:
            print(f"❌ ERROR cargando imagen: {e}")
            continue
        
        # Mostrar anotaciones originales
        print(f"\n📋 ANOTACIONES ORIGINALES:")
        for key, value in ground_truth.items():
            print(f"  {key}: '{value}'")
        
        # Procesar anotaciones (sanitizar)
        sanitized_gt = sanitize_ground_truth(ground_truth)
        print(f"\n🧹 ANOTACIONES SANITIZADAS:")
        for key, value in sanitized_gt.items():
            if key in ground_truth:
                if str(ground_truth[key]).strip() != str(value).strip():
                    print(f"  {key}: '{ground_truth[key]}' → '{value}' ⚠️ MODIFICADO")
                else:
                    print(f"  {key}: '{value}' ✅")
            else:
                print(f"  {key}: '{value}' 🆕 AGREGADO")
        
        # Convertir a secuencia de tokens
        local_tokens = []
        token_sequence = json2token(sanitized_gt, local_tokens)
        full_sequence = f"<s>{token_sequence}</s>"
        
        print(f"\n🔤 SECUENCIA TOKENIZADA:")
        print(f"Longitud: {len(full_sequence)} caracteres")
        print(f"Secuencia: {full_sequence}")
        
        # Tokenizar con el processor
        inputs = processor.tokenizer(
            full_sequence,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        
        input_ids = inputs.input_ids.squeeze(0)
        num_tokens = (input_ids != processor.tokenizer.pad_token_id).sum().item()
        
        print(f"\n🎯 ANÁLISIS DE TOKENS:")
        print(f"Tokens reales: {num_tokens}/{MAX_LENGTH}")
        print(f"¿Se truncó?: {'❌ SÍ' if num_tokens == MAX_LENGTH else '✅ NO'}")
        
        # Decodificar tokens para verificar
        decoded = processor.tokenizer.decode(input_ids, skip_special_tokens=False)
        print(f"Secuencia decodificada: {decoded[:200]}...")
        
        # Procesar imagen
        print(f"\n🖼️ PROCESAMIENTO DE IMAGEN:")
        processed = processor(image, return_tensors="pt")
        pixel_values = processed.pixel_values
        
        print(f"Forma original: {image.size} (ancho x alto)")
        print(f"Tensor procesado: {pixel_values.shape} (batch, canales, alto, ancho)")
        print(f"Rango de valores: [{pixel_values.min():.3f}, {pixel_values.max():.3f}]")
        
        # Verificación final de correlación
        print(f"\n✅ VERIFICACIÓN DE CORRELACIÓN:")
        print(f"  📁 Archivo imagen: {file_name}")
        print(f"  🔢 Número de campos anotados: {len([k for k, v in ground_truth.items() if v])}")
        print(f"  📏 Tokens usados: {num_tokens}/{MAX_LENGTH}")
        print(f"  🖼️ Imagen procesada: {pixel_values.shape}")
        
        # Mostrar imagen procesada visualmente
        mostrar_imagen_procesada(image, pixel_values, i+1)

def mostrar_imagen_procesada(original_image, processed_tensor, muestra_num):
    """Muestra comparación visual de imagen original vs procesada"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Imagen original
    ax1.imshow(original_image)
    ax1.set_title(f'Imagen Original (Muestra {muestra_num})\n{original_image.size}')
    ax1.axis('off')
    
    # Imagen procesada (convertir tensor a imagen)
    # processed_tensor shape: [1, 3, 1536, 384]
    processed_np = processed_tensor.squeeze(0).permute(1, 2, 0).numpy()
    
    # Normalizar para visualización (Donut usa normalización específica)
    # Necesitamos revertir la normalización de ImageNet que usa Donut
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    processed_np = processed_np * std + mean
    processed_np = np.clip(processed_np, 0, 1)
    
    ax2.imshow(processed_np)
    ax2.set_title(f'Imagen Procesada por Donut\n{processed_tensor.shape[2]}x{processed_tensor.shape[3]}')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'verificacion_muestra_{muestra_num}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Imagen guardada como: verificacion_muestra_{muestra_num}.png")

def analizar_longitudes_dataset():
    """Analiza las longitudes de todas las secuencias del dataset"""
    
    print("\n📊 ANÁLISIS DE LONGITUDES DEL DATASET COMPLETO")
    print("=" * 60)
    
    ds_train_raw = json.loads(open("train.json", encoding='utf-8').read())
    longitudes = []
    
    for entry in ds_train_raw:
        gt = entry["ground_truth"]
        sanitized_gt = sanitize_ground_truth(gt)
        temp_tokens = []
        token_sequence = json2token(sanitized_gt, temp_tokens)
        full_sequence = f"<s>{token_sequence}</s>"
        longitudes.append(len(full_sequence))
    
    print(f"Total de muestras: {len(longitudes)}")
    print(f"Longitud mínima: {min(longitudes)} caracteres")
    print(f"Longitud máxima: {max(longitudes)} caracteres")
    print(f"Longitud promedio: {np.mean(longitudes):.1f} caracteres")
    print(f"Longitud mediana: {np.median(longitudes):.1f} caracteres")
    
    # Análisis de truncamiento
    muestras_truncadas = sum(1 for l in longitudes if l > MAX_LENGTH)
    print(f"\n⚠️ ANÁLISIS DE TRUNCAMIENTO (MAX_LENGTH={MAX_LENGTH}):")
    print(f"Muestras que se truncarán: {muestras_truncadas}/{len(longitudes)} ({muestras_truncadas/len(longitudes)*100:.1f}%)")
    
    # Histograma
    plt.figure(figsize=(12, 6))
    plt.hist(longitudes, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(MAX_LENGTH, color='red', linestyle='--', linewidth=2, label=f'MAX_LENGTH={MAX_LENGTH}')
    plt.xlabel('Longitud de secuencia (caracteres)')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de longitudes de secuencias')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('distribucion_longitudes.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("💾 Histograma guardado como: distribucion_longitudes.png")

if __name__ == "__main__":
    print("🚀 INICIANDO VERIFICACIÓN COMPLETA DE DATOS DONUT")
    print("=" * 60)
    
    # 1. Verificar correlación de muestras específicas
    verificar_correlacion_datos(num_muestras=3)
    
    # 2. Análisis completo del dataset
    analizar_longitudes_dataset()
    
    print("\n✅ VERIFICACIÓN COMPLETA TERMINADA")