from transformers import DonutProcessor
from PIL import Image
import torch

# Cargar processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

# Configurar como tu código actual
processor.feature_extractor.size = {"height": 1536, "width": 384}
processor.feature_extractor.do_resize = True

def test_image_processing(image_path, align_long_axis_setting):
    # Cargar imagen
    image = Image.open(image_path).convert("RGB")
    print(f"\n=== CONFIGURACIÓN do_align_long_axis = {align_long_axis_setting} ===")
    print(f"Imagen original: {image.size} (ancho x alto)")
    
    # Configurar el setting
    processor.feature_extractor.do_align_long_axis = align_long_axis_setting
    
    # Procesar
    processed = processor(image, return_tensors="pt")
    pixel_values = processed.pixel_values
    
    print(f"Resultado procesado: {pixel_values.shape} -> [batch, canales, alto, ancho]")
    print(f"Dimensiones finales: {pixel_values.shape[2]} x {pixel_values.shape[3]}")
    
    # Calcular relación de aspecto
    original_ratio = image.size[0] / image.size[1]  # ancho/alto
    processed_ratio = pixel_values.shape[3] / pixel_values.shape[2]  # ancho/alto
    
    print(f"Relación aspecto original: {original_ratio:.3f}")
    print(f"Relación aspecto procesado: {processed_ratio:.3f}")
    print(f"¿Se preservó la relación? {'✅ SÍ' if abs(original_ratio - processed_ratio) < 0.1 else '❌ NO'}")
    
    return pixel_values

# Usar una de tus imágenes
image_path = "C:\\Users\\sanmi\\Documents\\Proyectos\\DonutModel\\train_images\\archivo_203.jpg"

# Probar con ambas configuraciones
test_image_processing(image_path, False)  # Tu configuración actual
test_image_processing(image_path, True)   # La configuración correcta