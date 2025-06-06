import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import json

def quick_test_model(model_path, test_image_path, expected_sequence):
    """
    Prueba rápida del modelo con una imagen
    """
    print("🧪 PRUEBA RÁPIDA DEL MODELO")
    print("="*40)
    
    try:
        # Cargar modelo
        print(f"📥 Cargando modelo: {model_path}")
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        processor = DonutProcessor.from_pretrained(model_path)
        
        # Configurar device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Cargar imagen
        print(f"📷 Cargando imagen: {test_image_path}")
        image = Image.open(test_image_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
        
        # Generar con parámetros simples
        print("🔮 Generando...")
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=pixel_values,
                decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids("<s>"),
                max_new_tokens=100,  # Más corto para prueba rápida
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.2,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        
        # Decodificar
        generated_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        print(f"\n📝 RESULTADO:")
        print(f"Generado: {generated_text}")
        print(f"Esperado: {expected_sequence[:100]}...")
        
        # Verificar si está mejorando
        if "<s_CUIT_EMISOR>" in generated_text:
            print("✅ Estructura de tokens detectada!")
        else:
            print("❌ Estructura de tokens NO detectada")
            
        if "</s_CUIT_EMISOR>" in generated_text:
            print("✅ Tokens de cierre detectados!")
        else:
            print("❌ Tokens de cierre NO detectados")
            
        return generated_text
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_during_training():
    """
    Función para probar el modelo durante el entrenamiento
    """
    # Rutas de ejemplo (ajustar según tu setup)
    model_path = "/content/drive/MyDrive/donut_project/models/modelo-final-v3"
    test_image = "/content/train_images/archivo_203.jpg"  # Usar una imagen de entrenamiento
    expected = "<s><s_CUIT_EMISOR>20213102827</s_CUIT_EMISOR><s_REGIMEN>IVA Responsable Inscripto</s_REGIMEN>..."
    
    result = quick_test_model(model_path, test_image, expected)
    return result

if __name__ == "__main__":
    test_during_training() 