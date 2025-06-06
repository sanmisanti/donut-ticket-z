import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer, TrainingArguments
from PIL import Image
import json
from datasets import Dataset, Features, Image, Value
from transformers import default_data_collator



NEW_SPECIAL_TOKENS = [] # Lista global de nuevos tokens
CONST_TARGET_SIZE = {"height": 1536, "width": 384}
REQUIRED_FIELDS = [
    "CUIT_EMISOR", "REGIMEN", "FECHA", "NUMERO_DOCUMENTO",
    "PRIMER_COMPROBANTE", "ULTIMO_COMPROBANTE",
    "GRAVADO", "NO_GRAVADO", "EXENTO", "DESCUENTOS",
    "COMP_GENERADOS", "COMP_CANCELADOS", "IVA", "TOTAL"
]
MAX_LENGTH = 300


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

ds_train_raw = json.loads(open("C:\\Users\\sanmi\\Documents\\Proyectos\\DonutModel\\train.json", encoding='utf-8').read())
processed = []
for entry in ds_train_raw:
    gt = entry["ground_truth"]
    sanitized_gt = sanitize_ground_truth(gt)
    seq = "<s>" + json2token(sanitized_gt, NEW_SPECIAL_TOKENS) + "</s>"
    processed.append({"file_name": entry["file_name"], "text": seq})

# Crear listas de rutas de imagen y textos
image_paths = [f"C:\\Users\\sanmi\\Documents\\Proyectos\\DonutModel\\train_images\\{e['file_name']}" for e in processed]
texts = [e['text'] for e in processed]

# Definir schema con imagen y texto
features = Features({"image": Image(), "text": Value("string")})

# Crear Dataset
ds_train = Dataset.from_dict({"image": image_paths, "text": texts},features=features)

ds_val_raw = json.loads(open("C:\\Users\\sanmi\\Documents\\Proyectos\\DonutModel\\val.json", encoding='utf-8').read())
processed = []
for entry in ds_val_raw:
    gt = entry["ground_truth"]
    sanitized_gt = sanitize_ground_truth(gt)
    seq = "<s>" + json2token(sanitized_gt, NEW_SPECIAL_TOKENS) + "</s>"
    processed.append({"file_name": entry["file_name"], "text": seq})

# Crear listas de rutas de imagen y textos
image_paths = [f"C:\\Users\\sanmi\\Documents\\Proyectos\\DonutModel\\train_images\\{e['file_name']}" for e in processed]
texts = [e['text'] for e in processed]

# Definir schema con imagen y texto
features = Features({"image": Image(), "text": Value("string")})

# Crear Dataset
ds_val = Dataset.from_dict({"image": image_paths, "text": texts},features=features)

ds_test_raw = json.loads(open("C:\\Users\\sanmi\\Documents\\Proyectos\\DonutModel\\test.json", encoding='utf-8').read())
processed = []
for entry in ds_test_raw:
    gt = entry["ground_truth"]
    sanitized_gt = sanitize_ground_truth(gt)
    seq = "<s>" + json2token(sanitized_gt, NEW_SPECIAL_TOKENS) + "</s>"
    processed.append({"file_name": entry["file_name"], "text": seq})

# Crear listas de rutas de imagen y textos
image_paths = [f"C:\\Users\\sanmi\\Documents\\Proyectos\\DonutModel\\train_images\\{e['file_name']}" for e in processed]
texts = [e['text'] for e in processed]

# Definir schema con imagen y texto
features = Features({"image": Image(), "text": Value("string")})

# Crear Dataset
ds_test = Dataset.from_dict({"image": image_paths, "text": texts},features=features)

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

# Agregar tokens especiales y tokens de inicio/fin de secuencia
task_start = "<s>"
eos_token = "</s>"
all_special = NEW_SPECIAL_TOKENS + [task_start, eos_token]
processor.tokenizer.add_special_tokens({"additional_special_tokens": all_special})

# Ajustar resolución de imágenes (ancho, alto)
processor.feature_extractor.size = {"height": CONST_TARGET_SIZE["height"], "width": CONST_TARGET_SIZE["width"]}
processor.feature_extractor.do_resize = True
processor.feature_extractor.do_align_long_axis = True


def transform(sample):
    """
    Función de transformación mejorada con debugging
    """
    # 1) Obtener objeto PIL.Image
    img_data = sample["image"]
    if isinstance(img_data, str):
        # Si es ruta en disco
        image = Image.open(img_data).convert("RGB")
    else:
        # Si ya es PIL.Image
        image = img_data.convert("RGB")

    # 2) Procesar imagen con configuración específica
    processed = processor(
        image, 
        return_tensors="pt"
    )
    pixel_values = processed.pixel_values.squeeze(0)
    
    # 3) Procesar texto - CRÍTICO: usar add_special_tokens=False
    text = sample["text"]
    inputs = processor.tokenizer(
        text,
        add_special_tokens=False,  # No agregar <s> y </s> extra
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    input_ids = inputs.input_ids.squeeze(0)
    
    # 4) Crear etiquetas: copiar input_ids y poner -100 en los pads
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    # 5) Debug información (solo para las primeras muestras)
    if hasattr(transform, 'debug_count'):
        transform.debug_count += 1
    else:
        transform.debug_count = 1
        
    if transform.debug_count <= 3:  # Solo debug para las primeras 3 muestras
        print(f"\n🔍 DEBUG TRANSFORM #{transform.debug_count}")
        print(f"📷 Imagen size: {image.size}")
        print(f"🖼️ Pixel values shape: {pixel_values.shape}")
        print(f"📝 Texto original: {text[:100]}...")
        print(f"🔤 Tokens: {len(input_ids)} tokens")
        print(f"📊 Input IDs (primeros 10): {input_ids[:10].tolist()}")
        print(f"📊 Labels (primeros 10): {labels[:10].tolist()}")
    
    return {
        "pixel_values": pixel_values, 
        "labels": labels, 
        "target_sequence": text
    }


ds_train = ds_train.map(transform, remove_columns=["image","text"], batched=False)
ds_val = ds_val.map(transform, remove_columns=["image","text"])
ds_test = ds_test.map(transform, remove_columns=["image","text"])

# Cargar modelo pre-entrenado Donut (encoder = Swin, decoder = BART)
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# CONFIGURACIÓN CRÍTICA DEL MODELO
print("\n🔧 CONFIGURANDO MODELO...")

# 1) Expandir la capa de embeddings del decoder ANTES de configurar tokens
print(f"📊 Vocabulario original: {model.decoder.config.vocab_size}")
print(f"📊 Vocabulario processor: {len(processor.tokenizer)}")
model.decoder.resize_token_embeddings(len(processor.tokenizer))
print(f"✅ Embeddings redimensionados a: {model.decoder.config.vocab_size}")

# 2) Configurar tamaño de imagen del encoder
model.config.encoder.image_size = [CONST_TARGET_SIZE["height"], CONST_TARGET_SIZE["width"]]
print(f"🖼️ Imagen configurada a: {model.config.encoder.image_size}")

# 3) Configurar tokens críticos del decoder
task_start_id = processor.tokenizer.convert_tokens_to_ids(task_start)
pad_token_id = processor.tokenizer.pad_token_id
eos_token_id = processor.tokenizer.eos_token_id

# Verificar que los tokens son válidos
if task_start_id == processor.tokenizer.unk_token_id:
    print(f"⚠️ WARNING: task_start_token '<s>' no encontrado, usando BOS token")
    task_start_id = processor.tokenizer.bos_token_id

print(f"🔤 Task start token: '<s>' -> ID: {task_start_id}")
print(f"🔤 PAD token: -> ID: {pad_token_id}")
print(f"🔤 EOS token: '</s>' -> ID: {eos_token_id}")

# 4) Aplicar configuración al modelo
model.config.decoder_start_token_id = task_start_id
model.config.pad_token_id = pad_token_id
model.config.eos_token_id = eos_token_id
model.config.decoder.max_length = MAX_LENGTH

# 5) Configuraciones adicionales importantes
model.config.decoder.early_stopping = True
model.config.decoder.length_penalty = 1.0
model.config.decoder.no_repeat_ngram_size = 3

print(f"📐 Max length configurado: {MAX_LENGTH}")
print(f"✅ Configuración del modelo completada")

# Verificar que estás usando el token correcto
decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s>")
print(f"🔧 Decoder start token ID: {decoder_start_token_id}")

# VERIFICACIONES ADICIONALES DE DEBUGGING
print("\n" + "="*50)
print("🔍 VERIFICACIONES DE DEBUGGING")
print("="*50)

# 1. Verificar tokens especiales
print(f"📋 Total de tokens especiales agregados: {len(NEW_SPECIAL_TOKENS)}")
print(f"📋 Primeros 10 tokens especiales: {NEW_SPECIAL_TOKENS[:10]}")
print(f"📋 Últimos 10 tokens especiales: {NEW_SPECIAL_TOKENS[-10:]}")

# 2. Verificar tamaño del vocabulario
print(f"📊 Tamaño del vocabulario: {len(processor.tokenizer)}")

# 3. Verificar una muestra de datos procesados
print(f"\n🔬 MUESTRA DE DATOS PROCESADOS:")
sample_text = processed[0]['text'] if processed else "No hay datos"
print(f"📝 Texto de muestra: {sample_text[:200]}...")

# 4. Verificar tokenización
if processed:
    sample_tokens = processor.tokenizer.tokenize(sample_text)
    print(f"🔤 Primeros 20 tokens: {sample_tokens[:20]}")
    sample_ids = processor.tokenizer.convert_tokens_to_ids(sample_tokens[:20])
    print(f"🔢 IDs correspondientes: {sample_ids}")

# 5. Verificar configuración del modelo
print(f"\n⚙️ CONFIGURACIÓN DEL MODELO:")
print(f"🔧 Decoder start token ID: {model.config.decoder_start_token_id}")
print(f"🔧 PAD token ID: {model.config.pad_token_id}")
print(f"🔧 EOS token ID: {model.config.eos_token_id}")
print(f"🔧 Max length: {model.config.decoder.max_length}")

print("="*50 + "\n")

training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/donut_project/models/donut-ticket-fiscal-v1",
    num_train_epochs=10,  # Reducido para evitar overfitting
    per_device_train_batch_size=1,  # Reducido para estabilidad
    per_device_eval_batch_size=1,
    learning_rate=3e-5,  # Aumentado ligeramente
    weight_decay=0.01,
    logging_steps=50,
    eval_steps=200,  # Evalúa menos frecuentemente
    save_strategy="steps",
    save_steps=200,  # Guarda menos frecuentemente
    eval_strategy="steps",
    save_total_limit=2,  # Menos checkpoints
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    warmup_steps=100,  # Más warmup
    gradient_accumulation_steps=4,  # Compensar batch size menor
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    # Parámetros adicionales para estabilidad
    max_grad_norm=1.0,  # Gradient clipping
    lr_scheduler_type="cosine",  # Scheduler más suave
    report_to=None,  # Desactivar wandb/tensorboard
)

def collate_fn(batch):
    """
    Data collator personalizado para modelo Donut/VisionEncoderDecoder
    """
    pixel_values = []
    labels = []
    
    for item in batch:
        # Obtener pixel_values y labels
        pv = item["pixel_values"]
        lb = item["labels"]
        
        # Asegurar que son tensores
        if not torch.is_tensor(pv):
            pv = torch.tensor(pv)
        if not torch.is_tensor(lb):
            lb = torch.tensor(lb)
            
        pixel_values.append(pv)
        labels.append(lb)
    
    # Apilar en batches
    pixel_values = torch.stack(pixel_values)
    labels = torch.stack(labels)
    
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }

# ✅ Crear clase de Trainer personalizada
class DonutTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Función de pérdida personalizada para Donut
        """
        labels = inputs.pop("labels")
        
        # Forward pass
        outputs = model(**inputs, labels=labels)
        
        # Extraer pérdida
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

# ✅ Usar Trainer básico en lugar de Seq2SeqTrainer
trainer = DonutTrainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer,  # Para logging
)

# ENTRENAR EL MODELO
print("\n🚀 INICIANDO ENTRENAMIENTO...")
print("="*50)

trainer.train()

print("\n✅ ENTRENAMIENTO COMPLETADO!")
print("="*50)

# GUARDAR MODELO Y PROCESSOR
save_dir = "/content/drive/MyDrive/donut_project/models/modelo-final-v3"
print(f"💾 Guardando modelo en: {save_dir}")

# Asegurar que el directorio existe
import os
os.makedirs(save_dir, exist_ok=True)

# Guardar modelo y processor
model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)

print(f"✅ Modelo guardado exitosamente!")

# VERIFICACIÓN FINAL
print(f"\n🔍 VERIFICACIÓN FINAL:")
print(f"📁 Directorio: {save_dir}")
saved_files = os.listdir(save_dir)
print(f"📄 Archivos guardados: {saved_files}")

# Verificar configuración guardada
test_model = VisionEncoderDecoderModel.from_pretrained(save_dir)
test_processor = DonutProcessor.from_pretrained(save_dir)

print(f"📊 Vocabulario guardado: {len(test_processor.tokenizer)}")
print(f"🔧 Decoder start token ID: {test_model.config.decoder_start_token_id}")
print(f"🔧 Max length: {test_model.config.decoder.max_length}")

print(f"\n🎉 ¡PROCESO COMPLETADO EXITOSAMENTE!")
print("="*50)

