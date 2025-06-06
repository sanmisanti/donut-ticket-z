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
  # 0) Obtener objeto PIL.Image
  img_data = sample["image"]
  if isinstance(img_data, str):
      # Si es ruta en disco
      image = Image.open(img_data).convert("RGB")
  else:
      # Si ya es PIL.Image
      image = img_data.convert("RGB")

  # Procesar imagen
  print("Original size:", image.size)  # (ancho, alto)
  # pixel_values = processor(image, random_padding=(False),return_tensors="pt").pixel_values.squeeze()
  processed = processor(image, return_tensors="pt")
  pixel_values = processed.pixel_values.squeeze(0)
  print("Procesado shape:", pixel_values.shape)


  inputs = processor.tokenizer(sample["text"],add_special_tokens=False,padding="max_length",truncation=True,max_length=MAX_LENGTH,return_tensors="pt")
  input_ids = inputs.input_ids.squeeze(0)
  # Crear etiquetas: copiar input_ids y poner -100 en los pads para ignorarlos
  labels = input_ids.clone()
  labels[labels == processor.tokenizer.pad_token_id] = -100
  return {"pixel_values": pixel_values, "labels": labels, "target_sequence":sample["text"]}


ds_train = ds_train.map(transform, remove_columns=["image","text"], batched=False)
ds_val = ds_val.map(transform, remove_columns=["image","text"])
ds_test = ds_test.map(transform, remove_columns=["image","text"])

# Cargar modelo pre-entrenado Donut (encoder = Swin, decoder = BART)
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# Expandir la capa de embeddings del decoder para los nuevos tokens
model.decoder.resize_token_embeddings(len(processor.tokenizer))

# Configurar el tamaño de entrada de imágenes (largo, ancho)
model.config.encoder.image_size = [CONST_TARGET_SIZE["height"], CONST_TARGET_SIZE["width"]] # (alto, ancho) ✅

print("patch_size:", model.config.encoder.patch_size)
print("window_size:", model.config.encoder.window_size)

# Configurar el token de inicio de generación (<s>) y pad
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(task_start)
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.config.decoder.max_length = MAX_LENGTH

print("Model Encoder Image Size:", model.config.encoder.image_size) # <-- Añadir verificación
print("Model Encoder Patch Size:", model.config.encoder.patch_size) # <-- Añadir verificación
print("Model Encoder Window Size:", model.config.encoder.window_size) # <-- Añadir verificación

# Verificar que estás usando el token correcto
decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s>")
print(f"🔧 Decoder start token ID: {decoder_start_token_id}")

training_args = TrainingArguments(
  output_dir="/content/drive/MyDrive/donut_project/models/donut-ticket-fiscal-v1",
  num_train_epochs=15,
  per_device_train_batch_size=2,
  per_device_eval_batch_size=2,
  learning_rate=1e-5,
  weight_decay=0.01,
  logging_steps=25,
  eval_steps=36,               # evalúa al final de cada época
  save_strategy="steps",
  save_steps=36,
  eval_strategy="steps",
  save_total_limit=3,
  fp16=True, # usar media precisión si la GPU lo soporta
  load_best_model_at_end=True,
  metric_for_best_model="eval_loss",
  warmup_steps=30,
  gradient_accumulation_steps=2,
  dataloader_pin_memory=False,
  remove_unused_columns=False,  # ✅ Importante
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

trainer.train()
# Asumiendo que `model` y `processor` son los objetos que usaste para entrenar
save_dir = "C:\\Users\\sanmi\\Documents\\Proyectos\\DonutModel\\models\\modelo-final-v0"
model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)

