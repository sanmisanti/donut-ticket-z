import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from PIL import Image
import json
from datasets import Dataset, Features, Image, Value



NEW_SPECIAL_TOKENS = [] # Lista global de nuevos tokens
CONST_TARGET_SIZE = {"height": 1536, "width": 384}
REQUIRED_FIELDS = [
    "CUIT_EMISOR", "REGIMEN", "FECHA", "NUMERO_DOCUMENTO",
    "PRIMER_COMPROBANTE", "ULTIMO_COMPROBANTE",
    "GRAVADO", "NO_GRAVADO", "EXENTO", "DESCUENTOS",
    "COMP_GENERADOS", "COMP_CANCELADOS", "IVA", "TOTAL"
]


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
    """
     Convierte un objeto JSON de anotaciones en una secuencia Donut con tokens
    <s_KEY>valor</s_KEY>.
     new_special_tokens será llenado con los tokens usados.
     """
    if isinstance(obj, dict):
        output = ""
        for k in sorted(obj.keys(), reverse=True):
            # Agregar tokens especiales para esta clave si no existen
            start_token = f"<s_{k}>"
            end_token = f"</s_{k}>"
            if start_token not in new_special_tokens:
                new_special_tokens.extend([start_token, end_token])
            # Llamada recursiva para soportar objetos anidados (no aplicará aquí si los valores son strings/números)
            output += start_token + json2token(obj[k], new_special_tokens) + end_token
        return output
    elif isinstance(obj, list):
        # En caso de listas (por si acaso), unir con separador <sep/>
        return "<sep/>".join(json2token(item, new_special_tokens) for item in obj)
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
processor.feature_extractor.do_align_long_axis = False


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
  pixel_values = processed.pixel_values

  print("Procesado shape:", pixel_values.shape)

  inputs = processor.tokenizer(sample["text"],add_special_tokens=False,padding="max_length",truncation=True,max_length=86,return_tensors="pt")
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
model.config.decoder.max_length = 86

print("Model Encoder Image Size:", model.config.encoder.image_size) # <-- Añadir verificación
print("Model Encoder Patch Size:", model.config.encoder.patch_size) # <-- Añadir verificación
print("Model Encoder Window Size:", model.config.encoder.window_size) # <-- Añadir verificación

training_args = Seq2SeqTrainingArguments(
  output_dir="C:\\Users\\sanmi\\Documents\\Proyectos\\DonutModel\\models\\donut-ticket-fiscal-v0",
  num_train_epochs=10,
  per_device_train_batch_size=2,
  per_device_eval_batch_size=2,
  learning_rate=3e-5,
  weight_decay=0.01,
  predict_with_generate=True,
  generation_max_length=86,
  generation_num_beams=1,
  logging_steps=50,
  eval_steps=72,               # evalúa al final de cada época
  save_strategy="steps",
  save_steps=72,
  eval_strategy="steps",
  save_total_limit=2,
  fp16=True, # usar media precisión si la GPU lo soporta
  load_best_model_at_end= True,
  metric_for_best_model="eval_loss",
  warmup_steps=50,
  gradient_accumulation_steps=2  # duplica el batch size efectivo
)

def collate_fn(batch):
    pixel_list = []
    label_list = []

    for x in batch:
        # Obtener el objeto y desempaquetarlo si viene en lista
        pv = x["pixel_values"]
        lb = x["labels"]

        # Si vienen en listas (incluso anidadas), convertirlos a tensor
        if not torch.is_tensor(pv):
            pv = torch.tensor(pv)
        if not torch.is_tensor(lb):
            lb = torch.tensor(lb)

        pixel_list.append(pv)
        label_list.append(lb)

    # Ahora sí apilamos en un batch tensorial
    pixel_values = torch.stack(pixel_list)
    labels       = torch.stack(label_list)

    return {"pixel_values": pixel_values, "labels": labels}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    tokenizer=processor.tokenizer,  # todavía útil para logging/generación
    data_collator=collate_fn        # solucionamos el error aquí
)

trainer.train()
# Asumiendo que `model` y `processor` son los objetos que usaste para entrenar
save_dir = "C:\\Users\\sanmi\\Documents\\Proyectos\\DonutModel\\models\\modelo-final-v0"
model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)

