# 0) Configura tu dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Carga processor + modelo desde la misma carpeta
save_dir = "/content/drive/MyDrive/donut_project/dataset/modelo-final-10epochs"
processor = DonutProcessor.from_pretrained(save_dir)
model     = VisionEncoderDecoderModel.from_pretrained(save_dir)
model.to(device)

# 2) Función de predicción corregida
def predict_debug(sample):
    # 1) Carga y preprocesa la imagen
    img_path = f"/content/drive/MyDrive/donut_project/dataset/train_images/{sample['file_name']}"
    image = Image.open(img_path).convert("RGB")
    # Inspeccionar antes del procesamiento
    print("Original size:", image.size)  # (ancho, alto)

    # Procesar
    processed = processor(image, return_tensors="pt")
    pixel_values = processed.pixel_values.to(device)

    # Inspeccionar después
    print("Procesado shape:", pixel_values.shape)  # (1, 3, H, W)

    # 2) Generación
    decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s>")
    eos_token_id           = processor.tokenizer.eos_token_id
    outputs = model.generate(
        pixel_values=pixel_values,
        decoder_start_token_id=decoder_start_token_id,
        max_length=model.config.decoder.max_length,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=eos_token_id,
        no_repeat_ngram_size=2
    )

    # 3) Decodifica **sin** eliminar los tokens especiales
    raw_seq = processor.tokenizer.decode(outputs[0], skip_special_tokens=False)
    print("=== Secuencia RAW ===")
    print(raw_seq)
    print("=====================")

    # 4) Aplica token2json sobre esa secuencia cruda
    json_out = processor.token2json(raw_seq)
    return json_out

# 3) Prueba con un ejemplo
sample = {"file_name": "archivo_148.jpg"}
pred = predict_debug(sample)
print(pred)