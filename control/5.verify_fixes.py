from transformers import DonutProcessor
import json

# Verificar que los cambios funcionan
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

with open("C:\\Users\\sanmi\\Documents\\Proyectos\\DonutModel\\train.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# Simular nueva configuración
sample_gt = data[0]["ground_truth"]
REQUIRED_FIELDS = ["CUIT_EMISOR", "REGIMEN", "FECHA", "NUMERO_DOCUMENTO", "PRIMER_COMPROBANTE", "ULTIMO_COMPROBANTE", "GRAVADO", "NO_GRAVADO", "EXENTO", "DESCUENTOS", "COMP_GENERADOS", "COMP_CANCELADOS", "IVA", "TOTAL"]

seq = "<s>"
for k in REQUIRED_FIELDS:
    if k in sample_gt:
        seq += f"<s_{k}>{sample_gt[k]}</s_{k}>"
seq += "</s>"

tokens = processor.tokenizer(seq, add_special_tokens=False, max_length=300)
print(f"✅ Nueva longitud: {len(tokens['input_ids'])} tokens")
print(f"✅ ¿Cabe en 300? {'SÍ' if len(tokens['input_ids']) <= 300 else 'NO'}")
print(f"✅ Secuencia completa: {seq[:100]}...")