import json
from transformers import DonutProcessor

# Tu configuración actual
REQUIRED_FIELDS = [
    "CUIT_EMISOR", "REGIMEN", "FECHA", "NUMERO_DOCUMENTO",
    "PRIMER_COMPROBANTE", "ULTIMO_COMPROBANTE",
    "GRAVADO", "NO_GRAVADO", "EXENTO", "DESCUENTOS",
    "COMP_GENERADOS", "COMP_CANCELADOS", "IVA", "TOTAL"
]

def json2token_current(obj, new_special_tokens):
    """Tu función actual (con orden alfabético reverso)"""
    if isinstance(obj, dict):
        output = ""
        for k in sorted(obj.keys(), reverse=True):  # PROBLEMA: orden alfabético
            start_token = f"<s_{k}>"
            end_token = f"</s_{k}>"
            if start_token not in new_special_tokens:
                new_special_tokens.extend([start_token, end_token])
            output += start_token + str(obj[k]) + end_token
        return output
    else:
        return str(obj)

def json2token_fixed(obj, new_special_tokens):
    """Función corregida (orden lógico)"""
    if isinstance(obj, dict):
        output = ""
        for k in REQUIRED_FIELDS:  # ORDEN FIJO Y LÓGICO
            if k in obj:
                start_token = f"<s_{k}>"
                end_token = f"</s_{k}>"
                if start_token not in new_special_tokens:
                    new_special_tokens.extend([start_token, end_token])
                output += start_token + str(obj[k]) + end_token
        return output
    else:
        return str(obj)

# Cargar un ejemplo de tus datos
with open("C:\\Users\\sanmi\\Documents\\Proyectos\\DonutModel\\train.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

sample_gt = data[0]["ground_truth"]
print("=== DATOS DE EJEMPLO ===")
print(f"Ground truth: {sample_gt}")

# Generar secuencias con ambos métodos
tokens_current = []
tokens_fixed = []

seq_current = "<s>" + json2token_current(sample_gt, tokens_current) + "</s>"
seq_fixed = "<s>" + json2token_fixed(sample_gt, tokens_fixed) + "</s>"

print(f"\n=== SECUENCIA ACTUAL (orden alfabético reverso) ===")
print(f"Longitud: {len(seq_current)} caracteres")
print(f"Secuencia: {seq_current}")

print(f"\n=== SECUENCIA CORREGIDA (orden lógico) ===")
print(f"Longitud: {len(seq_fixed)} caracteres")  
print(f"Secuencia: {seq_fixed}")

# Tokenizar con DonutProcessor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

# Probar diferentes max_length
for max_len in [86, 100, 128, 150]:
    tokens_current_ids = processor.tokenizer(seq_current, add_special_tokens=False, truncation=True, max_length=max_len)
    tokens_fixed_ids = processor.tokenizer(seq_fixed, add_special_tokens=False, truncation=True, max_length=max_len)
    
    print(f"\n=== TOKENIZACIÓN max_length={max_len} ===")
    print(f"Secuencia actual: {len(tokens_current_ids['input_ids'])} tokens")
    print(f"Secuencia corregida: {len(tokens_fixed_ids['input_ids'])} tokens")
    print(f"¿Se truncó secuencia actual? {'❌ SÍ' if len(tokens_current_ids['input_ids']) == max_len else '✅ NO'}")
    print(f"¿Se truncó secuencia corregida? {'❌ SÍ' if len(tokens_fixed_ids['input_ids']) == max_len else '✅ NO'}")