import json

REQUIRED_FIELDS = [
    "CUIT_EMISOR", "REGIMEN", "FECHA", "NUMERO_DOCUMENTO",
    "PRIMER_COMPROBANTE", "ULTIMO_COMPROBANTE", 
    "GRAVADO", "NO_GRAVADO", "EXENTO", "DESCUENTOS",
    "COMP_GENERADOS", "COMP_CANCELADOS", "IVA", "TOTAL"
]

# Cargar datos
with open("C:\\Users\\sanmi\\Documents\\Proyectos\\DonutModel\\train.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

sample_gt = data[0]["ground_truth"]

print("=== ANÁLISIS DEL ORDEN ===")
print(f"Campos en sample: {list(sample_gt.keys())}")
print(f"Orden alfabético: {sorted(sample_gt.keys())}")
print(f"Orden alfabético reverso (TU CÓDIGO): {sorted(sample_gt.keys(), reverse=True)}")
print(f"Orden lógico propuesto: {REQUIRED_FIELDS}")

print(f"\n=== ¿POR QUÉ EL ORDEN ALFABÉTICO REVERSO ES MALO? ===")
print("Tu orden actual:", sorted(sample_gt.keys(), reverse=True))
print("Problema: Los campos más importantes (como TOTAL, CUIT_EMISOR) aparecen")
print("en posiciones inconsistentes, confundiendo al modelo.")

print(f"\n=== ORDEN LÓGICO MEJOR ===")
print("1. Identificación: CUIT_EMISOR, REGIMEN, FECHA, NUMERO_DOCUMENTO")
print("2. Comprobantes: PRIMER_COMPROBANTE, ULTIMO_COMPROBANTE")  
print("3. Montos: GRAVADO, NO_GRAVADO, EXENTO, DESCUENTOS")
print("4. Contadores: COMP_GENERADOS, COMP_CANCELADOS")
print("5. Final: IVA, TOTAL")