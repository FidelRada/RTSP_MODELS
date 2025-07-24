import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def predict_NuExtract(texts, template, batch_size=1, max_length=10_000, max_new_tokens=4_000):
    model_name = "numind/NuExtract-1.5-tiny"
    device = "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True, attn_implementation="eager").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    template = json.dumps(json.loads(template), indent=4)
    prompts = [f"""<|input|>\n### Plantilla sigue el tipo de los campos de la plantilla para extraer los datos de este texto de un carnet de identidad boliviano:\n{template}\n### Texto:\n{text}\n<|output|>""" for text in texts]

    outputs = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_encodings = tokenizer(batch_prompts, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(model.device)
            pred_ids = model.generate(**batch_encodings, max_new_tokens=max_new_tokens)
            outputs += tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    return [output.split("<|output|>")[1] for output in outputs]


'''
def predict_NuExtract(texts, template, batch_size=1, max_length=10_000, max_new_tokens=4_000):
    
    model_name = "numind/NuExtract-1.5-tiny"
    device = "cpu" #"cuda"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32, 
        trust_remote_code=True,
        attn_implementation="eager"
        ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    template = json.dumps(json.loads(template), indent=4)
    #prompts = [f"""<|input|>\n### Template:\n{template}\n### Text:\n{text}\n\n<|output|>""" for text in texts]
    
    prompts = [f"""<|input|>
### Plantilla sigue el tipo de los campos de la plantilla para extraer los datos de este texto de un carnet de identidad boliviano:
{template}
### Texto:
{text}
<|output|>""" for text in texts]
    
    outputs = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_encodings = tokenizer(batch_prompts, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(model.device)

            pred_ids = model.generate(**batch_encodings, max_new_tokens=max_new_tokens)
            outputs += tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    return [output.split("<|output|>")[1] for output in outputs]

text = ''
EN

DE AD

ESTADO PLURINACIONA E BOLIVIA CE
NERAL DE ID

SERIE SECCIÓN o
44333 43222 N? 9800914
NOMBRES
ANDRES FIDEL
APELLIDOS
RADA ROJAS
FECHA DE NACIMIENTO
04/02/1995
FECHA DE EMISIÓN FECHA DE EXPIRACIÓN
14/04/2025 14/04/2030

0605387-BO e
d

y

s
FIRMA DEL TITULAR


''
template = ''{
    "Nombre": "",
    "Apellido": ""
}''


prediction = predict_NuExtract([text], template)[0]
print(prediction)'''
