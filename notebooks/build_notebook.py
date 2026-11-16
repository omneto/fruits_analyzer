"""Gera ``treino_frutas.ipynb``. Edite aqui e rode ``python notebooks/build_notebook.py``."""

from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md(
        """# Treino do detector de frutas e maturação

Este notebook treina um **YOLO11 nano** para encontrar **banana, laranja e tomate**
e dizer o estágio de maturação de cada fruta (**verde, amadurecendo, maduro, passado**).
No fim, ele gera o arquivo `frutas_yolo11n.onnx`, que a app usa na VPS sem GPU.

**Antes de rodar**

1. *Ambiente de execução > Alterar o tipo de ambiente de execução* > **GPU T4**.
2. Crie uma conta gratuita no [Roboflow](https://app.roboflow.com) e copie a chave da API
   (*Settings > API Keys*).
3. No Colab, abra o ícone de chave (**Secrets**) na barra lateral, crie o segredo
   `ROBOFLOW_API_KEY` com a chave e ative o acesso para este notebook.
4. *Ambiente de execução > Executar tudo*.

O treino completo leva cerca de 1 a 2 horas na T4, dependendo do tamanho dos datasets.

**Depois:** baixe `frutas_yolo11n.onnx` (a última célula oferece o download) e coloque o
arquivo em `models/` no repositório. A app passa a usar o modelo treinado sozinha."""
    ),
    code(
        """import os

# Teste rápido fora do Colab (dados sintéticos e CPU). Deixe False no Colab.
SMOKE_TEST = os.getenv("SMOKE_TEST") == "1"

EPOCHS = 2 if SMOKE_TEST else 100   # o treino para antes se parar de melhorar (PATIENCE)
PATIENCE = 20
IMGSZ = 320 if SMOKE_TEST else 640
BATCH = 4 if SMOKE_TEST else 16
BASE_MODEL = "yolo11n.pt"           # nano: o único que roda bem na VPS de 1 vCPU
SAVE_TO_DRIVE = False               # True para salvar uma cópia no Google Drive"""
    ),
    code(
        """import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    !nvidia-smi -L
    !pip install -q ultralytics roboflow onnx onnxslim
    if not Path("/content/fruits_analyzer").exists():
        !git clone -q https://github.com/omneto/fruits_analyzer.git /content/fruits_analyzer
    %cd /content/fruits_analyzer
    !git pull -q

sys.path.insert(0, str(Path.cwd()))
from training.datasets import CLASS_NAMES, DEFAULT_SOURCES, download_roboflow, merge_datasets, read_class_names, stats_table

WORK_DIR = Path("/content/work") if IN_COLAB else Path("work")
print("Classes do modelo:", CLASS_NAMES)"""
    ),
    md(
        """## 1. Datasets

Os datasets padrão são públicos no Roboflow Universe (licença CC BY 4.0). Cada um tem nomes de
classe próprios, que são convertidos para o padrão do projeto (`banana_maduro`, `tomate_verde`...).
O mapeamento fica em `training/datasets.py`.

Para trocar ou acrescentar um dataset, edite a lista abaixo. Classes fora do `class_map` são
deduzidas pelo nome (`unripe` → verde, `rotten` → passado...) ou descartadas."""
    ),
    code(
        """from training.datasets import DatasetSource

SOURCES = list(DEFAULT_SOURCES)

# Exemplo para acrescentar outro dataset de banana:
# SOURCES.append(DatasetSource(
#     name="banana2", fruit="banana",
#     roboflow_workspace="<workspace>", roboflow_project="<projeto>",
#     class_map={"green": "verde", "yellow": "maduro", "brown": "passado"},
# ))

for s in SOURCES:
    print(f"{s.fruit:8} {s.url}  ({s.license})")"""
    ),
    code(
        """RAW_DIR = WORK_DIR / "raw"
downloaded = []
if SMOKE_TEST:
    from training.synthetic import make_synthetic_sources
    downloaded = make_synthetic_sources(RAW_DIR)
else:
    from google.colab import userdata
    api_key = userdata.get("ROBOFLOW_API_KEY")
    for src in SOURCES:
        path = download_roboflow(src, api_key, RAW_DIR)
        downloaded.append((src, path))

for src, path in downloaded:
    print(f"{src.name}: classes originais = {read_class_names(path)}")"""
    ),
    code(
        """stats = merge_datasets(downloaded, WORK_DIR / "merged")
print(stats_table(stats))"""
    ),
    md(
        """Confira a tabela: classes com poucas caixas (menos de ~100) terão resultado fraco.
A laranja, por exemplo, só tem "verde" e "maduro" no dataset padrão."""
    ),
    md("## 2. Treino"),
    code(
        """from ultralytics import YOLO

RUNS = (WORK_DIR / "runs").resolve()  # caminho absoluto: a Ultralytics muda caminhos relativos
model = YOLO(BASE_MODEL)
model.train(
    data=stats["data_yaml"],
    epochs=EPOCHS,
    patience=PATIENCE,
    imgsz=IMGSZ,
    batch=BATCH,
    project=str(RUNS),
    name="frutas",
    exist_ok=True,
    workers=0 if SMOKE_TEST else 2,
    plots=not SMOKE_TEST,
)
BEST = Path(model.trainer.best)
print("Melhor modelo:", BEST)"""
    ),
    md("## 3. Avaliação"),
    code(
        """best = YOLO(str(BEST))
metrics = best.val(data=stats["data_yaml"], imgsz=IMGSZ, batch=BATCH, plots=not SMOKE_TEST, workers=0, project=str(RUNS), name="val", exist_ok=True)
print(f"mAP50 geral: {metrics.box.map50:.3f}   mAP50-95: {metrics.box.map:.3f}")
for i, cls_id in enumerate(metrics.box.ap_class_index):
    print(f"  {CLASS_NAMES[int(cls_id)]:<22} mAP50 = {metrics.box.ap50[i]:.3f}")"""
    ),
    code(
        """from IPython.display import Image as ShowImage, display

for name in ["confusion_matrix_normalized.png", "val_batch0_pred.jpg"]:
    for found in RUNS.glob(f"**/{name}"):
        display(ShowImage(filename=str(found), width=900))
        break"""
    ),
    md("## 4. Exportar para ONNX (para a VPS)"),
    code(
        """import shutil
from pathlib import Path

import onnx
from ultralytics import YOLO

# Funciona mesmo depois de reiniciar a sessão: procura o best.pt salvo em disco.
if "best" not in globals():
    found = sorted(Path.cwd().parent.glob("**/frutas/weights/best.pt")) or sorted(Path.cwd().glob("**/frutas/weights/best.pt"))
    if not found:
        raise FileNotFoundError("Nenhum best.pt encontrado: rode o treino (seção 2) antes de exportar.")
    BEST = found[-1]
    print("Usando", BEST)
    best = YOLO(str(BEST))
IMGSZ = globals().get("IMGSZ", 640)

onnx_path = best.export(format="onnx", imgsz=IMGSZ, simplify=True, opset=17)
MODEL_OUT = Path("models/frutas_yolo11n.onnx")
MODEL_OUT.parent.mkdir(exist_ok=True)
shutil.copy(onnx_path, MODEL_OUT)

# Confere as classes gravadas no arquivo (só com o pacote onnx, sem onnxruntime).
meta = {p.key: p.value for p in onnx.load(str(MODEL_OUT)).metadata_props}
print("Classes gravadas no modelo:", meta.get("names"))
print(f"Arquivo: {MODEL_OUT} ({MODEL_OUT.stat().st_size / 1e6:.1f} MB)")"""
    ),
    code(
        """import sys

if "google.colab" in sys.modules:
    from google.colab import files
    if globals().get("SAVE_TO_DRIVE"):
        from google.colab import drive
        drive.mount("/content/drive")
        dest = Path("/content/drive/MyDrive/fruits_analyzer")
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(MODEL_OUT, dest)
        shutil.copy(BEST, dest / "frutas_yolo11n.pt")
    files.download(str(MODEL_OUT))"""
    ),
    md(
        """## Próximo passo

Coloque `frutas_yolo11n.onnx` em `models/` no repositório e rode `docker compose up -d --build`
na VPS. A app detecta o modelo treinado, libera o tomate e passa a mostrar
"Maturação (modelo treinado)". Para voltar ao modelo antigo, apague o arquivo ou defina
`MODEL_PATH=/app/models/coco_yolo11n.onnx`.

Guarde também o `best.pt` (opção `SAVE_TO_DRIVE`): ele permite continuar o treino depois com
mais fotos, inclusive fotos suas anotadas no Roboflow."""
    ),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}
out = Path(__file__).with_name("treino_frutas.ipynb")
nbf.write(nb, out)
print(f"Notebook salvo em {out}")
