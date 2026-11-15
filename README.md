# Analisador de Frutas

Protótipo de app web que conta as frutas de uma foto e classifica cada uma por
**maturação** (verde, amadurecendo, maduro, passado) e **tamanho** (pequeno,
médio, grande). Funciona no PC, tablet e celular, e roda em uma VPS pequena
(1 vCPU, 2 GB de RAM) com Docker.

![Tela do protótipo](docs/screenshot.png)

## Modelos

A app escolhe o modelo sozinha ao iniciar:

| Modelo | Arquivo | Frutas | Maturação |
|--------|---------|--------|-----------|
| Treinado (etapa 2) | `models/frutas_yolo11n.onnx` | banana, laranja, tomate | vem do modelo |
| COCO pré-treinado (etapa 1) | `models/coco_yolo11n.onnx` | banana, laranja | regras de cor |

O modelo treinado é usado quando o arquivo existe. Para forçar outro, defina
`MODEL_PATH`.

- **Contagem:** YOLO11n exportado para ONNX (cerca de 10 MB). Roda só com
  ONNX Runtime, sem PyTorch.
- **Maturação:** com o modelo treinado, cada fruta já vem classificada. Com o
  COCO, regras de cor em HSV no centro de cada fruta (`backend/app/ripeness.py`).
- **Tamanho:** relativo às outras frutas da mesma foto
  (`backend/app/size.py`). Uma foto sozinha não informa o tamanho real.
- **Modo detalhado:** divide a imagem em blocos para achar frutas pequenas em
  fotos de árvore ou drone. É cerca de 5 vezes mais lento.

## Treinar o modelo (Google Colab)

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omneto/fruits_analyzer/blob/main/notebooks/treino_frutas.ipynb)

Abra [`notebooks/treino_frutas.ipynb`](notebooks/treino_frutas.ipynb) no Colab
pelo botão acima e siga as instruções da primeira célula. Resumo:

1. Ambiente com **GPU T4**.
2. Chave gratuita do Roboflow no segredo `ROBOFLOW_API_KEY`.
3. *Executar tudo*. O notebook baixa os datasets, unifica as classes, treina,
   mostra as métricas e oferece o download de `frutas_yolo11n.onnx`.
4. Coloque o arquivo em `models/` e rode `docker compose up -d --build`.

Datasets padrão (Roboflow Universe, CC BY 4.0), configurados em
`training/datasets.py`:

| Fruta | Dataset | Classes originais |
|-------|---------|-------------------|
| Banana | [banana-ripening-process](https://universe.roboflow.com/fruit-ripening/banana-ripening-process) | freshunripe, unripe, freshripe, ripe, overripe, rotten |
| Laranja | [orange_ripeness](https://universe.roboflow.com/yashas/orange_ripeness) | unripe_orange, ripe_orange |
| Tomate | [tomatoes-growth-by-laboro](https://universe.roboflow.com/tomatoes-dectection-riped/tomatoes-growth-by-laboro) | b/l_green, b/l_half_ripened, b/l_fully_ripened |

O notebook também pode ser testado sem GPU nem internet, com dados sintéticos:
`SMOKE_TEST=1 jupyter nbconvert --execute notebooks/treino_frutas.ipynb`.
Para editar o notebook, altere `notebooks/build_notebook.py` e rode-o.

## Limitações conhecidas

- O modelo COCO costuma marcar um **cacho de bananas como uma só caixa**, e
  folhas dentro da caixa puxam a cor para "verde". O modelo treinado resolve
  as duas coisas na medida em que os datasets tiverem fotos parecidas com as
  suas.
- Os datasets padrão têm poucas fotos de árvore inteira e de drone. Para esse
  uso, vale anotar algumas fotos próprias no Roboflow e incluí-las no treino.
- O dataset de laranja só tem "verde" e "maduro".

## Rodar com Docker

```bash
docker compose up -d --build
# abra http://localhost:8000
```

O container usa cerca de 170 MB de RAM e analisa uma foto em 0,1 a 1 s
(medido com limite de 1 CPU). As análises são feitas uma por vez.

Na VPS de 2 GB, recomenda-se criar swap antes do build:

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Configurações por variável de ambiente (`docker-compose.yml`):
`CONF_THRESHOLD`, `IOU_THRESHOLD`, `MAX_IMAGE_SIDE`, `MAX_UPLOAD_MB`,
`ONNX_THREADS`, `MODEL_PATH`.

## Desenvolvimento local

```bash
cd backend
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload     # http://localhost:8000
pytest
```

## API

- `GET /api/fruits`: frutas e disponibilidade.
- `POST /api/analyze` (multipart): `file` (imagem), `fruit` (`banana`,
  `laranja`), `tiled` (`true`/`false`). Retorna total, contagem por maturação e
  tamanho, cada detecção e a imagem anotada em base64.
- Documentação interativa em `/docs`.

## Estrutura

```
backend/app/     API (FastAPI), detector ONNX, regras de cor e tamanho
backend/tests/   testes (pytest)
training/        preparação dos datasets para o treino
notebooks/       notebook do Colab
frontend/        app web (HTML, CSS e JS, sem build)
models/          modelos ONNX
scripts/         utilitários (exportar modelo)
```

## Próximas etapas

3. Cadastro dinâmico de novas frutas a partir de fotos de referência.
4. Relatórios, histórico e guia de deploy completo.

## Licença do modelo

O YOLO11 é da Ultralytics, sob licença **AGPL-3.0**. Usar os pesos ou o código
deles num serviço web exige disponibilizar o código-fonte do serviço, ou obter
uma licença comercial da Ultralytics.
