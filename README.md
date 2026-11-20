<p align="center"><img src="assets/logo.png" alt="InoSlice" width="160"></p>

# InoSlice

Extrai recortes (crops) de imagens a partir de anotações de detecção de objetos (**YOLO** ou **COCO**) e monta um dataset de **classificação** pronto para treino, no formato `ImageFolder` do PyTorch.

![InoSlice](docs/screenshot.png)

## Funcionalidades

- **Entradas:** imagens `.jpg .jpeg .png .bmp .tif .tiff` com anotações YOLO (`.txt`) ou COCO (`.json`). O arquivo de classes (`classes.txt`, `*.names` ou `data.yaml`) é opcional.
- **Margem de contexto:** aumenta a caixa em uma porcentagem de cada lado, sem passar da borda da imagem.
- **Filtros:** descarta caixas menores que um tamanho mínimo (padrão 32×32 px) e detecções abaixo de uma confiança mínima.
- **Modos de recorte:** retangular (proporção original) ou quadrado com *letterbox* cinza ou preto, que não distorce o objeto.
- **Redimensionamento opcional:** 224, 256, 299, 384, 448, 512 px ou um tamanho personalizado.
- **Pré-visualização:** mostra a caixa original, a caixa com margem e o recorte final em tamanho real, exatamente como será salvo, com filtro por classe.
- **Rastreabilidade:** um `metadata_manifest.csv` liga cada recorte à imagem de origem (coluna `group_id`). Assim dá para dividir treino e teste sem que recortes da mesma imagem caiam nos dois lados (*data leakage*).
- **Resumo:** o `harvest_summary.json` traz totais por classe, descartes, tempo de execução e os parâmetros usados.
- **Desempenho:** o processamento roda em várias threads, mostra o progresso e pode ser cancelado.
- **Modo linha de comando:** roda sem interface, para servidores e pipelines.

### Saída gerada

```text
<pasta de saída>/
├── dataset_crops/
│   ├── abelha/
│   │   └── campo_000_crop_000.jpg
│   └── formiga/ ...
├── metadata_manifest.csv
└── harvest_summary.json
```

## Como rodar

Requer **Python 3.10+**.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

### Interface gráfica

```bash
python main.py
```

1. Escolha ou **arraste** a pasta de imagens. A pasta de labels e a pasta de saída são sugeridas automaticamente e as anotações carregam sozinhas.
2. Ajuste a margem, o formato, o tamanho de saída e os filtros. A pré-visualização mostra o recorte em **tamanho real**, então dá para ver o efeito do redimensionamento.
3. Clique em **Extrair recortes** (ou `Ctrl+Enter`).

Atalhos: `←`/`→` navegam entre as detecções (com a pré-visualização em foco), `R` sorteia uma detecção, `Ctrl+L` mostra os logs.

### Linha de comando

```bash
python main.py --images dados/images --labels dados/labels --classes dados/data.yaml --output saida --padding 15 --mode letterbox --target-size 224
```

Veja todas as opções com `python main.py --help`.

## Testes

```bash
python -m pytest -v
```

## Gerar o executável

No Windows, com o ambiente virtual ativo:

```bash
python build_exe.py
```

O executável sai em `dist\InoSlice.exe`, como arquivo único e sem janela de console. Opções:

| Opção | Efeito |
|---|---|
| `--onedir` | Gera uma pasta `dist\InoSlice\` em vez de um arquivo único (abre mais rápido) |
| `--console` | Mantém o console visível, útil para depuração |
| `--no-clean` | Reaproveita o cache do build anterior |

O PyInstaller não faz compilação cruzada: para ter o `.exe`, rode o script no Windows. Em Linux ou macOS ele gera o binário da própria plataforma.

## Estrutura

```text
src/core/    # leitura das anotações, geometria dos recortes e exportação (não depende do Qt)
src/gui/     # interface PySide6
src/utils/   # logs e validações
tests/       # testes pytest
assets/      # logo e ícone
main.py      # ponto de entrada (interface ou linha de comando)
build_exe.py # empacotamento com PyInstaller
```
