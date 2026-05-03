Fusão de Sensores para Detecção e Classificação de Drones

Este projeto explora o Drone Detection Dataset e implementa pipelines para preparação de dados, geração de anotações no formato YOLO, extração de atributos de imagem/áudio e avaliação de classificadores com fusão de sensores.
Os sensores considerados são:
IR: vídeo infravermelho;
V: vídeo visível;
AUDIO: sinais de áudio.
As classes globais usadas na etapa multiclasse são:
```text
AIRPLANE
BACKGROUND
BIRD
DRONE
HELICOPTER
```
---
1. Objetivo
   
O notebook `FUSAO.ipynb` tem quatro objetivos principais:
Explorar a estrutura do dataset compactado em `.zip`;
Organizar e resumir os arquivos por classe, modalidade e amostra;
Converter anotações de vídeo para o formato YOLO;
Avaliar métodos de fusão de sensores para classificação binária e multiclasse.
---
2. Estrutura geral do pipeline
   
O fluxo implementado no notebook é:
```text
Drone-detection-dataset-master.zip
        |
        v
Exploração do ZIP e geração de manifestos
        |
        v
Leitura de arquivos .mat, .xlsx e scripts MATLAB
        |
        v
Preparação dos dados de detecção YOLO
        |
        v
Extração de features IR, V e AUDIO
        |
        v
Classificação binária DRONE vs NON-DRONE
        |
        v
Classificação multiclasse com fusão de sensores
```
---
3. Configuração de caminhos
Antes de executar o notebook, ajuste os caminhos conforme o ambiente local.
Exemplos usados no notebook:
```python
ZIP_PATH = Path(r"C:\Users\Luis\Downloads\Drone-detection-dataset-master.zip")

DATA_ROOT = Path(
    r"C:\Users\Luis\Downloads\DroneDetectionDataset\Drone-detection-dataset-master\Data"
)

DETECT_ROOT = Path(
    r"C:\Users\Luis\Downloads\exploracao_drone_dataset\detector_training_data"
)

FEATURE_ROOT = Path(
    r"C:\Users\Luis\Downloads\exploracao_drone_dataset\fusion_3sensors_baseline"
)
```
---
4. Dependências
O notebook usa Python 3 e as seguintes bibliotecas principais:
```text
numpy
pandas
scikit-learn
matplotlib
Pillow
librosa
soundfile
scipy
openpyxl
```
Modelos opcionais:
```text
xgboost
lightgbm
catboost
```
Instalação sugerida:
```bash
pip install numpy pandas scikit-learn matplotlib pillow librosa soundfile scipy openpyxl
```
Para usar os modelos opcionais:
```bash
pip install xgboost lightgbm catboost
```
---
5. Exploração do dataset
A primeira etapa lê o arquivo `.zip` sem extraí-lo completamente e gera uma visão geral dos arquivos internos.
Saídas principais:
```text
exploracao_drone_dataset/
├── arquivos_no_zip.csv
├── pastas_no_zip.csv
└── estrutura_pastas.txt
```
Esses arquivos contêm:
lista completa de arquivos do ZIP;
extensão, tamanho e pasta de cada arquivo;
árvore de diretórios do dataset.
---
6. Manifestos e resumos do dataset
O notebook identifica automaticamente a modalidade de cada arquivo:
`audio`;
`video_ir`;
`video_visible`;
`label_mat`;
`spreadsheet`;
`matlab_script`;
`readme`;
`other`.
Também normaliza os nomes das amostras, por exemplo:
```text
V_DRONE_001.mp4       -> DRONE_001
IR_DRONE_001.mp4      -> DRONE_001
DRONE_001_LABELS.mat  -> DRONE_001
DRONE_001.wav         -> DRONE_001
```
Saídas geradas:
```text
exploracao_drone_dataset/
├── resumo_por_pasta_classe.csv
├── resumo_classes.csv
├── dataset_manifest_limpo.csv
├── resumo_classes_limpo.csv
├── resumo_pares_video_label.csv
└── arquivos_sem_par.csv
```
Esses arquivos permitem verificar:
número de arquivos por classe;
número de arquivos por modalidade;
presença de áudio, vídeo IR, vídeo visível e labels `.mat`;
amostras com vídeo sem label ou label sem vídeo.
---
7. Inspeção de arquivos `.mat`, `.xlsx` e scripts MATLAB
O notebook também inspeciona os arquivos de anotação e metadados do dataset.
Saídas:
```text
exploracao_drone_dataset/
├── resumo_conteudo_mat_direto_zip.csv
└── resumo_xlsx_direto_zip.txt
```
O arquivo `.xlsx` contém metadados como:
```text
SENSOR
CLASS
NUMBER
DISTANCE BIN
INTER BIN NUMBER
DRONE TYPE / INTERNET SOURCE
```
---
8. Preparação do dataset YOLO
A etapa de detecção converte as anotações dos CSVs `trainingData_IR.csv` e `trainingData_V.csv` para o formato YOLO.
Classes usadas:
```text
0: AIRPLANE
1: BIRD
2: DRONE
3: HELICOPTER
```
A conversão transforma caixas no formato MATLAB:
```text
x, y, width, height
```
para o formato YOLO normalizado:
```text
class_id x_center y_center width height
```
A divisão é feita por `sample_id`, evitando vazamento entre treino, validação e teste.
Proporção dos splits:
```text
train: 70%
val:   15%
test:  15%
```
Saídas:
```text
detector_training_data/
└── yolo_drone_detection/
    ├── annotations_long.csv
    ├── classes_summary.csv
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```
---
9. Extração de features
9.1. Features de imagem
Para os sensores IR e V, cada frame é convertido para escala de cinza, redimensionado para `32 x 32` e representado por:
pixels achatados;
histograma de intensidade;
histograma de magnitude do gradiente;
estatísticas globais da imagem.
As features são agregadas no nível da amostra por média e desvio-padrão.
9.2. Features de áudio
Para cada arquivo `.wav`, o notebook extrai:
log-mel spectrogram;
MFCC;
RMS;
zero-crossing rate;
spectral centroid;
spectral bandwidth;
spectral rolloff.
As features temporais são agregadas por média e desvio-padrão.
Saídas:
```text
fusion_3sensors_baseline/
├── features_IR.csv
├── features_V.csv
└── features_AUDIO.csv
```
---
10. Classificação binária: DRONE vs NON-DRONE
A primeira avaliação supervisionada trata o problema como classificação binária:
```text
DRONE      -> 1
NON-DRONE  -> 0
```
Combinações avaliadas:
```text
IR
V
AUDIO
IR+V
IR+AUDIO
V+AUDIO
IR+V+AUDIO
```
Modelos usados:
Regressão logística;
Random Forest.
Métricas calculadas:
accuracy;
balanced accuracy;
precision;
recall;
F1-score;
ROC-AUC;
matriz de confusão.
Saídas:
```text
fusion_3sensors_baseline/
├── combination_sizes.csv
├── fusion_metrics.csv
└── fusion_predictions.csv
```
---
11. Classificação multiclasse com fusão de sensores
A versão multiclasse avalia as seguintes classes globais:
```text
AIRPLANE
BACKGROUND
BIRD
DRONE
HELICOPTER
```
Como nem todos os sensores contêm todas as classes, o protocolo usa fusão tolerante a modalidades ausentes. O dataset combinado inclui indicadores de presença de sensores, como:
```text
has_IR
has_V
has_AUDIO
```
Combinações avaliadas:
```text
IR
V
AUDIO
IR+V
IR+AUDIO
V+AUDIO
IR+V+AUDIO
```
Estratégias avaliadas:
unimodal;
early fusion por concatenação;
late fusion por média;
late fusion ponderada por predições OOF;
late fusion por stacking.
Modelos usados:
Dummy prior;
Logistic Regression L2;
Linear SVM;
Random Forest;
Extra Trees;
HistGradientBoosting;
MLP.
Modelos opcionais, se habilitados:
XGBoost;
LightGBM;
CatBoost.
Métricas calculadas:
balanced accuracy;
macro-F1 calculado apenas sobre classes presentes no teste;
macro-F1 global para auditoria;
weighted-F1;
matriz de confusão do melhor modelo.
Saídas:
```text
multiclass_fusion_audio_results_fixed/
├── combination_sizes_multiclass.csv
├── metrics_all_repeats_multiclass.csv
├── metrics_summary_multiclass.csv
├── predictions_all_multiclass.csv
├── global_classes.json
└── confusion_matrix_best_model.png
```
---
12. Resultados observados no notebook
Na avaliação multiclasse corrigida, os melhores resultados médios observados foram obtidos principalmente com a combinação `IR+V+AUDIO`.
Exemplo de melhores configurações:
```text
IR+V+AUDIO | early_concat_missing | hist_gb
Balanced Accuracy média: 0.9084
Macro-F1 médio:          0.9072

IR+V+AUDIO | late_stacking | logreg_meta
Balanced Accuracy média: 0.9072
Macro-F1 médio:          0.9084

IR+V+AUDIO | early_concat_missing | extra_trees
Balanced Accuracy média: 0.9018
Macro-F1 médio:          0.9032
```
Resumo de amostras por combinação:
```text
IR:          365 amostras
V:           285 amostras
AUDIO:        90 amostras
IR+V:        371 amostras
IR+AUDIO:    395 amostras
V+AUDIO:     315 amostras
IR+V+AUDIO:  401 amostras
```
---
13. Observações metodológicas
As combinações avaliadas não representam exatamente a mesma tarefa de classificação, pois cada sensor cobre subconjuntos diferentes do espaço global de classes.
Por isso, os resultados devem ser interpretados como uma avaliação de fusão em cenário heterogêneo e tolerante a modalidades ausentes, e não como uma comparação perfeitamente pareada entre sensores.
Uma limitação importante é que os indicadores de presença de modalidade podem codificar padrões de disponibilidade do dataset. Avaliações futuras devem incluir:
ablação sem indicadores `has_IR`, `has_V` e `has_AUDIO`;
validação em dados estritamente pareados;
calibração de confiança;
análise por classe;
avaliação em base externa.
---
14. Como executar
Baixe ou mantenha disponível o arquivo:
```text
Drone-detection-dataset-master.zip
```
Ajuste os caminhos no notebook `FUSAO.ipynb`.
Execute as células na seguinte ordem:
```text
1. Exploração do ZIP
2. Resumos e manifestos
3. Inspeção de .mat e .xlsx
4. Geração do dataset YOLO
5. Extração de features IR, V e AUDIO
6. Classificação binária
7. Classificação multiclasse corrigida
```
Verifique os resultados nos diretórios:
```text
exploracao_drone_dataset/
detector_training_data/yolo_drone_detection/
fusion_3sensors_baseline/
multiclass_fusion_audio_results_fixed/
```
---
15. Reprodutibilidade
O notebook usa semente fixa:
```python
RANDOM_STATE = 42
```
Na etapa multiclasse, o número de repetições é controlado por:
```python
N_REPEATS = 10
```
Para testes rápidos, recomenda-se usar:
```python
N_REPEATS = 3
```
Para resultados finais, recomenda-se:
```python
N_REPEATS = 10
```
ou
```python
N_REPEATS = 20
```
---
16. Organização recomendada do repositório
```text
.
├── FUSAO.ipynb
├── README.md
├── data/
│   └── Drone-detection-dataset-master.zip
├── outputs/
│   ├── exploracao_drone_dataset/
│   ├── yolo_drone_detection/
│   ├── fusion_3sensors_baseline/
│   └── multiclass_fusion_audio_results_fixed/
└── requirements.txt
```
Exemplo de `requirements.txt`:
```text
numpy
pandas
scikit-learn
matplotlib
pillow
librosa
soundfile
scipy
openpyxl
```
Para modelos opcionais:
```text
xgboost
lightgbm
catboost
```
---
17. Licença e citação
Verifique a licença original do Drone Detection Dataset antes de redistribuir os dados ou publicar resultados derivados.
Ao usar este código em trabalhos acadêmicos, recomenda-se citar:
a fonte original do dataset;
as bibliotecas utilizadas;
a metodologia de fusão de sensores adotada.
