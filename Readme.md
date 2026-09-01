# Cam-Security

Sistema de monitoramento em tempo real via visão computacional. Detecta pessoas, analisa comportamento (colisões, socos, poses proibidas), reidentifica indivíduos e gera evidências de vídeo/imagem com retenção automática.

---

# PARTE 1 — DOCUMENTAÇÃO TEÓRICA

## 1. O problema que resolve

Monitoramento manual por câmeras é cansativo, ineficiente e perde eventos importantes. Operadores de segurança têm limite de atenção e costumam reagir tarde a incidentes.

Cam-Security automatiza esse trabalho: recebe o vídeo, identifica cada pessoa na cena, acompanha sua movimentação, classifica seu comportamento e dispara alertas com evidências concretas (vídeo + snapshot + metadados) no momento do incidente.

## 2. Conceitos fundamentais

### 2.1 Pipeline de visão computacional para monitoramento

Um pipeline de segurança típico em CV segue essa ordem lógica:

1. **Aquisição de imagem** — frames chegam de câmera ou arquivo.
2. **Detecção de objetos** — localiza *o que* e *onde* está cada pessoa no frame.
3. **Tracking de múltiplos objetos (MOT)** — atribui IDs únicos e mantém a continuidade de identidade entre frames.
4. **Extração de atributos** — pose, face, embedding, velocidade.
5. **Classificação de comportamento / eventos** — regras ou modelos decidem se algo é suspeito.
6. **Decisão / auditoria** — dispara alerta, grava evidência, envia notificação.

Cam-Security implementa essas 6 etapas com componentes desacoplados.

### 2.2 Detecção de pessoas: Object Detection com YOLOv8

**YOLO (You Only Look Once)** é uma arquitetura *single-stage*: prediz bounding boxes e classes em uma única passagem pela rede neural, o que a torna rápida o suficiente para FPS em tempo real.

Usamos **YOLOv8n (nano)** — o menor modelo da família (apenas ~3.2M parâmetros). Esse tradeoff é deliberado: em troca de um pouco menos de acurácia, ganhamos velocidade para rodar em CPU comum junto com MediaPipe e detecção facial.

- **Entrada**: frame RGB 640x480.
- **Classe filtrada**: apenas `class 0` (pessoa). Ignora carros, animais, etc.
- **Limiar de confiança**: `0.50` (50%). Caixas abaixo disso são descartadas.
- **NMS (Non-Maximum Suppression)**: pós-processamento com IoU threshold `0.45` que remove caixas sobrepostas redundantes de uma mesma pessoa.
- **Fallback**: se YOLO não carregar (ex: ausência do modelo `.pt`), o detector usa *bounding boxes inferidas dos landmarks de pose* (retângulo que engloba todos os pontos de pose + margem de 20px). Nunca fica 100% sem detecção.

### 2.3 Tracking: manter a identidade temporal

O MOT (Multi-Object Tracking) responde à pergunta: *essa pessoa no frame N é a mesma do frame N-1?*

Usamos um tracking híbrido por **IoU + distância centróide normalizada + ReID facial de fallback**:

1. **Matching por IoU** (Intersection over Union): se a nova caixa de detecção sobrepõe a anterior acima de `0.15`, assume-se ser o mesmo objeto.
2. **Matching por distância relativa**: se IoU não bateu, mas a distância entre os centros das caixas é menor que 40% da altura média da pessoa, ainda assim é considerado match (útil para pessoas com movimento rápido ou oclusão parcial).
3. **Score ponderado**: `score = IoU + max(0, 1 - dist/altura)`. Combina as duas métricas, depois ordena do melhor para o pior.
4. **Guloso (greedy) assignment**: atribui os matches pegando do maior score para o menor, sem reusar IDs nem caixas.
5. **Fallbacks**:
   - Boxes restantes sem match tentam **Face ReID** (corta a pessoa, extrai embedding facial, consulta o storage de pessoas conhecidas).
   - Se não houver match facial, é declarado um **novo ID** via `IDManager.get_next_id()`.
6. **TTL de track**: inativo por mais de 2 segundos → some da lista de candidatos; inativo por 5 minutos (300s) → expira completamente (exceto se o track estiver marcado como `triggered`, ou seja, participou de um evento recente).

### 2.4 Estimativa de Pose com MediaPipe Pose Landmarker

**Pose estimation** é a tarefa de predizer a localização de *keypoints* anatômicos (articulações, cabeça, mãos, pés) em 2D/3D.

Usamos **MediaPipe Tasks — Pose Landmarker** em modo `RunningMode.VIDEO`. A biblioteca Google entrega landmarks com **33 keypoints COCO-style** (nariz, ombros, cotovelos, pulsos, quadris, joelhos, tornozelos) e atributos extras como `visibility` (confiança de que o ponto não está ocluído) e `presence`.

Detalhes importantes:

- **Pool de landmarkers por track**: cada ID de pessoa tem sua própria instância de `PoseLandmarker`. Isso evita conflitos internos de estado do modelo e garante estabilidade temporal dos landmarks. Quando um track expira, seu landmarker é fechado.
- **Rodar no crop da pessoa, não no frame inteiro**: a detecção YOLO dá a box; fazemos um recorte expandido (18% de padding nas laterais, 20% extra no topo para cabeças) e rodamos pose *apenas na região da pessoa*. Resultado: landmarks muito mais precisos, menor uso de CPU.
- **Mapeamento inverso (crop → frame)**: landmarks são retornados em coordenadas normalizadas [0..1] do crop, convertemos de volta para o sistema de coordenadas do frame completo para desenho e cálculos.
- **PoseHold com TTL 0.35s**: suaviza flickering momentâneo de landmarks — se um frame falhar em detectar pose, reaproveita o último resultado por até 350ms.
- **Confiança mínima de visibilidade `0.30`**: landmarks abaixo desse limiar são ignorados no desenho do esqueleto e em cálculos geométricos.

### 2.5 Detecção de comportamento: lógica baseada em regras heurísticas

Nenhum modelo adicional de classificação de ação. O sistema usa **física simples + geometria** derivada dos landmarks e das boxes. Isso dá interpretabilidade e zero custo extra de inferência.

#### 2.5.1 Velocidade
Velocidade do centro da bounding box (ou do pulso direito, para detecção de soco) em pixels/segundo:
```
v = distância_px(pos_atual, pos_anterior) / delta_tempo
```
Classes de velocidade na risk score: `< 150 px/s` (baixa), `150–300 px/s` (média), `> 300 px/s` (alta).

#### 2.5.2 Proximidade entre pessoas
Distância normalizada: dividimos a distância euclidiana entre os centros pela **altura média das duas caixas** (isso escala com a distância da câmera). Limiar:
- `< 1.0` → **PERTO**
- `1.0 – 3.0` → **MÉDIO**
- `> 3.0` → **LONGE**

#### 2.5.3 Colisão
IoU > `0.05` entre duas bounding boxes de pessoas distintas. Esse limiar baixo captura toques/aproximações fortes sem depender de oclusão total.

#### 2.5.4 Soco / ataque rápido
Quatro condições devem ser verdadeiras **ao mesmo tempo**:
1. `pessoas na cena >= 2` (nunca dispara para pessoa sozinha — evita falsos positivos com alguém se espreguiçando)
2. Pessoas estão **PERTO ou em colisão**
3. **Velocidade do pulso direito >= 500 px/s** (movimento rápido do braço)
4. **Ângulo do cotovelo >= 150°** (braço quase estendido — posição final de soco)

O alerta só dispara depois de **3 detecções consecutivas** cumulativas no contador por track. 1 ou 2 picos isolados são descartados (filtro digital tipo "contador de debounce").

#### 2.5.5 Ângulo do braço
Cálculo geométrico com produto escalar. Dados 3 pontos (ombro `a`, cotovelo `b`, pulso `c`):
```
ba = a - b;  bc = c - b
ângulo = arccos( (ba · bc) / (|ba| * |bc|) )
```
Resultado em graus. 180° = braço totalmente reto; 90° = dobrado em L.

#### 2.5.6 Poses proibidas (com janela de confirmação)

`PoseEstimator` exige que a condição seja verdadeira por **N frames consecutivos** (padrão N=5). Cada par `(track_id, pose_name)` tem um contador independente; quebrar a sequência zera o contador. Evita falsos positivos de frames instantâneos.

Poses detectadas:

| Pose | Regra com Landmarks (MediaPipe idx) |
|---|---|
| **ARM_RAISED** (braço erguido) | Qualquer pulso acima do ombro correspondente por 8% de margem no eixo Y **E** acima do cotovelo por 4% |
| **HANDS_UP** (mãos pra cima) | **Ambos** os pulsos 10% acima dos ombros correspondentes |
| **FALLEN** (caído / deitado) | Nariz (0) muito próximo do quadril médio (23,24) no eixo Y (< 10% de diferença) **E** quadril abaixo de 70% da altura do frame (próximo ao chão) |

### 2.6 Biometria facial e Reidentificação (ReID)

#### 2.6.1 Detecção facial
**Haar Cascade (Viola-Jones)** em `haarcascade_frontalface_default.xml` — leve, roda em CPU, não requer modelo adicional. Desvantagem: só detecta face frontal.

Parâmetros:
- `scaleFactor = 1.1` (aumenta janela de busca em 10% por passo)
- `minNeighbors = 4` (exige 4 detecções sobrepostas para confirmar — reduz falsos)
- `minSize = 28x28px` (ignora rostos minúsculos/muito distantes)

Retorna os rostos ordenados por **área decrescente** (pega o maior primeiro).

#### 2.6.2 Embedding facial — Histograma de Gradientes Orientados (HOG) customizado
Feature vector **128-dimensional** extraído via HOG manual no grayscale da face, sem dependência de modelos de deep learning.

Como é feito:
1. Redimensiona o crop facial para 64×64.
2. Converte para tons de cinza.
3. Aplica filtros **Sobel 3×3** em X e Y para achar gradientes.
4. Converte magnitude e ângulo de cada pixel via `cartToPolar`.
5. Divide a imagem em **4×4 = 16 células** de 16×16.
6. Cada célula gera **8 bins de histograma angular (0–360°)** ponderados pela magnitude do gradiente.
7. Concatena tudo → **16 × 8 = 128 floats**.
8. Normaliza L2 o vetor final.

Embeddings HOG são simples e rápidos, mas carecem de semântica ("é a mesma pessoa?"). Para cenas pequenas (até ~20 pessoas distintas) e tempo curto, funcionam bem.

#### 2.6.3 Similaridade entre embeddings
**Cosine Similarity** (produto interno de vetores unitários):
```
sim = cosθ = (q · r) / (‖q‖ · ‖r‖)
```
Intervalo [-1, +1], onde +1 = idênticos. Limiar `0.70` é o ponto de corte; acima de `0.82` é classificado como **HIGH_CONFIDENCE**.

#### 2.6.4 Hierarquia de armazenamento facial
`FaceStorage` tem três camadas:

1. **Cache de detecção em `FaceCapture`** — último resultado por track por 0.5s (evita rodar detector 30x/seg).
2. **Faces temporárias** (`_temp`) — track novo sem identidade definida. Atualizadas com a melhor qualidade (`MIN_QUALITY_SCORE = 0.15`). Expiram após 15 minutos sem uso.
3. **Faces permanentes por pessoa** (`_permanent`) — associadas a um `person_id` reconhecido. Cada pessoa guarda no máximo **20 amostras** (mantém as 20 de melhor qualidade, descarta as piores).

#### 2.6.5 Cache de ReID
Por 30 segundos após um match positivo, não rodamos ReID novamente para o mesmo track (economia importante — o matching por força bruta O(N) não escala).

### 2.7 Sistema de alertas com debouncing (cooldown)

`AlertManager` recebe pedidos de disparo, mas evita spam do mesmo evento a cada frame. Estratégia:

- Chave composta `(track_id, event_type)` → último horário de disparo.
- Janela `cooldown_seconds` (padrão 3s) — dentro desse intervalo, pedidos iguais são rejeitados silenciosamente.
- Quando o alerta passa: gera `event_id`, loga auditoria, despacha notificação JSON, aciona o `EvidenceManager` e toca o beep de áudio.

### 2.8 Captura de evidência com buffer pré-trigger

Requisito em segurança: quando o alerta acontece, o vídeo já começou **antes** do momento da detecção.

Solução: **buffer circular de frames** (`collections.deque` com tamanho máximo pré-calculado):

1. **TODOS os frames** entram no `VideoBuffer.push()` em tempo real. O deque joga fora os frames mais velhos automaticamente.
2. Tamanho do buffer: `5s * 30fps * 1.5 = ~225 frames`. Margem de 1.5× para câmeras que rodem acima de 30fps.
3. Quando `trigger(event_id)` dispara:
   - Coleta do buffer todos os frames de `(triggered_at − 5s)` até `triggered_at` → **pré-frames**.
   - Inicia uma thread background que espera até `triggered_at + 15s`, acumulando frames novos em `post_frames`.
   - Quando o deadline bate (ou +2s de margem de segurança), roda o callback `on_ready(pre, post, event_id, triggered_at)`.
4. Apenas UMA captura ativa por câmera de cada vez (evita consumo excessivo de memória).

### 2.9 Geração de vídeo MP4 via FFmpeg

Não usamos `cv2.VideoWriter` — é limitado e dá problema de codecs em Windows. Pipeline:

1. Escreve cada frame da sequência como JPEG individual em `storage/temp/evt_{event_id}/frame_NNNNNN.jpg` (qualidade 85).
2. Invoca **FFmpeg subprocess**:
   - Entrada: sequência de imagens (`image2` demuxer).
   - Framerate: calculado automaticamente do intervalo dos timestamps dos frames.
   - Codec: **H.264 (libx264)**.
   - CRF 23 (qualidade padrão, equilíbrio tamanho/qualidade).
   - `preset=fast` (codificação mais rápida em troca de pouco tamanho extra).
   - `pix_fmt=yuv420p` e `movflags=+faststart` (compatibilidade máxima com navegadores/players).
3. Lê tamanho final do arquivo.
4. **Finalmente**: apaga a pasta temporária (shutil.rmtree).

Snapshot do trigger é separado: o frame com timestamp mais próximo de `triggered_at` é salvo como JPEG qualidade 90.

### 2.10 Persistência de metadados (banco)

Camada `evidence_store.py` faz **fallback automático**:

1. Se existe variável `CAM_DB_URL` (ex: PostgreSQL): conecta via SQLAlchemy e grava na tabela `evidences`.
2. Se não existe (ou a conexão falhou): grava em JSONL em `storage/evidences/evidences_fallback.jsonl`.

### 2.11 Limpeza automática do storage

Sem gerenciamento, vídeos e snapshots lotam disco. `StorageCleaner` usa duas políticas (OR — remove se satisfizer qualquer uma):

1. **Idade máxima 7 dias** (qualquer coisa mais antiga que `now − 7*86400` s apaga).
2. **Tamanho total máximo 500 MB** (se somatório do storage passar de 500MB, apaga os arquivos do mais antigo para o mais novo até ficar abaixo do limite).

Também remove **diretórios vazios** remanescentes (árvores YYYY/MM/DD limpas).

### 2.12 Snapshots periódicos

Captura autônoma independente de alertas. Regra de disparo é **OR (o que ocorrer primeiro)**:

- 40 frames processados desde a última captura **OU**
- 2 segundos desde a última captura

Resultado prático: câmera rápida (60fps) → snapshot a cada ~0.67s (pelos frames). Câmera lenta / CPU sobrecarregada → snapshot a cada 2s garantido pelo relógio.

Salva em `storage/periodic_snapshots/YYYY/MM/DD/` com qualidade JPEG 80.

---

# PARTE 2 — DOCUMENTAÇÃO TÉCNICA & ARQUITETURA

## 3. Stack e versões

| Componente | Versão mínima | Finalidade |
|---|---|---|
| Python | 3.10 | Runtime principal |
| opencv-python | 4.8 | Leitura de vídeo, detecção Haar, I/O de imagens |
| numpy | 1.24 | Álgebra linear (ângulos, embeddings HOG) |
| ultralytics | 8.0 | YOLOv8 — detecção de pessoas |
| mediapipe | 0.10 | Pose Landmarker — 33 keypoints por pessoa |
| PyYAML | 6.0 | Parsing do arquivo de configuração de câmera |
| sqlalchemy | 2.0 | ORM para metadados de evidências |
| psycopg2-binary | 2.9 | Driver PostgreSQL |
| loguru | 0.7 | Logger com rotação + retenção de arquivos |
| **FFmpeg** | 4.0+ | Codec H.264 para gerar MP4 (binário externo) |

## 4. Instalação passo a passo

### 4.1 Windows

```powershell
# 1. Instale FFmpeg (se ainda não tiver):
choco install ffmpeg -y
# OU baixe de https://ffmpeg.org/download.html e adicione no PATH.
# O app também busca automaticamente nesses caminhos comuns:
#   C:\ffmpeg\ffmpeg.exe
#   C:\Program Files\ffmpeg\bin\ffmpeg.exe
#   C:\ProgramData\chocolatey\bin\ffmpeg.exe
#   C:\Users\Usuario\Downloads\evolution-api\node_modules\@ffmpeg-installer\win32-x64\ffmpeg.exe

# 2. Clone / entre no projeto:
cd Cam-Security

# 3. Crie e ative ambiente virtual:
python -m venv venv
.\venv\Scripts\activate

# 4. Instale dependências:
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.2 Linux / macOS

```bash
sudo apt update && sudo apt install -y ffmpeg     # Debian/Ubuntu
# brew install ffmpeg                             # macOS

cd Cam-Security
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Configuração

### 5.1 Arquivo de câmera — `config/config_camera.yaml`

| Campo | Tipo | Default | Descrição |
|---|---|---|---|
| `camera_id` | int | `0` | Índice da câmera física no OpenCV (0 = webcam padrão). |
| `video_path` | string | `""` | Caminho para arquivo MP4/AVI. Se existir, sobrepõe `camera_id` e roda playback no FPS real do vídeo. |
| `width` | int | `640` | Resize de saída (largura). |
| `height` | int | `480` | Resize de saída (altura). |
| `fps` | int | `60` | Fallback se fonte não reportar FPS válido. |
| `use_synthetic` | bool | `false` | Reservado (geração de frames sintéticos). |

### 5.2 Variáveis de ambiente

| Variável | Obrigatória | Exemplo | Finalidade |
|---|---|---|---|
| `CAM_DB_URL` | Não | `postgresql://user:pass@localhost:5432/cam_security` | Se definida, metadados de evidência vão pro PostgreSQL. Senão, cai para JSONL em `storage/evidences/`. |

## 6. Execução

```bash
python main.py
```

Janela OpenCV `"Cam-Security | Visão Real"` abre. Pressione **ESC** para encerrar corretamente (libera câmera, junta threads, fecha landmarkers).

## 7. Arquitetura de módulos

```
Cam-Security/
├── camera/
│   └── capture.py              # CameraCapture — thread de leitura com lock,
│                                # sequencialização, playback com throttling de FPS
├── config/
│   ├── config_camera.yaml
│   ├── config_face.yaml
│   └── config_tracking.yaml
├── detection/
│   ├── person_detector.py      # YOLO + NMS + expand_box + draw_annotations (skeleton + HUD)
│   ├── face_detector.py        # Haar Cascade + HOG embedding 128D
│   ├── mediapipe_detector.py   # Pool de PoseLandmarker por track, mapeamento crop→frame, PoseHold
│   └── pose_estimation.py      # 3 regras de pose proibida com janela de N frames consecutivos
├── tracking/
│   ├── id_manager.py           # Contador monotônico de IDs
│   └── object_tracker.py       # Matching IoU+distância, fallback FaceReID, TTL 300s
├── face_biometry/
│   ├── face_capture.py         # Throttle 0.5s por track + insights brilho/nitidez
│   ├── face_storage.py         # Temp faces (TTL 15min) + Permanent por pessoa (max 20 amostras)
│   └── face_reid.py            # Cosine sim, caches, MATCH 0.70 / HIGH_CONF 0.82
├── events/
│   ├── alerts.py               # AlertManager: cooldown, UUID event_id, som, integra EvidenceManager
│   ├── event_logger.py         # Auditoria JSON
│   └── notification.py         # Dispatcher serializado (integração futura com backend)
├── evidence/
│   ├── evidence_capture.py     # FFmpeg pipeline + StorageCleaner (7d/500MB) + PeriodicSnapshotter (40f/2s)
│   ├── evidence_manager.py     # Dicionário de buffers por câmera, roteia on_event
│   ├── evidence_store.py       # SQLAlchemy ↔ JSONL fallback
│   └── video_buffer.py         # Deque circular pré-trigger + thread de captura pós-trigger
├── utils/
│   ├── logger.py               # loguru, rotation="10 MB", retention="7 days"
│   └── math_utils.py           # Geometria, velocidade, IoU, risco 0-100
├── storage/                    # Runtime (gerado automaticamente, git-ignorado)
│   ├── events/YYYY/MM/DD/event_{id}.mp4
│   ├── snapshots/event_{id}_trigger.jpg
│   ├── periodic_snapshots/YYYY/MM/DD/snap_{HHMMSS}_{seq}.jpg
│   ├── temp/                   # Apagado logo após gerar o MP4
│   └── evidences/evidences_fallback.jsonl
├── main.py                     # Pipeline 2-threads (inference + render)
├── requirements.txt
├── .gitignore
└── Readme.md
```

## 8. Fluxo de dados detalhado (thread de inferência)

Cada iteração do loop em [main.py](file:///c:/Users/Usuario/VSprojects/Cam-security/Cam-Security/main.py#L145-L354) executa esses passos em ordem:

| Passo | Função / Objeto | Saída |
|---|---|---|
| 1 | `camera.get_frame()` | `(frame, seq, captured_at)` com cópia thread-safe |
| 2 | `evidence_manager.push_frame()` | Frame entra no buffer circular de evidências |
| 3 | `snapshotter.on_frame()` | Snapshot periódico salvo (se 40 frames OU 2s) |
| 4 | Limpezas periódicas | `face_storage.expire_temp_faces()` a cada 120s; `storage_cleaner.run_cleanup()` a cada 300s |
| 5 | `person_detector.detect_persons()` YOLO | Lista de `[{box, confidence}]` classe=0 pessoa |
| 6 | `object_tracker.update()` + ReID callback | Dicionário `{track_id: {box, age, identity, face_box, ...}}` |
| 7 | Prune caches inativos | Pose pool, PoseHold, FaceCapture cache, ReID cache |
| 8 | Loopa tracks → `crop_person()` | Crop expandido por pessoa |
| 9 | `pose_detector.process_for_track()` | 33 landmarks mapeados para coordenadas do frame |
| 10 | `face_capture.capture_face_insights()` (throttled) | Box, embedding, brilho, nitidez |
| 11 | `face_storage.save_embedding()` → `face_reid.identify_track()` | Atualiza identidade, promove temp → permanent |
| 12 | Cálculos físicos | Velocidade corpo, velocidade pulso, proximidade, colisão, ângulo braço |
| 13 | `detect_punch()` / `pose_estimator.evaluate()` | Contadores por track |
| 14 | `alert_manager.trigger_alert()` se contadores passarem | Beep, log, notificação, **`EvidenceManager.on_event()` dispara o buffer** |
| 15 | Performance counters | A cada 10s loga médias YOLO/Track/Pose/Face/Total/FPS |
| 16 | `overlay_lock` update | Snapshot thread-safe do estado para a UI |

## 9. Fluxo da thread de renderização (principal)

- Dorme até que exista um `overlay["frame"]` novo (`id()` diferente).
- Lê `tracks` e `alert_track_ids` do overlay (congelados — thread-safe).
- [person_detector.draw_annotations()](file:///c:/Users/Usuario/VSprojects/Cam-security/Cam-Security/detection/person_detector.py#L95-L146) desenha:
  - **Bounding boxes**: verde = normal, vermelha = alerta.
  - **Labels**: número do track, identidade, status facial, tag `[ALERTA]`.
  - **Esqueleto MediaPipe** (linhas + pontos ciano/amarelo com confiança mínima 0.30).
  - **Face box** azul claro com confiança.
- **HUD** no canto superior esquerdo: `"Pessoas: N | Dist: LONGE/MEDIO/PERTO | Colisao: SIM/NAO"`.
- `cv2.waitKey(1) == 27` (ESC) → quebra ambos os loops.

## 10. Diagrama de threads e sincronização

```
┌───────────────────────────────────────────────────────────┐
│  Thread principal (render + UI)                           │
│     - espera overlay novo via id(render_frame)            │
│     - desenha → cv2.imshow → ESC detectado → camera.stop │
└────────▲──────────────────────────────────────────────────┘
         │ overlay_lock (dict com frame + tracks congelados)
         │
┌────────┴───────────────────────────────────────────────────┐
│  Thread daemon: inference_loop()                           │
│     - camera.get_frame() lock                             │
│     - YOLO → Tracker → Pose → Face → Regras → Alerta     │
│     - push em EvidenceManager (deque c/ seu lock)          │
│     - snapshotter.on_frame()                               │
└───────────────────────┬────────────────────────────────────┘
                        │
                        ├── background threads: on_ready callbacks do VideoBuffer
                        │   geram vídeos FFmpeg sem bloquear a inferência
                        │
                        └── CameraCapture._read_loop (thread daemon própria)
                            lê hardware → atualiza self.frame / lock
```

**Locks importantes** para não dar race condition:
- `CameraCapture.lock` → leitura/escrita do frame compartilhado.
- `VideoBuffer._lock` + `_capture_lock` → deque e captura ativa.
- `overlay_lock` → estado entre inferência e UI.
- `FaceStorage._lock` (RLock) → reads/writes de embeddings e cache de identidade.

## 11. Constantes de tuning importantes

| Constante | Arquivo | Valor | Efeito |
|---|---|---|---|
| `MIN_TRACK_AGE_FOR_EVENTS` | main.py | 10 frames | Não dispara alertas em tracks "recém-nascidos" (evita falsos positivos de detecção instável) |
| `PRE_TRIGGER_SECONDS` | video_buffer.py | 5.0 | Quanto de vídeo ANTES do alerta vai na evidência |
| `POST_TRIGGER_SECONDS` | video_buffer.py | 15.0 | Quanto de vídeo DEPOIS do alerta |
| `iou_threshold` (tracking) | object_tracker.py | 0.15 | Overlap mínimo para herdar ID no próximo frame |
| `ttl_seconds` (tracking) | object_tracker.py | 300.0 (5 min) | Tempo sem aparição até expirar ID |
| `FACE_CHECK_INTERVAL` | face_capture.py | 0.5 s | Não roda detector facial mais de 2x/s por track |
| `TEMP_FACE_TTL_SECONDS` | face_storage.py | 900 (15 min) | Faces temporárias sem promoção viram pó |
| `MAX_SAMPLES_PER_PERSON` | face_storage.py | 20 | Teto de embeddings permanentes por pessoa |
| `MATCH_THRESHOLD` | face_reid.py | 0.70 | Similaridade mínima cosine para match |
| `PUNCH_VELOCITY_THRESHOLD` | math_utils.py | 500 px/s | Pulso mais rápido que isso = candidato a soco |
| `PUNCH_ANGLE_THRESHOLD` | math_utils.py | 150° | Cotovelo menos dobrado que isso = extensão de soco |
| `DIST_PERTO / DIST_MEDIO` | math_utils.py | 1.0 / 3.0 | Limiar de proximidade normalizado por altura |
| `PERIODIC_FRAME_THRESHOLD` | evidence_capture.py | 40 frames | Condição 1 do snapshot periódico |
| `PERIODIC_TIME_THRESHOLD` | evidence_capture.py | 2.0 s | Condição 2 (OR) do snapshot periódico |
| `STORAGE_MAX_AGE_DAYS` | evidence_capture.py | 7 | Idade máxima de arquivo no storage |
| `STORAGE_MAX_SIZE_MB` | evidence_capture.py | 500 | Tamanho máximo agregado do storage |
| `FFMPEG_CRF` | evidence_capture.py | 23 | Qualidade do vídeo (menor = melhor, 23 é padrão) |

## 12. Políticas de retenção resumidas

| Recurso | Local | Retenção | Por quê? |
|---|---|---|---|
| Logs de aplicação | `cam_security.log` | 7 dias / rotaciona a cada 10 MB | loguru built-in |
| Frames de buffer | memória (deque) | ~5–7 segundos | Apenas para gerar vídeo de evento |
| Faces temporárias | memória (FaceStorage) | 15 min sem acesso | Evita acumular tracks de passantes |
| Vídeos MP4, snapshots, fotos periódicas | `storage/events`, `storage/snapshots`, `storage/periodic_snapshots` | **7 dias OU quando storage > 500 MB** | `StorageCleaner` roda a cada 5 min |
| Metadados (SQL) | tabela `evidences` no Postgres | ∞ (gerenciado pelo DBA) | Auditoria permanente |
| Metadados (fallback) | `storage/evidences/evidences_fallback.jsonl` | Mesma regra do storage (7 d / 500 MB) | Cai no `StorageCleaner` junto |

## 13. Observabilidade

Logs são a única interface de observabilidade (não há métricas Prometheus). Níveis:

- `INFO`: inicializações, alertas disparados, IDs de evento, criação/expiração de tracks, limpeza de storage, match ReID, vídeos gerados.
- `WARNING`: caminhos alternativos, FFmpeg não encontrado, arquivo não pôde ser apagado, qualidade de face baixa.
- `ERROR`: try/except de cada módulo com stacktrace (sem crashar a aplicação — erro é isolado e a iteração continua).
- `DEBUG`: a cada 90 frames MediaPipe diagnóstico de landmarks, snapshots periódicos salvos, faces temporárias expiradas individualmente.

**Log de performance** a cada 10s, formato:
```
[Perf/312frames] YOLO=41ms Track=1ms Pose=88ms Face=14ms Total=151ms FrameAge=24ms FPS_inf=31.2
```
Mostra gargalos instantâneos: se `YOLO` crescer ou `Total` passar de ~200ms, FPS de inferência cai abaixo de 5.

## 14. Pontos de melhoria / roadmap

- **Embeddings faciais reais**: trocar HOG 128D por FaceNet / ArcFace / insightface-buffered — aumenta acurácia do ReID.
- **Batch YOLO**: rodar detector em batch único ao invés de frame-a-frame (câmeras múltiplas).
- **Multi-câmera**: `EvidenceManager` já suporta múltiplos `camera_id` via dicionário; falta criar múltiplas instâncias `CameraCapture`.
- **Banco para eventos/alertas**: `EventLogger` e `NotificationDispatcher` hoje só salvam JSON; mesma estratégia de `EvidenceStore` (PostgreSQL fallback JSONL).
- **RTSP/streaming real**: `cv2.VideoCapture` já aceita URLs RTSP; adicionar reconnecting com backoff.
- **Web UI**: Substituir janela OpenCV por Flask/FastAPI servindo MJPEG stream e dashboard.
