# Cam-Security

Sistema inteligente de monitoramento em tempo real baseado em visão computacional, com foco em detecção de comportamentos suspeitos, geração de alertas e captura de evidências.





## Objetivo

Auxiliar operadores de segurança a identificar rapidamente situações suspeitas, reduzindo a dependência de monitoramento manual contínuo e aumentando a eficiência na resposta a incidentes.



## ⚙️ Funcionalidades

- 🎥 Monitoramento de câmeras em tempo real
- 👤 Detecção de pessoas com bounding boxes
- 🦴 Estimativa de pose (esqueleto)
- 🧠 Análise de comportamento:
  - Velocidade
  - Direção
  - Colisões
  - Quedas
- 🚨 Sistema de alerta em tempo real
- 📸 Captura inteligente de rostos
- 🔁 Reidentificação de indivíduos
- 💾 Registro de eventos e evidências



## Arquitetura do Sistema

```text
Câmera → Detecção → Tracking → Análise → Decisão → Evento → Interface


<!-- OQUE DEVE SER FEITO -->


Mode                 LastWriteTime         Length Name                                                                                                                                                   
----                 -------------         ------ ----                                                                                                                                                   
d-----        21/07/2026     14:23                bahavior                                                                                                                                               
d-----        21/07/2026     14:29                camera                                                                                                                                                 
d-----        21/07/2026     14:23                config                                                                                                                                                 
d-----        21/07/2026     14:29                detection                                                                                                                                              
d-----        21/07/2026     14:30                events                                                                                                                                                 
d-----        21/07/2026     14:23                face_biometry                                                                                                                                          
d-----        21/07/2026     14:23                tracking                                                                                                                                               
d-----        21/07/2026     14:32                ui                                                                                                                                                     
d-----        21/07/2026     14:32                utils                                                                                                                                                  
d-----        21/07/2026     14:25                venv                                                                                                                                                   
-a----        21/07/2026     14:23            241 .gitignore                                                                                                                                             
-a----        21/07/2026     14:32           3361 main.py                                                                                                                                                
-a----        21/07/2026     14:23            978 Readme.md                                                                                                                                              
-a----        21/07/2026     14:23            301 requirements.txt                                                                                                                                       


Camera: Refatorar e simplificar codigo

Detection
Adicionar Face_detection
Otimizar Mediapipe_detection
Person detector deve desenhar bounding boxe
Pose Estimation deve Guardar poses proibidas

Events

 Refatorar ALERTAS
Event Logger deve guardar Logs de auditoria	
Notification deve ser o que direciona o front pro backend de alertas

Face_Biometry

Face Capture deve usar face_detection e capturar insights do rosto
Face_reid deve redetectar um rosto ja conhecido
Face_storage guarda embbedings de face organizados por id

Tracking

ID_Manager: Atribui os IDS
Object_Tracker: Ele Vai gerar o tracking ID unico e vai salvar por 5 minutos a menos que hajam gatilhos

UI (Apagar)

UTILS
Logger (Migrar loggings pra ca com self.sleep)


ULTIMO SPRINT

Adicionar Banco de dados com embbedings e otimizações de imagens e video para detecção facial

PRAZO: 1 Semana
Entrega Esperada: Versão MVP Backend funcional. Um pouco lenta mas ainda funcional

