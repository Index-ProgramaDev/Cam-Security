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