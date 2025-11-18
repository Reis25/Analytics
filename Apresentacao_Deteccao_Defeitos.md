# 🍾 Detecção de Defeitos em Garrafas de Coca-Cola
## Análise Técnica do Sistema de Visão Computacional

---

## 📊 Visão Geral do Projeto

**Objetivo:** Detectar automaticamente defeitos em garrafas de Coca-Cola usando Deep Learning

**Problema:** Classificação Multi-Label
- Uma garrafa pode ter múltiplos defeitos simultâneos
- 8 tipos de defeitos possíveis
- Dataset real de linha de produção

**Dataset:** 141 imagens (120x160 pixels, RGB)

---

## 🎯 Tipos de Defeitos Detectados

### 8 Classes de Defeitos:

1. **CONTENT_HIGH** - Nível de líquido acima do padrão
2. **CONTENT_LOW** - Nível de líquido abaixo do padrão
3. **COVER_NONE** - Tampa ausente
4. **BOTTLE_SMASHED** - Garrafa quebrada/amassada
5. **LABEL_WHITE** - Rótulo em branco/sem impressão
6. **LABEL_MISPLACED** - Rótulo mal posicionado
7. **LABEL_NONE** - Rótulo ausente
8. **BOTTLE_NONE** - Garrafa ausente

> **Característica especial:** Uma única garrafa pode ter vários defeitos (ex: tampa ausente + rótulo mal posicionado)

---

## 🖼️ Processamento de Imagens

### Técnicas Aplicadas:

**1. Redimensionamento**
- Dimensões padronizadas: 120x160 pixels
- Algoritmo LANCZOS para preservar qualidade
- Mantém proporção original das garrafas

**2. Normalização**
- Pixels convertidos para escala 0-1 (divisão por 255)
- Facilita convergência do modelo
- Padrão em redes neurais

**3. Multi-Hot Encoding**
- Converte labels para formato binário
- Permite múltiplos defeitos simultâneos
- Exemplo: [1, 0, 1, 0, 0, 0, 0, 0] = defeitos 1 e 3 presentes

---

## 🔄 Data Augmentation

### Por que usar?
Dataset pequeno (141 imagens) → Risco de overfitting

### Transformações Aplicadas (apenas no treino):

| Técnica | Configuração | Justificativa |
|---------|--------------|---------------|
| **Rotação** | ±10° | Limitado - garrafas são verticais |
| **Deslocamento** | ±10% (H/V) | Simula posições variadas na esteira |
| **Zoom** | ±10% | Simula diferentes distâncias da câmera |
| **Brilho** | 80-120% | Simula variações de iluminação |
| **Flip Horizontal** | ❌ DESATIVADO | Garrafas não devem ser espelhadas |

> **Resultado:** Aumenta artificialmente o dataset, melhorando generalização

---

## 📂 Divisão dos Dados

### Estratégia de Separação:

```
┌─────────────────────────────────┐
│  Dataset Total: 141 imagens     │
└─────────────────────────────────┘
           ↓
    ┌──────┴───────┬──────────┐
    ↓              ↓          ↓
  Treino         Validação   Teste
   64%            16%         20%
(90 imgs)      (23 imgs)   (28 imgs)
    ↓
Com Data
Augmentation
```

**Estratificação:** Mantém proporção de classes em cada conjunto

**Validação:** Monitora performance durante treinamento

**Teste:** Avaliação final imparcial (nunca visto pelo modelo)

---

## 🧠 Modelo 1: CNN Custom (Baseline)

### Arquitetura Simples mas Eficaz

```
INPUT (120x160x3)
    ↓
[Bloco Conv 1] → 32 filtros → MaxPool → Dropout(0.25)
    ↓
[Bloco Conv 2] → 64 filtros → MaxPool → Dropout(0.25)
    ↓
[Bloco Conv 3] → 128 filtros → MaxPool → Dropout(0.30)
    ↓
Flatten → Dense(256) → Dropout(0.5)
    ↓
OUTPUT: Dense(8, sigmoid)
```

**Características:**
- Totalmente treinável desde zero
- ~256k parâmetros
- Boa baseline para comparação
- Rápido para treinar

**Vantagem:** Adaptado especificamente para garrafas

---

## 🚀 Modelo 2: ResNet50 (Transfer Learning)

### Aproveitando Conhecimento Pré-Treinado

```
ResNet50 (ImageNet)
[23M parâmetros CONGELADOS]
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, relu)
    ↓
Dropout(0.5)
    ↓
OUTPUT: Dense(8, sigmoid)
```

**O que é Transfer Learning?**
- Usa rede treinada em milhões de imagens
- Camadas iniciais detectam features genéricas (bordas, texturas)
- Apenas últimas camadas são retreinadas para defeitos

**Vantagens:**
- Excelente para datasets pequenos
- Aprende rápido (pesos pré-treinados)
- Alta precisão

**Desvantagens:**
- ~25M parâmetros (pesado)
- Mais lento para inferência

---

## 📱 Modelo 3: MobileNetV2 (Leve e Rápido)

### Otimizado para Dispositivos Móveis

```
MobileNetV2 (ImageNet)
[2.2M parâmetros CONGELADOS]
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, relu)  ← Menor que ResNet
    ↓
Dropout(0.4)
    ↓
OUTPUT: Dense(8, sigmoid)
```

**Por que MobileNet?**
- Arquitetura eficiente (depthwise separable convolutions)
- 10x mais leve que ResNet50
- Ideal para implantação em produção

**Trade-off:**
- Performance ligeiramente inferior
- Muito mais rápido e leve
- Pode rodar em dispositivos embarcados

---

## ⚙️ Configuração de Treinamento

### Hiperparâmetros Principais:

| Parâmetro | CNN Custom | ResNet50/MobileNet |
|-----------|------------|-------------------|
| **Loss** | Binary Crossentropy | Binary Crossentropy |
| **Optimizer** | Adam (lr=0.001) | Adam (lr=0.0001) |
| **Ativação Final** | Sigmoid | Sigmoid |
| **Épocas Máximas** | 100 | 100 |
| **Batch Size** | 16 | 16 |

### Por que Binary Crossentropy?
- Apropriado para classificação multi-label
- Cada classe é tratada independentemente
- Diferente de Categorical Crossentropy (exclusivo)

### Por que Sigmoid (não Softmax)?
- Sigmoid: Cada saída entre 0-1 independentemente
- Softmax: Soma das saídas = 1 (exclusão mútua)
- Multi-label precisa de predições independentes

---

## 🎚️ Callbacks Inteligentes

### 1. Early Stopping
```python
EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)
```
**Função:** Para treinamento quando não há melhora
- Monitora loss de validação
- Espera 15 épocas antes de parar
- Restaura melhores pesos automaticamente
- **Evita overfitting**

### 2. ReduceLROnPlateau
```python
ReduceLROnPlateau(
    factor=0.5,
    patience=7,
    min_lr=1e-7
)
```
**Função:** Ajusta learning rate dinamicamente
- Reduz LR em 50% quando estagna
- Ajuda modelo a "afinar" predições
- Melhora convergência

---

## 📈 Métricas de Avaliação

### Métricas Multi-Label Específicas:

**1. Accuracy (Acurácia)**
- Porcentagem de predições totalmente corretas
- Métrica básica de performance

**2. AUC (Area Under Curve)**
- Área sob curva ROC
- Robusta para classes desbalanceadas
- **Métrica principal para comparação**
- Varia de 0 a 1 (quanto maior, melhor)

**3. Hamming Loss**
- Fração de labels incorretas
- Específico para multi-label
- Quanto menor, melhor
- Exemplo: 2 erros em 8 labels = 0.25

**4. Precision & Recall (por classe)**
- Precision: Das predições positivas, quantas estão corretas?
- Recall: Dos defeitos reais, quantos foram detectados?

---

## 🔍 Análise Exploratória dos Dados

### Insights Descobertos:

**Distribuição de Defeitos:**
- Classes desbalanceadas (alguns defeitos mais raros)
- Necessidade de métricas robustas (AUC)

**Defeitos por Garrafa:**
- Média de defeitos por imagem
- Algumas garrafas sem defeitos
- Outras com múltiplos defeitos simultâneos

**Visualizações:**
- Amostras de cada tipo de defeito
- Gráficos de distribuição
- Matriz de co-ocorrência de defeitos

> **Importância:** Entender dados antes de modelar

---

## 🏆 Comparação dos Modelos

### Tabela Comparativa:

| Modelo | Parâmetros | AUC | Hamming Loss | Velocidade | Uso |
|--------|------------|-----|--------------|------------|-----|
| **CNN Custom** | 256k | ⭐⭐⭐ | Médio | ⚡⚡⚡ | Baseline/Prototipagem |
| **ResNet50** | 25M | ⭐⭐⭐⭐⭐ | Menor | ⚡ | Alta precisão |
| **MobileNetV2** | 3M | ⭐⭐⭐⭐ | Baixo | ⚡⚡⚡ | Produção/Embarcados |

### Recomendações:

**Para Laboratório/Pesquisa:** ResNet50
- Máxima precisão
- Recursos computacionais disponíveis

**Para Produção/Linha de Fabricação:** MobileNetV2
- Ótimo balanço precisão/velocidade
- Pode rodar em hardware limitado
- Custo-benefício superior

**Para Prototipagem Rápida:** CNN Custom
- Rápido para treinar e ajustar
- Entender viabilidade do problema

---

## 🛠️ Stack Tecnológico

### Bibliotecas Utilizadas:

**Deep Learning:**
- `TensorFlow/Keras` - Framework principal
- `keras.applications` - Modelos pré-treinados
- `keras.callbacks` - Controle de treinamento

**Processamento de Dados:**
- `NumPy` - Operações matriciais
- `Pandas` - Manipulação de dataframes
- `PIL` - Processamento de imagens

**Análise e Visualização:**
- `Matplotlib` - Gráficos
- `Seaborn` - Visualizações estatísticas
- `scikit-learn` - Métricas e divisão de dados

---

## ✨ Destaques Técnicos

### Decisões Importantes de Projeto:

**1. Multi-Label em vez de Multi-Class**
- Permite detecção de múltiplos defeitos
- Mais realista para inspeção industrial
- Sigmoid + Binary Crossentropy

**2. Transfer Learning**
- Compensa dataset pequeno (141 imagens)
- Aproveita conhecimento de milhões de imagens
- Base congelada, apenas cabeça treinável

**3. Adaptações Específicas para Garrafas**
- Sem flip horizontal (mantém orientação)
- Rotação limitada (garrafas são verticais)
- Augmentation controlado

**4. Data Augmentation Essencial**
- Aumenta dataset artificialmente
- Previne overfitting
- Melhora generalização

---

## 🎯 Resultados e Conclusões

### Pontos Fortes do Sistema:

✅ Detecta múltiplos defeitos simultâneos
✅ Três opções de modelo (precisão vs. velocidade)
✅ Robusto a variações de iluminação e posição
✅ Transfer Learning eficaz para dataset pequeno
✅ Métricas adequadas para avaliação multi-label

### Limitações:

⚠️ Dataset pequeno (141 imagens)
⚠️ Possível bias em classes raras
⚠️ Depende de qualidade das imagens de entrada

### Próximos Passos:

1. **Coletar mais dados** - Expandir dataset
2. **Fine-tuning** - Destravar camadas da base
3. **Ensemble** - Combinar predições dos 3 modelos
4. **Threshold Optimization** - Ajustar por classe
5. **Deploy** - Integrar em linha de produção

---

## 💡 Aplicações Práticas

### Cenários de Uso:

**1. Linha de Produção**
- Inspeção automática 24/7
- Redução de falhas humanas
- Aumento de throughput

**2. Controle de Qualidade**
- Rastreamento de defeitos por lote
- Análise de tendências
- Melhoria contínua de processos

**3. Alertas em Tempo Real**
- Notificação imediata de defeitos
- Intervenção rápida
- Redução de desperdício

**4. Relatórios Automatizados**
- Estatísticas de produção
- KPIs de qualidade
- Auditoria e compliance

---

## 📚 Referências e Recursos

### Arquiteturas Utilizadas:

- **ResNet50:** He et al., 2015 - "Deep Residual Learning for Image Recognition"
- **MobileNetV2:** Sandler et al., 2018 - "Inverted Residuals and Linear Bottlenecks"

### Conceitos Aplicados:

- Transfer Learning
- Data Augmentation
- Multi-Label Classification
- Binary Cross-Entropy Loss
- Early Stopping & Learning Rate Scheduling

### Dataset:

- Ground Truth: `bottles_ground_truth.csv`
- Imagens reais de linha de produção
- 8 categorias de defeitos anotadas

---

## 🙏 Obrigado!

### Resumo Final:

Este projeto demonstra como **Deep Learning** pode ser aplicado eficazmente em **inspeção industrial**, mesmo com **datasets limitados**, usando técnicas como:

- 🔄 Data Augmentation
- 🧠 Transfer Learning
- 📊 Classificação Multi-Label
- ⚖️ Balanço entre precisão e eficiência

**Resultado:** Sistema robusto e prático para detecção automática de defeitos em garrafas

---

### Contato e Perguntas

**Documentação Completa:** `Deteccao_Defeitos_Garrafas_CocaCola.ipynb`

**Modelos Disponíveis:**
- CNN Custom (baseline)
- ResNet50 (máxima precisão)
- MobileNetV2 (produção)

---
