# 🍾 Detecção de Defeitos em Garrafas de Coca-Cola
## Análise Técnica do Sistema de Visão Computacional

---

## 🤖 Machine Learning vs Deep Learning

### O que é Machine Learning?

**Machine Learning (ML)** é um subcampo da Inteligência Artificial que permite aos computadores aprenderem com dados sem serem explicitamente programados.

**Características:**
- Requer engenharia manual de features (características)
- Algoritmos clássicos: Decision Trees, SVM, Random Forest, Naive Bayes
- Humano define quais características são importantes
- Bom para dados estruturados/tabulares

**Exemplo em inspeção de qualidade:**
```
Humano define: "Cor da tampa", "Altura do líquido", "Presença de rótulo"
↓
Algoritmo aprende: "Se altura < 10cm E cor_tampa = branco → Defeito"
```

### O que é Deep Learning?

**Deep Learning (DL)** é um subconjunto do Machine Learning baseado em redes neurais artificiais profundas (múltiplas camadas).

**Características:**
- **Aprendizado automático de features** (não precisa de engenharia manual)
- Arquiteturas em camadas (input → camadas ocultas → output)
- Excelente para dados não estruturados (imagens, texto, áudio)
- Requer mais dados e poder computacional

**Exemplo em inspeção de qualidade:**
```
Imagem da garrafa → [Rede Neural Profunda] → Defeitos detectados
                      ↑
              (aprende sozinha a identificar:
               bordas, texturas, formas, padrões)
```

### Comparação Lado a Lado:

| Aspecto | Machine Learning | Deep Learning |
|---------|------------------|---------------|
| **Features** | Manual (engenharia de features) | Automático (aprende sozinho) |
| **Arquitetura** | Algoritmos simples | Redes neurais profundas |
| **Dados necessários** | Poucos a médios | Muitos |
| **Hardware** | CPU suficiente | GPU recomendado |
| **Interpretabilidade** | Alta | Baixa ("caixa preta") |
| **Performance em imagens** | Limitada | Excelente |
| **Tempo de desenvolvimento** | Rápido (mas requer expertise) | Mais lento (mas automatizado) |

### Por que Deep Learning para Detecção de Defeitos?

✅ **Complexidade Visual:** Defeitos em garrafas envolvem padrões visuais complexos
✅ **Múltiplas Variações:** Iluminação, ângulo, posição variam constantemente
✅ **Features Hierárquicas:** DL aprende automaticamente desde bordas básicas até conceitos complexos
✅ **Estado da Arte:** CNNs são superiores em visão computacional
✅ **Transfer Learning:** Compensa datasets pequenos

**Neste projeto:** Usamos **Deep Learning** (CNNs) porque:
- Imagens têm alta dimensionalidade (120×160×3 = 57,600 valores)
- Defeitos visuais são difíceis de descrever manualmente
- Transfer Learning permite usar conhecimento pré-existente
- CNNs são o estado da arte em inspeção visual industrial

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

## 📐 Fundamentos Matemáticos: Redes Neurais Convolucionais (CNN)

### O que é uma CNN?

**Convolutional Neural Network (CNN)** é uma arquitetura de rede neural especializada em processar dados com estrutura de grade (como imagens).

### Operações Matemáticas Principais:

#### 1. **Convolução 2D**

A operação fundamental que extrai features da imagem:

```
Saída[i,j] = Σ Σ Entrada[i+m, j+n] × Kernel[m,n] + bias
             m n
```

**Onde:**
- `Entrada`: Imagem ou mapa de features (H×W×C)
- `Kernel/Filtro`: Matriz de pesos aprendíveis (3×3, 5×5, etc.)
- `Saída`: Feature map resultante
- `Stride`: Passo do filtro (geralmente 1)
- `Padding`: Preenchimento nas bordas

**Exemplo visual:**
```
Imagem 5×5          Filtro 3×3         Feature Map 3×3
┌─────────┐         ┌─────┐            ┌─────┐
│1 2 3 4 5│         │1 0 1│            │ 12  │
│2 3 4 5 6│    *    │0 1 0│    =       │ 16  │
│3 4 5 6 7│         │1 0 1│            │ 20  │
│4 5 6 7 8│         └─────┘            └─────┘
│5 6 7 8 9│
└─────────┘
```

**Por que convolução?**
- Detecta padrões locais (bordas, texturas)
- Compartilhamento de parâmetros (mesmo filtro em toda imagem)
- Invariância translacional (detecta padrão em qualquer posição)

#### 2. **Função de Ativação (ReLU)**

Introduz não-linearidade no modelo:

```
ReLU(x) = max(0, x) = { x,  se x > 0
                      { 0,  se x ≤ 0
```

**Por que ReLU?**
- Simples e eficiente computacionalmente
- Evita vanishing gradient
- Esparsidade (muitos neurônios "desligados")

#### 3. **Max Pooling**

Reduz dimensionalidade preservando features importantes:

```
Para cada janela 2×2, pega o valor máximo:

┌────┐         ┌──┐
│1 3│         │4 │
│2 4│    →    └──┘
└────┘
```

**Matematicamente:**
```
Saída[i,j] = max{ Entrada[2i+m, 2j+n] | m,n ∈ {0,1} }
```

**Benefícios:**
- Reduz parâmetros (evita overfitting)
- Invariância a pequenos deslocamentos
- Mantém features mais salientes

#### 4. **Dropout**

Regularização que desativa neurônios aleatoriamente durante treino:

```
Durante treino (p = 0.5):
Saída = { 0,        com probabilidade p
        { x/(1-p),  com probabilidade 1-p

Durante teste:
Saída = x  (todos neurônios ativos)
```

**Por que Dropout?**
- Previne overfitting
- Cria ensemble implícito de redes
- Força redundância nas representações

#### 5. **Camada Densa (Fully Connected)**

Neurônio clássico conectado a todos inputs:

```
y = σ(W·x + b)

Onde:
- x: vetor de entrada (n dimensões)
- W: matriz de pesos (m×n)
- b: vetor de bias (m dimensões)
- σ: função de ativação
- y: saída (m dimensões)
```

**Para classificação multi-label:**
```
Output[k] = sigmoid(Σ w[k,i] × x[i] + b[k])
                    i
```

#### 6. **Sigmoid (Ativação Final)**

Para classificação multi-label (predições independentes):

```
σ(z) = 1 / (1 + e^(-z))

Propriedades:
- Saída entre 0 e 1
- Cada classe é independente
- Threshold: y > 0.5 → classe presente
```

**Diferença do Softmax:**
```
Softmax (multi-class):  P(y=k) = e^(z_k) / Σ e^(z_i)  → Soma = 1
Sigmoid (multi-label):  P(y_k) = 1/(1+e^(-z_k))      → Independentes
```

---

## 🧠 Modelo 1: CNN Custom (Baseline)

### Resumo Matemático da Arquitetura

**Fluxo de transformações dimensionais:**

```
INPUT: X ∈ ℝ^(120×160×3)
   ↓
BLOCO 1:
  Conv2D(32):  X₁ = ReLU(W₁ * X + b₁)     → (120×160×32)
  Conv2D(32):  X₂ = ReLU(W₂ * X₁ + b₂)    → (120×160×32)
  MaxPool(2×2): X₃ = MaxPool(X₂)          → (60×80×32)
  Dropout(0.25): X₄ = Dropout(X₃, p=0.25) → (60×80×32)
   ↓
BLOCO 2:
  Conv2D(64):  X₅ = ReLU(W₅ * X₄ + b₅)    → (60×80×64)
  Conv2D(64):  X₆ = ReLU(W₆ * X₅ + b₆)    → (60×80×64)
  MaxPool(2×2): X₇ = MaxPool(X₆)          → (30×40×64)
  Dropout(0.25): X₈ = Dropout(X₇, p=0.25) → (30×40×64)
   ↓
BLOCO 3:
  Conv2D(128): X₉ = ReLU(W₉ * X₈ + b₉)    → (30×40×128)
  MaxPool(2×2): X₁₀ = MaxPool(X₉)         → (15×20×128)
  Dropout(0.3):  X₁₁ = Dropout(X₁₀, p=0.3) → (15×20×128)
   ↓
CLASSIFICADOR:
  Flatten:     X₁₂ = Flatten(X₁₁)          → (38400,)
  Dense(256):  X₁₃ = ReLU(W₁₃·X₁₂ + b₁₃)  → (256,)
  Dropout(0.5): X₁₄ = Dropout(X₁₃, p=0.5) → (256,)
  Dense(8):    ŷ = σ(W₁₄·X₁₄ + b₁₄)       → (8,)
   ↓
OUTPUT: ŷ ∈ [0,1]^8  (probabilidades por classe)
```

**Função de perda (Binary Cross-Entropy):**

```
L = -1/N Σ Σ [y_ik·log(ŷ_ik) + (1-y_ik)·log(1-ŷ_ik)]
        i k

Onde:
- N: número de amostras
- k: classe (8 classes)
- y_ik: label real (0 ou 1)
- ŷ_ik: predição (entre 0 e 1)
```

**Otimização (Adam):**

```
m_t = β₁·m_{t-1} + (1-β₁)·∇L        (primeiro momento)
v_t = β₂·v_{t-1} + (1-β₂)·(∇L)²    (segundo momento)
θ_t = θ_{t-1} - α·m̂_t/√(v̂_t + ε)  (atualização)

Onde:
- α = 0.001 (learning rate)
- β₁ = 0.9, β₂ = 0.999 (padrões Adam)
- ε = 10⁻⁸ (estabilidade numérica)
```

**Número total de parâmetros:**
```
Convoluções: ~150k parâmetros
Dense layers: ~100k parâmetros
Total: ~256k parâmetros treináveis
```

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

## 📐 Fundamentos Matemáticos: ResNet (Residual Networks)

### O Problema do Vanishing Gradient

Em redes muito profundas, gradientes diminuem exponencialmente durante backpropagation:

```
∂L/∂W₁ = ∂L/∂Wₙ · ∂Wₙ/∂Wₙ₋₁ · ... · ∂W₂/∂W₁

Se cada termo < 1 → produto → 0 (gradiente desaparece)
```

**Consequência:** Camadas iniciais não aprendem, rede profunda performa pior que rasa.

### A Solução: Conexões Residuais (Skip Connections)

**Ideia revolucionária:** Ao invés de aprender H(x), aprenda F(x) = H(x) - x

```
Bloco Tradicional:
  x → [Conv] → [ReLU] → [Conv] → H(x)

Bloco Residual:
       ┌────────────────────┐
       │                    │
  x →  │  [Conv→ReLU→Conv]  │  →  F(x) + x  → ReLU → saída
       │         ↓          │
       └─────────────F(x)───┘
              (identidade)
```

**Matematicamente:**
```
y = F(x, {Wᵢ}) + x

Onde:
- x: entrada
- F(x, {Wᵢ}): transformação residual (camadas conv)
- y = F(x) + x: saída do bloco
- Identidade: x passa direto (atalho)
```

**Por que funciona?**

1. **Gradiente sempre flui:**
   ```
   ∂y/∂x = ∂F/∂x + 1  → sempre tem componente = 1
   ```

2. **Mais fácil aprender identidade:**
   - Se F(x) = 0 → y = x (identidade perfeita)
   - Camadas podem "se desligar" se não precisarem aprender nada

3. **Permite redes muito profundas:**
   - ResNet50: 50 camadas
   - ResNet152: 152 camadas
   - Sem degradação de performance

### ResNet50: Arquitetura Detalhada

**Estrutura em blocos:**

```
INPUT (224×224×3)
    ↓
[Conv 7×7, stride=2] → (112×112×64)
    ↓
[MaxPool 3×3, stride=2] → (56×56×64)
    ↓
[Bloco Residual ×3]  Conv1_x → (56×56×256)
    ↓
[Bloco Residual ×4]  Conv2_x → (28×28×512)
    ↓
[Bloco Residual ×6]  Conv3_x → (14×14×1024)
    ↓
[Bloco Residual ×3]  Conv4_x → (7×7×2048)
    ↓
[GlobalAveragePool] → (2048,)
    ↓
[Dense(1000, softmax)] → ImageNet classes
```

**Bloco Bottleneck (usado em ResNet50):**

```
        x (entrada)
        │
    ┌───┴────────────────────┐
    │                        │
    │  [Conv 1×1, 64]  ──────┼──> Reduz dimensionalidade
    │        ↓               │
    │  [Conv 3×3, 64]  ──────┼──> Processa features
    │        ↓               │
    │  [Conv 1×1, 256] ──────┼──> Expande dimensionalidade
    │        │               │
    └────────┴───────────────┘
             │
        F(x) + x
             │
          ReLU
             ↓
        saída (256 canais)
```

**Matematicamente:**
```
F(x) = W₃·ReLU(W₂·ReLU(W₁·x + b₁) + b₂) + b₃

Onde:
- W₁: Conv 1×1 (redução: 256→64 canais)
- W₂: Conv 3×3 (processamento: 64→64)
- W₃: Conv 1×1 (expansão: 64→256 canais)

Saída: y = ReLU(F(x) + x)
```

**Total de parâmetros ResNet50:**
```
~25.6 milhões de parâmetros
```

### Transfer Learning: Matemática da Adaptação

**Pré-treinamento no ImageNet:**
```
f_ImageNet: ℝ^(224×224×3) → ℝ^1000

Aprende: θ* = argmin Σ L(f(x_i; θ), y_i)
                θ    i∈ImageNet
```

**Adaptação para Garrafas:**
```
1. Congela base: θ_base = θ* (fixo)

2. Nova cabeça: g: ℝ^2048 → ℝ^8

3. Treina apenas g:
   θ_head* = argmin Σ L(g(f_base(x_i)), y_i)
              θ     i∈Garrafas

4. Modelo final:
   f_garrafas(x) = g(f_base(x; θ*); θ_head*)
                    └─────fixo────┘ └─treina┘
```

**Por que funciona?**

Features de baixo nível são universais:
- Camadas iniciais: bordas, cores, texturas
- Camadas médias: formas, padrões
- Camadas finais: conceitos específicos (ImageNet)

**Vantagem:**
```
Garrafas (141 imgs) + ImageNet (1.4M imgs) >> Garrafas sozinhas
```

---

## 🚀 Modelo 2: ResNet50 (Transfer Learning)

### Resumo Matemático da Adaptação

**Arquitetura completa para detecção de defeitos:**

```
INPUT: X ∈ ℝ^(120×160×3)
   ↓
[Resize para 224×224] → X' ∈ ℝ^(224×224×3)
   ↓
┌──────────────────────────────────────────┐
│ ResNet50 Base (CONGELADA)                │
│ - 49 camadas convolucionais              │
│ - 16 blocos residuais                    │
│ - Pesos pré-treinados: θ* (ImageNet)    │
│                                          │
│ f_base: ℝ^(224×224×3) → ℝ^(7×7×2048)    │
└──────────────────────────────────────────┘
   ↓
[GlobalAveragePooling2D]
   ↓
X_pool ∈ ℝ^2048  (vetor de features)
   ↓
┌──────────────────────────────────────────┐
│ Cabeça Customizada (TREINÁVEL)          │
│                                          │
│ Dense(256): h = ReLU(W₁·X_pool + b₁)     │
│            h ∈ ℝ^256                     │
│    ↓                                     │
│ Dropout(0.5): h' = Dropout(h, p=0.5)     │
│    ↓                                     │
│ Dense(8): ŷ = σ(W₂·h' + b₂)              │
│          ŷ ∈ [0,1]^8                     │
└──────────────────────────────────────────┘
   ↓
OUTPUT: Probabilidades de 8 defeitos
```

**Parâmetros treináveis:**
```
GlobalAvgPool: 0 parâmetros
Dense(256): 2048×256 + 256 = 524,544 parâmetros
Dense(8): 256×8 + 8 = 2,056 parâmetros
─────────────────────────────────────────
Total treinável: ~527k parâmetros
Total congelado: ~23M parâmetros (ResNet50)
Total geral: ~25M parâmetros
```

**Função de perda (mesma que CNN):**
```
L = -1/N Σ Σ [y_ik·log(ŷ_ik) + (1-y_ik)·log(1-ŷ_ik)]
        i k

Mas gradientes apenas fluem pelas camadas treináveis:
∂L/∂θ_base = 0  (congelado)
∂L/∂θ_head ≠ 0  (treinável)
```

**Otimização (Adam com LR menor):**
```
α = 0.0001  (10x menor que CNN custom)

Por quê?
- Features pré-treinadas já são boas
- Ajuste fino requer passos menores
- Evita "esquecer" conhecimento do ImageNet
```

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

## 📐 Fundamentos Matemáticos: MobileNetV2

### O Desafio: Eficiência Computacional

**Problema:** CNNs tradicionais são muito pesadas para dispositivos móveis

```
Conv 3×3 tradicional com 128 canais:
Custo: 3×3×128×128 = 147,456 operações por pixel
```

### Solução 1: Depthwise Separable Convolution

**Ideia:** Separar convolução espacial da convolução de canais

#### **Passo 1: Depthwise Convolution**
Aplica filtro separadamente para cada canal:

```
Entrada: X ∈ ℝ^(H×W×C_in)

Para cada canal c:
  Y_c = Conv2D_3×3(X_c)  (filtro independente)

Saída: Y ∈ ℝ^(H×W×C_in)
```

**Custo computacional:**
```
3×3×C_in (um filtro 3×3 por canal)
```

#### **Passo 2: Pointwise Convolution**
Combina canais usando Conv 1×1:

```
Entrada: Y ∈ ℝ^(H×W×C_in)

Z = Conv2D_1×1(Y)  (C_out filtros)

Saída: Z ∈ ℝ^(H×W×C_out)
```

**Custo computacional:**
```
1×1×C_in×C_out
```

#### **Economia Total:**

```
Tradicional: 3×3×C_in×C_out = 9·C_in·C_out
Separável: 3×3×C_in + 1×1×C_in×C_out = 9·C_in + C_in·C_out

Razão: (9·C_in + C_in·C_out) / (9·C_in·C_out)
     = 1/C_out + 1/9
     ≈ 1/9  (se C_out grande)
```

**Resultado: ~8-9x mais eficiente!**

### Solução 2: Inverted Residual Block (MobileNetV2)

**Problema do MobileNetV1:**
- Depthwise conv opera em poucos canais (informação limitada)

**Solução:** Inverted Bottleneck
- Expande canais ANTES de depthwise
- Comprime DEPOIS

#### **Bloco Linear Bottleneck:**

```
        x (t·C canais)
        │
    ┌───┴────────────────────┐
    │                        │
    │  [Conv 1×1, expand]    │  → Expansão (t×C canais)
    │  [t·C = 6×C típico]    │     t = expansion factor
    │        ↓               │
    │  [ReLU6]               │  → Ativação
    │        ↓               │
    │  [DepthConv 3×3]       │  → Convolução espacial
    │        ↓               │
    │  [ReLU6]               │
    │        ↓               │
    │  [Conv 1×1, project]   │  → Projeção (C canais)
    │        ↓               │
    │  [Linear] ← SEM ReLU!  │  → Preserva informação
    │        │               │
    └────────┴───────────────┘
             │
        F(x) + x  (se stride=1)
             ↓
```

**Matematicamente:**

```
1. Expansão:
   X₁ = ReLU6(Conv1×1_expand(x))
   X₁ ∈ ℝ^(H×W×(t·C))

2. Depthwise:
   X₂ = ReLU6(DepthConv3×3(X₁))
   X₂ ∈ ℝ^(H×W×(t·C))

3. Projeção linear:
   X₃ = Conv1×1_project(X₂)  ← Linear, SEM ativação!
   X₃ ∈ ℝ^(H×W×C)

4. Residual (se stride=1 e C_in=C_out):
   y = X₃ + x
```

**Por que Linear (sem ReLU) na projeção?**

```
ReLU(x) = max(0, x) → perde informação negativa

Em baixa dimensionalidade: ReLU destrói muita informação
Solução: Projeção linear preserva toda informação
```

**ReLU6 (ativação limitada):**
```
ReLU6(x) = min(max(0, x), 6)

Vantagens:
- Robustez em precisão baixa (quantização)
- Previne valores muito grandes
- Ideal para dispositivos móveis
```

### MobileNetV2: Arquitetura Completa

```
INPUT (224×224×3)
    ↓
[Conv 3×3, stride=2] → (112×112×32)
    ↓
[Inverted Residual ×1, t=1] → (112×112×16)
    ↓
[Inverted Residual ×2, t=6] → (56×56×24)
    ↓
[Inverted Residual ×3, t=6] → (28×28×32)
    ↓
[Inverted Residual ×4, t=6] → (14×14×64)
    ↓
[Inverted Residual ×3, t=6] → (14×14×96)
    ↓
[Inverted Residual ×3, t=6] → (7×7×160)
    ↓
[Inverted Residual ×1, t=6] → (7×7×320)
    ↓
[Conv 1×1] → (7×7×1280)
    ↓
[GlobalAveragePool] → (1280,)
    ↓
[Dense(1000, softmax)] → ImageNet classes
```

**Estatísticas:**
```
Total de parâmetros: ~3.4M
Operações (MAdds): ~300M
Tamanho: ~14MB

Comparação:
- ResNet50: 25M parâmetros, ~4B MAdds
- MobileNetV2: 3.4M parâmetros, 300M MAdds
→ MobileNetV2 é ~13x mais eficiente!
```

### Comparação: ResNet50 vs MobileNetV2

| Aspecto | ResNet50 | MobileNetV2 |
|---------|----------|-------------|
| **Bloco básico** | Bottleneck clássico | Inverted Residual |
| **Expansão** | Reduz → Processa → Expande | Expande → Processa → Reduz |
| **Convolução** | Tradicional 3×3 | Depthwise Separable |
| **Ativação final** | ReLU | Linear (bottleneck) |
| **Parâmetros** | 25M | 3.4M |
| **Eficiência** | Baseline | 8-13x mais eficiente |
| **Uso** | Servidores/GPUs | Dispositivos móveis |

---

## 📱 Modelo 3: MobileNetV2 (Leve e Rápido)

### Resumo Matemático da Adaptação

**Arquitetura completa para detecção de defeitos:**

```
INPUT: X ∈ ℝ^(120×160×3)
   ↓
[Resize para 224×224] → X' ∈ ℝ^(224×224×3)
   ↓
┌──────────────────────────────────────────┐
│ MobileNetV2 Base (CONGELADA)             │
│ - 53 camadas                             │
│ - 17 blocos Inverted Residual            │
│ - Pesos pré-treinados: θ* (ImageNet)    │
│ - Depthwise Separable Convolutions      │
│                                          │
│ f_base: ℝ^(224×224×3) → ℝ^(7×7×1280)    │
└──────────────────────────────────────────┘
   ↓
[GlobalAveragePooling2D]
   ↓
X_pool ∈ ℝ^1280  (vetor de features)
   ↓
┌──────────────────────────────────────────┐
│ Cabeça Customizada (TREINÁVEL)          │
│ [Menor que ResNet para eficiência]      │
│                                          │
│ Dense(128): h = ReLU(W₁·X_pool + b₁)     │
│            h ∈ ℝ^128                     │
│    ↓                                     │
│ Dropout(0.4): h' = Dropout(h, p=0.4)     │
│    ↓                                     │
│ Dense(8): ŷ = σ(W₂·h' + b₂)              │
│          ŷ ∈ [0,1]^8                     │
└──────────────────────────────────────────┘
   ↓
OUTPUT: Probabilidades de 8 defeitos
```

**Parâmetros treináveis:**
```
GlobalAvgPool: 0 parâmetros
Dense(128): 1280×128 + 128 = 163,968 parâmetros
Dense(8): 128×8 + 8 = 1,032 parâmetros
─────────────────────────────────────────
Total treinável: ~165k parâmetros
Total congelado: ~2.2M parâmetros (MobileNetV2)
Total geral: ~3.4M parâmetros
```

**Comparação de eficiência:**
```
                 Parâmetros  Inferência  Memória
CNN Custom:         256k        Rápida     Baixa
ResNet50:           25M         Lenta      Alta
MobileNetV2:        3.4M        Rápida     Média

Trade-off MobileNetV2:
✓ 7x mais leve que ResNet50
✓ Performance próxima (~95% do ResNet)
✓ Ideal para produção
```

**Otimização (mesma que ResNet50):**
```
α = 0.0001  (learning rate baixo)
Loss: Binary Cross-Entropy
Optimizer: Adam

Filosofia idêntica ao ResNet:
- Features pré-treinadas de qualidade
- Apenas ajuste fino da cabeça
- Preserva conhecimento do ImageNet
```

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

## 🎚️ Callbacks Inteligentes: Controle Automático do Treinamento

### O que são Callbacks?

**Callbacks** são funções executadas automaticamente em momentos específicos do treinamento:
- Início/fim de cada época
- Início/fim de cada batch
- Durante validação

**Objetivo:** Automatizar decisões de treinamento sem intervenção manual

---

### 1. Early Stopping (Parada Antecipada)

#### **Configuração:**
```python
EarlyStopping(
    monitor='val_loss',          # Métrica a monitorar
    patience=15,                 # Épocas sem melhora
    restore_best_weights=True,   # Restaura melhores pesos
    mode='min',                  # Minimizar val_loss
    min_delta=0.0001            # Melhora mínima significativa
)
```

#### **Como Funciona (Matemática):**

```
Para cada época t:
  1. Calcular métrica: M(t) = val_loss(t)

  2. Comparar com melhor valor histórico:
     if M(t) < M_best - δ:  (δ = min_delta)
         M_best = M(t)
         contador = 0
         salva_pesos(θ_best = θ(t))
     else:
         contador += 1

  3. Verificar paciência:
     if contador >= patience:
         PARA_TREINAMENTO()
         restaura_pesos(θ = θ_best)
```

#### **Exemplo Visual:**

```
Época | Val_Loss | Melhor | Contador | Ação
------|----------|--------|----------|------------------
  1   |  0.850   | 0.850  |    0     | Novo melhor ✓
  2   |  0.720   | 0.720  |    0     | Novo melhor ✓
  3   |  0.680   | 0.680  |    0     | Novo melhor ✓
  4   |  0.685   | 0.680  |    1     | Sem melhora
  5   |  0.690   | 0.680  |    2     | Sem melhora
  6   |  0.650   | 0.650  |    0     | Novo melhor ✓
  ...  |   ...    |  ...   |   ...    | ...
  20  |  0.655   | 0.650  |   13     | Sem melhora
  21  |  0.658   | 0.650  |   14     | Sem melhora
  22  |  0.660   | 0.650  |   15     | PARA! Paciência esgotada

Restaura pesos da época 6 (melhor val_loss = 0.650)
```

#### **Benefícios:**

1. **Previne Overfitting:**
   ```
   Train_Loss continua caindo → modelo memorizando
   Val_Loss para de cair/sobe → modelo não generaliza
   → Early stopping PARA antes do overfitting
   ```

2. **Economiza Tempo:**
   ```
   Sem callback: 100 épocas × 5 min = 500 min
   Com callback: Para em 45 épocas = 225 min
   Economia: 55% de tempo!
   ```

3. **Garante Melhor Modelo:**
   - Sempre restaura pesos da época com melhor validação
   - Não precisa monitorar manualmente

#### **Por que patience=15?**
```
Dataset pequeno (141 imagens):
- Alta variância entre épocas
- Pode ter flutuações aleatórias
- Patience alta evita parar prematuramente

Dataset grande:
- Menor variância
- Patience menor (5-10 épocas)
```

---

### 2. ReduceLROnPlateau (Ajuste de Learning Rate)

#### **Configuração:**
```python
ReduceLROnPlateau(
    monitor='val_loss',    # Métrica a monitorar
    factor=0.5,            # Multiplicador de redução
    patience=7,            # Épocas sem melhora
    min_lr=1e-7,          # LR mínimo permitido
    mode='min',            # Minimizar val_loss
    cooldown=0,           # Épocas de espera após redução
    verbose=1             # Imprimir mensagens
)
```

#### **Como Funciona (Matemática):**

```
Para cada época t:
  1. Calcular métrica: M(t) = val_loss(t)

  2. Verificar estagnação:
     if M(t) < M_recent_best - δ:
         M_recent_best = M(t)
         contador_lr = 0
     else:
         contador_lr += 1

  3. Reduzir learning rate se estagnado:
     if contador_lr >= patience:
         α_new = max(α_old × factor, min_lr)
         if α_new != α_old:
             α = α_new
             contador_lr = 0
             print(f"LR reduzido: {α_old} → {α_new}")
```

#### **Exemplo Visual:**

```
Época | Val_Loss | LR      | Contador | Ação
------|----------|---------|----------|---------------------
  1   |  0.850   | 0.001   |    0     | Início
  5   |  0.680   | 0.001   |    0     | Melhorando
  10  |  0.650   | 0.001   |    0     | Melhorando
  15  |  0.648   | 0.001   |    1     | Estagnando
  20  |  0.647   | 0.001   |    5     | Estagnando
  22  |  0.647   | 0.001   |    7     | LR → 0.0005 (×0.5)
  25  |  0.640   | 0.0005  |    0     | Melhorando (LR menor)
  35  |  0.638   | 0.0005  |    7     | LR → 0.00025 (×0.5)
  45  |  0.636   | 0.00025 |    0     | Melhorando
  60  |  0.635   | 0.00025 |    7     | LR → 0.000125 (×0.5)
```

#### **Por que Funciona?**

**Analogia da Bola em um Vale:**

```
Learning Rate Alto (início):
    ↓ ↓ ↓
   /     \     Pulos grandes → descida rápida
  /   •   \   mas impreciso no fundo
 /         \

Learning Rate Baixo (fim):
         ↓
    /       \   Pulos pequenos → encontra
   /    •    \  exatamente o fundo
  /___________\
```

**Matematicamente:**

```
Gradiente descendente: θ_new = θ_old - α·∇L

LR alto (α grande):
- Passos grandes
- Converge rápido inicialmente
- Oscila próximo ao mínimo

LR baixo (α pequeno):
- Passos pequenos
- Converge devagar
- Preciso no mínimo local
```

#### **Schedule de Learning Rate:**

```
Época 1-22:   α = 0.001    (exploração rápida)
Época 23-34:  α = 0.0005   (refinamento)
Época 35-59:  α = 0.00025  (ajuste fino)
Época 60+:    α = 0.000125 (convergência precisa)
```

**Redução exponencial:**
```
α(n) = α₀ × factor^n

Exemplo com α₀=0.001, factor=0.5:
α(0) = 0.001
α(1) = 0.0005
α(2) = 0.00025
α(3) = 0.000125
α(4) = 0.0000625
...
α(∞) ≥ min_lr = 1e-7
```

#### **Benefícios:**

1. **Convergência Melhor:**
   ```
   LR fixo: Pode nunca alcançar ótimo (oscila)
   LR adaptativo: Ajusta automaticamente para convergir
   ```

2. **Automático:**
   ```
   Manual: Testar diferentes LR schedules
   Callback: Ajusta sozinho baseado no progresso
   ```

3. **Previne Platôs:**
   ```
   Modelo estagnado em mínimo local → reduz LR
   → Passos menores permitem escapar/refinar
   ```

---

### 3. Combinação: Early Stopping + ReduceLROnPlateau

**Estratégia de dois níveis:**

```
ReduceLROnPlateau (patience=7):
    ├─> Detecta estagnação RÁPIDA
    └─> Tenta resolver reduzindo LR

EarlyStopping (patience=15):
    ├─> Detecta estagnação PROLONGADA
    └─> Para treinamento se LR reduzido não ajudar
```

**Fluxo de decisão:**

```
                 Época t
                    ↓
            Calcular val_loss
                    ↓
         ┌──────────┴──────────┐
         │                     │
    Melhorou?              Estagnado?
         │                     │
        Sim                   Não
         │                     │
    Resetar                Contador++
    contadores                 │
         │              ┌──────┴──────┐
         │              │             │
         │         Contador=7    Contador=15
         │              │             │
         │        Reduz LR       Para treino
         │              │             │
         └──────────────┴─────────────┘
                        ↓
                  Próxima época
```

#### **Exemplo Completo:**

```
Época | Val_Loss | LR      | Cnt_LR | Cnt_ES | Ação
------|----------|---------|--------|--------|------------------
  1   |  0.850   | 0.001   |   0    |   0    | Início
  10  |  0.650   | 0.001   |   0    |   0    | Melhorando
  15  |  0.648   | 0.001   |   0    |   1    | Estagnando
  20  |  0.647   | 0.001   |   5    |   5    | Estagnando
  22  |  0.647   | 0.001   |   7    |   7    | LR→0.0005 ✓
  23  |  0.645   | 0.0005  |   0    |   0    | Melhorou! ✓
  30  |  0.640   | 0.0005  |   0    |   0    | Melhorando
  37  |  0.639   | 0.0005  |   7    |   7    | LR→0.00025 ✓
  38  |  0.6385  | 0.00025 |   0    |   1    | Leve melhora
  45  |  0.6384  | 0.00025 |   6    |   7    | Estagnando
  46  |  0.6384  | 0.00025 |   7    |   8    | LR→0.000125 ✓
  47  |  0.6384  | 0.000125|   1    |   9    | Sem melhora
  ...  |   ...    |  ...    |  ...   |  ...   | ...
  61  |  0.6384  | 0.000125|   5    |  15    | PARA! ✗

Restaura pesos da época 38 (melhor val_loss = 0.6385)
```

---

### 4. Por que Esses Callbacks São Essenciais?

#### **Sem Callbacks:**
```python
# Código manual (trabalhoso e propenso a erros)
best_loss = float('inf')
patience_counter = 0
lr = 0.001

for epoch in range(100):
    train()
    val_loss = validate()

    # Lógica manual complexa
    if val_loss < best_loss - 0.0001:
        best_loss = val_loss
        save_weights()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 7:
            lr *= 0.5
            if patience_counter >= 15:
                break
```

#### **Com Callbacks:**
```python
# Limpo, automático, testado
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=7)
]

model.fit(X, y, validation_data=(X_val, y_val),
          epochs=100, callbacks=callbacks)
```

**Vantagens:**
✅ Código limpo e legível
✅ Testado por milhões de usuários
✅ Sem bugs de implementação
✅ Fácil de ajustar hiperparâmetros
✅ Integrado ao TensorFlow/Keras

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
