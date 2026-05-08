# Trabalho AV2 - Redes Neurais

## Requisitos

- Python ou Python3
- `numpy`
- `matplotlib`

## Como rodar o projeto

```bash
python3 -m venv IA_venv
source IA_venv/bin/activate
python3 -m pip install -r requirements.txt
python3 main.py
```

Ao final da execucao, os arquivos sao gerados dentro da pasta `resultados/`.

## Resultados Gerados

```text
resultados/
├── fronteiras/
│   ├── perceptron/
│   └── adaline/
└── monte_carlo/
    └── comparacao_perceptron_adaline/
        ├── resumo_*.csv
        └── casos_extremos/
```

### Fronteiras

`resultados/fronteiras/` guarda os graficos das fronteiras lineares encontradas pelos modelos.

- `perceptron/fronteira_linear.png`: mostra a reta gerada pelo Perceptron Simples.
- `adaline/fronteira_linear.png`: mostra a reta gerada pelo ADALINE.

Essas imagens servem para visualizar como cada modelo tenta separar as duas classes do conjunto `spiral_d.csv`.

### Monte Carlo

`resultados/monte_carlo/comparacao_perceptron_adaline/` guarda a comparacao estatistica entre Perceptron Simples e ADALINE.

A validacao usa `500` rodadas. Em cada rodada, os dados sao embaralhados e divididos em `80%` para treino e `20%` para teste.

### Tabelas Resumo

Os arquivos `resumo_*.csv` apresentam uma tabela para cada metrica:

- `resumo_acuracia.csv`
- `resumo_sensibilidade.csv`
- `resumo_especificidade.csv`
- `resumo_precisao.csv`
- `resumo_f1_score.csv`

Cada tabela mostra, para cada modelo, a media, o desvio-padrao, o maior valor e o menor valor obtidos nas rodadas de Monte Carlo.

### Casos Extremos

`casos_extremos/` guarda os melhores e piores casos de cada metrica para cada modelo.

A organizacao segue este formato:

```text
casos_extremos/modelo/metrica/melhor/
casos_extremos/modelo/metrica/pior/
```

Cada pasta possui:

- `matriz_confusao.png`: mostra os acertos e erros da rodada selecionada.
- `curva_aprendizado.png`: mostra a evolucao do treinamento.

No Perceptron, a curva representa os erros por epoca. No ADALINE, a curva representa o EQM por epoca.
