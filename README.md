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
│   ├── dados_iniciais.png
│   ├── perceptron/
│   ├── adaline/
│   └── mlp/
├── mlp/
│   └── topologias/
└── monte_carlo/
    └── comparacao_modelos/
        ├── resumo_*.csv
        └── casos_extremos/
```

### Fronteiras

`resultados/fronteiras/` guarda os graficos das fronteiras encontradas pelos modelos.

- `dados_iniciais.png`: mostra o grafico de espalhamento original do conjunto, sem fronteira de decisao.
- `perceptron/fronteira_linear.png`: mostra a reta gerada pelo Perceptron Simples.
- `adaline/fronteira_linear.png`: mostra a reta gerada pelo ADALINE.
- `mlp/fronteira_decisao.png`: mostra a regiao de decisao nao linear gerada pela MLP.

Essas imagens servem para visualizar a distribuicao inicial dos dados e como cada modelo tenta separar as duas classes do conjunto `spiral_d.csv`.

### Topologias Da MLP

`resultados/mlp/topologias/` guarda o estudo de topologias da MLP quando `run_mlp_topology_study` esta ativado no `main.py`.

Os casos configurados sao `subdimensionado`, `referencia` e `superdimensionado`. Cada caso gera matriz de confusao, curva de aprendizado e fronteira de decisao.

### Monte Carlo

`resultados/monte_carlo/comparacao_modelos/` guarda a comparacao estatistica entre os modelos ativados no Monte Carlo.

A validacao usa `500` rodadas. Em cada rodada, os dados sao embaralhados e divididos em `80%` para treino e `20%` para teste. Por padrao, a MLP fica fora do Monte Carlo para evitar execucoes longas; para inclui-la, ative `include_mlp_in_monte_carlo` no `main.py`.

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

No Perceptron, a curva representa os erros por epoca. No ADALINE e na MLP, a curva representa o EQM por epoca.
