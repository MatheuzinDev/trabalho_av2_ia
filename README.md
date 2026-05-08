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

## Estrutura

```text
trabalho_av2_ia/
  main.py
  dados.py
  metricas.py
  monte_carlo.py
  spiral_d.csv
  modelos/
    perceptron.py
    adaline.py
    mlp.py
  resultados/
    perceptron/
    adaline/
```

## Saidas

Ao executar o projeto, os artefatos do Perceptron Simples sao gerados em `resultados/perceptron/`.

A imagem da fronteira linear do ADALINE e gerada separadamente em `resultados/adaline/`.
