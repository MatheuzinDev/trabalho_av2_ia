import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from dados_recfac import carregar_imagens_recfac
from modelos.mlp_multiclasse import MLPMulticlasse
from modelos.perceptron_multiclasse import PerceptronMulticlasse
from modelos.adaline_multiclasse import AdalineMulticlasse

def particionar_dados_multiclasse(X, Y, proporcao_treino=0.8):
    N = X.shape[1]
    indices = np.random.permutation(N)
    limite = int(proporcao_treino * N)
    
    idx_treino, idx_teste = indices[:limite], indices[limite:]
    X_train, Y_train = X[:, idx_treino], Y[:, idx_treino]
    X_test, Y_test = X[:, idx_teste], Y[:, idx_teste]
    
    Y_test_ids = np.argmax(Y_test, axis=0)
    
    return X_train, Y_train, X_test, Y_test_ids

def plotar_matriz_seaborn(y_real, y_pred, nomes_classes, titulo, caminho_salvar):
    matriz = np.zeros((len(nomes_classes), len(nomes_classes)), dtype=int)
    for r, p in zip(y_real, y_pred):
        matriz[r, p] += 1
        
    plt.figure(figsize=(14, 12))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
                xticklabels=nomes_classes, yticklabels=nomes_classes)
    plt.title(titulo, fontsize=16)
    plt.xlabel('Predição', fontsize=14)
    plt.ylabel('Realidade', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(caminho_salvar, dpi=150)
    plt.close()

def plotar_curva(curva, titulo, caminho_salvar, ylabel):
    plt.figure(figsize=(8, 5))
    plt.plot(curva, color='red')
    plt.title(titulo)
    plt.xlabel('Épocas')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(caminho_salvar)
    plt.close()

def main():
    print(">>> Carregando imagens da pasta RecFac...")
    X, Y, mapa_classes = carregar_imagens_recfac()
    
    nomes_pessoas = [nome for nome, id_ in sorted(mapa_classes.items(), key=lambda x: x[1])]
    num_atributos = X.shape[0]
    num_classes = Y.shape[0]
    
    pasta_resultados = Path("resultados_etapa2")
    pasta_resultados.mkdir(exist_ok=True)
    
    rodadas = 10
    
    resultados = {
        'Perceptron': {'accs': [], 'melhor_acc': -1, 'pior_acc': float('inf'), 'melhor_pred': None, 'pior_pred': None, 'y_real_melhor': None, 'y_real_pior': None, 'curva_melhor': [], 'curva_pior': []},
        'ADALINE':    {'accs': [], 'melhor_acc': -1, 'pior_acc': float('inf'), 'melhor_pred': None, 'pior_pred': None, 'y_real_melhor': None, 'y_real_pior': None, 'curva_melhor': [], 'curva_pior': []},
        'MLP':        {'accs': [], 'melhor_acc': -1, 'pior_acc': float('inf'), 'melhor_pred': None, 'pior_pred': None, 'y_real_melhor': None, 'y_real_pior': None, 'curva_melhor': [], 'curva_pior': []}
    }
    
    print(f"\n>>> Iniciando Monte Carlo ({rodadas} rodadas)...")
    
    for r in range(rodadas):
        X_train, Y_train, X_test, Y_test_ids = particionar_dados_multiclasse(X, Y)
        
        # 1. Treinando Perceptron
        perc = PerceptronMulticlasse(num_atributos, num_classes, max_epochs=100, learning_rate=0.01)
        perc.fit(X_train, Y_train)
        pred_perc = perc.predict(X_test)
        acc_perc = np.mean(pred_perc == Y_test_ids)
        
        # 2. Treinando ADALINE
        ada = AdalineMulticlasse(num_atributos, num_classes, max_epochs=100, learning_rate=0.001)
        ada.fit(X_train, Y_train)
        pred_ada = ada.predict(X_test)
        acc_ada = np.mean(pred_ada == Y_test_ids)
        
        # 3. Treinando MLP (50 neurônios na camada oculta como teste)
        mlp = MLPMulticlasse(num_atributos, num_ocultos=50, num_saidas=num_classes, max_epochs=100, learning_rate=0.005)
        mlp.fit(X_train, Y_train)
        pred_mlp = mlp.predict(X_test)
        acc_mlp = np.mean(pred_mlp == Y_test_ids)
        
        print(f"Rodada {r+1:02d} | Perceptron Acc: {acc_perc:.4f} | ADALINE Acc: {acc_ada:.4f} | MLP: {acc_mlp:.4f}")
        
        # Atualizando melhores/piores casos MLP
        resultados['MLP']['accs'].append(acc_mlp)
        if acc_mlp > resultados['MLP']['melhor_acc']:
            resultados['MLP'].update({'melhor_acc': acc_mlp, 'melhor_pred': pred_mlp, 'y_real_melhor': Y_test_ids, 'curva_melhor': mlp.learning_curve})
        if acc_mlp < resultados['MLP']['pior_acc']:
            resultados['MLP'].update({'pior_acc': acc_mlp, 'pior_pred': pred_mlp, 'y_real_pior': Y_test_ids, 'curva_pior': mlp.learning_curve})
            
        # Atualizando melhores/piores casos Perceptron
        resultados['Perceptron']['accs'].append(acc_perc)
        if acc_perc > resultados['Perceptron']['melhor_acc']:
            resultados['Perceptron'].update({'melhor_acc': acc_perc, 'melhor_pred': pred_perc, 'y_real_melhor': Y_test_ids, 'curva_melhor': perc.learning_curve})
        if acc_perc < resultados['Perceptron']['pior_acc']:
            resultados['Perceptron'].update({'pior_acc': acc_perc, 'pior_pred': pred_perc, 'y_real_pior': Y_test_ids, 'curva_pior': perc.learning_curve})
            
        # Atualizando melhores/piores casos ADALINE
        resultados['ADALINE']['accs'].append(acc_ada)
        if acc_ada > resultados['ADALINE']['melhor_acc']:
            resultados['ADALINE'].update({'melhor_acc': acc_ada, 'melhor_pred': pred_ada, 'y_real_melhor': Y_test_ids, 'curva_melhor': ada.learning_curve})
        if acc_ada < resultados['ADALINE']['pior_acc']:
            resultados['ADALINE'].update({'pior_acc': acc_ada, 'pior_pred': pred_ada, 'y_real_pior': Y_test_ids, 'curva_pior': ada.learning_curve})

    print("\n>>> Resumo Estatístico (Acurácia):")
    for modelo, dados in resultados.items():
        media, dp = np.mean(dados['accs']), np.std(dados['accs'])
        maximo, minimo = np.max(dados['accs']), np.min(dados['accs'])
        print(f"{modelo:10} -> Média: {media:.4f} | Desvio: {dp:.4f} | Maior: {maximo:.4f} | Menor: {minimo:.4f}")
        
        # Gerando Matrizes e Curvas
        plotar_matriz_seaborn(dados['y_real_melhor'], dados['melhor_pred'], nomes_pessoas, f"{modelo} - MELHOR Caso (Acc: {dados['melhor_acc']:.4f})", pasta_resultados / f"matriz_melhor_{modelo}.png")
        plotar_matriz_seaborn(dados['y_real_pior'], dados['pior_pred'], nomes_pessoas, f"{modelo} - PIOR Caso (Acc: {dados['pior_acc']:.4f})", pasta_resultados / f"matriz_pior_{modelo}.png")
        
        y_label_curva = "Erros" if modelo == "Perceptron" else "EQM"
        plotar_curva(dados['curva_melhor'], f"Curva Melhor Caso - {modelo}", pasta_resultados / f"curva_melhor_{modelo}.png", y_label_curva)
        plotar_curva(dados['curva_pior'], f"Curva Pior Caso - {modelo}", pasta_resultados / f"curva_pior_{modelo}.png", y_label_curva)

    print("\n>>> Sucesso! Gráficos salvos na pasta 'resultados_etapa2'.")

if __name__ == "__main__":
    main()