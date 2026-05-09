from pathlib import Path
import cv2
import numpy as np

def carregar_imagens_recfac(caminho_base="RecFac", tamanho=(40, 40)):
    pasta = Path(caminho_base)
    if not pasta.is_absolute():
        pasta = Path(__file__).resolve().parent / pasta

    if not pasta.exists() or not pasta.is_dir():
        raise FileNotFoundError(f"Pasta principal nao encontrada: {pasta.resolve()}")

    imagens_achatadas = []
    rotulos_originais = []
    mapa_classes = {}
    id_classe_atual = 0

    extensoes_validas = {'.pgm', '.png', '.jpg', '.jpeg'}

    for subpasta in sorted(pasta.iterdir()):
        if subpasta.is_dir():
            nome_pessoa = subpasta.name 
            
            if nome_pessoa not in mapa_classes:
                mapa_classes[nome_pessoa] = id_classe_atual
                id_classe_atual += 1
            
            for arquivo in sorted(subpasta.iterdir()):
                if arquivo.is_file() and arquivo.suffix.lower() in extensoes_validas:
                    img = cv2.imread(str(arquivo.resolve()), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                        
                    img_redimensionada = cv2.resize(img, tamanho)
                    
                    imagens_achatadas.append(img_redimensionada.flatten())
                    
                    rotulos_originais.append(mapa_classes[nome_pessoa])

    X_bruto = np.array(imagens_achatadas, dtype=float)
    N = X_bruto.shape[0]
    C = len(mapa_classes)
    
    X_normalizado = X_bruto / 255.0 

    bias = np.ones((N, 1))
    X_com_bias = np.hstack((bias, X_normalizado))
    
    X = X_com_bias.T 

    Y = -np.ones((C, N))
    
    for i, classe_id in enumerate(rotulos_originais):
        Y[classe_id, i] = 1

    return X, Y, mapa_classes