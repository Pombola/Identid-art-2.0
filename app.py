import os
from flask import Flask, request, render_template
import cv2
import numpy as np

app = Flask(__name__)

# Dicionário que mapeia nomes de imagens de referência 
imagens_referencia = {
    'mona.jpg': '"Mona Lisa" de Da Vinci: Pintada entre 1503 e 1506, durante o Renascimento italiano, refletindo a técnica realista e a busca pela representação humanística que caracterizaram esse movimento artístico.',
    'vangogh.jpg': '"A Noite Estrelada" de Van Gogh: Criada em 1889, durante o período em que Van Gogh estava internado em um hospital psiquiátrico, reflete o estilo pós-impressionista do artista, caracterizado por pinceladas expressivas e cores intensas, demonstrando emoções e visões do artista sobre o céu noturno.',
    'lavie.png': 'La Vie de Picasso: Criada em 1903 durante o período chamado "Período Azul" de Picasso, onde ele explorou temas de pobreza, solidão e tristeza. Este foi um período em que predominavam tons azuis e verdes em suas obras.',
}

#método identificador
def identificar_obra_de_arte(imagem_de_entrada):
    resultados = []
    imagem_disponivel = False

    for imagem_referencia_nome, descricao in imagens_referencia.items():
        imagem_referencia = cv2.imread(os.path.join('imgs', imagem_referencia_nome))

        #transformando em branco e preto
        if imagem_referencia is not None:
            imagem_disponivel = True
            imagem_de_entrada_gray = cv2.cvtColor(imagem_de_entrada, cv2.COLOR_BGR2GRAY)
            imagem_referencia_gray = cv2.cvtColor(imagem_referencia, cv2.COLOR_BGR2GRAY)

            sift = cv2.SIFT_create()
            keypoints_entrada, descritores_entrada = sift.detectAndCompute(imagem_de_entrada_gray, None)
            keypoints_referencia, descritores_referencia = sift.detectAndCompute(imagem_referencia_gray, None)

            bf = cv2.BFMatcher()
            correspondencias = bf.knnMatch(descritores_entrada, descritores_referencia, k=2)

            correspondencias_boas = []
            for m, n in correspondencias:
                if m.distance < 0.15 * n.distance:
                    correspondencias_boas.append(m)

            if len(correspondencias_boas) > 10:
                resultados.append(f"A imagem corresponde à obra: {descricao}")

    return resultados
#configuraçoes flask
@app.route('/', methods=['GET', 'POST'])
def index():
    resultados = None
    aviso = None

    if request.method == 'POST':
        imagem_de_entrada = request.files['imagem']
        if imagem_de_entrada:
            imagem = cv2.imdecode(np.frombuffer(imagem_de_entrada.read(), np.uint8), cv2.IMREAD_COLOR)
            resultados = identificar_obra_de_arte(imagem)
            if not resultados:
              aviso = "Ainda não conseguimos reconhecer esta obra :( "
            

    return render_template('index.html', resultados=resultados, aviso=aviso)

if __name__ == '__main__':
    app.run(debug=True)
