from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

class Modelo:

    def __init__(self,features,alvo):
        self.df_colunas = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.features = features
        self.alvo = alvo

    # Dividir o conjunto de dados em treinamento (80%) e teste (20%)
    def prepararDados(self):
        self.features_treino, self.features_teste, self.alvo_treino,self.alvo_teste = train_test_split(self.features, self.alvo, test_size=0.2, random_state=42)

    def iniciarModelo(self):
        self.modelo_knn = KNeighborsClassifier()

    # treinando o modelo com os dados de treinamento
    def treinar(self):
       self.modelo_knn.fit(self.features_treino, self.alvo_treino)

    # Calcular a acurácia do modelo
    def calcularAcuracia(self):
        previsoes = self.modelo_knn.predict(self.features_teste)
        acuracia = accuracy_score(self.alvo_teste, previsoes)
        print(f'\n Acurácia do modelo: {acuracia:.2f}')

    def prever(self, entrada):
        previsao = self.modelo_knn.predict(entrada)
        if(previsao[0]== 1):
            print("Vitória!")
        else:
            print("Derrota!")
    
    def obterAmostraUsuario(self):
        try:
            entrada_usuario = input("Insira 9 números (0, 1 ou -1) separados por espaço: ")
            # Converter a entrada em uma lista de números
            array_numerico = [int(numero) for numero in entrada_usuario.split()]

            # Verificar se foram inseridos exatamente 9 números
            if len(array_numerico) != 9:
                raise ValueError("Você deve inserir exatamente 9 números.")

            # Criar um DataFrame do Pandas com a entrada do usuário
            df_usuario = pd.DataFrame([array_numerico], columns=self.df_colunas)
            return df_usuario
        except ValueError as e:
            print(f"Erro: {e}")
            return None