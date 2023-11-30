import pandas as pd
from modelo import Modelo

# Fazendo a leitura dos dados.
df = pd.read_csv('https://raw.githubusercontent.com/marcelovca90-inatel/AG002/main/tic-tac-toe.csv')
mapeamento = {'o': -1, 'b': 0, 'x': 1, 'negativo': -1, 'positivo': 1}
# Aplicando o mapeamento
df.replace(mapeamento, inplace=True)
# Preparando os dados
features = df.drop('resultado', axis=1)
alvo = df['resultado']

modelo_tic_tac = Modelo(features, alvo)
modelo_tic_tac.iniciarModelo()
modelo_tic_tac.prepararDados()
modelo_tic_tac.treinar()
modelo_tic_tac.calcularAcuracia()

amostra_usuario = modelo_tic_tac.obterAmostraUsuario()

modelo_tic_tac.prever(amostra_usuario)