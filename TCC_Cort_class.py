
#--------------------------Parte 1------------------------------------

from minisom import MiniSom #Importando a classe MiniSom.
import numpy as np #Importando o numpy, pois será utilizado arrays.        
import matplotlib.pyplot as plt #Importando matplot
                                            #para a visualização.

#Abaixo será coletado os dados do csv, excluindo as 
    #duas primeiras linhas e a coluna '0'. Eles 
        #serão convertidos para um array.
data= np.genfromtxt('area_cortical.csv', delimiter= ';',
                     usecols= (1,2,3,4,5,6,7,8,9,10,11),
                                    skip_header= 2, dtype= float)

#Aqui será feito a mesma coisa que a anterior, mas
    #coletará apenas a primeira coluna.
labels= np.genfromtxt('area_cortical.csv', delimiter =';',
                       usecols= (0), skip_header= 2, dtype= str)

data #Realizar o print do array de dados.

print("Dados inseridos!! \n")

#--------------------------Parte 2------------------------------------

linha = 13
coluna = 13 #O tamanho x e y da rede.
itera = 85500
sigma = 0.4 #Largura efetiva da vizinhança.
apren = 0.1 #A taxa de aprendizagem.

#Instancia um objeto como classe MiniSom.
    #O '11' é local para se colocar o tamanho da entrada.

som = MiniSom(linha, coluna, 11, sigma, apren)

print("Rede SOM criada!! \n")

#--------------------------Parte 3------------------------------------

print("Gráfico das distâncias: 1")
#Função que exibe um gráfico
    #contendo a distância entre os
        #neurônios da rede.
#Aqui os pesos não foram distribuídos ainda na rede.
frequencies = som.activation_response(data)
plt.pcolor(frequencies.T, cmap='Blues') 
plt.colorbar()
plt.show()

#--------------------------Parte 4------------------------------------

#Inicia os pesos aleatoriamente da rede.
som.random_weights_init(data)

#--------------------------Parte 5------------------------------------

print("Gráfico das distâncias: 2")
# Segunda exibição do gráfico, após o 
    #distribuimento dos pesos.
frequencies = som.activation_response(data)
plt.pcolor(frequencies.T, cmap='Blues') 
plt.colorbar()
plt.show()

#--------------------------Parte 6------------------------------------

#Função que treina a rede sequencialmente. 
    #O treinamento será executado pelo número
        #de iterações escolhido. O 'verbose' faz
            #o print da execução do treinamento.
som.train_batch(data, itera, verbose= 1)

#--------------------------Parte 7------------------------------------

#Função que mostra a contagem dos
    #neurônios vencedores dentro da rede.
        #Essa exibição é feita em formato de matriz.
            #E coordenada que tem um vencedor é 
                #acrescentado em '1' sua contagem.
print ("\n",som.activation_response(data),"\n")

#--------------------------Parte 8------------------------------------

#Função que explicita o neurônio
    #vencedor e sua respectiva classificação.
print ("/n",som.labels_map(data, labels),"/n")

#--------------------------Parte 9------------------------------------

print("\n Gráfico das distâncias: 3")
#Terceira exibição do gráfico, feita após
    #a rede ter sido treinada.
frequencies = som.activation_response(data)
plt.pcolor(frequencies.T, cmap='Blues') 
plt.colorbar()
plt.show()

#--------------------------Parte 10------------------------------------
import pickle

#Esse comando abaixo grava o estado da rede SOM.
    #Portanto é bastante útil para guardar estado.
with open('som.p', 'wb') as outfile:
    pickle.dump(som, outfile)
    
#--------------------------Parte 11------------------------------------

#Esse comando é usado para carregar a rede SOM 
    #que foi gravada anteriormente.
with open('som.p', 'rb') as infile:
    som2 = pickle.load(infile)
    
print ("Rede pronta para o uso!!")


#--------------------------Parte 12------------------------------------

#Essa função foi feita para salvar as coordenadas
    #do neurônio vencedor para cada uma das áreas.
        #Elas são salvas num vetor de posições.
def area_cort(som, data):
    
    area = []
    
    for i in range(43):
        
        area.insert(i,som2.winner((data[i])))
            
    return area


#--------------------------Parte 13------------------------------------

#Essa função que faz a classificação de um determinado estímulo.
    #como parâmetros temos a rede SOM, os dados que serão utilizados e
        # o vetor com as áreas.

def classificar(som, data, area):
    
    #É escolhido um valor aleatoriamente dentro dos dados
        #e é calculado seu neurônio vencedor.
    estimulo = som.winner(random.choice(data))
   
    print("Coordenada do estímulo na rede: " + str(estimulo) + "\n") 
    
    #Então é comparado o vencedor de cada área com o vencedor do estímulo.
    if area[0] == estimulo:
        print ("Esta é a área 1 do Córtex Cerebral, também conhecida como Córtex Somatossensorial Primário ou S1.\n" +
               "Tem a função de processar as informações sobre as sensações do corpo.")
        return
    elif area[1] == estimulo:
        print ("Esta é a área 2 do Córtex Cerebral, também conhecida como Córtex Somatossensorial Secundário ou S2.\n" +
              "Tem a função de processar as informações sobre as sensações do corpo.")
        return
    elif area[2] == estimulo:
        print ("Esta é a área 3 do Córtex Cerebral, também conhecida como Córtex Somatossensorial Terciário ou S3.\n" +
              "Tem a função de processar as informações sobre as sensações do corpo.")
        return
    elif area[3] == estimulo:
        print ("Esta é a área 4 do Córtex Cerebral, também conhecida como Córtex Motor Primário.\n" +
              "Tem a função de controlar os movimentos voluntários do corpo.")
        return
    elif area[4] == estimulo:
        print ("Esta é a área 5 do Córtex Cerebral, também conhecida como Córtex Somatossensorial de ordem superior.\n" +
              "Tem a função de reconhecer o entorno por meio do tato.")
        return
    elif area[5] == estimulo:
        print ("Esta é a área 6 do Córtex Cerebral, também conhecida como Córtex Motor Auxiliar.\n" +
              "Tem a função de realizar o planejamento dos movimentos e dos olhos.")
        return
    elif area[6] == estimulo:
        print ("Esta é a área 7 do Córtex Cerebral, também conhecida como ́Área de Associação Parietal Posterior.\n" +
              "Tem a função de controlar a consciência espacial visuomotora e a percepção.") 
        return
    elif area[7] == estimulo:
        print ("Esta é a área 8 do Córtex Cerebral, também conhecida como os Campos Visuais Frontais.\n" +
              "Tem a função de controlar os movimentos sacádicos dos olhos.") 
        return
    elif area[8] == estimulo:
        print ("Esta é a área 9 do Córtex Cerebral, também conhecida como Córtex de Associação Pré-frontal Dorsolateral.\n" +
              "Tem a função de controlar as atividades cognitivas e da memória.") 
        return
    elif area[9] == estimulo:
        print ("Esta é a área 10 do Córtex Cerebral, também conhecida como Córtex de Associação Pré-frontal Anterior.\n" +
              "Tem a função de realizar atividades cognitivas e de planejamento.") 
        return
    elif area[10] == estimulo:
        print ("Esta é a área 11 e 12 do Córtex Cerebral, também conhecida como Córtex Orbifrontal.\n" +
              "Tem a função de realizar reconhecimento espacial com a visão.") 
        return
    elif area[11] == estimulo:
        print ("Esta é a área 13 do Córtex Cerebral, também conhecida como Córtex Insular.\n" +
              "Tem a função em atividades de sensação, decisão e movimento.")
        return
    elif area[12] == estimulo:
        print ("Esta é a área 17 do Córtex Cerebral, também conhecida como Córtex Visual Primário.\n" +
              "Tem a função de realizar processamentos envolvendo a visão.")
        return
    elif area[13] == estimulo:
        print ("Esta é a área 18 do Córtex Cerebral, também conhecida como Córtex Visual Secundário.\n" +
              "Tem a função de realizar processamentos envolvendo a visão e profundidade.")
        return
    elif area[14] == estimulo:
        print ("Esta é a área 19 do Córtex Cerebral, também conhecida como Córtex Visual de Ordem Superior.\n" +
              "Tem a função de realizar processamentos envolvendo a visão, profundidade, cor e objetos em movimento.")
        return
    elif area[15] == estimulo:
        print ("Esta é a área 20 do Córtex Cerebral, também conhecida como ́Área Inferotemporal Visual.\n" +
              "Tem a função de identificar formas usando a memória.")
        return
    elif area[16] == estimulo:
        print ("Esta é a área 21 do Córtex Cerebral, também conhecida como Área Inferotemporal Visual.\n" +
              "Tem a função de processar formas visuais e linguagem.") 
        return
    elif area[17] == estimulo:
        print ("Esta é a área 22 do Córtex Cerebral, também conhecida como Córtex Auditivo de Ordem Superior.\n" +
              "Tem a função de processar sons e compreensão da fala.") 
        return
    elif area[18] == estimulo:
        print ("Esta é a área 23 do Córtex Cerebral, também conhecida como Córtex de Associação Límbico.\n" +
              "Tem a função de realizar processamentos envolvendo as emoções e planejamento.") 
        return
    elif area[19] == estimulo:
        print ("Esta é a área 24 do Córtex Cerebral, também conhecida como Córtex de Associação Límbico.\n" +
              "Tem a função de realizar processamentos envolvendo as emoções e planejamento.")
        return
    elif area[20] == estimulo:
        print ("Esta é a área 25 do Córtex Cerebral, também conhecida como Córtex de Associação Límbico.\n" +
              "Tem a função de realizar processamentos envolvendo as emoções.") 
        return
    elif area[21] == estimulo:
        print ("Esta é a área 26 do Córtex Cerebral, também conhecida como Córtex de Associação Límbico.\n" +
               "Tem funções envolvidas com o aprendizagem motora.")
        return
    elif area[22] == estimulo:
        print ("Esta é a área 27 do Córtex Cerebral, também conhecida como Córtex de Associação Límbico.\n" +
              "Tem a função em processamentos de memória e de olfato.")
        return
    elif area[23] == estimulo:
        print ("Esta é a área 28 do Córtex Cerebral, também conhecida como Córtex Olfatório Primário.\n" +
              "Tem a função de atividades relacionadas ao olfato, emoções, memória e aprendizado.")
        return
    elif area[24] == estimulo:
        print ("Esta é a área 29 do Córtex Cerebral, também conhecida como Córtex Cingulado.\n" +
              "Tem a função de memória e localização espacial.")
        return
    elif area[25] == estimulo:
        print ("Esta é a área 30 do Córtex Cerebral, também conhecida como Córtex Cingulado.\n" +
              "Tem a função de regulação emocional.")
        return
    elif area[26] == estimulo:
        print ("Esta é a área 31 do Córtex Cerebral, também conhecida como Córtex Cingulado.\n" +
              "Tem a função de controlar muitas funções cognitivas e emoções.") 
        return
    elif area[27] == estimulo:
        print ("Esta é a área 32 do Córtex Cerebral, também conhecida como Córtex Cingulado.\n" +
              "Tem função em processos comunicativos.") 
        return
    elif area[28] == estimulo:
        print ("Esta é a área 33 do Córtex Cerebral, também conhecida como Córtex Cingulado.\n" +
              "Tem a função de processar estímulos envolvendo emoções.") 
        return
    elif area[29] == estimulo:
        print ("Esta é a área 34 do Córtex Cerebral, também conhecida como Córtex de Associação Límbico.\n" +
              "Tem funções envolvidas com a memória.") 
        return
    elif area[30] == estimulo:
        print ("Esta é a área 35 do Córtex Cerebral, também conhecida como Córtex de Associação Límbico.\n" +
              "Tem a função de lidar com atividades envolvidas com a memória e o olfato.") 
        return
    elif area[31] == estimulo:
        print ("Esta é a área 36 do Córtex Cerebral, também conhecida como Córtex de Associação Límbico.\n" +
              "Tem a função de processar estímulos envolvendo emoções e memória.")
        return
    elif area[32] == estimulo:
        print ("Esta é a área 37 do Córtex Cerebral, também conhecida como Córtex de Associação Parietal-temporal-occipital.\n" +
              "Tem diversas funções envolvendo a visão com atividades de leitura e percepção, além da fala.")
        return
    elif area[33] == estimulo:
        print ("Esta é a área 38 do Córtex Cerebral, também conhecida como Polo Temporal.\n" +
              "Tem a função de trabalhar com estímulos envolvendo olfato, emoções e personalidade.")
        return
    elif area[34] == estimulo:
        print ("Esta é a área 39 do Córtex Cerebral, também conhecida como Giro Angular.\n" +
              "Tem funções relacionadas a fala, memória, atenção e leitura.")
        return
    elif area[35] == estimulo:
        print ("Esta é a área 40 do Córtex Cerebral, também conhecida como Giro Supra-marginal.\n" +
              "Tem a função de atividades envolvendo a fala e emoções.")
        return
    elif area[36] == estimulo:
        print ("Esta é a área 41 do Córtex Cerebral, também conhecida como Córtex Auditivo Primário.\n" +
              "Tem a função de processar estímulos auditivos.") 
        return
    elif area[37] == estimulo:
        print ("Esta é a área 42 do Córtex Cerebral, também conhecida como Córtex Auditivo Secundário.\n" +
              "Tem a função de processar estímulos auditivos.") 
        return
    elif area[38] == estimulo:
        print ("Esta é a área 43 do Córtex Cerebral, também conhecida como Córtex Gustatório.\n" +
              "Tem a função de ser o principal processador do paladar.") 
        return
    elif area[39] == estimulo:
        print ("Esta é a área 44 do Córtex Cerebral, também conhecida como Área de Broca.\n" +
              "Tem a função de fala e planejamento dos movimentos.") 
        return
    elif area[40] == estimulo:
        print ("Esta é a área 45 do Córtex Cerebral, também conhecida como Córtex de Associação Pré-frontal.\n" +
              "Tem funções relacionadas a fala, cognição, pensamentos e planejamento de comportamento.") 
        return
    elif area[41] == estimulo:
        print ("Esta é a área 46 do Córtex Cerebral, também conhecida como Córtex Pré-frontal Dorsolateral.\n" +
              "Tem as função de lidar com a memória, cognição, pensamentos e movimentos dos olhos.")
        return
    elif area[42] == estimulo:
        print ("Esta é a área 47 do Córtex Cerebral, também conhecida como Opérculo Frontal.\n" +
              "Tem a função de trabalhar com estímulos da fala, de pensamentos e da cognição.")
        return
    else:
        print ("Não foi possível localizar o destino desse estímulo.")


#--------------------------Parte 14------------------------------------

#A instanciação da função área_cortical,
    #para que possa ser salva em um
        #vetor, que será usado na próxima função.
area = area_cort(som2, data)


#--------------------------Parte 15------------------------------------
import random #É necessário o random para escolher
                    #aleatoriamente dentro da função.

classificar(som2, data, area)


#--------------------------Testes------------------------------------

#Função feita para a realização do teste 3.
def teste3(som, data, area): 
    estimulo = som.winner(random.choice(data))

    for i in range(43):
        if area[i] == estimulo:
            return 1
        
    print("Ruído encontrado!")       
    return 0

#Execução do teste 3.
for x in range(100000):
    estado = teste3(som2, data, area)
    if estado == 0:
        break
if estado == 1:
    print("Não foi encontrado ruído!")

#Execução do teste 4.
data_teste= np.genfromtxt('teste4.csv', delimiter= ';',
                     usecols= (1,2,3,4,5,6,7,8,9,10,11),
                                    skip_header= 2, dtype= float)

classificar(som2, data_teste, area)
data_teste

