import controller_template as controller_template
import numpy

class Controller(controller_template.Controller):
    def __init__(self, track, evaluate=True, bot_type=None):
        super().__init__(track, evaluate=evaluate, bot_type=bot_type)
        self.old_sensors = []

    #######################################################################
    ##### METHODS YOU NEED TO IMPLEMENT ###################################
    #######################################################################

    

    def take_action(self, parameters: list) -> int:
        """
        :param parameters: Current weights/parameters of your controller
        :return: An integer corresponding to an action:
        1 - Right
        2 - Left
        3 - Accelerate
        4 - Brake
        5 - Nothing
        """
        features = self.compute_features(self.sensors)
        Q = []
        Q.append(0);
        for i in range(0, 5*len(features), len(features)):
            Q.append(
                parameters[i]*features['distToGrass'] 
                + parameters[i+1]*features['checkpointDist'] 
                + parameters[i+2]*features['onTrack']
                + parameters[i+3]*features['middle']
                )

        return Q.index(max(Q))

        #raise NotImplementedError("This Method Must Be Implemented")



    def compute_features(self, sensors):
        """
        :param sensors: Car sensors at the current state s_t of the race/game
        contains (in order):
        0    track_distance_left: 1-100
        1    track_distance_center: 1-100
        2    track_distance_right: 1-100
        3    on_track: 0 or 1
        4    checkpoint_distance: 0-???
        5    car_velocity: 10-200
        6    enemy_distance: -1 or 0-???
        7    position_angle: -180 to 180
        8    enemy_detected: 0 or 1
          (see the specification file/manual for more details)
        :return: A list containing the features you defined
        """
        self.normalize_sensors(sensors)
        features = {}
        
        features['distToGrass'] = sensors[1]
        features['checkpointDist'] = sensors[4]
        features['onTrack'] = sensors[3]
        features['middle'] = sensors[2] - sensors[0]

        if len(self.old_sensors) != 0:
            features['distToGrass'] = self.old_sensors[1] - sensors[1]
            features['checkpointDist'] = self.old_sensors[4] - sensors[4]
            features['onTrack'] = self.old_sensors[3] - sensors[3]
            features['middle'] = sensors[2] - sensors[0]


        self.old_sensors = sensors
        return features
        #raise NotImplementedError("This Method Must Be Implemented")
 


    def learn(self, weights) -> list:
        """
        IMPLEMENT YOUR LEARNING METHOD (i.e. YOUR LOCAL SEARCH ALGORITHM) HERE

        HINT: you can call self.run_episode (see controller_template.py) to evaluate a given set of weights
        :param weights: initial weights of the controller (either loaded from a file or generated randomly)
        :return: the best weights found by your learning algorithm, after the learning process is over
        """
        individuos_na_populacao = 100
        quantidade_de_melhores = 50
        quantidade_de_features, V = self.gera_populacao_inicial(individuos_na_populacao)
        maximo_iteracoes = 5
        iteracoes = 1
        media_antiga = 0
        porcentagem_de_mutacao = 0.1
        while True:

            #avalia populacao            
            for i in range(individuos_na_populacao):
                V[i]["pontuacao"] = self.run_episode(V[i]["estado"])

            V = sorted(V, key=lambda vizinho: vizinho["pontuacao"], reverse=True)
            # calcula final do algoritmo
            if(iteracoes == maximo_iteracoes):
                return V[0]["estado"]

            media_atual = 0
            for i in range(len(V)):
                media_atual += V[i]["pontuacao"]
            media_atual = media_atual/len(V) 
            
            #selecao elitista
            melhores = []      
            for i in range(quantidade_de_melhores):
                melhores.append(V[i].copy())             

            #reproducao
            mascara = []
            for i in range(5*quantidade_de_features):
                if i%3 == 0:
                    mascara.append(1)
                else: 
                    mascara.append(0)
            
            V = []
            for i in range(0,quantidade_de_melhores,2):
                par1 = {"estado": [], "pontuacao": 0}
                par2 = {"estado": [], "pontuacao": 0}
                for j in range(5*quantidade_de_features):
                    if mascara[j] == 1:
                        par1["estado"].append(melhores[i]["estado"][j])
                        par2["estado"].append(melhores[i+1]["estado"][j])
                    else:
                        par1["estado"].append(melhores[i+1]["estado"][j])
                        par2["estado"].append(melhores[i]["estado"][j])    
                V.append(melhores[i])
                V.append(melhores[i+1])
                V.append(par1)
                V.append(par2)

            #se não variou muito aumenta a chance de mutacao
            if media_antiga != 0 and 0.9 <= abs(media_atual)/abs(media_antiga) <= 1.1:
                porcentagem_de_mutacao += 0.05
           
            media_antiga = media_atual
            #muta
            for i in range(len(V)):
                if numpy.random.ranf() < porcentagem_de_mutacao:
                    j = numpy.random.randint(0,len(V[i]["estado"])-1)
                    rnd = numpy.random.normal(0.0, 1.0)
                    V[i]["estado"][j] += rnd
            
            iteracoes += 1
            
        #raise NotImplementedError("This Method Must Be Implemented")

    def normalize_sensors(self, sensors):
        sensors[0] = (sensors[0] - 1) / (100-1)
        sensors[1] = (sensors[1] - 1) / (100-1)
        sensors[2] = (sensors[2] - 1) / (100-1)
        if sensors[4] > 400:
            sensors[4] = 400
        sensors[4] = (sensors[4] - 0) / (400)

    def gera_populacao_inicial(self, n):
        sensors = [7, 45, 73, 1, 1, 1, 1, 1, 1]
        features = self.compute_features(sensors)
        V_inicial = []
        media = 0.0
        desvio_padrao = 1

        pontuacao_inicial = 0
        for i in range(n):
            par = {"estado": [], "pontuacao": pontuacao_inicial}
            v = []
            for j in range(5*len(features)):
                rnd = numpy.random.normal(media, desvio_padrao)
                v.append(rnd)
            par["estado"] = v.copy()
            V_inicial.append(par)
        self.old_sensors = []
        return (len(features), V_inicial)
                