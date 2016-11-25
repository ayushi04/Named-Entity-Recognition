#!/usr/bin/python

#Viterbi PoS Tagging
#correct=0
#total=0
class Viterbi(object):
    correct=0
    total=0
    def __init__(self, input_params):
        self.states = input_params['states']
        self.observations = input_params['observations']
        self.start_probability = input_params['start_probability']
        self.transition_probability = input_params['transition_probability']
        self.emission_probability = input_params['emission_probability']
        #Executing Viterbi on the above parameters
        #self.viterbi()
    def viterbi(self):
        obs = self.observations
        states = self.states
        start_p = self.start_probability
        trans_p = self.transition_probability
        emit_p = self.emission_probability
        V = [{}]
        path = {}
        # Initialize base cases (t == 0)
        for y in states:
            temp=emit_p[y].get(obs[0],0)
            if(start_p.has_key(y)):
                V[0][y] = start_p[y] * temp
                path[y] = [y]
            else:
                V[0][y] = 0.0001 * temp
                path[y] = [y]
            # Run Viterbi for t > 0
        for t in range(1, len(obs)):
            V.append({})
            newpath = {}
            for y in states:
                temp=emit_p[y].get(obs[t],0)
                #temp2=trans_p[y0].get
                (prob, state) = max((V[t-1][y0] * trans_p[y0].get(y,0) * temp, y0) for y0 in states)
                V[t][y] = prob
                newpath[y] = path[state] + [y]
            path = newpath
        # if only one element is observed max is sought in the initialization values
        n = 0
        if len(obs)!=1:
            n = t
        (prob, state) = max((V[n][y], y) for y in states)
        toReturnObj = [prob, path[state]]
        finalPath= path[state]
        #print (finalPath)
        return toReturnObj
            

