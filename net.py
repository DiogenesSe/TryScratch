class Network:
    def __init__(self, list):
        self.list = list

    def anwenden(self, data):
        zw = data
        for i in range(len(self.list)):
            zw = self.list[i].forward(zw)
        return zw

    def train(self, data, y):
        predictions = [data]
        for i in range(len(self.list)):
            predictions.append(self.list[i].forward(predictions[i]))
        predictions.reverse()
        #predictions.append([data])
        output_delta = []
        self.list.reverse()
        output_delta.append(self.list[0].backward(predictions[1], y, predictions[0],0,0 ))
        for i in range(1, len(self.list)):
            output_delta.append(self.list[i].backward(predictions[i+1],y,predictions[i], output_delta[i-1], self.list[i-1].getMatrix() ))
        self.list.reverse()
        #hidden_1 = neuron.forward(data)
        #output = neuron2.forward(hidden_1)
        #output_delta = neuron2.backward(hidden_1, y, output, 0, 0)
        #hidden_1 = neuron.backward(data, y, hidden_1, output_delta, neuron2.getMatrix())