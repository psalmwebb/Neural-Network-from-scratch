import numpy as np


class NeuralNets:

    def __init__(self,inputs,hidden,outputs,lr=0.3):
        self.bias_ih = np.random.randn(hidden,1)
        self.bias_ho = np.random.randn(outputs,1)
        self.weights_ih = np.random.randn(hidden,inputs)
        self.weights_ho = np.random.randn(outputs,hidden)
        self.lr = lr
        self.error = 0
        print("Hey :)")

    def sigmoid(self,x):
        return  1 / ( 1 + np.exp(-x))

    def relu(self,x,d=False):
        if d:
          # if d is true,this function returns the derivative of the rectilinear unit activation function.
          return 1 * (x > 0)
        else:
          # if not,it squishes it to a number between 0 and positive infinity
          return x * (x > 0)

    def softmax(self,x):
        #return number that sums up to 1.0
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)

    def train(self,x,targets):
         x=x.reshape(1,len(x))
         # Reshaping the targets output favouring the rows
         if type(targets) == np.ndarray:
             targets = targets.reshape(len(targets),1)
             # targets = targets.reshape(1,len(targets))
         hidden = self.sigmoid(np.dot( self.weights_ih,x.T) + self.bias_ih)

         outputs = self.softmax(np.dot(self.weights_ho,hidden) + self.bias_ho)
         # Calculating the error in the output
         outputs_error = (targets - outputs)
         # print(outputs_error)

         hidden_error = np.dot(self.weights_ho.T,outputs_error)

         # Finding the optimum weights and bias where the cost function is minimum
          # y = mx + b
         hidden_output_w_g = np.dot(self.lr * outputs_error,hidden.T)

         hidden_output_b_g = self.lr * outputs_error

         self.weights_ho+=hidden_output_w_g
         self.bias_ho+=hidden_output_b_g

#              input_hidden_w_g = np.dot(self.lr * hidden_error * self.relu(hidden,d=True),x)
#              input_hidden_b_g = self.lr * hidden_error * self.relu(hidden,d=True)
         input_hidden_w_g = np.dot(self.lr * hidden_error * hidden * (1 - hidden),x)
         input_hidden_b_g = self.lr * hidden_error * hidden * (1 - hidden)

         self.weights_ih+=input_hidden_w_g

         self.bias_ih+=input_hidden_b_g
         self.error = np.square(outputs_error)

    def pred(self,x,one=False):
         x=x.reshape(1,len(x))
         hidden = self.sigmoid(np.dot(self.weights_ih,x.T) + self.bias_ih)
         outputs = self.softmax(np.dot(self.weights_ho,hidden) + self.bias_ho)
         # return np.argmax(outputs,axis=1)
         if one:
            return np.argmax(outputs)
         return outputs

    def predict(self,X):
        all_pred=np.zeros(len(X))
        for i,x in enumerate(X):
            p=np.argmax(self.pred(x))
            all_pred[i] = p

        return all_pred.astype(int)


