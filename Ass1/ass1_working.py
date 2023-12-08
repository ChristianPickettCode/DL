import math
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + (math.exp(-x)))

def softmax(i, outputs):
    den = 0
    for j in range(len(outputs)):
        den += math.e ** outputs[j]
        
    if den == 0:
        return 0
    
    return (math.e ** outputs[i]) / den

def cross_entropy(y, t):
    ce = -1 * t * math.log(y)
    return ce

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = arr.max() - arr.min()    
    for i in arr:
        temp = (((i - arr.min())*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def target_arr(num_cls, t):
    arr = [0] * num_cls
    arr[t] = 1
    return arr

class MLP:
    def __init__(self):
        self.logs = False
    
    def forward(self, x_inputs, w_weights, b_weights, v_weights, c_weights):
        k_pre = [0.0, 0.0, 0.0]
        h_hidden = [0.0, 0.0, 0.0]
        o_raw = [0.0, 0.0]
        y_outputs = [0.0, 0.0]
        
        if self.logs: print(f'x_inputs: {x_inputs}')
        
        for j in range(3): 
            for i in range(2):
                k_pre[j] += w_weights[i][j] * x_inputs[i]
            k_pre[j] += b_weights[j]
        if self.logs: print(f'k_pre: {k_pre}') 
        
        for i in range(3):
            h_hidden[i] = sigmoid(k_pre[i])
        if self.logs: print(f'h_hidden: {h_hidden}') 
        
        for j in range(2): 
            for i in range(3): 
                o_raw[j] += v_weights[i][j] * h_hidden[i] 
            o_raw[j] += c_weights[j]
        if self.logs: print(f'o_raw: {o_raw}') 
        
        
        for i in range(2):
            y_outputs[i] = softmax(i, o_raw)
        if self.logs: print(f'y_outputs: {y_outputs}') 
        
        return y_outputs, h_hidden
            
        
    def loss(self, y_outputs, t_targets):
        # print(y_outputs, t_targets)
        loss = 0
        for i in range(2):
            loss += cross_entropy(y_outputs[i], t_targets[i])
        if self.logs: print(f'loss: {loss}')  
        
        return loss
        
    def backward(self, y_outputs, t_targets, h_hidden, v_weights, x_inputs, dldw, dldb, dldv, dldc):
        dldh = [0.0, 0.0, 0.0]
        dldk = [0.0, 0.0, 0.0]
        dldo = [0.0, 0.0]
        # dldy = [0.0, 0.0]
        
        for i in range(2):
            dldo[i] += y_outputs[i] - t_targets[i]
        if self.logs: print(f'dldo: {dldo}')
        
        for j in range(3):
            for i in range(2):
                dldv[j][i] += dldo[i] * h_hidden[j]
                dldh[j] += dldo[i] * v_weights[j][i]      
        if self.logs: print(f'dldv: {dldv}')
        if self.logs: print(f'dldh: {dldh}')
        
        dldc = dldo
        if self.logs: print(f'dldc: {dldc}')
        
        for i in range(3):
            dldk[i] += dldh[i] * h_hidden[i] * (1 - h_hidden[i]) 

        if self.logs: print(f'dldk: {dldk}')
            
        # print('stuff: ', dldk, x_inputs)
        for j in range(3):
            for i in range(2):
                dldw[i][j] += dldk[j] * x_inputs[i] 
            
        dldb = dldk

        if self.logs: print(f'dldw: {dldw}')
        if self.logs: print(f'dldb: {dldb}')
        
        return dldw, dldb, dldv, dldc
        
    def set_inputs(self, x_inputs):
        self.x_inputs = x_inputs
        
    def set_targets(self, t_targets):
        self.t_targets = t_targets
        
    def set_learning_rate(self, lr):
        self.lr = lr
        
    def sgd(self, lr, w_weights, b_weights, v_weights, c_weights, dldw, dldb, dldv, dldc):
        for j in range(3):
            for i in range(2):
                diff = -lr * dldv[j][i]
                # if logs: print(f'diff v: {diff}')
                v_weights[j][i] += diff
                
        for j in range(2):
            for i in range(3):
                diff = -lr * dldw[j][i]
                # if logs: print(f'diff w: {diff}')
                w_weights[j][i] += diff
                
        for j in range(2):
            diff = -lr * dldc[j]
            # if logs: print(f'diff w: {diff}')
            c_weights[j] += diff
        
        for j in range(3):
            diff = -lr * dldb[j]
            # if logs: print(f'diff w: {diff}')
            b_weights[j] += diff
            
        return v_weights, w_weights, c_weights, b_weights
    
from data import load_synth
(xtrain, ytrain), (xval, yval), num_cls = load_synth()

a_learning_rate = 1e-4
mlp = MLP()
mlp.set_learning_rate(a_learning_rate)

losses = []
epochs = 1

dldh = [0.0, 0.0, 0.0]
dldk = [0.0, 0.0, 0.0]
dldo = [0.0, 0.0]
dldy = [0.0, 0.0]

dldw = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
dldv = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
dldb = [0.0, 0.0, 0.0]
dldc = [0.0, 0.0]
     
w_weights = [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]
b_weights = [0.0, 0.0, 0.0] 
v_weights = [[1.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]]  # rand_weights(3, 2) # # 
c_weights = [0.0, 0.0]

norm_train_x = normalize(xtrain, 0, 1)
norm_val_x = normalize(xval, 0, 1)

norm_train_y = normalize(ytrain, 0, 1)
norm_val_y = normalize(yval, 0, 1)

for j in range(1):
    loss_avg = []
    
    dldw = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    dldb = [0.0, 0.0, 0.0]
    dldv = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    dldc = [0.0, 0.0]
    
    for i in range(len(norm_train_x)):
        x1, x2 = norm_train_x[i]
        y = 0 if norm_train_y[i] == 1 else 1 
        t_targets = target_arr(num_cls, y)
        
        x_inputs = [x1, x2]
        
        y_outputs, h_hidden = mlp.forward(x_inputs, w_weights, b_weights, v_weights, c_weights)
        loss = mlp.loss(y_outputs, t_targets)
        
        w_pri, b_pri, v_pri, c_pri = mlp.backward(y_outputs, t_targets, h_hidden, v_weights, x_inputs, dldw, dldb, dldv, dldc)
        
        dldw += w_pri
        dldb += b_pri
        dldv += v_pri
        dldc += c_pri
        
        
        if i % 100 == 0:
            print(f'Run : {i}')
            losses.append(loss)
            
        
            
    v_weights, w_weights, c_weights, b_weights = mlp.sgd(a_learning_rate, w_weights, b_weights, v_weights, c_weights, dldw, dldb, dldv, dldc)
    
    

loss_train = losses
episodes = range(0,len(losses))
plt.plot(episodes, loss_train, 'g', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
    