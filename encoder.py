import numpy as np




class NeuralNetwork:
    def __init__(self,d_model,N,d_hidden,X):
        
        self.d_model=d_model
        self.d_hidden=d_hidden
        self.N=N
        self.X=X
        head=2

        #Initialize weights
        self.w_q=np.random.rand(head,self.d_model,self.d_model)
        self.w_k=np.random.rand(head,self.d_model,self.d_model)
        self.w_v=np.random.rand(head,self.d_model,self.d_model)
        self.wsigma=np.random.rand(self.d_model,1)
        self.w0=np.random.rand(head*self.d_model,self.d_model)

        self.w1=np.random.rand(self.d_model,self.d_hidden)
        self.bias1=np.zeros(self.d_hidden)
        self.w2=np.random.rand(self.d_hidden,self.d_model)
        self.bias2=np.zeros(self.d_model)

        

        #|i-j| in prior association
        self.rcd=np.abs(np.indices((N,N))[0]-np.indices((N,N))[1])

    def Relu(self,x):
        return np.maximum(0,x)


    def symmetric_kl_divergence(self,p_row,s_row):
        epsilon=1e-9
        p_row=p_row+epsilon
        s_row=s_row+epsilon
        kl_ps=np.sum(p_row*np.log(p_row/s_row))
        kl_sp=np.sum(s_row*np.log(s_row/p_row))
    
        return kl_ps+kl_sp
    def feedforward(self,X):
        x_hidden = np.dot(X,self.w1)+self.bias1
        x_hidden_activated=self.Relu(x_hidden)
        output=np.dot(x_hidden_activated,self.w2)+self.bias2
        return output
    
    def backward_min():
        pass




    def attention(self,X,head):
        for epochs in range(10):
            sigma=np.dot(X,self.wsigma)
            sigma=np.maximum(sigma,1e-9)
            normalize=1/(np.sqrt(2*np.pi)*sigma)
            self.priass=normalize*np.exp(-0.5*((self.rcd/sigma)**2))
            consf=[]
            ass_dis_lis=[]
            for i in range(head):
                q=np.dot(X,self.w_q[i])
                k=np.dot(X,self.w_k[i])
                v=np.dot(X,self.w_v[i])
    
                s=np.dot(q,k.T)/np.sqrt(self.d_model)
                s=np.exp(s)
                s=s/np.sum(s,axis=-1)


                ass_dis=np.array([self.symmetric_kl_divergence(self.priass[j,:],s[j,:])for j in range(self.priass.shape[0])])/len(self.priass)
                ass_dis_lis.append(ass_dis)


                consf.append(np.dot(s,v))
                
            Z=np.concatenate(consf,axis=-1)
            Z=np.dot(Z,self.w0)
            
            avg_assdiss=np.mean(np.array(ass_dis_lis),axis=0)
            _X=self.feedforward(Z)

            print("assdiss:",avg_assdiss)
            print("output _X:",np.round(_X,4))
            if epochs%2==0:
                #minimization phase
                L_t=np.square(X-_X) + lamda*np.abs(avg_assdiss) 
                self.backward_min(L_t,X)
                #here only updation of the weights also takes place
            else:
                L_t=np.square(X-_X) - lamda*np.abs(avg_assdiss)   
                self.backward_max()
                #here only the updation of the weights takes place     














X=np.array([
    [1.0, 0.9, 1.2, 1.1],    # Normal
    [1.1, 0.95, 1.3, 1.05],  # Normal
    [1.2, 1.0, 1.1, 0.9],    # Normal
    [2.5, 3.0, 2.8, 2.7],    # Anomaly
    [1.15, 0.97, 1.25, 1.0], # Normal
    [1.18, 1.02, 1.22, 1.1], # Normal
    [1.2, 1.1, 1.0, 0.95],   # Normal
    [0.8, 0.7, 1.5, 3.5],    # Anomaly
    [1.1, 0.92, 1.3, 1.05],  # Normal
    [1.05, 1.0, 1.15, 0.98], # Normal
])
model=NeuralNetwork(4,10,12,X)
model.attention(X,2)