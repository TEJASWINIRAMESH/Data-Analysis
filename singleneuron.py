# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
x=np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
d=np.array([0,1,1,1])

# %%
epochs=1000
mse=[]

w=2*np.random.random((3,1))-1
for _ in range(epochs):
       y=1/(1+np.exp(-(np.dot(x,w))))
       e=d.reshape(-1,1)-y
       dw=0.9*np.dot(x.T,(y*(1-y))*e)
       w+=dw
       mse.append(np.mean(e**2))
       

# %%
plt.plot(mse)
plt.title("mse vs epochs")
plt.show()

# %%


# %%



