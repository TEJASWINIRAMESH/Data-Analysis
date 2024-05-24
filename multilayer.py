# %%
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# %%
x=np.array([[0,0,1],
            [1,0,0],
            [1,1,0],
            [1,0,1]])
d=np.array([0,1,1,0])

# %%
model=Sequential(
       [Dense(4,input_dim=3,activation="relu"),
        Dense(4,activation="relu"),
        Dense(1,activation="sigmoid")]
)

# %%
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])

# %%
history1=model.fit(x,d,epochs=1000,validation_split=0.2,verbose=0)

# %%
loss,accuracy=model.evaluate(x,d,verbose=0)
print(loss,accuracy)

# %%
plt.plot(history1.history['loss'],label="training loss")
plt.plot(history1.history['val_loss'],label="validation loss")
plt.show()


