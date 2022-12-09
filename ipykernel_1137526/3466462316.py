from numpy import random
n_dimension = train_data.shape[1]
init_A = random.random(size=(n_dimension,n_dimension))
ref_after =  np.dot(ref_train_data, init_A)
print(ref_after.shape)