
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
# tf.compat.v1.enable_eager_execution()

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import math
from matplotlib import pyplot as plt
import gc

from celluloid import Camera


from scipy.stats import truncnorm

def get_truncated_normal(mean=5.5, sd=2, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


# SPATIAL_DOMAIN = [0,10]
# TEMPORAL_DOMAIN = [0,0]
L = 80
T = 5

rho = 8.92
Cp = 0.092
k = 0.95

c_square = k/(rho*Cp)

lambda_sqaure = (c_square*math.pi**2)/L**2

# Sampler for sampling the points

def point_sampler(batch_size):

    # Sample the initial points uniformly in time and spatial domain

    x = tf.random.uniform(shape=[batch_size,1],minval=-30,maxval=L+30)
    # x = tf.random.truncated_normal(shape=[batch_size,1], mean=40, stddev=30, dtype=tf.dtypes.float32, seed=None, name=None)
    # x = tf.convert_to_tensor(get_truncated_normal(mean=40,sd=10,low=0,upp=L).rvs([batch_size,1]))
    # x = tf.cast(x, tf.float32)
    # print(x.shape)
    # x = np.random.uniform(low=0, high=L, size=[batch_size,1])
    # t = np.random.uniform(low=0, high=T, size=[batch_size,1])

    t = tf.random.uniform(shape=[batch_size,1],minval=0,maxval=T+3)
    # t = tf.random.truncated_normal(shape=[batch_size, 1], mean=2.5, stddev=1, dtype=tf.dtypes.float32, seed=None, name=None)
    # t = tf.convert_to_tensor(get_truncated_normal(mean=2.5,sd=0.5,low=0,upp=T).rvs([batch_size,1]))
    # t = tf.cast(t, tf.float32)
   #  print(t.shape)
    x = tf.Variable(x, trainable=True)
    t = tf.Variable(t, trainable=True)



    x_t_leftBound = tf.Variable(tf.zeros(shape=[batch_size,1]))

    x_t_rightBound = tf.Variable(tf.zeros(shape=[batch_size,1])+L)


    # print(f'The shape of x and t array is {x_t.shape}')
    # print(f'The shape of x and t array for initial condition is{x_t_initial.shape}')
    # print(f'The shape of x and t array for right boundary condition{x_t_rightBound.shape}')
    # print(f'The shape of x and t array for left boundary condition{x_t_leftBound.shape}')

    return x,t,x_t_rightBound, x_t_leftBound






# model creator

def create_model(nn_architecture):

    input1 = keras.Input(shape=nn_architecture[0], name="input_layer1")

    input2 = keras.Input(shape=nn_architecture[0], name="input_layer2")

    inputs = keras.layers.concatenate([input1, input2])

    x = keras.layers.Dense(nn_architecture[1],activation='tanh', name="dense_layer_1")(inputs)
    # x = keras.layers.LeakyReLU()(x)

    for i, nunits in enumerate(nn_architecture[2:-1]):

        x = keras.layers.Dense(nunits,activation='tanh', name=f"dense_layer_{i+2}")(x)
        # x = keras.layers.LeakyReLU()(x)

    outputs = keras.layers.Dense(nn_architecture[-1], name="output_layer")(x)

    model = keras.Model(inputs=[input1,input2], outputs=outputs)

    model.summary()

    return model


# Loss Calculator

def calculate_loss(model, batchsize):

    x, t, x_t_rightBound, x_t_leftBound = point_sampler(batch_size=batchsize)

    t_init = tf.zeros_like(t)

    y = model([x, t])
    print(y.shape)

    with tf.GradientTape() as tape1:


        with tf.GradientTape(persistent=True) as tape2:

            y = model([x,t])

        dt = tape2.gradient(y,t)

        dx = tape2.gradient(y,x)

    dxx = tf.linalg.tensor_diag_part(tape1.jacobian(dx, x))

    print(dt.shape, dx.shape, dxx.shape)




    heat_equatn_loss = tf.square(dt-c_square*dxx)

    ic_loss = tf.square((tf.math.sin(tf.divide(math.pi*x,L)) - model([x,t_init])))

    left_bc_loss = tf.square(model([x_t_leftBound,t]) - tf.zeros_like(x_t_leftBound))

    right_bc_loss = tf.square(model([x_t_rightBound,t]) - tf.zeros_like(x_t_rightBound))



    return tf.reduce_mean(heat_equatn_loss+ic_loss+left_bc_loss+right_bc_loss), tf.reduce_mean(heat_equatn_loss), tf.reduce_mean(ic_loss), tf.reduce_mean(left_bc_loss + right_bc_loss )





# training loop



def training(model, lr, epochs, batchsize):

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    loss = []
    h_loss_lst = []
    ic_loss_lst = []
    bc_loss_lst = []

    epochs_list = [i for i in range(epochs)]

    for epoch in range(epochs):

        print(f"\nStart of epoch {epoch}")

        with tf.GradientTape() as tape:

            loss_value, h_loss,ic_loss, bc_loss = calculate_loss(model=model,batchsize=batchsize)


        grads = tape.gradient(loss_value, model.trainable_weights)

        optimizer.apply_gradients(zip(grads, model.trainable_weights))


        print(f'Epoch No {epoch} completed\n')

        loss.append(loss_value.numpy())
        h_loss_lst.append(h_loss.numpy())
        ic_loss_lst.append(ic_loss.numpy())
        bc_loss_lst.append(bc_loss.numpy())

    plt.figure(1,figsize=(15,15))

    plt.subplot(2,2,1)
    plt.plot(epochs_list,loss,'-r')
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_list, h_loss_lst, '-r')
    plt.xlabel('Epochs')
    plt.ylabel('Heat Equation Loss')

    plt.subplot(2, 2, 3)
    plt.plot(epochs_list, ic_loss_lst, '-r')
    plt.xlabel('Epochs')
    plt.ylabel('Initial Condition Loss')

    plt.subplot(2, 2, 4)
    plt.plot(epochs_list, bc_loss_lst, '-r')
    plt.xlabel('Epochs')
    plt.ylabel('Boundary Condition Loss')

    plt.savefig('All_Losses_uniform_tanh_20_50_20_2000.png')


def prediction(model):

    x = np.linspace(0, L, 100).reshape(-1,1)

    final_temp_ana = []
    t = np.linspace(0, T, 5)

    u_x_t = lambda x, t: np.sin((x * np.pi) / L) * np.exp(-lambda_sqaure * t)

    fig, ax = plt.subplots()
    camera = Camera(fig)

    ax.set_xlabel('Length (m)')
    ax.set_ylabel('Tempreature ($^0$C)')

    for i,_t in enumerate(t):

        t_nn = np.ones_like(x) * i

        final_temp = model([x, t_nn])

        final_temp_ana.append(u_x_t(x, _t))

        print(final_temp.shape)

        ax.plot(x, final_temp, '-b', x.flatten(), np.array(final_temp_ana)[i,:],'-r')

        ax.text(0.5, 1.01, "Time = {} secs ".format(int(i)), transform=ax.transAxes)

        ax.legend(['Analytical', 'Deep Galerkin Method'])

        camera.snap()
    #         plt.savefig('Results.png')

    anim = camera.animate()

    anim.save('solution_DGM_vs_Analytical_uniform_tanh_20_50_20.gif')


model = create_model([1,20,50,20,1])


# point_sampler(10)
training(model,lr=0.0001,epochs=20000,batchsize=500)

model.save('myModel_20_50_20_2000.h5')
prediction(model)


gc.collect()
