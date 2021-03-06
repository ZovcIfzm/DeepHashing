This explanation doc will try to explain how the tensorflow model works
by going through what is run through each step of the dummy code.
# model(sample_data) Runthrough
  ``model = DeepHash([64,32,20],1,.001,.001, initialize_W(sample_data,64))``
Like normal code, this just instantiates the model

  ``y = model(sample_data)``
THIS IS NOT LIKE NORMAL PYTHON CODE. The important thing to note is that the
DeepHash model was instantiated as,
  ``class DeepHash(Model):``
Where Model was imported in,
  ``from tensorflow.keras.models import Model``
A property of Model is that it has adds a "build" and "call" functionality that
allows you to call the model like a function.

As a result, when you call
  ``y = model(sample_data)``
The __call__ function inherent to a tensorflow model is called, which then calls
the ``build`` function that create the layer weights if it hasn't already been called,
as well as the ``call`` function where you define what you want to happen when you call
the model like a function.

Inside "build", the layers are defined as a list in fc_layers, where each layer is
appended as a "Dense" layer, which is also imported in
  ``from tensorflow.keras.layers import Dense``
Within the Dense layer, various parameters are set, including the layer weights, layer biases, weight regularizers, and bias regularizers.
  ``kernal_initializer`` is the parameter that sets the layer weights.

Within our model class, the only other functions besides __init__, call, and build are functions that determine the custom weights we want which are defined in the paper. Thus, as far as I understand, how our model works is that we've defined the ``__init__`` class to save our model inputs, ``build`` to build our layers based on those inputs, and ``call`` to define what runs when we call the model like a function, e.g. ``model(sample_data)``. The rest of the functions like ``custom_W_init()`` are used within these functions when we're setting custom parameters as defined in the paper.

So when you call ``model(sample_data)``, ``__call__`` is called which calls ``build`` then ``call``. These functions train the model with sample_data. Inside the call function, there is an additional function inherent to tensorflow models which is add_loss, this just keeps tracks of losses.

# How loss is calculated
For both the unsupervised and supervised loss

# train_unsupervised runthrough
This runs the DH algorithm outlined in the paper.
  ``loss = tf.reduce_sum(model.losses)``
Finds the value of the objective function
  ``grad = tape.gradient(loss, model.trainable_variables)``
Obtains the gradients of the objective function (loss) w.r.t. the trainable variables (W, c)
  ``optimizer.apply_gradients(zip(grad, model.trainable_variables))``
Updates W^m and C^m
```
if i > 1 and abs(loss-old_loss) < conv_error:
              print("loss flattened below convergence error of", conv_error)
              return
```
Stops the function once the loss flattens

# train_supervised runthrough
To be added

