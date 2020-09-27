# Fast-Gradient-Sign-Perturbation

This github repo is to demonstrate the fast gradient sign method for generating perturbations that can fool deep neural networks used for image classification applications.  I do this by first creating a CNN based network for classifying the MNIST dataset.  Next, I pick an image from the dataset and calculate the gradient of the cost function for that image with respect to the image.  Then I add the scaled gradient to the original image and compare the performance of the classification algorithm on the original image to the perturbed image.  This will reveal a worse performance on the perturbed image.

In this readme I will step through part of the code to explain how it works

After performing all the necessary library imports as well as importing the MNIST dataset I one hot enode the labels for the train and test set.  I also normalize the pixels by scaling them from 0-255 to 0-1.

#One hot encode the labels
y_cat_test=to_categorical(y_test,num_classes=10)
y_cat_train=to_categorical(y_train,num_classes=10)

#scale features
X_train=X_train/255
X_test=X_test/255

Next I create a fairly simple CNN based network with early stopping to classify the images.

#Create CNN based model
model=Sequential()

model.add(Conv2D(filters=32,kernel_size=(4,4),strides=(1,1),padding='valid',
                input_shape=(28,28,1),activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

# Output Layer softmax-->multiclass

model.add(Dense(10,activation='softmax'))

#keras.io/metrics

model.compile(loss='categorical_crossentropy',optimizer='adam',
             metrics=['accuracy'])
             
#Use early stopping to prevent overfitting
from tensorflow.keras.callbacks import EarlyStopping
early_stop=EarlyStopping(monitor='val_loss',patience=1)
model.fit(X_train,y_cat_train,epochs=10,validation_data=(X_test,y_cat_test),
         callbacks=[early_stop])
         
After this I pick an image from the dataset to perform the fast gradient sign method on.

#Pick an image to use the fast gradient sign method on and convert to a tensorflow tensor
adversarial_test=X_test[26]
adversarial_test=tf.cast(adversarial_test,tf.float32)
adversarial_test=tf.image.resize(adversarial_test,(28,28))
adversarial_test=adversarial_test[None,...]

#Get label of image for creating the adversarial example
adversarial_test_answer=y_cat_test[26]
adversarial_test_answer=tf.reshape(adversarial_test_answer,(1,10))

Next I create my function that calculates the gradient of the cost function of an image with respect to the image.  It takes the input image and label as an input.

def create_perturbation(input_image, input_label):
    with tf.GradientTape() as tape:
        #Make independent variable the input image
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, prediction)
     # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    
    return signed_grad
    
Ill go through this function in a bit of detail to explain each step.  All of this is performed with the gradient tape method in tensorflow.  First I tell tensorflow to look at the input image because the input image is the independant variable in the gradient.  Next I tell my model to predict the classification of my input image.  I use this to then calculate the loss function between the predicted label and the actual label, which is then used to calculate the gradient of the loss function with respect to the input image.  Then I return the sign of the gradient.  I do this because it scales the gradient to a -1, 0, or 1 instead of the gradient being of a very small magnitude.  This makes the gradient easier to scale when adding to the original image.

After creating the function for calculating the gradient/ generating the perturbation I call the function to generate the perturbation.

#Generate perturbations
perturbations = create_perturbation(adversarial_test, adversarial_test_answer)

I can then introduce the perturbation to the original image scaled by whatever scaling factor I want.  In this case I use 0.08

#Introduce perturbation to image
adv_x=adversarial_test+0.08*perturbations

Then I can compare the prediction of the original image to the perturbed image

model(adversarial_test)
<tf.Tensor: shape=(1, 10), dtype=float32, numpy=
array([[9.9866611e-07, 9.1984106e-08, 7.9529582e-06, 3.2559488e-05,
        9.3120590e-08, 2.2030828e-07, 2.7885732e-12, 9.9994719e-01,
        2.9226507e-08, 1.0813650e-05]], dtype=float32)>
        
#Show model prediction of perturbation of image
model(adv_x)

<tf.Tensor: shape=(1, 10), dtype=float32, numpy=
array([[1.7176197e-05, 1.6535258e-05, 1.8394503e-03, 9.2570549e-01,
        6.5586551e-06, 2.9130737e-04, 4.9319338e-10, 6.9887474e-02,
        1.6402168e-05, 2.2196618e-03]], dtype=float32)>
        
As you can see the fast gradient sign method causes the algorithm to misclassify a 7 as a 3
