import tensorflow as tf
import time


devices = tf.config.list_physical_devices()
print("Available devices:", devices)

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data
X_train = train_images.reshape((60000, 784)).astype('float32') / 255
X_test = test_images.reshape((10000, 784)).astype('float32') / 255

y_train = tf.keras.utils.to_categorical(train_labels)
y_test = tf.keras.utils.to_categorical(test_labels)

# Define the model
with tf.device('/GPU:0'):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model on GPU
    
    history = model.fit(X_train, y_train, epochs=10, batch_size=1000, validation_data=(X_test, y_test))

    # Evaluate the model on GPU
    start = time.time()
    test_loss_gpu, test_accuracy_gpu = model.evaluate(X_test, y_test,verbose=0,batch_size=1000)
    end = time.time()
    print(X_test.shape)
    print("GPU Evaluation Time:", end-start)
    print('GPU Test Accuracy:', test_accuracy_gpu)
    print('GPU Test Loss:', test_loss_gpu)
