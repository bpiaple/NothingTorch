import tensorflow as tf

image_path = "5.png"

def print_prediction(pred):
    class_names = [str(i) for i in range(10)]
    predicted_class = class_names[tf.argmax(pred[0])]
    probability = pred[0][tf.argmax(pred[0])]
    confidence = probability * 100
    print(f"Predicted class: {predicted_class} with {confidence:.2f}% confidence.")

image = tf.io.read_file(image_path)
image = tf.image.decode_image(image, channels=1)
image = tf.image.resize(image, [28, 28])
image = tf.cast(image, tf.float32) / 255.0       

image = tf.squeeze(image, axis=2)
image = tf.expand_dims(image, 0)
# model = tf.keras.models.load_model("best_mnist_model.keras")
model = tf.keras.models.load_model("mnist_model.keras")

prediction = model.predict(image)
print(prediction)

print_prediction(prediction)


