from models.sudoku import Sudoku
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse


INIT_LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 128


argument_parser = argparse.ArgumentParser()
argument_parser.add_argument(
    '-m',
    '--model',
    required=True,
    help='Path to output model after training'
)

args = vars(argument_parser.parse_args())


print('Accessing MNIST...')
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

print('Configuring train and test data...')
# Add a channel dimension to the digist
trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))

# Scale data to the range (0 - 1)
trainData = trainData.astype('float32') / 255.0
testData = testData.astype('float32') / 255.0

# Convert the labels from integers to vectors
label_binarizer = LabelBinarizer()
trainLabels = label_binarizer.fit_transform(trainLabels)
testLabels = label_binarizer.transform(testLabels)

print('Compiling model...')
optimizer = Adam(lr=INIT_LEARNING_RATE)
model = Sudoku.build(width=28, height=28, depth=1, classes=10)
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

print('Train the network')
network = model.fit(
    trainData,
    trainLabels,
    validation_data=(testData, testLabels),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)

print('Evaluating network...')
predictions = model.predict(testData)
print(classification_report(
    testLabels.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=[str(x) for x in label_binarizer.classes_]
))

print('Serializing digit model...')
model.save(args['model'], save_format='h5')
