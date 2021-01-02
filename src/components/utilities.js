import * as tf from '@tensorflow/tfjs';

export const YOGA_CLASSES = ['yoga1', 'yoga2'];
export const YOGA_NAME_TO_LABEL = {
  yoga1: 0,
  yoga2: 1,
};
const NUM_YOGA_CLASSES = YOGA_CLASSES.length;

const testSplit = 0.15;

const convertToTensors = (data, targets, testSplit) => {
  const numExamples = data.length;
  if (numExamples !== targets.length) {
    throw new Error('data and targets have different number of examples');
  }

  // Randomly shuffle data and targets
  const indices = [];
  for (let i = 0; i < numExamples; i++) {
    indices.push(i);
  }

  tf.util.shuffle(indices);
  const shuffledData = [];
  const shuffledTargets = [];
  for (let i = 0; i < numExamples; i++) {
    shuffledData.push(data[indices[i]]);
    shuffledTargets.push(targets[indices[i]]);
  }

  // Split the data into a training set and a test set, based on 'testSplit'
  const numTestExamples = Math.round(numExamples * testSplit);
  const numTrainExamples = numExamples - numTestExamples;

  const xDims = shuffledData[0].length; // The number of features in each training point

  // Create a 2D tf.Tensor to hold the feature data
  const xs = tf.tensor2d(shuffledData, [numExamples, xDims]);

  // Create a 1D tf.Tensor to hold the labels (targets), and convert the number label
  // from the set {0, 1, ...., NUM_YOGA_CLASSES-1} into one-hot encoding (Eg.: 0 --> [1, 0, 0])
  const ys = tf.oneHot(tf.tensor1d(shuffledTargets).toInt(), NUM_YOGA_CLASSES);

  // Split the data into training and test sets using 'slice'
  const xTrain = xs.slice([0, 0], [numTrainExamples, xDims]);
  const xTest = xs.slice([numTrainExamples, 0], [numTestExamples, xDims]);
  const yTrain = ys.slice([0, 0], [numTrainExamples, NUM_YOGA_CLASSES]);
  const yTest = ys.slice(
    [numTrainExamples, 0],
    [numTestExamples, NUM_YOGA_CLASSES]
  );
  return [xTrain, yTrain, xTest, yTest];
};

export const getDataset = (X, Y) => {
  return tf.tidy(() => {
    // First clean X and Y, i.e.
    // convert the string values of X into proper numbers
    // and replace the class names in Y with values
    for (let i = 0; i < X.length; i++) {
      for (let j = 0; j < X[i].length; j++) {
        X[i][j] = +X[i][j] / 400;
      }
      Y[i][0] = YOGA_NAME_TO_LABEL[Y[i][0]];
    }
    // console.log(X);
    // Split the data into different arrays for each of the class, i.e.
    // segregate the data points according to their class
    // We do this so as to get same proportion of text split
    // from all the classes
    const dataByClass = [];
    const targetsByClass = [];
    for (let i = 0; i < YOGA_CLASSES.length; i++) {
      dataByClass.push([]);
      targetsByClass.push([]);
    }

    for (let i = 0; i < X.length; i++) {
      const target = Y[i][0];
      const data = X[i];
      dataByClass[target].push(data);
      targetsByClass[target].push(target);
    }

    const xTrains = [];
    const yTrains = [];
    const xTests = [];
    const yTests = [];
    // Convert these values into tensors
    for (let i = 0; i < YOGA_CLASSES.length; i++) {
      const [xTrain, yTrain, xTest, yTest] = convertToTensors(
        dataByClass[i],
        targetsByClass[i],
        testSplit
      );
      xTrains.push(xTrain);
      yTrains.push(yTrain);
      xTests.push(xTest);
      yTests.push(yTest);
    }

    // Get a linear set of tensors to feed into the training
    // instead of feeding the 2D one to reduce the overall complexity
    const concatAxis = 0;
    return [
      tf.concat(xTrains, concatAxis),
      tf.concat(yTrains, concatAxis),
      tf.concat(xTests, concatAxis),
      tf.concat(yTests, concatAxis),
    ];

    // return ['a', 'b', 'c', 'd'];
  });
};
