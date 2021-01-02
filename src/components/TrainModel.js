import React, { useState, useEffect } from 'react';
import Papa from 'papaparse';
import * as tf from '@tensorflow/tfjs';
import { getDataset } from './utilities';

const TrainModel = () => {
  // Load the X and Y dataset as soon as the component mounts
  // Create a model in TFjs
  // Train that model once the user presses on 'Train'
  // Add an option to download the trained model

  const [X, setX] = useState([]);
  const [Y, setY] = useState([]);

  const loadX = async () => {
    const response = await fetch('/data/X.csv');
    const reader = response.body.getReader();
    const result = await reader.read();
    const decoder = new TextDecoder('utf-8');
    const csv = decoder.decode(result.value);
    const results = Papa.parse(csv);
    const X = results.data;
    setX(X);
  };

  const loadY = async () => {
    const response = await fetch('/data/Y.csv');
    const reader = response.body.getReader();
    const result = await reader.read();
    const decoder = new TextDecoder('utf-8');
    const csv = decoder.decode(result.value);
    const results = Papa.parse(csv);
    const Y = results.data;
    setY(Y);
  };

  useEffect(() => {
    // Load the X and Y dataset from csv files
    loadX();
    loadY();

    // Prepare the dataset for training
  }, []);

  const handleTrainModel = async () => {
    console.log('Training');
    const learningRate = 0.01,
      numberOfEpochs = 10;

    const [xTrain, yTrain, xTest, yTest] = await getDataset(X, Y);

    // Define the topology of the model: two dense layers
    const model = tf.sequential();
    model.add(
      tf.layers.dense({
        units: 10,
        activation: 'relu',
        inputShape: [xTrain.shape[1]],
      })
    );
    model.add(
      tf.layers.dense({
        units: 2,
        activation: 'softmax',
      })
    );
    // model.summary();

    const optimizer = tf.train.adam(learningRate);
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    const history = await model.fit(xTrain, yTrain, {
      epochs: numberOfEpochs,
      validationData: [xTest, yTest],
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          console.log('Epoch: ' + epoch + ' Logs: ' + logs.loss);
          await tf.nextFrame();
        },
      },
    });

    // Do prediction on xTest and then compare the results
    // Calculate the accuracy using xTest
    const predictions = model.predict(xTest);
    const yPred = predictions.argMax(-1).dataSync();
    const yTrue = yTest.argMax(-1).dataSync();
    let correct = 0,
      wrong = 0;
    for (let i = 0; i < yTrue.length; i++) {
      if (yTrue[i] == yPred[i]) correct++;
      else wrong++;
    }
    console.log('Accuracy: ' + correct / yTrue.length);
  };

  return (
    <div>
      <button onClick={handleTrainModel}>Train the model</button>
    </div>
  );
};

export default TrainModel;
