import '@tensorflow/tfjs-node'
import mnist from 'mnist-data'
import MultinominalLogisticRegression from './multinominal-logistic-regression.mjs'
import _ from 'lodash'

const mnistData = mnist.training(0, 60000)
const [features, labels] = processData(mnistData)

const regression = new MultinominalLogisticRegression(features, labels, {
  learningRate: 1,
  iterations: 20,
  batchSize: 100,
})

regression.train()
regression.saveWeights()
regression.saveMean()
regression.saveVariance()

const testMnistData = mnist.testing(0, 1000)
const [testFeatures, testLabels] = processData(testMnistData)

const accuracy = regression.test(testFeatures, testLabels)
console.log(`Accuracy:`, accuracy)

// const predictions = regression.predict(flatFeatures(test.images.values))
// predictions.print()

function processData(mnistData) {
  const flattedFeatures = flatFeatures(mnistData.images.values)
  const encodedLabels = encodeLabels(mnistData.labels.values)
  return [flattedFeatures, encodedLabels]
}

function flatFeatures(features) {
  return features.map(identity(_.flatMap))
  function identity(fn) {
    return arg => fn(arg)
  }
}

function encodeLabels(labels) {
  return labels.map(label => {
    const row = new Array(10).fill(0)
    row[label] = 1
    return row
  })
}
