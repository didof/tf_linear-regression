import tf from '@tensorflow/tfjs'
import fs from 'fs'
import path from 'path'

const defaultOptions = {
  learningRate: 0.1,
  iterations: 100,
  batchSize: 100,
  decisionBoundary: 0.5,
}

class MultinominalLogisticRegression {
  constructor(features, labels, options) {
    this.options = Object.assign(defaultOptions, options)

    features = tf.tensor(features)

    const { mean, variance } = tf.moments(features, 0)
    this.mean = mean
    this.variance = variance

    this.features = this._processFeatures(features)
    this.labels = tf.tensor(labels)

    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]])

    this.costHistory = []
    this.bHistory = []
  }

  train() {
    const { iterations, batchSize } = this.options
    const batchsAmount = Math.floor(this.features.shape[0] / batchSize)
    const batchSlice = makeBatchSlicer(batchSize)

    console.info('Start Training')

    for (let i = 0; i < iterations; i++) {
      for (let j = 0; j < batchsAmount; j++) {
        const featuresBatch = batchSlice(this.features, j)
        const labelsBatch = batchSlice(this.labels, j)

        this._gradientDescent(featuresBatch, labelsBatch)
      }

      const cost = this._calculateCrossEntropy()
      this._adaptLearningRate(cost)

      console.info(`Iteration #` + i)
    }

    console.info('End Trainig')

    function makeBatchSlicer(batchSize) {
      return (tensor, j) => {
        return tensor.slice([batchSize * j, 0], [batchSize, -1])
      }
    }

    return this
  }

  saveWeights() {
    const filename = path.join(process.cwd(), 'mnist', 'public', 'weights.txt')
    const values = this.weights.dataSync()

    const stringWeights = this.weights
      .unstack()
      .map(row => row.dataSync().join(','))
      .join(':')

    fs.writeFile(filename, stringWeights, function (err) {
      if (err) return console.log(err)
      console.log(`Weights saved in ${filename}`)
    })
  }

  saveMean() {
    const filename = path.join(process.cwd(), 'mnist', 'public', 'means.txt')

    const arr = Array.from(this.mean.dataSync()).join(',')

    fs.writeFile(filename, arr, function (err) {
      if (err) return console.log(err)
      console.log(`Mean saved in ${filename}`)
    })
  }

  saveVariance() {
    const filename = path.join(
      process.cwd(),
      'mnist',
      'public',
      'variances.txt'
    )
    const arr = Array.from(this.variance.dataSync()).join(',')

    fs.writeFile(filename, arr, function (err) {
      if (err) return console.log(err)
      console.log(`Variance saved in ${filename}`)
    })
  }

  test(features, labels) {
    const predictions = this.predict(features)
    const testLabels = tf.tensor(labels).argMax(1)

    const incorrect = predictions.notEqual(testLabels).sum().get()

    const total = predictions.shape[0]
    return (total - incorrect) / total
  }

  predict(observations) {
    const features = tf.tensor(observations)
    const predictions = this._processFeatures(features)
      .matMul(this.weights)
      .softmax()
      .argMax(1)

    return predictions
  }

  print() {
    console.info('iterations amount:', this.options.iterations)
    console.info('learning rate:', this.options.learningRate)
    this.weights.unstack().forEach((w, i) => {
      const label = (!i ? 'b' : `m${i}`) + ':'
      console.info(label, this.weights.get(i, 0))
    })
  }

  _gradientDescent(features, labels) {
    const predictions = features.matMul(this.weights).softmax()

    const residuals = predictions.sub(labels)

    const derivatives = features
      .transpose()
      .matMul(residuals)
      .div(features.shape[0])

    this.weights = this.weights.sub(derivatives.mul(this.options.learningRate))
  }

  _calculateCrossEntropy() {
    const predictions = this.features.matMul(this.weights).softmax()

    const termOne = this.labels.transpose().matMul(predictions.log())

    const termTwo = this.labels
      .mul(-1)
      .add(1)
      .transpose()
      .matMul(predictions.mul(-1).add(1).log())

    return termOne.add(termTwo).div(this.features.shape[0]).mul(-1).get(0, 0)
  }

  _adaptLearningRate(cost) {
    if (cost > this.costHistory[0]) {
      this.options.learningRate /= 2
    } else {
      this.options.learningRate *= 1.01
    }
    this.costHistory.unshift(cost)
  }

  _prependColumnsOfOne(tensor) {
    const ones = tf.ones([tensor.shape[0], 1])
    return ones.concat(tensor, 1)
  }

  _processFeatures(features) {
    features = this._standardize(features)
    features = this._prependColumnsOfOne(features)

    return features
  }

  _standardize(features) {
    const flippedSignal = this.variance
      .cast('bool')
      .logicalNot()
      .cast('float32')
    return features
      .sub(this.mean)
      .div(flippedSignal.add(this.variance).pow(0.5))
  }
}

export default MultinominalLogisticRegression
