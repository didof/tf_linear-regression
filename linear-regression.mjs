import tf from '@tensorflow/tfjs'

const defaultOptions = {
  learningRate: 0.1,
  iterations: 100,
  batchSize: null,
}

export class BaseLinearRegression {
  constructor(features, labels, options) {
    this.features = features
    this.labels = labels

    this.options = Object.freeze(Object.assign(defaultOptions, options))

    this.b = 0
    this.m = 0
  }

  train() {
    const { iterations } = this.options

    for (let i = 0; i < iterations; i++) {
      this._gradientDescent()
    }

    return this
  }

  print() {
    console.log('m', this.m)
    console.log('b', this.b)
  }

  _gradientDescent() {
    const remainers = this.features.map(row => {
      return this.m * row[0] + this.b
    })

    const derivativeRespectToB =
      remainers
        .map((remainer, index) => {
          return remainer - this.labels[index][0]
        })
        .reduce((acc, cur) => acc + cur, 0) / this.labels.length

    const derivativeRespectToM =
      remainers
        .map((remainer, index) => {
          return (
            (this.labels[index][0] - remainer) * this.features[index][0] * -1
          )
        })
        .reduce((acc, cur) => acc + cur, 0) / this.labels.length

    const { learningRate } = this.options

    this.b = this.b - learningRate * derivativeRespectToB
    this.m = this.m - learningRate * derivativeRespectToM
  }
}

class LinearRegession {
  constructor(features, labels, options) {
    this.options = Object.assign(defaultOptions, options)

    features = tf.tensor(features)

    const { mean, variance } = tf.moments(features, 0)
    this.mean = mean
    this.variance = variance

    this.features = this._processFeatures(features)
    this.labels = tf.tensor(labels)

    this.weights = tf.zeros([this.features.shape[1], 1]) // [b, m1, m2, ..., mN]

    this.mseHistory = []
    this.bHistory = []
  }

  train() {
    const { iterations, batchSize } = this.options

    if (!batchSize) {
      for (let i = 0; i < iterations; i++) {
        this._gradientDescent(this.features, this.labels)

        const MSE = this._calculateMSE()
        this._adaptLearningRate(MSE)
        this.bHistory.push(this.weights.get(0, 0))
      }

      return
    }

    const batchsAmount = Math.floor(this.features.shape[0] / batchSize)
    const batchSlice = makeBatchSlicer(batchSize)

    for (let i = 0; i < iterations; i++) {
      for (let j = 0; j < batchsAmount; j++) {
        const featuresBatch = batchSlice(this.features, j)
        const labelsBatch = batchSlice(this.labels, j)

        this._gradientDescent(featuresBatch, labelsBatch)
      }

      const MSE = this._calculateMSE()
      this._adaptLearningRate(MSE)
      this._recordB()
    }

    return this

    function makeBatchSlicer(batchSize) {
      return (tensor, j) => {
        return tensor.slice([batchSize * j, 0], [batchSize, -1])
      }
    }
  }

  test(features, labels) {
    features = this._processFeatures(tf.tensor(features))
    labels = tf.tensor(labels)

    const SSfit = this._calculateSSfit(features, labels)
    const SSmean = this._calculateSSmean(labels)

    return 1 - SSfit / SSmean
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
    const predictions = features.matMul(this.weights)

    const residuals = predictions.sub(labels)

    const derivatives = features
      .transpose()
      .matMul(residuals)
      .div(features.shape[0])

    this.weights = this.weights.sub(derivatives.mul(this.options.learningRate))

    return residuals
  }

  _calculateMSE() {
    const predictions = this.features.matMul(this.weights)

    const residuals = predictions.sub(this.labels)

    return residuals.pow(2).sum().div(this.features.shape[0]).get()
  }

  _adaptLearningRate(MSE) {
    if (MSE > this.mseHistory[0]) {
      this.options.learningRate /= 2
    } else {
      this.options.learningRate *= 1.01
    }
    this.mseHistory.unshift(MSE)
  }

  _recordB() {
    this.bHistory.push(this.weights.get(0, 0))
  }

  _calculateSSmean(labels) {
    return labels.sub(labels.mean()).pow(2).sum().get()
  }

  _calculateSSfit(features, labels) {
    const predictions = features.matMul(this.weights)
    return labels.sub(predictions).pow(2).sum().get()
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
    return features.sub(this.mean).div(this.variance.pow(0.5))
  }
}

export default LinearRegession
