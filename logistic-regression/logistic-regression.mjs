import tf from '@tensorflow/tfjs'

const defaultOptions = {
  learningRate: 0.1,
  iterations: 100,
  batchSize: null,
  decisionBoundary: 0.5,
}

class LogisticRegression {
  constructor(features, labels, options) {
    this.options = Object.assign(defaultOptions, options)

    features = tf.tensor(features)

    const { mean, variance } = tf.moments(features, 0)
    this.mean = mean
    this.variance = variance

    this.features = this._processFeatures(features)
    this.labels = tf.tensor(labels)

    this.weights = tf.zeros([this.features.shape[1], 1]) // [b, m1, m2, ..., mN]

    this.costHistory = []
    this.bHistory = []
  }

  train() {
    const { iterations, batchSize } = this.options

    if (!batchSize) {
      for (let i = 0; i < iterations; i++) {
        this._gradientDescent(this.features, this.labels)

        const cost = this._calculateCrossEntropy()
        this._adaptLearningRate(cost)
      }
    } else {
      const batchsAmount = Math.floor(this.features.shape[0] / batchSize)
      const batchSlice = makeBatchSlicer(batchSize)

      for (let i = 0; i < iterations; i++) {
        for (let j = 0; j < batchsAmount; j++) {
          const featuresBatch = batchSlice(this.features, j)
          const labelsBatch = batchSlice(this.labels, j)

          this._gradientDescent(featuresBatch, labelsBatch)
        }

        const cost = this._calculateCrossEntropy()
        this._adaptLearningRate(cost)
      }

      function makeBatchSlicer(batchSize) {
        return (tensor, j) => {
          return tensor.slice([batchSize * j, 0], [batchSize, -1])
        }
      }
    }

    return this
  }

  test(features, labels) {
    const predictions = this.predict(features)
    const incorrect = predictions.sub(tf.tensor(labels)).abs().sum().get()
    const total = predictions.shape[0]

    return (total - incorrect) / total
  }

  predict(observations) {
    const features = tf.tensor(observations)

    return this._processFeatures(features)
      .matMul(this.weights)
      .sigmoid()
      .greater(this.options.decisionBoundary)
      .cast('float32')
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
    const predictions = features.matMul(this.weights).sigmoid()

    const residuals = predictions.sub(labels)

    const derivatives = features
      .transpose()
      .matMul(residuals)
      .div(features.shape[0])

    this.weights = this.weights.sub(derivatives.mul(this.options.learningRate))

    return residuals
  }

  _calculateCrossEntropy() {
    const guesses = this.features.matMul(this.weights).sigmoid()

    const termOne = this.labels.transpose().matMul(guesses.log())

    const termTwo = this.labels
      .mul(-1)
      .add(1)
      .transpose()
      .matMul(guesses.mul(-1).add(1).log())

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

  _recordB() {
    this.bHistory.push(this.weights.get(0, 0))
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

export default LogisticRegression
