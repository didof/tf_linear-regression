let preprocessed = {
  weights: null,
  means: null,
  variances: null,
}
getPreprocessedData().then(([weights, means, variances]) => {
  preprocessed = { weights, means, variances }
})

function getPreprocessedData() {
  function fetchPrepocessed(filename) {
    return fetch(`${filename}.txt`)
      .then(res => res.text())
      .then(data => {
        if (filename === 'weights') {
          return tf.tensor(data.split(':').map(row => row.split(',')))
        }

        return tf.tensor(data.split(','))
      })
  }

  const promises = Object.keys(preprocessed).map(fetchPrepocessed)

  return Promise.all(promises)
}

function demo(array) {
  let { weights, means, variances } = preprocessed

  if (Object.values(preprocessed).some(el => !Boolean(el))) {
    console.warn('The preprocessed data is not yet avaiable.')

    return
  }

  const features = tf.tensor(array)

  const processedFeatures = processFeatures(features)

  const predictions = processedFeatures
    .matMul(weights)
    .softmax()
    .argMax(1)
    .get(0)

  return predictions

  function prependColumnsOfOne(tensor) {
    const ones = tf.ones([tensor.shape[0], 1])
    return ones.concat(tensor, 1)
  }

  function processFeatures(features) {
    features = standardize(features)

    features = prependColumnsOfOne(features)

    return features
  }

  function standardize(features) {
    const flippedSignal = variances.cast('bool').logicalNot().cast('float32')

    return features
      .sub(means)
      .div(flippedSignal.add(variances).pow(tf.tensor(0.5)))
  }
}
