import '@tensorflow/tfjs-node'
import loadCSV from '../utils/load-csv.mjs'
import MultinominalLogisticRegression from './multinominal-logistic-regression.mjs'
import plot from 'node-remote-plot'
import _ from 'lodash'

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'displacement', 'weight'],
  labelColumns: ['mpg'],
  converters: {
    mpg: value => {
      const mpg = parseFloat(value)
      if (mpg < 15) return [1, 0, 0]
      if (mpg < 30) return [0, 1, 0]
      return [0, 0, 1]
    },
  },
})

const regression = new MultinominalLogisticRegression(
  features,
  _.flatMap(labels),
  {
    learningRate: 0.5,
    iterations: 20,
    batchSize: 10,
    decisionBoundary: 0.5,
  }
)

regression.train()
const accuracy = regression.test(testFeatures, _.flatMap(testLabels))
console.log(accuracy)

// regression.print()
// const accuracy = regression.test(testFeatures, testLabels)
// console.info(accuracy)

// plot({
//   x: regression.costHistory.reverse(),
//   xLabel: 'Iterations #',
//   yLabel: 'Cost Function (Cross Entropy)',
//   title: `CE minimization (${regression.options.batchSize} elms batching)`,
//   name: `CE-iterations.batch${regression.options.batchSize}.plot`,
// })

// plot({
//   x: regression.costHistory.reverse(),
//   xLabel: 'Iterations #',
//   yLabel: 'Cost Function (Cross Entropy)',
//   title: 'CE minimization (w/out batching)',
//   name: 'CE-iterations.noBatching.plot',
// })
