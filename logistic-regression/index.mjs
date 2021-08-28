import '@tensorflow/tfjs-node'
import loadCSV from '../utils/load-csv.mjs'
import LogisticRegression from './logistic-regression.mjs'
import plot from 'node-remote-plot'

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'displacement', 'weight'],
  labelColumns: ['passedemissions'],
  converters: {
    passedemissions: value => (value === 'TRUE' ? 1 : 0),
  },
})

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 20,
  batchSize: 10,
  decisionBoundary: 0.5,
})

regression.train()
regression.print()
const accuracy = regression.test(testFeatures, testLabels)
console.info(accuracy)

// const predictions = regression.predict([
//   [193, 304, 2.37],
//   [88, 97, 1.065],
// ])
// predictions.print()

plot({
  x: regression.costHistory.reverse(),
  xLabel: 'Iterations #',
  yLabel: 'Cost Function (Cross Entropy)',
  title: `CE minimization (${regression.options.batchSize} elms batching)`,
  name: `CE-iterations.batch${regression.options.batchSize}.plot`,
})

// plot({
//   x: regression.costHistory.reverse(),
//   xLabel: 'Iterations #',
//   yLabel: 'Cost Function (Cross Entropy)',
//   title: 'CE minimization (w/out batching)',
//   name: 'CE-iterations.noBatching.plot',
// })
