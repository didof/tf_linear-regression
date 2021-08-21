import '@tensorflow/tfjs-node'
import loadCSV from './load-csv.mjs'
import LinearRegession from './linear-regression.mjs'
import plot from 'node-remote-plot'

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'displacement', 'weight'],
  labelColumns: ['mpg'],
})

const regression = new LinearRegession(features, labels, {
  learningRate: 0.1,
  iterations: 30,
  batchSize: null,
})

regression.train()

const r2 = regression.test(testFeatures, testLabels)

regression.print()
console.info('r2:', r2)

const predictions = regression.predict([
  [90, 112, 1.4],
  [120, 380, 2],
])
predictions.print()

// plot({
//   x: regression.mseHistory.reverse(),
//   xLabel: 'Iterations #',
//   yLabel: 'Mean Squared Error',
//   title: 'MSE development (no batching)',
//   name: 'MSE-iterations.plot',
// })

// plot({
//   x: regression.bHistory,
//   y: regression.mseHistory.reverse(),
//   xLabel: 'Value of B',
//   yLabel: 'Mean Squared Error',
//   title: 'MSE respect to changes in B',
//   name: 'MSE-b.plot',
// })
