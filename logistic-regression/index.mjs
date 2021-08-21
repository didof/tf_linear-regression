import '@tensorflow/tfjs-node'
import loadCSV from '../utils/load-csv.mjs'
import LogisticRegression from './logistic-regression.mjs'

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'displacement', 'weight'],
  labelColumns: ['mpg'],
})
