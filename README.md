# Linear Regression

## How to use

First install dependencies with `npm install`

In `index.js` you can tweak the options of the instantiation:

```js
const regression = new LinearRegession(features, labels, {
  learningRate: 0.1,
  iterations: 30,
  batchSize: null, // number | null
})
```

You could also try to change the features and the class label:

```js
let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'displacement', 'weight'], // features
  labelColumns: ['mpg'], // labels
})
```

Finally, run `node index.js`. Some info will be displayed in console, a plot will be created/updated in the fs.

### Learning Rate Optimization

There are some opinionated methods, but in this demo I implemented a simple, custom one.

- Adam
- Adagrad
- RMSProp
- Momentum
