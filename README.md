## Linear Regression

### How it works

When you instantiate the `LinearRegression` class it is obviously necessary to provide it with the set of features (1) and the set of labels (2). Optionally it is possible to pass a configuration object as follows:

- **learningRate** (default 0.1): factor multiplied by the value of the derivative. It is used to decide how much to increase/decrease the weights for the next iteration of the gradient descent.

  - There are several methods considered to optimize the learning rate (_Adam_, _Adagrad_, _RMSProp_, _Momentum_) but in this demo a simple and custom method is implemented.

- **iterations** (default 100): the number of times the gradient descent is recalculated. Technically, the higher the number of iterations, the more weights are representative of the training dataSet (assuming an opportun learning rate).

- **batchSize** (default null): represents the number of items for each batch. If set to null, the gradient descent is calculated over the entire training dataset at each iteration. Setting it to any other number, at each iteration the dataset is sliced ​​in many batches.

The following methods can be called on the instance:

- **train**: uses the gradient descent to calculate the weights related to the dataset provided and the selected options.

- **test**: receives the testing dataset, returns R ^ 2 indicating the reliability of the line found.

- **predict**: receives a dataset of features and returns the corresponding predictions based on the line found during training.

### How to use

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

Finally, run `npm run regression:linear`.

In console you can find info about the weights and R^2. Uncomment the plot section to obtain plots in the fs.
