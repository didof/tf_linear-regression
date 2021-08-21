function extrapolateM(features, labels) {
  const x = tf.tensor(features)
  const y = tf.tensor(labels)

  const N = x.sum().mul(y.sum()).sub(x.mul(y, 1).sum().mul(2))

  const D = x.sum().pow(2).sub(x.pow(2).sum().mul(2))

  return N.div(D)
}
