function getElements(...identifiers) {
  return identifiers.map(identifiers => document.querySelector(identifiers))
}

function addEventListeners(canvas, dict) {
  Object.keys(dict).forEach(eventListenerLabel => {
    canvas.addEventListener(eventListenerLabel, dict[eventListenerLabel])
  })
}
