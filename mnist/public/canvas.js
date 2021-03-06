const canvas = document.querySelector('canvas')
const ctx = canvas.getContext('2d')
const guess = document.getElementById('guess')

window.addEventListener('scroll', event => event.preventDefault())

const offset = createOffsetObject()
const ctxOptions = createContextOptions(ctx)

const canvasActions = createDrawer({ canvas, ctx, offset })

enableButtons(canvasActions)

function createDrawer({ canvas, ctx, offset }) {
  let isDrawing = false

  const onTouchMove = createTouchAdapters(canvas)

  const dict = {
    mousemove: onMousemove,
    mouseleave: onMouseleave,
    mousedown: onMousedown,
    mouseup: onMouseup,
    touchstart: onMousedown,
    touchcancel: onMouseleave,
    touchend: onMouseleave,
    touchmove: onTouchMove,
  }

  addEventListeners(canvas, dict)
  createBackground()

  return {
    clear: createBackground,
  }

  function onMousemove({ offsetX, offsetY }) {
    offset.update(offsetX, offsetY)
    offset.render()

    if (!isDrawing) return

    ctx.lineTo(offset.x, offset.y)
    ctx.stroke()
  }

  function onMouseleave() {
    offset.clear()
    offset.render()
    onMouseup()
    ctx.beginPath()
  }

  function onMousedown() {
    isDrawing = true
    ctx.beginPath()
  }

  function onMouseup() {
    isDrawing = false
    ctx.closePath()
  }

  function createBackground() {
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    guess.innerHTML = ''
  }

  function createTouchAdapters(element) {
    const { left, top } = element.getBoundingClientRect()

    return onTouchMove

    function onTouchMove(event) {
      const { pageX, pageY } = event.targetTouches[0]
      const offsetX = Math.floor(pageX - left)
      const offsetY = Math.floor(pageY - top)
      onMousemove({ offsetX, offsetY })
    }
  }
}

function createOffsetObject() {
  const offset = {
    x: null,
    y: null,
    update,
    clear,
    render: createRenderer(),
  }

  return offset

  function update(x, y) {
    offset.x = x
    offset.y = y
  }

  function clear() {
    update(null, null)
  }

  function createRenderer() {
    const [offsetXSpan, offsetYSpan] = getElements('#offset_x', '#offset_y')

    return function renderOffsetObject() {
      offsetXSpan.innerHTML = offset.x ? `x: ${offset.x}` : ''
      offsetYSpan.innerHTML = offset.y ? `y: ${offset.y}` : ''
    }
  }
}

function createContextOptions(ctx) {
  const ctxOptionsObject = decorate({
    lineWidth: {
      value: 8,
    },
    lineCap: {
      value: 'round',
    },
  })

  bootstrap()

  return ctxOptionsObject

  function bootstrap() {
    Object.keys(ctxOptionsObject).forEach(key => {
      ctxOptionsObject[key].update()
      ctxOptionsObject[key].render()
    })
  }

  function decorate(dict) {
    return Object.keys(dict).reduce((acc, cur) => {
      acc[cur].update = createUpdater(cur)
      acc[cur].render = createRenderer(cur)
      return acc
    }, dict)

    function createUpdater(key) {
      return function update(value = ctxOptionsObject[key].value) {
        ctxOptionsObject[key].value = value
        ctx[key] = value
      }
    }

    function createRenderer(key) {
      const el = document.getElementById(key)
      return function render() {
        el.innerHTML = ctxOptionsObject[key].value
      }
    }
  }
}

function enableButtons(actions) {
  const [printBtn, clearBtn, demoBtn] = getElements(
    '#print_btn',
    '#clear_btn',
    '#demo_btn'
  )

  enableClearButton()
  enablePrintButton()
  enableDemoButton()

  function enableClearButton() {
    clearBtn.addEventListener('click', actions.clear)
  }

  function enablePrintButton() {
    printBtn.addEventListener('click', onPrintBtnClick)

    function onPrintBtnClick() {
      const url = canvas.toDataURL('image/jpeg', 1.0)
      const image = new Image()
      image.src = url

      window.open('').document.write(image.outerHTML)
    }
  }

  function enableDemoButton() {
    demoBtn.addEventListener('click', execDemo)

    function execDemo() {
      const [canvas, ctx] = generatePhantom()
      const data = getData(canvas, ctx)

      const prediction = demo([data])
      guess.innerHTML = prediction
    }
  }

  function generatePhantom() {
    const P = 28
    const phantomCanvas = document.createElement('canvas')
    phantomCanvas.width = P
    phantomCanvas.height = P
    const phantomCtx = phantomCanvas.getContext('2d')
    phantomCtx.fillStyle = 'red'
    phantomCtx.fillRect(0, 0, P, P)

    phantomCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, P, P)

    return [phantomCanvas, phantomCtx]
  }

  function getData(canvas, ctx) {
    const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height)

    return isolateAndInvertAlpha(data)

    function isolateAndInvertAlpha(array) {
      let isolated = []

      for (let i = 0; i < array.length; i++) {
        if (i % 4) continue
        isolated.push(array[i] ^ 0xff)
      }

      return isolated
    }
  }
}
