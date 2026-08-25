import sharp from 'sharp'
import fs from 'node:fs'

// ---------- background removal via border flood fill ----------
async function cutout(file) {
  const { data, info } = await sharp(file).ensureAlpha().raw().toBuffer({ resolveWithObject: true })
  const { width: w, height: h } = info
  const bg = new Uint8Array(w * h) // 1 = background
  const queue = []

  const idx = (x, y) => y * w + x
  const col = (i) => [data[i * 4], data[i * 4 + 1], data[i * 4 + 2]]

  // Global background model: both source images have desaturated
  // (gray/white) backgrounds while the characters are colorful or dark.
  // A pixel can be background only if it is low-chroma AND bright enough.
  // Flood fill from the borders provides connectivity, so desaturated
  // areas INSIDE the character (white tee, teeth) are never touched.
  const isBgColor = (i) => {
    const [r, g, b] = col(i)
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b)
    const chroma = mx - mn
    const lum = (r + g + b) / 3
    return chroma < 30 && lum > 105
  }

  // seed from all border pixels that match the background model
  for (let x = 0; x < w; x++) {
    for (const i of [idx(x, 0), idx(x, h - 1)]) {
      if (!bg[i] && isBgColor(i)) { bg[i] = 1; queue.push(i) }
    }
  }
  for (let y = 0; y < h; y++) {
    for (const i of [idx(0, y), idx(w - 1, y)]) {
      if (!bg[i] && isBgColor(i)) { bg[i] = 1; queue.push(i) }
    }
  }

  let head = 0
  while (head < queue.length) {
    const i = queue[head++]
    const x = i % w, y = (i / w) | 0
    const neighbors = []
    if (x > 0) neighbors.push(i - 1)
    if (x < w - 1) neighbors.push(i + 1)
    if (y > 0) neighbors.push(i - w)
    if (y < h - 1) neighbors.push(i + w)
    for (const n of neighbors) {
      if (!bg[n] && isBgColor(n)) { bg[n] = 1; queue.push(n) }
    }
  }

  // apply alpha with 1px edge feather (average of neighbors)
  const out = Buffer.from(data)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = idx(x, y)
      if (bg[i]) { out[i * 4 + 3] = 0; continue }
      // feather: if any 4-neighbor is background, soften alpha
      let bgN = 0, tot = 0
      for (const [dx, dy] of [[-1,0],[1,0],[0,-1],[0,1]]) {
        const nx = x + dx, ny = y + dy
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue
        tot++; if (bg[idx(nx, ny)]) bgN++
      }
      if (bgN > 0) out[i * 4 + 3] = Math.round(255 * (1 - bgN / (tot + 1)))
    }
  }

  // bounding box of character
  let minX = w, minY = h, maxX = 0, maxY = 0
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (!bg[idx(x, y)]) {
      if (x < minX) minX = x; if (x > maxX) maxX = x
      if (y < minY) minY = y; if (y > maxY) maxY = y
    }
  }

  const img = sharp(out, { raw: { width: w, height: h, channels: 4 } }).png()
  return { img, box: { minX, minY, maxX, maxY, w, h } }
}

// ---------- build one DP variant ----------
async function buildDP({ boy, girl, bgColor, outFile }) {
  const SIZE = 1600 // hi-res master; YouTube wants >= 98x98, 800x800 recommended

  // head-and-shoulders crop: from top of head down to a fraction of char height.
  // Crops extend deep enough that, after scaling, the cut edge falls BELOW the
  // canvas — so no flat cut line is ever visible inside the circle.
  async function bust(cut, frac) {
    const { minX, minY, maxX, maxY } = cut.box
    const charH = maxY - minY
    const cropH = Math.round(charH * frac)
    const buf = await cut.img.png().toBuffer()
    return sharp(buf).extract({
      left: minX, top: minY,
      width: maxX - minX + 1, height: Math.min(cropH, cut.box.h - minY),
    }).png().toBuffer()
  }

  const boyBust = await bust(boy, 0.62)
  const girlBust = await bust(girl, 0.70)

  // scale: both heads roughly equal size, filling the frame
  const boyMeta = await sharp(boyBust).metadata()
  const girlMeta = await sharp(girlBust).metadata()

  // heads keep the same visual size as before; extra crop depth just runs
  // off the bottom of the canvas (boy: 0.20 + 0.84 > 1, girl: 0.30 + 0.76 > 1)
  const targetBoyH = Math.round(SIZE * 0.84)
  const targetGirlH = Math.round(SIZE * 0.76)
  const boyR = await sharp(boyBust).resize({ height: targetBoyH }).png().toBuffer()
  const girlR = await sharp(girlBust).resize({ height: targetGirlH }).png().toBuffer()
  const bm = await sharp(boyR).metadata()
  const gm = await sharp(girlR).metadata()

  // Position: boy slightly left/up, girl right, overlapping like a duo poster.
  // Safe zone: circle of diameter SIZE centered — keep faces well inside.
  const boyLeft = Math.round(SIZE * 0.30 - bm.width / 2)
  const boyTop = Math.round(SIZE * 0.20)
  const girlLeft = Math.round(SIZE * 0.65 - gm.width / 2)
  const girlTop = Math.round(SIZE * 0.30)

  const canvas = sharp({
    create: { width: SIZE, height: SIZE, channels: 4, background: bgColor },
  })

  await canvas
    .composite([
      { input: boyR, left: boyLeft, top: boyTop },
      { input: girlR, left: girlLeft, top: girlTop },
    ])
    .png()
    .toFile(outFile)

  // circular preview (what YouTube shows)
  const circleMask = Buffer.from(
    `<svg width="${SIZE}" height="${SIZE}"><circle cx="${SIZE/2}" cy="${SIZE/2}" r="${SIZE/2}" fill="white"/></svg>`
  )
  await sharp(outFile)
    .composite([{ input: circleMask, blend: 'dest-in' }])
    .png()
    .toFile(outFile.replace('.png', '-circle.png'))

  console.log('[v0] wrote', outFile)
}

const boy = await cutout('public/characters/boy.png')
const girl = await cutout('public/characters/girl.png')
console.log('[v0] boy box', boy.box, 'girl box', girl.box)

fs.mkdirSync('public/dp', { recursive: true })
await buildDP({ boy, girl, bgColor: { r: 255, g: 199, b: 44, alpha: 1 }, outFile: 'public/dp/dp-amber.png' })
await buildDP({ boy, girl, bgColor: { r: 16, g: 42, b: 67, alpha: 1 }, outFile: 'public/dp/dp-navy.png' })
await buildDP({ boy, girl, bgColor: { r: 200, g: 16, b: 46, alpha: 1 }, outFile: 'public/dp/dp-red.png' })
console.log('[v0] done')
