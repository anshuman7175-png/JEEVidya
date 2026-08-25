import { generateText } from 'ai'
import { writeFileSync, mkdirSync } from 'node:fs'

const BOY_URL =
  'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/character1a-GbKTyNzzVIx1djUhDnTZIxFxiID2v3.png'
const GIRL_URL =
  'https://hebbkx1anhila5yf.public.blob.vercel-storage.com/character2a-7LiJA2Qy7BYjeQJ736nIgl3H1D9GrF.png'

const prompt = `You are given two reference character images: a boy (round black glasses, swept dark hair, orange shirt over white tee) and a girl (brown high ponytail with red hair tie, big brown eyes, red hoodie).

TASK: Create a professional YouTube channel profile picture (square 1:1 canvas) featuring BOTH characters together.

ABSOLUTE RULES — CHARACTER FIDELITY:
- The characters' faces, hairstyles, eyes, skin tone, proportions, and outfits must be IDENTICAL to the reference images. Do NOT redesign, restyle, age, or alter them in any way. Treat the references as the single source of truth.
- Same rendering style as the references: soft matte 3D animation style, NOT glossy or plastic.

COMPOSITION (designed for YouTube's circular crop):
- A perfect large circle centered on the canvas is the design. Inside the circle: a single FLAT solid warm golden-yellow background (#F5B301), completely plain — no gradients, no sparkles, no particles, no glow, no light rays.
- Outside the circle: pure white, so the circular design is clearly visible.
- The two characters appear head-and-shoulders (chest-up), side by side and slightly overlapping, filling roughly 70% of the circle's height. Boy slightly behind on the left with arms crossed, girl in front on the right smiling. Both looking at the viewer with warm friendly expressions.
- Faces positioned in the vertical center of the circle so nothing important is near the circle edge.

STRICTLY FORBIDDEN (these make it look AI-generated and unprofessional):
- NO YouTube play buttons or logos of any kind
- NO text or lettering
- NO neon rings, glowing outlines, sparkles, bokeh, lens flares, or light streaks
- NO gradient or split-color backgrounds
- Clean, even, soft studio lighting only. Subtle soft shadow under the characters is fine.

The result should look like an official character poster crop from a professional animation studio — simple, bold, instantly readable at 98 pixels.`

async function main() {
  console.log('[v0] Generating DP with character references...')
  const result = await generateText({
    model: 'google/gemini-3-pro-image',
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: prompt },
          { type: 'image', image: new URL(BOY_URL) },
          { type: 'image', image: new URL(GIRL_URL) },
        ],
      },
    ],
  })

  const images = result.files.filter((f) => f.mediaType.startsWith('image/'))
  console.log('[v0] Text output:', result.text?.slice(0, 200))
  console.log('[v0] Images returned:', images.length)
  if (images.length === 0) {
    console.error('[v0] No image generated')
    process.exit(1)
  }
  mkdirSync('public', { recursive: true })
  writeFileSync('public/youtube-dp.png', images[0].uint8Array)
  console.log('[v0] Saved to public/youtube-dp.png')
}

main().catch((e) => {
  console.error('[v0] Error:', e.message)
  process.exit(1)
})
