import os, math, random, subprocess, textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# =========================
# SETTINGS
# =========================
W, H = 1920, 1080
FPS = 30
N_SLIDES = 40
CLIP_DUR = 2.25         # seconds per slide
TRANS_DUR = 0.50        # transition seconds
STEP = CLIP_DUR - TRANS_DUR
TOTAL_DUR = N_SLIDES * CLIP_DUR - (N_SLIDES - 1) * TRANS_DUR  # ~70.5s

OUT_DIR = "fiverr_video_v2"
SLIDES_DIR = os.path.join(OUT_DIR, "slides")
os.makedirs(SLIDES_DIR, exist_ok=True)

OUT_VIDEO = os.path.join(OUT_DIR, "fiverr_gig_website_video_v2_70s.mp4")
AUDIO_WAV = os.path.join(OUT_DIR, "ambient.wav")

# =========================
# HELPERS
# =========================
def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout

def font(size, bold=False):
    # Works on most Windows installs. Adjust if needed.
    candidates = [
        ("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        ("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for f in candidates:
        if os.path.exists(f):
            return ImageFont.truetype(f, size=size)
    return ImageFont.load_default()

def wrap(draw, text, fnt, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=fnt) <= max_w:
            cur = test
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

def gradient_bg(w, h, c1, c2):
    img = Image.new("RGB", (w, h), c1)
    px = img.load()
    for y in range(h):
        t = y / (h - 1)
        r = int(c1[0] + (c2[0] - c1[0]) * t)
        g = int(c1[1] + (c2[1] - c1[1]) * t)
        b = int(c1[2] + (c2[2] - c1[2]) * t)
        for x in range(w):
            px[x, y] = (r, g, b)
    return img

def add_glow_blobs(base, blobs):
    overlay = Image.new("RGBA", base.size, (0,0,0,0))
    d = ImageDraw.Draw(overlay)
    for (cx, cy, rad, col, alpha) in blobs:
        d.ellipse((cx-rad, cy-rad, cx+rad, cy+rad), fill=(*col, alpha))
    overlay = overlay.filter(ImageFilter.GaussianBlur(80))
    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGBA")

def round_rect(draw, box, r, fill, outline=None, width=2):
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)

def browser_mockup(size=(1200, 720), variant=0):
    """Create a clean website UI screenshot (no real brands)."""
    w, h = size
    img = Image.new("RGBA", (w, h), (255,255,255,0))
    d = ImageDraw.Draw(img)

    # Outer
    round_rect(d, (0,0,w,h), 28, fill=(255,255,255,235), outline=(255,255,255,255), width=2)

    # Top bar
    round_rect(d, (18,18,w-18,80), 20, fill=(240,243,249,255))
    # Window dots
    dots = [(40,49,(255,95,110)), (70,49,(255,200,90)), (100,49,(70,210,140))]
    for x,y,c in dots:
        d.ellipse((x-9,y-9,x+9,y+9), fill=c)

    # Nav
    f_nav = font(26, bold=True)
    d.text((140, 38), "AgencyPro", font=f_nav, fill=(15,25,40,255))
    nav_items = ["Services","Work","Pricing","Contact"]
    f_item = font(22, bold=False)
    x = w - 440
    for it in nav_items:
        d.text((x, 40), it, font=f_item, fill=(80,95,120,255))
        x += 90

    # Hero
    accent = [(70,140,255),(120,80,255),(0,210,170),(255,120,170)][variant % 4]
    d.text((60, 120), "Premium Business Website", font=font(44, True), fill=(15,25,40,255))
    d.text((60, 180), "Modern layout • Fast loading • Built to convert", font=font(26, False), fill=(70,85,110,255))
    # CTA button
    round_rect(d, (60, 240, 310, 305), 24, fill=(*accent,255))
    d.text((90, 257), "Get a Quote", font=font(24, True), fill=(255,255,255,255))
    round_rect(d, (330, 240, 520, 305), 24, fill=(230,235,245,255))
    d.text((360, 257), "View Work", font=font(24, True), fill=(15,25,40,255))

    # Feature cards
    y0 = 340
    for i in range(3):
        x0 = 60 + i*360
        round_rect(d, (x0, y0, x0+320, y0+160), 26, fill=(248,250,255,255), outline=(230,235,245,255), width=2)
        d.ellipse((x0+24, y0+26, x0+54, y0+56), fill=(*accent,255))
        d.text((x0+70, y0+22), ["Mobile-First","SEO-Ready","Secure + Clean"][i], font=font(26, True), fill=(15,25,40,255))
        d.text((x0+70, y0+64), ["Responsive UI on all devices","Clean structure + speed","Audit-friendly build quality"][i],
               font=font(20, False), fill=(85,100,125,255))

    # Lower section varies (portfolio / dashboard / ecommerce)
    base_y = 540
    if variant % 5 == 0:
        d.text((60, base_y), "Recent Work", font=font(28, True), fill=(15,25,40,255))
        for i in range(4):
            x0 = 60 + i*280
            round_rect(d, (x0, base_y+50, x0+240, base_y+160), 18, fill=(235,240,248,255))
            d.line((x0+20, base_y+90, x0+220, base_y+90), fill=(210,220,235,255), width=3)
            d.line((x0+20, base_y+120, x0+180, base_y+120), fill=(210,220,235,255), width=3)
    elif variant % 5 == 1:
        d.text((60, base_y), "Analytics Dashboard", font=font(28, True), fill=(15,25,40,255))
        round_rect(d, (60, base_y+55, w-60, h-40), 22, fill=(245,248,255,255), outline=(230,235,245,255), width=2)
        # chart
        cx0, cy0 = 100, base_y+100
        for i in range(7):
            bh = random.randint(60, 180)
            bx = cx0 + i*130
            round_rect(d, (bx, cy0+210-bh, bx+70, cy0+210), 14, fill=(*accent,220))
        d.line((100, cy0+210, w-100, cy0+210), fill=(200,210,230,255), width=3)
    elif variant % 5 == 2:
        d.text((60, base_y), "Checkout + Payment Ready", font=font(28, True), fill=(15,25,40,255))
        # product grid
        for r in range(2):
            for c in range(3):
                x0 = 60 + c*360
                y1 = base_y+55 + r*190
                round_rect(d, (x0, y1, x0+320, y1+160), 22, fill=(245,248,255,255), outline=(230,235,245,255), width=2)
                round_rect(d, (x0+22, y1+22, x0+118, y1+118), 18, fill=(235,240,248,255))
                d.text((x0+140, y1+28), "Product", font=font(22, True), fill=(15,25,40,255))
                d.text((x0+140, y1+62), "$49", font=font(22, False), fill=(85,100,125,255))
                round_rect(d, (x0+140, y1+100, x0+280, y1+136), 18, fill=(*accent,255))
                d.text((x0+160, y1+108), "Add to cart", font=font(18, True), fill=(255,255,255,255))
    elif variant % 5 == 3:
        d.text((60, base_y), "Booking + Lead Capture", font=font(28, True), fill=(15,25,40,255))
        round_rect(d, (60, base_y+55, w-60, h-40), 22, fill=(245,248,255,255), outline=(230,235,245,255), width=2)
        # form fields
        fx, fy = 100, base_y+110
        for i, lab in enumerate(["Name","Email","Service","Message"]):
            y = fy + i*95
            d.text((fx, y-28), lab, font=font(18, True), fill=(80,95,120,255))
            round_rect(d, (fx, y, w-140, y+52), 16, fill=(255,255,255,255), outline=(220,230,245,255), width=2)
        round_rect(d, (fx, h-120, 340, h-70), 20, fill=(*accent,255))
        d.text((fx+45, h-109), "Book Now", font=font(20, True), fill=(255,255,255,255))
    else:
        d.text((60, base_y), "Speed + SEO Optimized", font=font(28, True), fill=(15,25,40,255))
        # speed meters
        for i, label in enumerate(["Performance","SEO","Best Practices","Accessibility"]):
            x0 = 60 + i*450
            round_rect(d, (x0, base_y+60, x0+410, base_y+180), 22, fill=(245,248,255,255), outline=(230,235,245,255), width=2)
            score = random.randint(90, 99)
            d.text((x0+24, base_y+80), label, font=font(20, True), fill=(80,95,120,255))
            d.text((x0+24, base_y+120), f"{score}/100", font=font(34, True), fill=(*accent,255))

    return img

def make_slide(idx, title, bullets, theme):
    # theme = (bg1, bg2, accent)
    bg1, bg2, accent = theme
    base = gradient_bg(W, H, bg1, bg2).convert("RGBA")
    base = add_glow_blobs(base, [
        (260, 200, 360, accent, 90),
        (1700, 240, 420, (255, 120, 170), 55),
        (1500, 900, 520, (0, 220, 170), 45),
        (420, 920, 520, (120, 80, 255), 40),
    ])

    d = ImageDraw.Draw(base)

    # Header line
    d.line((80, 90, 680, 90), fill=(255,255,255,70), width=3)
    d.line((80, 96, 420, 96), fill=(*accent,200), width=5)

    # Title
    f_t = font(64, True)
    f_b = font(28, False)
    d.text((80, 130), title, font=f_t, fill=(255,255,255,245))

    # Bullets
    y = 230
    maxw = 720
    for b in bullets:
        lines = wrap(d, b, f_b, maxw)
        d.text((110, y), "•", font=f_b, fill=(255,255,255,230))
        d.text((140, y), lines[0], font=f_b, fill=(230,240,255,230))
        y += 42
        for ln in lines[1:]:
            d.text((140, y), ln, font=f_b, fill=(230,240,255,220))
            y += 36
        y += 12

    # Browser mockup (varies each slide)
    shot = browser_mockup((1120, 700), variant=idx)
    # Device shadow
    shadow = Image.new("RGBA", (shot.size[0]+40, shot.size[1]+40), (0,0,0,0))
    sd = ImageDraw.Draw(shadow)
    round_rect(sd, (20, 24, shadow.size[0]-20, shadow.size[1]-12), 32, fill=(0,0,0,150))
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))

    px, py = 780, 250
    base.alpha_composite(shadow, (px-20, py-20))
    base.alpha_composite(shot, (px, py))

    # Small footer tag
    f_s = font(22, True)
    tag = f"Slide {idx+1:02d} / {N_SLIDES}"
    tw = d.textlength(tag, font=f_s)
    round_rect(d, (80, H-120, 80+tw+40, H-70), 22, fill=(0,0,0,90), outline=(255,255,255,70), width=2)
    d.text((100, H-106), tag, font=f_s, fill=(255,255,255,235))

    out = os.path.join(SLIDES_DIR, f"slide_{idx+1:03d}.png")
    base.convert("RGB").save(out, "PNG", optimize=True)
    return out

# =========================
# SLIDE SCRIPT (40)
# =========================
themes = [
    ((10, 14, 28), (18, 48, 110), (0, 180, 255)),
    ((18, 10, 30), (95, 30, 150), (120, 80, 255)),
    ((8, 20, 24), (0, 110, 90), (0, 220, 170)),
    ((20, 10, 18), (130, 30, 60), (255, 120, 170)),
]

slides_text = [
("Premium Business Website",
 ["Modern agency-style design that looks expensive and trustworthy.",
  "Built to convert visitors into leads with clear CTA structure.",
  "Fast, responsive, and clean across all devices."], 0),

("Hero Section That Converts",
 ["Strong headline + subheadline to instantly explain your offer.",
  "CTA buttons that guide users to message, book, or buy.",
  "Clean spacing, premium typography, and modern layout."], 1),

("Services + Value Blocks",
 ["Clear service cards that make your offer easy to understand.",
  "Trust signals, icons, and feature highlights.",
  "Perfect for agencies, consultants, startups, and businesses."], 2),

("Portfolio / Case Studies",
 ["Show your best work with clean thumbnails and summaries.",
  "Case-study style sections that increase trust and conversions.",
  "Built for both desktop and mobile viewing."], 3),

("Testimonials & Social Proof",
 ["Review blocks that feel real and professional.",
  "Add logos, ratings, and credibility sections.",
  "Helps buyers decide faster with confidence."], 0),

("Pricing Section",
 ["Premium pricing layout with clear packages and benefits.",
  "Highlights what’s included, what’s optional, and next steps.",
  "Designed to reduce confusion and increase orders."], 1),

("Contact & Lead Capture",
 ["Clean contact forms with validation and good UX.",
  "Opt-in newsletter forms if needed.",
  "Lead-focused layout that keeps friction low."], 2),

("Booking & Scheduling",
 ["Booking blocks for calls, meetings, or appointments.",
  "Calendar-style layout and simple flow.",
  "Perfect for service businesses and consultants."], 3),

("Analytics Dashboard Preview",
 ["Dashboard UI for metrics, KPIs, and reporting screens.",
  "Clean charts and cards for business insights.",
  "Ideal for web apps or internal portals."], 0),

("Data-Driven Layout",
 ["Sections designed to support tracking and conversion.",
  "Analytics-ready structure.",
  "Optional event tracking and goal setup."], 1),

("E-commerce Ready",
 ["Product sections designed for clean browsing.",
  "Cart/checkout flow preview.",
  "Optimized for product clarity and trust."], 2),

("Payment Integration",
 ["Stripe/PayPal-ready purchase flow.",
  "Clear checkout UI and confirmation screens.",
  "Secure, professional, and conversion-focused."], 3),

("Responsive Design",
 ["Desktop, tablet, and mobile optimized.",
  "Mobile-first layout behavior.",
  "Looks premium on every screen size."], 0),

("Speed Optimization",
 ["Lightweight structure and optimized assets.",
  "Performance-focused build choices.",
  "Optional advanced optimization if needed."], 1),

("SEO-Friendly Structure",
 ["Clean headings, structure, and semantic layout.",
  "SEO-ready page sections and metadata support.",
  "Better indexing and better first impression."], 2),

("Security + Clean Build",
 ["Strong structure and safe integrations.",
  "Clean code and professional best practices.",
  "Built for reliability."], 3),
]

# Expand to 40 slides with detailed variations
while len(slides_text) < 40:
    base = slides_text[len(slides_text) % 16]
    t = base[0]
    bullets = base[1][:]
    bullets.append(random.choice([
        "Smooth navigation + premium UI details.",
        "Professional structure that feels genuine.",
        "Clean layout designed for real buyers.",
        "Modern sections that improve trust instantly.",
        "Consistent spacing and visual hierarchy."
    ]))
    slides_text.append((t, bullets[:3], (len(slides_text) % 4)))

# Force a strong last CTA slide
slides_text[-1] = ("Ready to Launch?",
                   ["Message me on Fiverr to start your premium website.",
                    "Tell me your goal + reference links, I’ll handle the rest.",
                    "Fast delivery • Clean quality • High standards"], 0)

# =========================
# GENERATE 40 SLIDES
# =========================
slide_paths = []
for i, (t, b, th) in enumerate(slides_text[:40]):
    slide_paths.append(make_slide(i, t, b, themes[th]))

print(f"Generated {len(slide_paths)} slides in: {SLIDES_DIR}")

# =========================
# GENERATE AMBIENT AUDIO (ROYALTY-FREE)
# =========================
sr = 44100
dur = TOTAL_DUR
t = np.linspace(0, dur, int(sr*dur), endpoint=False)

# Gentle pad: layered sines + slow LFO + soft noise
chords = [220, 277.18, 329.63]  # A3, C#4, E4
sig = np.zeros_like(t)

for f in chords:
    sig += 0.20*np.sin(2*np.pi*f*t) + 0.08*np.sin(2*np.pi*(f*2)*t)

lfo = 0.6 + 0.4*np.sin(2*np.pi*0.08*t)
sig *= lfo

noise = np.random.normal(0, 1, size=t.shape) * 0.01
sig += noise

# Fade in/out
fade = int(sr*2.0)
env = np.ones_like(sig)
env[:fade] *= np.linspace(0,1,fade)
env[-fade:] *= np.linspace(1,0,fade)
sig *= env

# Normalize and write WAV (16-bit)
sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.35
wav = (sig * 32767).astype(np.int16)

import wave
os.makedirs(OUT_DIR, exist_ok=True)
with wave.open(AUDIO_WAV, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(wav.tobytes())

print("Audio created:", AUDIO_WAV)

# =========================
# BUILD FFMPEG FILTERGRAPH (40 inputs + xfade chain)
# =========================
transitions = [
    "fade","wipeleft","wiperight","wipeup","wipedown",
    "slideleft","slideright","slideup","slidedown",
    "circleopen","circleclose","rectcrop",
    "fadeblack","fadewhite","radial",
    "smoothleft","smoothright","smoothup","smoothdown",
    "horizopen","horizclose","vertopen","vertclose",
    "diagbl","diagbr","diagtl","diagtr",
    "hlslice","hrslice","vuslice","vdslice",
    "pixelize","dissolve","hblur","vblur","zoomin","zoomout",
]
# Ensure 39 transitions
while len(transitions) < (N_SLIDES-1):
    transitions.append(random.choice(transitions))

# Per-slide motion directions (alive feel)
motions = ["in","out","in","in","out","in","out","in","out","in"]

filter_lines = []
for i in range(N_SLIDES):
    # zoompan: subtle motion
    # Different motion per slide to avoid boredom
    if motions[i % len(motions)] == "in":
        zexpr = "min(zoom+0.0020,1.12)"
    else:
        zexpr = "max(zoom-0.0018,1.00)"
    dframes = int(CLIP_DUR * FPS)

    filter_lines.append(
        f"[{i}:v]scale={W}:{H},format=rgba,"
        f"zoompan=z='{zexpr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={dframes}:s={W}x{H}:fps={FPS},"
        f"trim=duration={CLIP_DUR},setpts=PTS-STARTPTS[v{i}]"
    )

# xfade chain
offset = STEP
filter_lines.append(f"[v0][v1]xfade=transition={transitions[0]}:duration={TRANS_DUR}:offset={offset}[x1]")
for k in range(2, N_SLIDES):
    offset += STEP
    tr = transitions[k-1]
    filter_lines.append(f"[x{k-1}][v{k}]xfade=transition={tr}:duration={TRANS_DUR}:offset={offset}[x{k}]")

# Audio: fade + volume
filter_lines.append(f"[{N_SLIDES}:a]afade=t=in:st=0:d=2,afade=t=out:st={TOTAL_DUR-2:.2f}:d=2,volume=0.9[aout]")

filter_complex = ";".join(filter_lines)

# =========================
# RENDER (with fallback if transitions unsupported)
# =========================
inputs = []
for p in slide_paths:
    inputs += ["-loop","1","-t",str(CLIP_DUR),"-i",p]
inputs += ["-i", AUDIO_WAV]

cmd = ["ffmpeg","-y"] + inputs + [
    "-filter_complex", filter_complex,
    "-map", f"[x{N_SLIDES-1}]",
    "-map", "[aout]",
    "-r", str(FPS),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-crf", "18",
    "-preset", "medium",
    "-c:a", "aac",
    "-b:a", "192k",
    "-shortest",
    OUT_VIDEO
]

print("Rendering:", OUT_VIDEO)
rc, log = run(cmd)
if rc != 0:
    print("\nFFmpeg failed (likely unsupported transition on your build). Falling back to safe transitions...\n")
    safe = ["fade","wipeleft","wiperight","wipeup","wipedown","slideleft","slideright","slideup","slidedown","dissolve","pixelize","fadeblack","fadewhite"]
    while len(safe) < (N_SLIDES-1):
        safe.append(random.choice(safe))
    # rebuild xfade only
    filter_lines2 = [l for l in filter_lines if not l.startswith("[v0][v1]xfade") and "xfade=transition=" not in l]
    offset = STEP
    filter_lines2.append(f"[v0][v1]xfade=transition={safe[0]}:duration={TRANS_DUR}:offset={offset}[x1]")
    for k in range(2, N_SLIDES):
        offset += STEP
        filter_lines2.append(f"[x{k-1}][v{k}]xfade=transition={safe[k-1]}:duration={TRANS_DUR}:offset={offset}[x{k}]")
    filter_complex2 = ";".join(filter_lines2)
    cmd2 = ["ffmpeg","-y"] + inputs + [
        "-filter_complex", filter_complex2,
        "-map", f"[x{N_SLIDES-1}]",
        "-map", "[aout]",
        "-r", str(FPS),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        OUT_VIDEO
    ]
    rc2, log2 = run(cmd2)
    if rc2 != 0:
        print(log2)
        raise SystemExit("Render failed. Run: ffmpeg -h filter=xfade and reduce transition list.")
else:
    print("Render OK.")

print("\nDONE:", OUT_VIDEO)
