"""Yayın sürümünü üretir: drafts/hero-parcacik/kaydirma  ->  site/

Kaynak (geliştirilen sürüm) drafts altında kalır; site/ her çalıştırmada
baştan üretilir, elle düzenlenmez. Ayıklananlar:
  - geliştirme paneli (HTML, CSS, JS bağları)
  - test kancaları (?hep=1, window.__kareler, __durum, __sis, __adPiksel ...)
  - kullanılmayan dosyalar (yedek SVG'ler, ham portre PNG'si, kullanılmayan ikonlar)
Her değişiklik tam bir kez eşleşmek zorunda: kaynak değişip bir desen
tutmazsa betik durur, yarım ayıklanmış bir site üretmez.

Kullanım:  python araclar/yayinla.py
"""
import io
import re
import shutil
import subprocess
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
KAYNAK = KOK / "drafts" / "hero-parcacik" / "kaydirma"
HEDEF = KOK / "site"
DOSYALAR = ["Parcaciklar.js", "ad-imza.svg", "oyna-dugmesi.svg", "ismet-portre.webp"]
# Oyun ayrı projede geliştirilir; yayında site/oyun/ olur, PLAY düğmesi onu açar.
OYUN = KOK.parent / "Hypercasual_game" / "oyun"
OYUN_DOSYALAR = ["index.html", "core.js",
                 # Karakterler: yalnız oyunun yüklediği dosyalar (.blend kaynakları ve eski GLB'ler gitmez)
                 "assets/ismet-rigged.glb", "assets/acelya-rigged.glb",
                 "assets/ismet-icon.png", "assets/acelya-icon.png"]


def degistir(s, eski, yeni, ad):
    n = s.count(eski)
    if n != 1:
        sys.exit(f"HATA [{ad}]: desen {n} kez bulundu (1 bekleniyordu)")
    return s.replace(eski, yeni, 1)


def regex(s, desen, yeni, ad, bayrak=re.S):
    s2, n = re.subn(desen, yeni, s, flags=bayrak)
    if n != 1:
        sys.exit(f"HATA [{ad}]: desen {n} kez eşleşti (1 bekleniyordu)")
    return s2


def ayikla(s):
    # --- panel HTML ---
    s = regex(s, r'<div class="panel" id="panel">.*?\n</div>\n\n(?=<script type="importmap">)', "", "panel-html")
    # --- panel CSS: bolum basligindan .olcum satirina kadar (dahil) ---
    s = regex(s, r"/\* -+ gelistirme paneli \(gercek sitede olmayacak\) -+ \*/\n.*?\.olcum\{[^\n]*\}\n", "", "panel-css")
    s = degistir(s, '.panel{font-family:"Fragment Mono",monospace}\n', "", "panel-font")
    # --- JS: panel baglari ---
    s = degistir(s, "  document.getElementById('panel').style.display = 'none'\n", "", "yedek-panel")
    s = regex(s, r"    document\.getElementById\('sayiAd'\)\.textContent = [^\n]*\n    // panel dugmesi[^\n]*\n"
                 r"    document\.querySelectorAll\('#sayi button'\)[^\n]*\n[^\n]*aria-pressed[^\n]*\n", "", "sayi-gosterge")
    s = regex(s, r"  // \?hep=1 [^\n]*\n  const hepCiz = [^\n]*\n", "  const hepCiz = false\n", "hep")
    s = degistir(s, "  const fpsEl = document.getElementById('fps'), msEl = document.getElementById('ms')\n", "", "fps-el")
    s = degistir(s, "  const gecisEl = document.getElementById('gecisAd')\n", "", "gecis-el")
    s = regex(s, r"    gecisEl\.textContent = [^\n]*\n", "", "gecis-yaz")
    s = regex(s, r"      msEl\.textContent = [^\n]*\n      fpsEl\.textContent = [^\n]*\n", "", "fps-yaz")
    s = regex(s, r"\n  const grup = \(id, ozn, fn\) => .*?p\.classList\.contains\('kapali'\) \? '\+' : '—' \}\)\n(?=\}\n</script>)",
              "\n", "panel-dugmeleri")
    # --- test kancalari ---
    s = regex(s, r"  /\* Test icin: arka plan sekmesinde[^*]*\*/\n  window\.__sis = [^\n]*\n  window\.__kareler = [^\n]*\n", "", "kareler")
    s = regex(s, r"  window\.__durum = \(\) => \(\{.*?\}\)   // test icin\n", "", "durum")
    s = regex(s, r"    window\.__adPiksel = [^\n]*\n", "", "ad-piksel")
    s = regex(s, r"    window\.__zamanPiksel = [^\n]*\n", "", "zaman-piksel")
    # --- son kontrol: panel ya da test izi kalmasin ---
    for iz in ['id="panel"', "__kareler", "__durum", "__sis", "hep=1", "sayiAd", "fpsEl", "gecisEl"]:
        if iz in s:
            sys.exit(f"HATA: ayıklamadan sonra '{iz}' kaldı")
    return s


def js_denetle(html, ad="index.html", dosyalar=()):
    """Sayfadaki her <script> bloğunu ve verilen .js dosyalarını node ile sözdizimi açısından denetler."""
    node = shutil.which("node")
    if not node:
        print("uyarı: node yok, JS sözdizimi denetlenmedi")
        return
    tmp = HEDEF / "_denetim"
    tmp.mkdir()
    try:
        for i, (tur, kod) in enumerate(re.findall(r'<script( type="module")?>(.*?)</script>', html, re.S)):
            f = tmp / f"b{i}{'.mjs' if tur else '.js'}"
            f.write_text(kod, encoding="utf-8")
            r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
            if r.returncode:
                sys.exit(f"HATA: {ad} script {i} sözdizimi:\n{r.stderr}")
        for f in dosyalar:
            r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
            if r.returncode:
                sys.exit(f"HATA: {f.name} sözdizimi:\n{r.stderr}")
    finally:
        shutil.rmtree(tmp)


def oyun_kopyala():
    """Hypercasual_game/oyun -> site/oyun. Yalnız listelenen dosyalar gider (demo, not vb. gitmez)."""
    if not OYUN.is_dir():
        sys.exit(f"HATA: oyun klasörü yok: {OYUN}")
    hedef = HEDEF / "oyun"
    hedef.mkdir()
    for ad in OYUN_DOSYALAR:
        (hedef / ad).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OYUN / ad, hedef / ad)
    html = (hedef / "index.html").read_text(encoding="utf-8")
    # Sayfanın andığı her assets/ dosyası listede olmalı, yoksa canlıda 404 verir
    for yol in sorted(set(re.findall(r"assets/[\w.-]+", html))):
        if yol not in OYUN_DOSYALAR:
            sys.exit(f"HATA: oyun/index.html '{yol}' kullanıyor ama OYUN_DOSYALAR listesinde yok")
    js_denetle(html, "oyun/index.html", [p for p in hedef.glob("*.js")])
    # PLAY düğmesinin açtığı yol ile oyunun kapatma mesajı birbirini tutmalı
    if "oyun-kapat" not in html:
        sys.exit("HATA: oyun/index.html 'oyun-kapat' mesajını göndermiyor, katman kapanmaz")


def main():
    html = (KAYNAK / "index.html").read_text(encoding="utf-8")
    html = ayikla(html)

    if HEDEF.exists():
        shutil.rmtree(HEDEF)
    HEDEF.mkdir()
    js_denetle(html)
    with io.open(HEDEF / "index.html", "w", encoding="utf-8", newline="") as f:
        f.write(html)
    for ad in DOSYALAR:
        shutil.copy2(KAYNAK / ad, HEDEF / ad)
    ikonlar = sorted(set(re.findall(r"ikon/([a-z0-9-]+\.svg)", html)))
    (HEDEF / "ikon").mkdir()
    for ad in ikonlar:
        shutil.copy2(KAYNAK / "ikon" / ad, HEDEF / "ikon" / ad)
    if 'src="oyun/"' in html:
        oyun_kopyala()

    boyut = sum(p.stat().st_size for p in HEDEF.rglob("*") if p.is_file())
    print(f"site/ üretildi: index.html + {len(DOSYALAR)} dosya + {len(ikonlar)} ikon, toplam {boyut/1024:.0f} KB")


if __name__ == "__main__":
    main()
