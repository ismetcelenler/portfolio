# ismetcelenler.com

Kişisel portföy sitesi — İsmet Çelenler, AI-focused software engineer
(computer vision + ML + AI workflows).

- Saf HTML, CSS ve JavaScript; derleme adımı yok.
- Hero ve kaydırma geçişleri: [three.js](https://threejs.org) ile WebGL2 GPGPU parçacık sistemi
  (`site/Parcaciklar.js`).
- İletişim formu: [Web3Forms](https://web3forms.com).

## Yapı

```
site/              yayınlanan site (Vercel bu klasörü sunar)
araclar/yayinla.py site/ klasörünü geliştirme sürümünden üretir
```

`site/` elle düzenlenmez: geliştirme sürümünden `python araclar/yayinla.py` ile
üretilir (geliştirme paneli ve test kancaları bu sırada ayıklanır).

## Yerelde çalıştırma

```
cd site
python -m http.server 8778
```

## Atıflar

- İkonlar: [Streamline Pixel](https://www.streamlinehq.com/icons/pixel) (CC BY 4.0),
  [Pixelarticons](https://pixelarticons.com) (MIT)
- Font: [VT323](https://fonts.google.com/specimen/VT323) (OFL)
