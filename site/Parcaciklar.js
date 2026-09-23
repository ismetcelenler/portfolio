/**
 * Parcaciklar — WebGL2 GPGPU parcacik sistemi
 *
 * Bruno Simon'un TSL/WebGPU sisteminin davranisini WebGL2'ye tasir.
 * Durum iki float dokusunda tutulur (pozisyon+omur, hiz) ve her kare
 * ping-pong ile guncellenir. Compute shader gerekmez, her tarayicida calisir.
 *
 * Preset alanlari birebir karsilanir:
 *   count, decayFrequency, velocityDamping,
 *   emitterRadius, emitterVelocityStrength, initialVelocity, initialRandomVelocity,
 *   turbulenceStrength, turbulenceTimeFrequeny, turbulencePositionFrequeny,
 *   gravity, floorY, floorDamping,
 *   colorIn, colorOut, fadeIn, fadeOut, size, glowSpread, solidRatio, solidAlpha, opacity,
 *   sparklingAlpha, sparklingFrequency, sparklingDuration
 */
import * as THREE from 'three'
import { GPUComputationRenderer } from 'three/examples/jsm/misc/GPUComputationRenderer.js'

/* ---------------------------------------------------------------- yardimcilar */

// Dave Hoskins hash'leri — ayni girdi her iki shader'da ayni sonucu verir,
// pozisyon ve hiz shader'larinin yeniden dogusta ayni rastgeleyi kullanmasi sart.
const HASH = /* glsl */`
float hash11(float p){ p = fract(p * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
vec3  hash32(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031,0.1030,0.0973)); p3 += dot(p3, p3.yxz + 33.33); return fract((p3.xxy + p3.yzz) * p3.zyx); }
`

// Ashima / Stefan Gustavson simplex 3D — curl gurultusunun tabani
const SIMPLEX = /* glsl */`
vec3 mod289(vec3 x){ return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 mod289(vec4 x){ return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 permute(vec4 x){ return mod289(((x*34.0)+1.0)*x); }
vec4 taylorInvSqrt(vec4 r){ return 1.79284291400159 - 0.85373472095314 * r; }
float snoise(vec3 v){
  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i  = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;
  i = mod289(i);
  vec4 p = permute(permute(permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}
vec3 snoiseVec3(vec3 x){
  return vec3(snoise(x),
              snoise(x + vec3(17.13, 9.27, 4.31)),
              snoise(x + vec3(31.71, 12.33, 8.91)));
}
// Iraksamasiz (divergence-free) alan: gercek girdap hissi buradan gelir
vec3 curlNoise(vec3 p){
  const float e = 0.22;
  vec3 dx = vec3(e, 0.0, 0.0), dy = vec3(0.0, e, 0.0), dz = vec3(0.0, 0.0, e);
  vec3 px0 = snoiseVec3(p - dx), px1 = snoiseVec3(p + dx);
  vec3 py0 = snoiseVec3(p - dy), py1 = snoiseVec3(p + dy);
  vec3 pz0 = snoiseVec3(p - dz), pz1 = snoiseVec3(p + dz);
  float x = (py1.z - py0.z) - (pz1.y - pz0.y);
  float y = (pz1.x - pz0.x) - (px1.z - px0.z);
  float z = (px1.y - px0.y) - (py1.x - py0.x);
  return normalize(vec3(x, y, z) / (2.0 * e));
}
`

// Yeniden dogus kosulu ve rastgelesi — iki shader'da birebir ayni olmali
const DOGUS = /* glsl */`
vec3 dogusYonu(vec2 uv, float tohum){
  vec3 r = hash32(uv * 51.7 + tohum);
  vec3 d = r * 2.0 - 1.0;
  float u = length(d);
  return u > 0.0001 ? d / u : vec3(0.0, 1.0, 0.0);
}
`

/* ---------------------------------------------------------------- alanlar
 * uMod: 0 fiskiye · 1 gradyan inisi (3B arazi, arayuzde yok) · 2 k-ortalama
 *       3 Aizawa cekicisi · 4 Lorenz cekicisi · 5 gradyan inisi (2B kesit)
 */
const ALANLAR = /* glsl */`
// --- 1) kayip yuzeyi: coklu minimumlu, merkeze dogru cukur ---
float kayip(vec2 q){
  return 0.55 * sin(1.55 * q.x) * cos(1.55 * q.y)
       + 0.13 * sin(3.70 * q.x + 1.1) * cos(3.10 * q.y - 0.6)   // ince sirtlar
       + 0.10 * dot(q, q);                                       // genis kase
}
vec2 kayipGradyan(vec2 q){
  return vec2(
     0.8525 * cos(1.55 * q.x) * cos(1.55 * q.y)
   + 0.4810 * cos(3.70 * q.x + 1.1) * cos(3.10 * q.y - 0.6) + 0.20 * q.x,
    -0.8525 * sin(1.55 * q.x) * sin(1.55 * q.y)
   - 0.4030 * sin(3.70 * q.x + 1.1) * sin(3.10 * q.y - 0.6) + 0.20 * q.y
  );
}

// --- 5) kesit profili: tek boyutlu kayip egrisi ---
// Bilerek asimetrik: sol taraftan inen once SIG bir yerel cukura duser,
// devam edebilmek icin bir sirti asmasi gerekir, kuresel minimum sagdadir.
float kesitKayip(float x){
  // Bes cukur: dordu yerel, biri kuresel. Kuresel soldan gelirken DORDUNCU
  // sirada (x=1.29, L=-1.375) — once uc yerel cukuru gecmek gerekiyor.
  return 0.045 * x * x
       - 0.52 * exp(-4.2 * (x - (-2.95)) * (x - (-2.95)))
       - 0.70 * exp(-4.6 * (x - (-1.55)) * (x - (-1.55)))
       - 0.58 * exp(-4.4 * (x - (-0.15)) * (x - (-0.15)))
       - 1.45 * exp(-3.2 * (x - (1.30)) * (x - (1.30)))
       - 0.62 * exp(-4.8 * (x - (2.80)) * (x - (2.80)));
}
float kesitTurev(float x){
  return 0.09 * x
       + 4.368 * (x - (-2.95)) * exp(-4.2 * (x - (-2.95)) * (x - (-2.95)))
       + 6.440 * (x - (-1.55)) * exp(-4.6 * (x - (-1.55)) * (x - (-1.55)))
       + 5.104 * (x - (-0.15)) * exp(-4.4 * (x - (-0.15)) * (x - (-0.15)))
       + 9.280 * (x - (1.30)) * exp(-3.2 * (x - (1.30)) * (x - (1.30)))
       + 5.952 * (x - (2.80)) * exp(-4.8 * (x - (2.80)) * (x - (2.80)));
}

// --- 2) optik akis: yumusak hareket vektoru alani ---
vec3 akisAlani(vec3 p, float t){
  return vec3(
     sin(p.y * 1.45 + t * 0.35) * 0.85 + 0.30,
     sin(p.x * 1.15 - t * 0.28) * 0.62,
     sin((p.x + p.y) * 0.70 + t * 0.20) * 0.28
  );
}

// --- 3) islem hatti: soldan saga dort kapi ---
vec3 hatAlani(vec3 p){
  vec3 f = vec3(0.75, 0.0, 0.0);               // tasiyici akis
  float x = p.x;
  if(x > -1.15 && x <= -0.15){                 // TESPIT — kadraja daral
    f.y += -p.y * 2.4;
    f.z += -p.z * 2.4;
  } else if(x > -0.15 && x <= 0.95){           // TAKIP — kimlik seritlerine ayril
    float serit = floor((p.y + 0.90) / 0.45);
    float hedef = serit * 0.45 - 0.90 + 0.225;
    f.y += (hedef - p.y) * 3.4;
    f.z += -p.z * 1.8;
  } else if(x > 0.95){                          // KARAR — tek noktada topla
    f.y += -p.y * 3.8;
    f.z += -p.z * 3.8;
  }
  return f;
}

// --- 2) k-ortalama: bes Gauss yigini. Parcacik = veri noktasi, yerinde durur. ---
vec3 yiginNoktasi(vec2 uv, float tohum){
  vec3 a = hash32(uv * 27.3 + tohum);
  vec3 b = hash32(uv * 91.7 + tohum + 5.0);
  int k = int(floor(a.x * 5.0));
  vec2 merkez = vec2(0.0);
  if(k == 0) merkez = vec2(-1.55,  0.62);
  else if(k == 1) merkez = vec2( 1.42,  0.78);
  else if(k == 2) merkez = vec2(-0.95, -0.82);
  else if(k == 3) merkez = vec2( 1.05, -0.70);
  else merkez = vec2( 0.10,  0.05);
  // kutu-muller yerine iki hash'in ortalamasi: ucuz ve yeterince Gauss
  vec2 sapma = (vec2(a.y + b.x, a.z + b.y) - 1.0) * 0.62;
  return vec3(merkez + sapma, (b.z - 0.5) * 0.25);
}

// --- 3) Aizawa cekicisi ---
// dx = (z-b)x - dy
// dy = dx + (z-b)y
// dz = c + az - z^3/3 - (x^2+y^2)(1+ez) + f z x^3
// (a=0.95 b=0.7 c=0.6 d=3.5 e=0.25 f=0.1)
// Cekici dogal olarak x,y icinde [-1.4, 1.4], z icinde [-0.7, 1.8] dolasir.
// Sahne esleme Lorenz'dekiyle ayni mantikta: cekicinin DIKEY ekseni z,
// sahnenin dikey ekseni y — ikisi takas ediliyor ki sekil ayakta dursun.
const float AS = 1.55;   // olcek: govde cerceveye sigsin
const float AZ = 0.62;   // z kaydirma: sekil ekranda ortalansin
vec3 aizawaHizi(vec3 p){
  vec3 A = vec3(p.x / AS, p.z / AS, p.y / AS + AZ);   // sahne -> cekici uzayi
  float x = A.x, y = A.y, z = A.z;
  float a = 0.95, b = 0.7, c = 0.6, d = 3.5, e = 0.25, f = 0.1;
  vec3 t = vec3(
    (z - b) * x - d * y,
    d * x + (z - b) * y,
    c + a * z - z * z * z / 3.0 - (x * x + y * y) * (1.0 + e * z) + f * z * x * x * x
  );
  return vec3(t.x, t.z, t.y) * AS;                    // cekici -> sahne
}

// --- 4) Lorenz cekicisi: uzayda iki spiral kanat ---
// dx=s(y-x), dy=x(r-z)-y, dz=xy-bz   (s=10, r=28, b=8/3)
// Sahne uzayina esleme: sahne.x = L.x/S, sahne.y = (L.z-Z0)/S, sahne.z = L.y/S
const float LS = 23.0;   // olcek: kelebek cerceveye sigsin
const float LZ = 25.0;   // z kaydirma: kelebek ekranda ortalansin
vec3 lorenzHizi(vec3 p){
  vec3 L = vec3(p.x * LS, p.z * LS, p.y * LS + LZ);
  vec3 d = vec3(10.0 * (L.y - L.x),
                L.x * (28.0 - L.z) - L.y,
                L.x * L.y - (8.0/3.0) * L.z);
  return vec3(d.x, d.z, d.y) / LS;
}

// --- moda gore dogum yeri ---
vec3 dogumYeri(float m, vec2 uv, float tohum, vec3 eP, vec3 eOnceki, float karis, float yaricap, vec3 yon){
  if(m < 0.5){
    return mix(eP, eOnceki, karis) + yon * yaricap;
  }
  vec3 r = hash32(uv * 83.1 + tohum) * 2.0 - 1.0;
  if(m < 1.5){                                    // gradyan: XZ arazisine dagit, yuzeye otur
    vec2 q = vec2(r.x * 3.20, r.z * 2.30);
    return vec3(q.x, kayip(q) + 0.035, q.y);
  }
  if(m < 2.5) return yiginNoktasi(uv, tohum);                    // k-ortalama verisi
  if(m < 3.5){                                  // Aizawa: cekici icinde kucuk bulut
    vec3 A0 = vec3(0.10, 0.00, 0.20) + r * 0.12;
    return vec3(A0.x * AS, (A0.z - AZ) * AS, A0.y * AS);
  }
  if(m < 4.5) return vec3(0.06, 0.06, 0.06) + r * 0.05;          // Lorenz: merkeze yakin kucuk bulut
  // 5 · kesit: z = 0 duzleminde, sol uctan
  float x0 = -3.35 + r.x * 0.34;
  return vec3(x0, kesitKayip(x0) + 0.03, 0.0);
}
`

/* ---------------------------------------------------------------- shader'lar */

const POZISYON_SHADER = /* glsl */`
uniform float dt, zaman, uMod, uSifirla;
uniform float decayFrequency, emitterRadius, floorY;
uniform vec3  emitterPosition, emitterPreviousPosition;
${HASH}
${DOGUS}
${ALANLAR}
void main(){
  vec2 uv = gl_FragCoord.xy / resolution.xy;
  vec4 pk  = texture2D(texturePosition, uv);
  vec3 hiz = texture2D(textureVelocity, uv).xyz;

  vec3  p    = pk.xyz;
  float omur = pk.w;
  float yeni = omur + dt * decayFrequency;

  /* Gradyan inisinde parcacik ancak BIR MINIMUMA OTURUNCA yeniden dogar.
     Yolda olan hicbiri omru bitti diye kaybolmaz — gercek algoritmada da
     iterasyon yakinsayana kadar surer. */
  float hiz2 = dot(hiz, hiz);
  bool gradyan = (uMod > 0.5 && uMod < 1.5) || uMod > 4.5;
  bool durdu   = gradyan && yeni > 0.22 && hiz2 < 0.00035;
  bool oldu    = (uSifirla > 0.5) || (gradyan ? durdu : (yeni > 1.0));

  if(oldu){
    float tohum = floor(zaman * 60.0);          // ayni karede iki shader ayni tohumu alir
    vec3  yon   = dogusYonu(uv, tohum);
    float karis = hash12(uv * 13.1 + tohum);
    p    = dogumYeri(uMod, uv, tohum, emitterPosition, emitterPreviousPosition, karis, emitterRadius, yon);
    yeni = gradyan ? 0.0 : fract(yeni);      // gradyanda omur sifirlanir, yoksa aninda yeniden 'durmus' sayilir
  } else if(uMod > 1.5 && uMod < 2.5){
    /* k-ortalama: veri noktalari sabittir, yalnizca RENKLERI degisir */
  } else {
    p += hiz * dt;
    if(uMod < 0.5) p.y = max(p.y, floorY);      // zemin yalniz fiskiye modunda
    /* 2B kesit: egri bir zemin. Parcacik ustunde hoplayabilir ama ALTINA
       inemez — hiz shader'indeki sekme bunun karsiligi. Tam kilitlemek
       hoplamayi da oldururdu; burada yalnizca alt yari kapatiliyor. */
    if(uMod > 4.5) p.y = max(p.y, kesitKayip(p.x) + 0.03);
  }
  if(gradyan) yeni = min(yeni, 0.999);          // omur renk icin kullaniliyor, tasmasin
  gl_FragColor = vec4(p, yeni);
}
`

const HIZ_SHADER = /* glsl */`
uniform float dt, zaman, uMod, uOgrenme, uSifirla;
uniform float decayFrequency, velocityDamping;
uniform float turbulenceStrength, turbulenceTimeFrequeny, turbulencePositionFrequeny;
uniform float floorY, floorDamping;
uniform float emitterVelocityStrength, initialRandomVelocity;
uniform vec3  gravity, initialVelocity, emitterVelocity, emitterPreviousVelocity;
${HASH}
${SIMPLEX}
${DOGUS}
${ALANLAR}
void main(){
  vec2 uv = gl_FragCoord.xy / resolution.xy;
  vec4 pk = texture2D(texturePosition, uv);
  vec3 v  = texture2D(textureVelocity, uv).xyz;

  vec3  p    = pk.xyz;
  float omur = pk.w;
  float yeni = omur + dt * decayFrequency;

  float hiz2 = dot(v, v);
  bool gradyan = (uMod > 0.5 && uMod < 1.5) || uMod > 4.5;
  bool durdu   = gradyan && yeni > 0.22 && hiz2 < 0.00035;
  bool oldu    = (uSifirla > 0.5) || (gradyan ? durdu : (yeni > 1.0));

  if(oldu){
    float tohum = floor(zaman * 60.0);
    vec3  yon   = dogusYonu(uv, tohum);
    float karis = hash12(uv * 13.1 + tohum);
    if(uMod < 0.5){
      vec3 temel = mix(emitterVelocity, emitterPreviousVelocity, karis);
      v = temel * emitterVelocityStrength
          + yon * initialRandomVelocity
          + initialVelocity;
    } else {
      v = yon * 0.06;                  // alan modlarinda hafif bir baslangic
    }
  }
  else if(uMod < 0.5){
    /* ---------------- 0 · FISKIYE (preset) ---------------- */
    vec3 girdapGirdi = p * turbulencePositionFrequeny + 12.34;
    v += curlNoise(girdapGirdi + zaman * turbulenceTimeFrequeny) * turbulenceStrength;
    v += gravity * dt;
    v *= (1.0 - velocityDamping);
    if(p.y <= floorY + 0.0001 && v.y < 0.0){
      v.y = -v.y * (1.0 - floorDamping);
      v.xz *= 0.88;
    }
  }
  else if(uMod < 1.5){
    /* ---------------- 1 · GRADYAN INISI -------------------
     * Yuzey XZ duzleminde, yukseklik Y. Kamera egik bakar ki
     * tepe ve vadi okunabilsin.                              */
    vec2 q = p.xz;
    vec2 g = kayipGradyan(q);
    // adim buyuklugu dogrudan ogrenme oranina bagli: buyuk eta -> uzun sicrama
    v.x += -g.x * uOgrenme * dt * 11.0;
    v.z += -g.y * uOgrenme * dt * 11.0;
    // yuzeye yapis: parcacik araziden kopmasin
    float h = kayip(q) + 0.035;
    v.y += (h - p.y) * 11.0 * dt;
    /* Sonum zayif tutuldu ki momentum kalsin: boylece bazi parcaciklar
       yerel minimumdan sicrayip cikar, bazilari icinde kalir. Gercek
       gradyan inisinde ogrenme oraninin yaptigi tam olarak budur. */
    v *= exp(-2.6 * dt);
  }
  else if(uMod < 2.5){
    /* 2 · k-ortalama: veri noktalari sabit, konum dogrudan yazilir */
    v = vec3(0.0);
  }
  else if(uMod < 3.5){
    /* ---------------- 3 · AIZAWA CEKICISI -----------------
       Lorenz gibi: hiz dogrudan alan, ivme birikmiyor. Zaman olcegi
       d = 3.5'lik donme terimine gore secildi — govde okunacak hizda
       sarilsin, burgu sicramali gorunmesin. */
    v = aizawaHizi(p) * 0.55;
  }
  else if(uMod < 4.5){
    /* ---------------- 4 · LORENZ CEKICISI ----------------- */
    v = lorenzHizi(p) * 0.34;          // zaman olcegi: kelebek okunacak hizda dolassin
  }
  else {
    /* ------- 5 · GRADYAN INISI, 2B KESIT (z = 0) ----------
       Topolojinin tek bir dilimi. Ders kitabindaki klasik gorsel:
       kayip egrisi ve uzerinde yuvarlanan noktalar. */
    /* Her parcacigin KENDI ogrenme orani ve sonumu var. Gercekte de her
       kosu farkli baslar ve farkli sonuclanir; hepsi ayni davranirsa
       tek bir parcacik izlemekten farki kalmaz. */
    float kisisel = 0.80 + hash12(uv * 7.73) * 2.70;      // 0.80x - 3.50x
    /* Sonum dusuk tutuldu: momentum kalmazsa hicbiri sirti asamaz ve
       hepsi ilk yerel cukurda yigiliyor. Dagilim genis -> bir kismi
       asar, bir kismi asamaz. Aranan ayrisma bu. */
    float sonum   = 0.22 + hash12(uv * 19.31) * 1.45;     // 0.22 - 1.67
    float g = kesitTurev(p.x);
    v.x += -g * uOgrenme * kisisel * dt * 11.0;
    float h = kesitKayip(p.x) + 0.03;
    /* Egriye dogru cekim. Yay az sonumlu (k=14, sonum 0.22-1.67 -> sonum
       orani 0.03-0.22) ve bu BILEREK boyle: hoplama buradan geliyor, buyuk
       ogrenme orani uzun sicrama demek. Sorun salinimin kendisi degildi,
       salinimin ALT yariydi — parcacik egrinin altina gecip kayboluyordu. */
    v.y += (h - p.y) * 14.0 * dt;
    /* Tek yonlu kisit: egri bir zemin. Ustunde serbest, altina gecemez.
       Carpma aninda dikey hiz ters cevrilip kisiliyor; parcacik seker,
       sekme sonumlenerek cukurda oturur. */
    if(p.y <= h && v.y < 0.0) v.y = -v.y * 0.42;
    v.z += -p.z * 8.0 * dt;            // dilimde kal
    v *= exp(-sonum * dt);
  }
  gl_FragColor = vec4(v, 1.0);
}
`

const KOSE_SHADER = /* glsl */`
uniform sampler2D texturePosition;
uniform sampler2D textureVelocity;
uniform float uBoyut, uFadeIn, uFadeOut, uSparklingFrequency, uSparklingDuration;
uniform float uPikselOran, uIz, uModR, uMerkezSayi;
uniform vec3 uMerkez[6];
/* Istasyonlar arasi gecis. Parcacik iki istasyon arasinda (A -> B)
   karistirilir; her uc ya CANLI simulasyondur (kamera o istasyonun
   kutusuna oturtulmus) ya da sabit bir SEKIL (ad, zaman cizgisi).
   Simulasyon hic durmaz; yalnizca cizilen konum karisir. uF = 0 ve
   A canli iken cikti birebir eski hero. */
uniform sampler2D uHedefA, uHedefB; // xy: kutu icinde hedef (CSS px), z: gecikme 0-1, w: tohum
uniform vec2  uKutuA, uKutuB;       // sekil kutusunun sol-ust kosesi (CSS px)
uniform float uAsim, uBsim;         // 1: o uc canli simulasyon
uniform vec2  uEkran;               // ekran eni-boyu (CSS px)
uniform float uF, uYayilma, uAdPx, uAkis, uZamanV;
varying float vAd;          // sekle ne kadar yakin (1: keskin beyaz nokta)
attribute vec2  aRef;      // parcacigin doku koordinati
attribute float aRastgele; // boyut cesitliligi
varying float vOmur;
varying float vKivilcim;
varying vec2  vUv;
varying float vKume;
void main(){
  vec4 pk = texture2D(texturePosition, aRef);
  vec3 merkez = pk.xyz;
  vOmur = pk.w;

  // fadeIn / fadeOut -> olcek
  float girisO = uFadeIn  > 0.0 ? clamp(vOmur / uFadeIn, 0.0, 1.0) : 1.0;
  float cikisO = uFadeOut > 0.0 ? clamp((1.0 - vOmur) / uFadeOut, 0.0, 1.0) : 1.0;
  float o = min(girisO, cikisO);
  o = o * o * (3.0 - 2.0 * o);                 // smoothstep

  /* Parcacigin kendi ilerlemesi. Gecikme varis seklinden geliyor (ad ve
     zaman cizgisinde soldan saga); varis canliysa kalkis seklinden. */
  vec4 hA = texture2D(uHedefA, aRef);
  vec4 hB = texture2D(uHedefB, aRef);
  float gecikme = uBsim > 0.5 ? hA.z : hB.z;
  float s = clamp((uF - gecikme * uYayilma) / (1.0 - uYayilma), 0.0, 1.0);
  s = s < 0.5 ? 4.0 * s * s * s : 1.0 - pow(-2.0 * s + 2.0, 3.0) * 0.5;   // ease-in-out
  float sekil = mix(1.0 - uAsim, 1.0 - uBsim, s);
  /* Sekilde omur solmasi kapanir: yoksa yeniden doganlar harfte goz kirpar. */
  o = mix(o, 1.0, sekil);
  vAd = sekil;
  float olcek = o * uBoyut * aRastgele;

  // kivilcim: omrun belli bir aninda kisa parlama
  float kivZaman = hash11(aRef.x * 731.7 + aRef.y * 129.3);
  float kivOmur  = fract(vOmur * uSparklingFrequency);
  vKivilcim = (kivOmur < kivZaman && kivOmur > kivZaman - uSparklingDuration * uSparklingFrequency) ? 1.0 : 0.0;

  // k-ortalama: en yakin merkezin indisi -> renk
  vKume = -1.0;
  if(uModR > 1.5 && uModR < 2.5){
    float enIyi = 1e9; float indis = 0.0;
    for(int i = 0; i < 6; i++){
      if(float(i) >= uMerkezSayi) break;
      float d = distance(merkez.xy, uMerkez[i].xy);
      if(d < enIyi){ enIyi = d; indis = float(i); }
    }
    vKume = indis;
  }

  vUv = uv;

  // kameraya bakan dortgen; uIz > 0 ise hiz yonunde uzatilir (hareket izi)
  vec4 goz = modelViewMatrix * vec4(merkez, 1.0);
  if(uIz > 0.0001){
    vec3 hiz   = texture2D(textureVelocity, aRef).xyz;
    vec3 hizG  = (modelViewMatrix * vec4(hiz, 0.0)).xyz;
    float uz   = length(hizG.xy);
    vec2 yon   = uz > 0.0001 ? hizG.xy / uz : vec2(1.0, 0.0);
    vec2 dik   = vec2(-yon.y, yon.x);
    float kat  = 1.0 + min(uz * uIz, 7.0);
    goz.xy += yon * (position.x * olcek * kat) + dik * (position.y * olcek);
  } else {
    goz.xy += position.xy * olcek;
  }
  gl_Position = projectionMatrix * goz;

  /* Karisim ekran uzayinda (NDC) yapilir: cekici 3B, sekil 2B. Dortgenin
     kose ofseti ayrica tasinir ki parcacik yolda bicimini korusun. */
  bool canliDuruyor = (uAsim > 0.5 && s <= 0.0) || (uBsim > 0.5 && s >= 1.0);
  if(!canliDuruyor){
    vec4 c0 = projectionMatrix * (modelViewMatrix * vec4(merkez, 1.0));
    vec2 nS = c0.xy / c0.w;
    vec2 ofsS = gl_Position.xy / gl_Position.w - nS;
    /* Sekilde boyut PIKSEL cinsinden: kamera derinligine bagli olcek orada
       anlamsiz, ve kuculunce cekirdek alt-piksele dusup parcacik
       gorunmez oluyordu. */
    vec2 ofsT = position.xy * uAdPx * (0.75 + 0.25 * aRastgele) * 2.0 / uEkran;
    vec2 pA = uKutuA + hA.xy, pB = uKutuB + hB.xy;
    vec2 nA = uAsim > 0.5 ? nS : vec2(pA.x / uEkran.x * 2.0 - 1.0, 1.0 - pA.y / uEkran.y * 2.0);
    vec2 nB = uBsim > 0.5 ? nS : vec2(pB.x / uEkran.x * 2.0 - 1.0, 1.0 - pB.y / uEkran.y * 2.0);
    vec2 n = mix(nA, nB, s);
    if(uAkis > 0.0 && s > 0.0 && s < 1.0){
      /* Akis: yolun ustundeki noktada curl gurultusu okunur. Yakin
         parcaciklar yakin deger alir, bu yuzden tek tek titremezler,
         dere gibi birlikte kivrilirlar. sin(pi s) kalkista ve varista
         sifir: baslangic ve varis yeri bozulmaz, yalniz yol akar. */
      vec2 px = (n * 0.5 + 0.5) * uEkran;
      vec3 girdap = curlNoise(vec3(px / 260.0, uZamanV * 0.12 + hB.w * 0.15));
      n += girdap.xy * (uAkis * 110.0 * sin(3.14159265 * s)) * 2.0 / uEkran;
    }
    gl_Position = vec4(n + mix(uAsim > 0.5 ? ofsS : ofsT, uBsim > 0.5 ? ofsS : ofsT, s), 0.0, 1.0);
  }
}
`.replace('${HASH}', '')

const PARCA_SHADER = /* glsl */`
uniform vec3  uColorIn, uColorOut;
uniform float uGlowSpread, uSolidRatio, uSolidAlpha, uOpacity, uSparklingAlpha, uAcik;
uniform float uGorunur;     // yaziya devirde parcaciklar soner (0-1)
varying float vOmur;
varying float vKivilcim;
varying vec2  vUv;
varying float vKume;
varying float vAd;
void main(){
  float d = length(vUv - 0.5);
  if(d > 0.5) discard;
  if(vAd >= 0.999){
    /* Adda: hale yok, kivilcim yok, beyaz ve keskin dolu disk. */
    float disk = 1.0 - smoothstep(0.34, 0.5, d);
    gl_FragColor = vec4(vec3(disk), disk) * uGorunur;
    return;
  }

  // cekirdek: solidRatio yaricapinda tam parlak
  float cekirdek = (1.0 - step(uSolidRatio * 0.5, d)) * uSolidAlpha;
  // parlama: merkeze yaklastikca artan hale
  float hale = uGlowSpread / max(d, 0.0001) - uGlowSpread * 2.0;
  hale *= (1.0 - cekirdek);

  float alfa = max(hale, cekirdek) * uOpacity;
  alfa *= vKivilcim * uSparklingAlpha * (1.0 - vAd) + 1.0;
  // yolda: hale profili kademeli olarak keskin diske doner
  alfa = mix(alfa, 1.0 - smoothstep(0.34, 0.5, d), vAd);
  if(alfa <= 0.0) discard;

  vec3 renk = mix(uColorIn, uColorOut, vOmur);
  if(vKume >= 0.0){                      // k-ortalama kume paleti
    int k = int(vKume + 0.5);
    if(k == 0)      renk = vec3(1.00, 0.82, 0.23);
    else if(k == 1) renk = vec3(0.29, 0.48, 1.00);
    else if(k == 2) renk = vec3(0.49, 0.97, 0.77);
    else if(k == 3) renk = vec3(1.00, 0.48, 0.42);
    else if(k == 4) renk = vec3(0.65, 0.55, 0.98);
    else            renk = vec3(0.90, 0.90, 0.92);
  }
  renk = mix(renk, vec3(1.0), vAd);        // ada yaklastikca beyaza
  if(uAcik > 0.5){
    /* Acik temada toplamali karisim ise yaramaz: beyaza isik eklenemez.
       Carpimsal karisimda parcacik kagida oturan murekkep gibi karartir. */
    /* Halenin uzamsal bicimi ayni kalsin diye yalnizca alfa tepkisi
       guclendiriliyor: toplamalida siyah zeminde goruneni, carpimsalda
       beyaz zeminde ayni buyuklukte gostermek icin gerekli. */
    float a = clamp(alfa * 4.5, 0.0, 1.0);
    gl_FragColor = vec4(mix(vec3(1.0), renk, a), 1.0);
  } else {
    gl_FragColor = vec4(renk * alfa, alfa) * uGorunur;   // toplamali karisim icin onceden carpilmis
  }
}
`

/* ---------------------------------------------------------------- sinif */

export default class Parcaciklar {
  constructor(renderer, sayi = 10000){
    this.renderer = renderer
    this.emitterPosition = new THREE.Vector3(0, 0.35, 0)
    this._oncekiEmitter = this.emitterPosition.clone()
    this._emitterHiz = new THREE.Vector3()
    this._oncekiEmitterHiz = new THREE.Vector3()

    this.ayar = {
      decayFrequency: 0.2, velocityDamping: 0.01,
      emitterRadius: 0.01, emitterVelocityStrength: 0.4, initialRandomVelocity: 0,
      initialVelocity: new THREE.Vector3(0, 0, 0),
      turbulenceStrength: 0.01, turbulenceTimeFrequeny: 0.1, turbulencePositionFrequeny: 3,
      gravity: new THREE.Vector3(0, -0.5, 0), floorY: -0.95, floorDamping: 0.1,
      colorIn: new THREE.Color('#ffd23a'), colorOut: new THREE.Color('#4a7bff'),
      fadeIn: 0.2, fadeOut: 0.2, size: 0.075, glowSpread: 0.02,
      solidRatio: 0.05, solidAlpha: 5, opacity: 1,
      sparklingAlpha: 4, sparklingFrequency: 1, sparklingDuration: 0.01,
      uMod: 0,          // 0 fiskiye · 1 gradyan inisi · 2 optik akis · 3 islem hatti
      uIz: 0,           // hiz yonunde uzatma: akis modlarinda izi gorunur kilar
      uOgrenme: 0.9     // gradyan inisinde ogrenme orani
    }

    this.kur(sayi)
  }

  kur(sayi){
    this.sayi = sayi
    // kare doku: yan uzunlugu
    this.en = Math.ceil(Math.sqrt(sayi))
    this.sayi = this.en * this.en

    this.gpu = new GPUComputationRenderer(this.en, this.en, this.renderer)
    /* Tam float render hedefi her yerde yok (ozellikle iOS Safari):
       EXT_color_buffer_float yoksa yarim duyarlikli hedefe dus. Konumlar
       +-3 civarinda, yarim duyarlilik (~0.002) gorunur fark yaratmiyor. */
    // ?yarim=1 : bu yolu float destegi olan tarayicida da zorla (test)
    const yarimZorla = new URLSearchParams(location.search).has('yarim')
    if(yarimZorla || this.renderer.capabilities.isWebGL2 !== true || !this.renderer.extensions.has('EXT_color_buffer_float')){
      this.gpu.setDataType(THREE.HalfFloatType)
    }

    const dPos = this.gpu.createTexture()
    const dHiz = this.gpu.createTexture()
    this._tohumla(dPos, dHiz)

    this.degPos = this.gpu.addVariable('texturePosition', POZISYON_SHADER, dPos)
    this.degHiz = this.gpu.addVariable('textureVelocity', HIZ_SHADER, dHiz)
    this.gpu.setVariableDependencies(this.degPos, [this.degPos, this.degHiz])
    this.gpu.setVariableDependencies(this.degHiz, [this.degPos, this.degHiz])

    const a = this.ayar
    Object.assign(this.degPos.material.uniforms, {
      dt: { value: 0 }, zaman: { value: 0 },
      decayFrequency: { value: a.decayFrequency },
      emitterRadius: { value: a.emitterRadius },
      floorY: { value: a.floorY },
      uSifirla: { value: 0 },
      emitterPosition: { value: this.emitterPosition.clone() },
      emitterPreviousPosition: { value: this._oncekiEmitter.clone() },
      uMod: { value: a.uMod }
    })
    Object.assign(this.degHiz.material.uniforms, {
      dt: { value: 0 }, zaman: { value: 0 },
      decayFrequency: { value: a.decayFrequency },
      velocityDamping: { value: a.velocityDamping },
      turbulenceStrength: { value: a.turbulenceStrength },
      turbulenceTimeFrequeny: { value: a.turbulenceTimeFrequeny },
      turbulencePositionFrequeny: { value: a.turbulencePositionFrequeny },
      floorY: { value: a.floorY }, floorDamping: { value: a.floorDamping },
      emitterVelocityStrength: { value: a.emitterVelocityStrength },
      initialRandomVelocity: { value: a.initialRandomVelocity },
      gravity: { value: a.gravity.clone() },
      initialVelocity: { value: a.initialVelocity.clone() },
      emitterVelocity: { value: this._emitterHiz.clone() },
      emitterPreviousVelocity: { value: this._oncekiEmitterHiz.clone() },
      uMod: { value: a.uMod },
      uOgrenme: { value: a.uOgrenme },
      uSifirla: { value: 0 }
    })

    const hata = this.gpu.init()
    if(hata !== null) throw new Error('GPUComputationRenderer: ' + hata)

    this._meshKur()
  }

  _tohumla(dPos, dHiz){
    const p = dPos.image.data, h = dHiz.image.data
    for(let i = 0; i < p.length; i += 4){
      p[i] = 0; p[i+1] = 99999; p[i+2] = 0     // goruntu disinda basla
      p[i+3] = Math.random()                    // omur dagilsin ki hep birlikte dogmasinlar
      h[i] = 0; h[i+1] = 0; h[i+2] = 0; h[i+3] = 1
    }
  }

  _meshKur(){
    const g = new THREE.InstancedBufferGeometry()
    // tek dortgen
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      -0.5,-0.5,0,  0.5,-0.5,0,  0.5,0.5,0,  -0.5,0.5,0
    ]), 3))
    g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([0,0, 1,0, 1,1, 0,1]), 2))
    g.setIndex([0,1,2, 0,2,3])

    const ref = new Float32Array(this.sayi * 2)
    const rnd = new Float32Array(this.sayi)
    let n = 0
    for(let y = 0; y < this.en; y++){
      for(let x = 0; x < this.en; x++){
        ref[n*2]   = (x + 0.5) / this.en
        ref[n*2+1] = (y + 0.5) / this.en
        rnd[n] = 0.25 + Math.random() * 0.75
        n++
      }
    }
    g.setAttribute('aRef', new THREE.InstancedBufferAttribute(ref, 2))
    g.setAttribute('aRastgele', new THREE.InstancedBufferAttribute(rnd, 1))
    g.instanceCount = this.sayi
    // kutu hesabi anlamsiz: konumlar dokudan geliyor
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 1e6)

    const a = this.ayar
    this.sekiller = []
    this._bosDoku = this._doku()
    this.material = new THREE.ShaderMaterial({
      vertexShader: HASH + SIMPLEX + KOSE_SHADER,
      fragmentShader: PARCA_SHADER,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: false,
      uniforms: {
        texturePosition: { value: null },
        textureVelocity: { value: null },
        uIz: { value: 0 },
        uModR: { value: 0 },
        uMerkezSayi: { value: 5 },
        uMerkez: { value: [0,1,2,3,4,5].map(() => new THREE.Vector3()) },
        uBoyut: { value: a.size }, uFadeIn: { value: a.fadeIn }, uFadeOut: { value: a.fadeOut },
        uSparklingFrequency: { value: a.sparklingFrequency },
        uSparklingDuration: { value: a.sparklingDuration },
        uSparklingAlpha: { value: a.sparklingAlpha },
        uColorIn: { value: a.colorIn.clone() }, uColorOut: { value: a.colorOut.clone() },
        uGlowSpread: { value: a.glowSpread }, uSolidRatio: { value: a.solidRatio },
        uSolidAlpha: { value: a.solidAlpha }, uOpacity: { value: a.opacity },
        uPikselOran: { value: 1 },
        uAcik: { value: 0 },
        uHedefA: { value: this._bosDoku }, uHedefB: { value: this._bosDoku },
        uKutuA: { value: new THREE.Vector2() }, uKutuB: { value: new THREE.Vector2() },
        uAsim: { value: 1 }, uBsim: { value: 0 },
        uEkran: { value: new THREE.Vector2(1, 1) },
        uF: { value: 0 }, uYayilma: { value: 0.45 },
        uAdPx: { value: 2.2 }, uAkis: { value: 1 }, uZamanV: { value: 0 },
        uGorunur: { value: 1 }
      }
    })

    this.mesh = new THREE.Mesh(g, this.material)
    this.mesh.frustumCulled = false
  }

  _doku(veri){
    const t = new THREE.DataTexture(veri || new Float32Array(this.sayi * 4),
      this.en, this.en, THREE.RGBAFormat, THREE.FloatType)
    t.needsUpdate = true
    return t
  }

  /** Bir sekil istasyonunun hedeflerini yukle. noktalar: [x0,y0, x1,y1, ...]
      CSS px, kutuya gore; en: kutunun eni (gecikme soldan saga). Noktalar
      karistirilip sirayla dagitilir: parcacik azsa her yerden esit secilir,
      coksa hepsi dolar. */
  sekilKur(no, noktalar, en){
    const n = noktalar.length / 2
    if(n === 0) return
    const sira = new Uint32Array(n)
    for(let i = 0; i < n; i++) sira[i] = i
    for(let i = n - 1; i > 0; i--){ const j = Math.floor(Math.random() * (i + 1)); const t = sira[i]; sira[i] = sira[j]; sira[j] = t }
    const d = new Float32Array(this.sayi * 4)
    for(let i = 0; i < this.sayi; i++){
      const k = sira[i % n]
      const x = noktalar[k*2] + (Math.random() - 0.5), y = noktalar[k*2+1] + (Math.random() - 0.5)
      d[i*4]   = x
      d[i*4+1] = y
      d[i*4+2] = Math.min(1, (x / en) * 0.92 + Math.random() * 0.08)
      d[i*4+3] = Math.random()
    }
    if(this.sekiller[no]) this.sekiller[no].dispose()
    this.sekiller[no] = this._doku(d)
  }

  /** Hangi iki istasyon arasindayiz. canli: o uc simulasyon mu. */
  bolum(a, b, canliA, canliB){
    const u = this.material.uniforms
    u.uHedefA.value = (!canliA && this.sekiller[a]) || this._bosDoku
    u.uHedefB.value = (!canliB && this.sekiller[b]) || this._bosDoku
    u.uAsim.value = canliA ? 1 : 0
    u.uBsim.value = canliB ? 1 : 0
  }

  /** Tek bir ayari degistir. Ornek: sis.ayarla('size', 0.2) */
  ayarla(ad, deger){
    const a = this.ayar
    if(a[ad] && a[ad].isColor){ a[ad].set(deger) }
    else if(a[ad] && a[ad].isVector3){ a[ad].copy(deger) }
    else { a[ad] = deger }

    const p = this.degPos.material.uniforms, h = this.degHiz.material.uniforms, m = this.material.uniforms
    if(ad in p) p[ad].value = a[ad] && a[ad].isVector3 ? a[ad].clone() : a[ad]
    if(ad in h) h[ad].value = a[ad] && a[ad].isVector3 ? a[ad].clone() : a[ad]
    const eslesme = {
      size:'uBoyut', fadeIn:'uFadeIn', fadeOut:'uFadeOut', glowSpread:'uGlowSpread',
      solidRatio:'uSolidRatio', solidAlpha:'uSolidAlpha', opacity:'uOpacity',
      sparklingAlpha:'uSparklingAlpha', sparklingFrequency:'uSparklingFrequency',
      sparklingDuration:'uSparklingDuration', colorIn:'uColorIn', colorOut:'uColorOut',
      uIz:'uIz', uMod:'uModR'
    }
    if(eslesme[ad]) m[eslesme[ad]].value = a[ad] && a[ad].isColor ? a[ad].clone() : a[ad]
  }

  /** Butun parcaciklari bastan dogurt. Ogrenme orani degisince cagrilir:
      mevcut konumlarindan devam etmeleri yaniltici olurdu. */
  sifirla(){ this._sifirlaKare = 2 }

  /** Acik/koyu tema: karisim kipini de degistirir */
  tema(acik){
    this.material.uniforms.uAcik.value = acik ? 1 : 0
    this.material.blending = acik ? THREE.MultiplyBlending : THREE.AdditiveBlending
    this.material.needsUpdate = true
  }

  /** k-ortalama merkezlerini gonder (en fazla 6) */
  merkezler(liste){
    const u = this.material.uniforms
    u.uMerkezSayi.value = Math.min(liste.length, 6)
    for(let i = 0; i < 6; i++){
      u.uMerkez.value[i].copy(liste[i] || liste[liste.length - 1] || new THREE.Vector3())
    }
  }

  update(dt, zaman){
    dt = Math.min(Math.max(dt, 0.0001), 0.05)

    // emitter hizi — bolme sifira karsi korumali
    this._emitterHiz.copy(this.emitterPosition).sub(this._oncekiEmitter).divideScalar(dt)

    const p = this.degPos.material.uniforms, h = this.degHiz.material.uniforms
    p.dt.value = dt; h.dt.value = dt
    p.zaman.value = zaman; h.zaman.value = zaman
    p.emitterPosition.value.copy(this.emitterPosition)
    p.emitterPreviousPosition.value.copy(this._oncekiEmitter)
    h.emitterVelocity.value.copy(this._emitterHiz)
    h.emitterPreviousVelocity.value.copy(this._oncekiEmitterHiz)

    const sif = this._sifirlaKare > 0 ? 1 : 0
    this.degPos.material.uniforms.uSifirla.value = sif
    this.degHiz.material.uniforms.uSifirla.value = sif
    if(this._sifirlaKare > 0) this._sifirlaKare--

    this.gpu.compute()
    this.material.uniforms.texturePosition.value =
      this.gpu.getCurrentRenderTarget(this.degPos).texture
    this.material.uniforms.textureVelocity.value =
      this.gpu.getCurrentRenderTarget(this.degHiz).texture

    this._oncekiEmitter.copy(this.emitterPosition)
    this._oncekiEmitterHiz.copy(this._emitterHiz)
  }

  dispose(){
    this.gpu.dispose()
    this.mesh.geometry.dispose()
    this.sekiller.forEach(t => t && t.dispose())
    this._bosDoku.dispose()
    this.material.dispose()
    this.mesh.removeFromParent()
  }
}
