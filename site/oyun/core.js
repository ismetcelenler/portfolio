// Jump Jump çekirdeği: oyun mantığı ve HUD. Görsel katman (3D / 2D) bunu okur.
(function () {
  const MAXD = 6.5;          // tam şarjda zıplama mesafesi (dünya birimi)
  const CHARGE_TIME = 1.5;   // tam şarj süresi (sn)
  const JUMP_DUR = 0.45;
  const JUMP_H = 2.2;
  const PLAT_H = 1;
  const PERFECT_R = 0.3;
  const PALETTE = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#c77dff', '#ff9f45'];

  const rand = (a, b) => a + Math.random() * (b - a);
  const pick = () => PALETTE[(Math.random() * PALETTE.length) | 0];
  const inside = (p, x, z) => Math.abs(x - p.x) <= p.w / 2 && Math.abs(z - p.z) <= p.w / 2;

  function loadBest() { try { return +localStorage.getItem('jj-best') || 0; } catch (e) { return 0; } }
  function saveBest(v) { try { localStorage.setItem('jj-best', v); } catch (e) {} }

  function create() {
    const g = { listeners: [], PLAT_H };
    let idc = 0;
    const emit = (e) => g.listeners.forEach((fn) => fn(e));
    g.on = (fn) => g.listeners.push(fn);

    function spawnNext() {
      const c = g.cur;
      const dir = Math.random() < 0.5 ? 'x' : 'z';
      // zorluk skora değil zıplama sayısına bağlı: kombo yapan oyuncu cezalanmasın
      const shrink = Math.min(0.6, g.jumps * 0.015);
      const w = Math.max(1.1, rand(1.4, 2.4) - shrink);
      const d = c.w / 2 + w / 2 + rand(0.6, 2.8);
      const n = { id: ++idc, x: c.x + (dir === 'x' ? d : 0), z: c.z + (dir === 'z' ? d : 0), w, color: pick(), squash: 1 };
      g.platforms.push(n);
      g.next = n;
      emit({ type: 'spawn', plat: n });
      if (g.platforms.length > 8) emit({ type: 'remove', plat: g.platforms.shift() });
    }

    g.reset = function () {
      g.platforms = [];
      g.score = 0; g.combo = 0; g.charge = 0; g.overT = 0; g.jumps = 0;
      g.best = loadBest();
      g.phase = 'idle';
      const p0 = { id: ++idc, x: 0, z: 0, w: 2.2, color: pick(), squash: 1 };
      g.platforms.push(p0);
      g.cur = p0;
      emit({ type: 'reset' });
      emit({ type: 'spawn', plat: p0 });
      spawnNext();
      g.player = { x: 0, z: 0, y: PLAT_H, sq: 1, rot: 0, dx: 1, dz: 0 };
      g.cam = { x: (p0.x + g.next.x) / 2, z: (p0.z + g.next.z) / 2 };
    };

    g.press = function () {
      if (g.phase === 'idle') { g.phase = 'charging'; g.charge = 0; emit({ type: 'charge' }); }
      else if (g.phase === 'over' && g.overT > 0.4) g.reset();
    };

    // Sekme gizlenir / pencere odağı kaybolursa yarım şarjı zıplamadan iptal et
    g.cancel = function () {
      if (g.phase === 'charging') { g.phase = 'idle'; g.charge = 0; }
    };

    g.release = function () {
      if (g.phase !== 'charging') return;
      const p = g.player;
      let dx = g.next.x - p.x, dz = g.next.z - p.z;
      const len = Math.hypot(dx, dz) || 1;
      dx /= len; dz /= len;
      const d = g.charge * MAXD;
      p.dx = dx; p.dz = dz;
      g.jump = { fx: p.x, fz: p.z, tx: p.x + dx * d, tz: p.z + dz * d, t: 0 };
      g.phase = 'jumping';
      emit({ type: 'jump' });
    };

    function resolve() {
      const p = g.player, n = g.next;
      if (inside(n, p.x, p.z)) {
        const perfect = Math.hypot(p.x - n.x, p.z - n.z) < PERFECT_R;
        let gain;
        if (perfect) { g.combo++; gain = 2 * g.combo; } else { g.combo = 0; gain = 1; }
        g.score += gain;
        g.jumps++;
        g.cur = n;
        g.phase = 'idle';
        emit({ type: 'land', perfect, gain, combo: g.combo, x: p.x, z: p.z });
        spawnNext();
      } else if (inside(g.cur, p.x, p.z)) {
        g.phase = 'idle';
        emit({ type: 'stay' });
      } else {
        g.phase = 'falling'; g.vy = 0;
        emit({ type: 'fall' });
      }
    }

    g.update = function (dt) {
      const p = g.player, k = Math.min(1, dt * 14);
      if (g.phase === 'charging') {
        g.charge = Math.min(1, g.charge + dt / CHARGE_TIME);
        p.sq = 1 - 0.45 * g.charge;
        g.cur.squash = 1 - 0.25 * g.charge;
      } else {
        p.sq += (1 - p.sq) * k;
        g.platforms.forEach((pl) => { pl.squash += (1 - pl.squash) * k; });
      }
      if (g.phase === 'idle' || g.phase === 'charging') {
        p.y = PLAT_H * g.cur.squash; p.rot = 0;
      } else if (g.phase === 'jumping') {
        const j = g.jump;
        j.t = Math.min(1, j.t + dt / JUMP_DUR);
        p.x = j.fx + (j.tx - j.fx) * j.t;
        p.z = j.fz + (j.tz - j.fz) * j.t;
        p.y = PLAT_H + JUMP_H * 4 * j.t * (1 - j.t);
        p.rot = j.t * Math.PI * 2;
        if (j.t >= 1) { p.rot = 0; resolve(); }
      } else if (g.phase === 'falling') {
        g.vy -= 30 * dt;
        p.y += g.vy * dt;
        if (p.y < -6) {
          g.phase = 'over'; g.overT = 0;
          const record = g.score > g.best;
          if (record) { g.best = g.score; saveBest(g.best); }
          emit({ type: 'over', score: g.score, record });
        }
      } else if (g.phase === 'over') {
        g.overT += dt;
      }
      const tx = (g.cur.x + g.next.x) / 2, tz = (g.cur.z + g.next.z) / 2;
      const ck = Math.min(1, dt * 4);
      g.cam.x += (tx - g.cam.x) * ck;
      g.cam.z += (tz - g.cam.z) * ck;
    };

    return g;
  }

  // Ortak giriş: fare, dokunma ve boşluk tuşu
  function bindInput(g, el) {
    let pid = null; // yalnız ilk parmak sayılır, ikinci parmak karıştırmaz
    el.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      if (pid !== null) return;
      pid = e.pointerId;
      g.press();
    });
    const up = (e) => { if (e.pointerId !== pid) return; pid = null; g.release(); };
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', (e) => { if (e.pointerId !== pid) return; pid = null; g.cancel(); });
    window.addEventListener('keydown', (e) => {
      if (e.code !== 'Space' || e.repeat || e.target.closest?.('button')) return;
      e.preventDefault();
      g.press();
    });
    window.addEventListener('keyup', (e) => { if (e.code === 'Space') g.release(); });
    window.addEventListener('blur', () => { pid = null; g.cancel(); });
    document.addEventListener('visibilitychange', () => { if (document.hidden) { pid = null; g.cancel(); } });
    document.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  // Ortak HUD: #score, #best, #pop, #msg
  function attachHud(g) {
    const $ = (id) => document.getElementById(id);
    let popTimer;
    const refresh = () => { $('score').textContent = g.score; $('best').textContent = 'REKOR ' + g.best; };
    const pop = (text) => {
      const el = $('pop');
      el.textContent = text; el.style.opacity = 1;
      clearTimeout(popTimer);
      popTimer = setTimeout(() => { el.style.opacity = 0; }, 650);
    };
    g.on((e) => {
      if (e.type === 'reset') { $('msg').innerHTML = 'basılı tut &rarr; bırak'; setTimeout(refresh); }
      if (e.type === 'charge') $('msg').textContent = '';
      if (e.type === 'land') { refresh(); pop(e.perfect ? 'MÜKEMMEL +' + e.gain : '+1'); }
      if (e.type === 'over') {
        refresh();
        $('msg').innerHTML = (e.record ? 'YENİ REKOR!' : 'OYUN BİTTİ') + '<br>skor ' + e.score + '<br><small>tekrar için dokun</small>';
      }
    });
  }

  window.JJ = { create, bindInput, attachHud, PALETTE };
})();
