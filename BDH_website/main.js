// === BULLETPROOF SHOWCASE ENGINE ===

// --- LOADER (FAIL-SAFE) ---
function initLoader() {
  const fill = document.getElementById('loader-fill');
  const loader = document.getElementById('loader');
  if(!loader) return;
  
  let p = 0;
  const interval = setInterval(() => {
    p += Math.random() * 20 + 5;
    if(p >= 100) {
      p = 100;
      clearInterval(interval);
      exitLoader();
    }
    if(fill) fill.style.width = p + '%';
  }, 100);

  // EMERGENCY BYPASS: Hide loader after 4s no matter what
  setTimeout(exitLoader, 4000);

  function exitLoader() {
    if(loader.classList.contains('hidden')) return;
    loader.style.opacity = '0';
    setTimeout(() => {
      loader.classList.add('hidden');
      startReveal();
    }, 600);
  }
}

// --- VISUAL REVEAL ---
function startReveal() {
  // Reveal elements on scroll
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if(entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.reveal-sota').forEach(el => {
    el.style.transition = 'all 1.2s cubic-bezier(0.16, 1, 0.3, 1)';
    observer.observe(el);
  });
  
  // Hero Entrance
  if(typeof gsap !== 'undefined') {
    gsap.from('.headline-huge', { y: 100, opacity: 0, duration: 2, ease: 'expo.out' });
  }
}

// --- THREE.JS BACKGROUND ---
function initThree() {
  const container = document.getElementById('neural-cloud-container');
  if(!container || typeof THREE === 'undefined') return;
  
  try {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.2));
    container.appendChild(renderer.domElement);

    const count = 2000;
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    for(let i=0; i<count; i++) {
      const r = 15 + Math.random() * 10;
      const t = Math.random() * Math.PI * 2;
      const p = Math.acos(2 * Math.random() - 1);
      pos[i*3] = r * Math.sin(p) * Math.cos(t);
      pos[i*3+1] = r * Math.sin(p) * Math.sin(t);
      pos[i*3+2] = r * Math.cos(p);
      if(Math.random() > 0.2) {
        colors[i*3] = 0.83; colors[i*3+1] = 0.68; colors[i*3+2] = 0.21;
      } else {
        colors[i*3] = 0.8; colors[i*3+1] = 0.8; colors[i*3+2] = 0.8;
      }
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const points = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.1, vertexColors: true, transparent: true, opacity: 0.3 }));
    scene.add(points);

    camera.position.z = 40;
    let mx=0, my=0, curMx=0, curMy=0;
    document.addEventListener('mousemove', e => { mx = (e.clientX/window.innerWidth-0.5); my = (e.clientY/window.innerHeight-0.5); });

    function animate() {
      requestAnimationFrame(animate);
      curMx += (mx - curMx) * 0.05;
      curMy += (my - curMy) * 0.05;
      points.rotation.y += 0.0005;
      points.rotation.x = curMy * 0.2;
      points.rotation.y += curMx * 0.2;
      renderer.render(scene, camera);
    }
    animate();
    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth/window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });
  } catch(e) {
    console.warn("Three.js Init Failed:", e);
  }
}

// --- INIT ---
document.addEventListener('DOMContentLoaded', () => {
  initLoader();
  initThree();
  
  // Spotlight
  document.addEventListener('mousemove', e => {
    document.documentElement.style.setProperty('--mx', e.clientX + 'px');
    document.documentElement.style.setProperty('--my', e.clientY + 'px');
  });
});
