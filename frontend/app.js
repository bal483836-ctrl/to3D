import * as THREE from 'three';
import { GLTFLoader } from './vendor/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from './vendor/jsm/controls/OrbitControls.js';

// --- 视图槽位定义（对应后端 ViewName 与 UI 布局） ---------------------------
const VIEWS = [
  { v: 'empty' }, { v: 'top', label: '顶图', ic: '⬒' }, { v: 'empty' },
  { v: 'left45', label: '左45°', ic: '◱' }, { v: 'front', label: '正图', ic: '▣', req: true }, { v: 'right45', label: '右45°', ic: '◲' },
  { v: 'left', label: '左图', ic: '◧' }, { v: 'empty' }, { v: 'right', label: '右图', ic: '◨' },
  { v: 'empty' }, { v: 'back', label: '背图', ic: '▤' }, { v: 'empty' },
  { v: 'empty' }, { v: 'bottom', label: '底图', ic: '⬓' }, { v: 'empty' },
];
const DIM_LABELS = {
  shape: ['器型', 'shape'], pattern: ['花纹', 'pattern'], bottom: ['底部', 'bottom'],
  height: ['高度', 'height'], width: ['宽度', 'width'], material: ['材质', 'material'],
};
const STAGE_SEQ = [
  ['preprocessing', '预处理 · 文字约束抽取'],
  ['generating', '图文联合生成（图 + 文同时）'],
  ['verifying', '一致性自检'],
  ['refining', '定向修正'],
  ['done', '完成'],
];

const images = {}; // view -> dataURL
let currentTaskId = null;

// --- 构建视图上传网格 --------------------------------------------------------
const grid = document.getElementById('viewgrid');
VIEWS.forEach((view) => {
  const el = document.createElement('div');
  if (view.v === 'empty') { el.className = 'slot empty'; grid.appendChild(el); return; }
  el.className = 'slot' + (view.req ? ' req' : '');
  el.innerHTML = `<span class="ic">${view.ic}</span>${view.label}${view.req ? ' *' : ''}`;
  el.dataset.view = view.v;
  el.addEventListener('click', () => pickImage(view.v, el, view));
  grid.appendChild(el);
});

function pickImage(v, el, view) {
  const input = document.createElement('input');
  input.type = 'file'; input.accept = 'image/*';
  input.onchange = () => {
    const file = input.files[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      images[v] = reader.result;
      el.classList.add('filled');
      el.innerHTML = `<img src="${reader.result}" alt="${view.label}"/>` +
        `<span class="rm" title="移除">×</span>`;
      el.querySelector('.rm').addEventListener('click', (e) => {
        e.stopPropagation(); delete images[v];
        el.classList.remove('filled');
        el.innerHTML = `<span class="ic">${view.ic}</span>${view.label}${view.req ? ' *' : ''}`;
        el.addEventListener('click', () => pickImage(v, el, view));
        updateCount();
      });
      updateCount();
    };
    reader.readAsDataURL(file);
  };
  input.click();
}
function updateCount() {
  document.getElementById('imgCount').textContent = `${Object.keys(images).length} / 8`;
}

// --- 表单交互 ----------------------------------------------------------------
document.querySelectorAll('.ex').forEach((b) =>
  b.addEventListener('click', () => { document.getElementById('prompt').value = b.textContent; }));
document.getElementById('threshold').addEventListener('input', (e) =>
  document.getElementById('thVal').textContent = Number(e.target.value).toFixed(2));
document.getElementById('modeSeg').addEventListener('click', (e) => {
  if (e.target.tagName !== 'SPAN') return;
  document.querySelectorAll('#modeSeg span').forEach((s) => s.classList.remove('on'));
  e.target.classList.add('on');
});

// --- 生成 --------------------------------------------------------------------
document.getElementById('genBtn').addEventListener('click', startGeneration);

async function startGeneration() {
  const err = document.getElementById('err'); err.textContent = '';
  const prompt = document.getElementById('prompt').value.trim();
  const imgList = Object.entries(images).map(([v, url]) => ({ view: v, url, required: v === 'front' }));

  if (!prompt) { err.textContent = '请填写文字描述。'; return; }
  if (imgList.length < 2) { err.textContent = '至少上传 2 张图片。'; return; }
  if (!images.front) { err.textContent = '正图为必填。'; return; }

  const body = {
    images: imgList,
    prompt,
    fusion: {
      strategy: document.querySelector('input[name=strategy]:checked').value,
      consistency_threshold: Number(document.getElementById('threshold').value),
      max_refine_rounds: Number(document.getElementById('maxRounds').value),
    },
    human_in_loop: document.getElementById('hil').checked,
  };

  const btn = document.getElementById('genBtn'); btn.disabled = true; btn.textContent = '生成中…';
  resetOutputs();
  try {
    const resp = await fetch('/api/v1/generation', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
    const { task_id } = await resp.json();
    currentTaskId = task_id;
    openSocket(task_id);
  } catch (e) {
    err.textContent = '创建任务失败：' + e.message;
    btn.disabled = false; btn.textContent = '✦ 立即生成';
  }
}

function openSocket(taskId) {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws/tasks/${taskId}`);
  ws.onmessage = (ev) => onState(JSON.parse(ev.data));
  ws.onclose = () => {
    const btn = document.getElementById('genBtn'); btn.disabled = false; btn.textContent = '✦ 立即生成';
  };
}

// --- 状态渲染 ----------------------------------------------------------------
function onState(s) {
  renderStages(s);
  if (s.diff_report) renderReport(s.diff_report);
  if (s.status === 'awaiting_user') renderDecision(s.diff_report);
  else document.getElementById('decision').hidden = true;
  if (s.status === 'done') { renderDownloads(s); loadMesh(currentTaskId); }
  if (s.status === 'failed') document.getElementById('err').textContent = '生成失败：' + (s.error || '');
}

function renderStages(s) {
  const box = document.getElementById('stages');
  const order = STAGE_SEQ.map((x) => x[0]);
  const idx = order.indexOf(s.status === 'awaiting_user' ? 'verifying' : s.status);
  box.innerHTML = STAGE_SEQ.map(([k, label], i) => {
    let cls = 'stage';
    if (i < idx || s.status === 'done') cls += ' done';
    if (i === idx && s.status !== 'done') cls += ' active';
    const r = (k === 'verifying' || k === 'refining') && s.round ? `第 ${s.round} 轮` : '';
    return `<div class="${cls}"><span class="b"></span>${label}<span class="r">${r}</span></div>`;
  }).join('');
}

// 未达阈值但仲裁为「以图为准」的维度：差异被有意保留，不是待修正项
const isKeptDiff = (d) => !d.passed && d.authority === 'image';

function renderReport(rep) {
  const box = document.getElementById('report'); box.hidden = false;
  const rows = rep.dimensions.map((d) => {
    const [zh, en] = DIM_LABELS[d.dimension];
    const color = d.passed ? 'var(--good)' : (isKeptDiff(d) ? 'var(--warn)' : 'var(--bad)');
    const authTxt = isKeptDiff(d) ? '以图为准 · 保留差异'
      : { image: '以图为准', text: '以文校正', low_confidence: '低置信' }[d.authority];
    return `<div class="dimrow">
        <div class="nm">${zh}<small>${en}</small></div>
        <div class="bar"><i style="width:${(d.score * 100).toFixed(0)}%;background:${color}"></i></div>
        <div class="v">${d.score.toFixed(2)}</div>
      </div>
      <div class="auth"><span class="tag ${d.authority}${isKeptDiff(d) ? ' kept' : ''}">${authTxt}</span> ${d.detail}</div>`;
  }).join('');
  // 「已一致」不等于六维全绿：以图为准而保留的差异要说清楚，否则红条与结论看着矛盾
  const kept = rep.dimensions.filter(isKeptDiff);
  const keptTxt = kept.length
    ? `<div class="note">${kept.length} 项未达阈值但有对应视角图，以图为准保留差异，未触发修正：` +
      `${kept.map((d) => DIM_LABELS[d.dimension][0]).join('、')}</div>`
    : '';
  const st = rep.passed ? '<span style="color:var(--good)">已一致</span>'
    : `<span style="color:var(--warn)">待修正 ${rep.actions.length} 项</span>`;
  box.innerHTML = `<h4>六维一致性差异报告</h4>
    <div class="ov">overall ${rep.overall.toFixed(2)} · 阈值 ${rep.threshold.toFixed(2)} · ${st}</div>
    ${keptTxt}${rows}`;
}

function renderDecision(rep) {
  const box = document.getElementById('decision'); box.hidden = false;
  const acts = (rep?.actions || []).join('，') || '无';
  box.innerHTML = `<h4>需要你的决策（人在环）</h4>
    <div style="color:var(--muted);font-size:12.5px;margin-bottom:10px">建议修正动作：${acts}</div>
    <div class="acts">
      <button class="primary" data-a="accept_all">全部修正</button>
      <button data-a="reject_keep_A">拒绝，保留当前图生模型</button>
    </div>`;
  box.querySelectorAll('button').forEach((b) =>
    b.addEventListener('click', () => sendDecision(b.dataset.a)));
}

async function sendDecision(action) {
  document.getElementById('decision').hidden = true;
  await fetch(`/api/v1/generation/${currentTaskId}/decision`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action }),
  });
}

function renderDownloads(s) {
  const box = document.getElementById('downloads'); box.hidden = false;
  const base = `/api/v1/generation/${currentTaskId}/mesh`;
  box.innerHTML =
    `<a href="${base}?format=glb" download>下载 GLB</a>` +
    `<a href="${base}?format=obj" download>下载 OBJ</a>`;

  const ds = document.getElementById('dataset'); ds.hidden = false;
  ds.innerHTML =
    `<button id="dsBtn" class="ds-btn">✦ 生成后续任务：50 视角数据集（Color/Depth/Normal/Mask）</button>` +
    `<div id="dsStatus" class="ds-status"></div>`;
  document.getElementById('dsBtn').addEventListener('click', startDataset);
}

async function startDataset() {
  const btn = document.getElementById('dsBtn');
  const status = document.getElementById('dsStatus');
  btn.disabled = true; status.className = 'ds-status'; status.textContent = '提交中…';
  try {
    const r = await fetch(`/api/v1/generation/${currentTaskId}/dataset`, { method: 'POST' });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    pollDataset((await r.json()).dataset_id);
  } catch (e) {
    status.textContent = '失败：' + e.message; btn.disabled = false;
  }
}

async function pollDataset(dsId) {
  const status = document.getElementById('dsStatus');
  const btn = document.getElementById('dsBtn');
  const tick = async () => {
    try {
      const s = await (await fetch(`/api/v1/dataset/${dsId}`)).json();
      if (s.status === 'running' || s.status === 'queued') {
        status.textContent = `渲染中… ${(s.progress * 100).toFixed(0)}%（50 视角 × 4 模态）`;
        setTimeout(tick, 2000);
      } else if (s.status === 'done') {
        status.innerHTML = `✓ 完成 · <a href="${s.download}" download>下载数据集 (zip)</a>`;
        btn.disabled = false;
      } else {
        status.textContent = '失败：' + (s.error || '');
        btn.disabled = false;
      }
    } catch (e) {
      status.textContent = '状态查询失败：' + e.message; btn.disabled = false;
    }
  };
  tick();
}

function resetOutputs() {
  ['report', 'decision', 'downloads', 'dataset', 'viewerWrap'].forEach((id) =>
    document.getElementById(id).hidden = true);
}

// --- Three.js 预览 -----------------------------------------------------------
let renderer, scene, camera, controls, currentModel;
function initViewer() {
  const wrap = document.getElementById('viewerWrap'); wrap.hidden = false;
  const canvas = document.getElementById('viewer');
  if (renderer) return;
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  camera.position.set(1.6, 1.4, 2.2);
  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x28323f, 1.3));
  const key = new THREE.DirectionalLight(0xffffff, 2.0); key.position.set(3, 5, 4); scene.add(key);
  const fill = new THREE.DirectionalLight(0x9fc3ff, 0.8); fill.position.set(-4, 2, -3); scene.add(fill);
  const resize = () => {
    const s = canvas.clientWidth;
    renderer.setSize(s, s, false); camera.aspect = 1; camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(canvas); resize();
  const tick = () => { controls.update(); renderer.render(scene, camera); requestAnimationFrame(tick); };
  tick();
}

function loadMesh(taskId) {
  initViewer();
  new GLTFLoader().load(`/api/v1/generation/${taskId}/mesh`, (gltf) => {
    if (currentModel) scene.remove(currentModel);
    const model = gltf.scene;
    const mat = new THREE.MeshStandardMaterial({
      color: 0x9fc3d6, roughness: 0.4, metalness: 0.1, side: THREE.DoubleSide,
    });
    model.traverse((o) => {
      if (o.isMesh) {
        o.geometry.computeVertexNormals(); // 修正回转体法线朝向
        o.material = mat;
      }
    });
    // 居中并归一化
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const scale = 1.6 / Math.max(size.x, size.y, size.z);
    model.scale.setScalar(scale);
    model.position.sub(center.multiplyScalar(scale));
    scene.add(model); currentModel = model;
  });
}

updateCount();
