// ─── Game Configuration ───────────────────────────────────────────────────────

const SNAKES = {
  99: 21,
  87: 24,
  74: 53,
  62: 19,
  54: 34,
  46: 5,
  43: 18,
  30: 6,
};

const LADDERS = {
  3: 22,
  8: 26,
  20: 41,
  28: 77,
  40: 59,
  51: 67,
  63: 81,
  71: 91,
};

const PLAYERS = [
  { name: 'Player 1', color: '#e74c3c', emoji: '🔴' },
  { name: 'Player 2', color: '#3498db', emoji: '🔵' },
];

const DICE_FACES = ['', '⚀', '⚁', '⚂', '⚃', '⚄', '⚅'];

// ─── State ────────────────────────────────────────────────────────────────────

let positions = [0, 0];
let currentPlayer = 0;
let isAnimating = false;
let gameOver = false;

// ─── DOM refs ─────────────────────────────────────────────────────────────────

const canvas = document.getElementById('board');
const ctx = canvas.getContext('2d');
const rollBtn = document.getElementById('roll-btn');
const diceEl = document.getElementById('dice');
const diceDisplay = document.getElementById('dice-display');
const turnInfo = document.getElementById('turn-info');
const moveInfo = document.getElementById('move-info');
const playerListEl = document.getElementById('player-list');
const winnerModal = document.getElementById('winner-modal');
const winnerText = document.getElementById('winner-text');
const restartBtn = document.getElementById('restart-btn');

// ─── Board Drawing ────────────────────────────────────────────────────────────

const COLS = 10;
const ROWS = 10;
const CELL = canvas.width / COLS;

/** Convert square number (1–100) to canvas {x, y} center */
function squareToXY(sq) {
  const idx = sq - 1;
  const row = Math.floor(idx / COLS); // 0 = bottom, 9 = top
  const col = idx % COLS;
  // Even rows go left→right, odd rows go right→left (snake numbering)
  const boardRow = ROWS - 1 - row; // canvas row (0 = top)
  const boardCol = row % 2 === 0 ? col : COLS - 1 - col;
  return {
    x: boardCol * CELL + CELL / 2,
    y: boardRow * CELL + CELL / 2,
  };
}

function drawBoard() {
  for (let sq = 1; sq <= 100; sq++) {
    const { x, y } = squareToXY(sq);
    const row = Math.floor((sq - 1) / COLS);
    const col = (sq - 1) % COLS;
    const boardRow = ROWS - 1 - row;
    const boardCol = row % 2 === 0 ? col : COLS - 1 - col;

    // Cell background
    const isLight = (boardRow + boardCol) % 2 === 0;
    ctx.fillStyle = isLight ? '#1e3a5f' : '#16213e';
    ctx.fillRect(boardCol * CELL, boardRow * CELL, CELL, CELL);

    // Square number
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.font = `bold ${CELL * 0.22}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(sq, boardCol * CELL + CELL / 2, boardRow * CELL + CELL * 0.25);
  }

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= COLS; i++) {
    ctx.beginPath();
    ctx.moveTo(i * CELL, 0);
    ctx.lineTo(i * CELL, canvas.height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i * CELL);
    ctx.lineTo(canvas.width, i * CELL);
    ctx.stroke();
  }
}

function drawSnakesAndLadders() {
  for (const [bottom, top] of Object.entries(LADDERS)) {
    drawLadder(Number(bottom), Number(top));
  }
  for (const [head, tail] of Object.entries(SNAKES)) {
    drawSnake(Number(head), Number(tail));
  }
}

function drawLadder(bottomSq, topSq) {
  const bottom = squareToXY(bottomSq);
  const top = squareToXY(topSq);

  const dx = top.x - bottom.x;
  const dy = top.y - bottom.y;
  const len = Math.sqrt(dx * dx + dy * dy);

  const halfW = 7;
  const px = -(dy / len) * halfW;
  const py = (dx / len) * halfW;

  ctx.save();

  // Shadow / depth
  ctx.strokeStyle = 'rgba(0,0,0,0.5)';
  ctx.lineWidth = 5;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(bottom.x + px + 2, bottom.y + py + 2);
  ctx.lineTo(top.x + px + 2, top.y + py + 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(bottom.x - px + 2, bottom.y - py + 2);
  ctx.lineTo(top.x - px + 2, top.y - py + 2);
  ctx.stroke();

  // Rails
  ctx.strokeStyle = '#c8862a';
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(bottom.x + px, bottom.y + py);
  ctx.lineTo(top.x + px, top.y + py);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(bottom.x - px, bottom.y - py);
  ctx.lineTo(top.x - px, top.y - py);
  ctx.stroke();

  // Rail highlight
  ctx.strokeStyle = '#f0b84a';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(bottom.x + px * 0.5, bottom.y + py * 0.5);
  ctx.lineTo(top.x + px * 0.5, top.y + py * 0.5);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(bottom.x - px * 0.5, bottom.y - py * 0.5);
  ctx.lineTo(top.x - px * 0.5, top.y - py * 0.5);
  ctx.stroke();

  // Rungs
  const numRungs = Math.max(3, Math.round(len / 26));
  for (let i = 1; i < numRungs; i++) {
    const t = i / numRungs;
    const r1x = (bottom.x + px) + ((top.x + px) - (bottom.x + px)) * t;
    const r1y = (bottom.y + py) + ((top.y + py) - (bottom.y + py)) * t;
    const r2x = (bottom.x - px) + ((top.x - px) - (bottom.x - px)) * t;
    const r2y = (bottom.y - py) + ((top.y - py) - (bottom.y - py)) * t;

    ctx.strokeStyle = 'rgba(0,0,0,0.4)';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(r1x + 1, r1y + 1);
    ctx.lineTo(r2x + 1, r2y + 1);
    ctx.stroke();

    ctx.strokeStyle = '#c8862a';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(r1x, r1y);
    ctx.lineTo(r2x, r2y);
    ctx.stroke();

    ctx.strokeStyle = '#f0b84a';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(r1x, r1y);
    ctx.lineTo(r2x, r2y);
    ctx.stroke();
  }

  // End caps
  for (const pt of [bottom, top]) {
    ctx.fillStyle = '#f0b84a';
    ctx.beginPath();
    ctx.arc(pt.x + px, pt.y + py, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(pt.x - px, pt.y - py, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

function drawSnake(headSq, tailSq) {
  const head = squareToXY(headSq);
  const tail = squareToXY(tailSq);

  const dx = tail.x - head.x;
  const dy = tail.y - head.y;
  const len = Math.sqrt(dx * dx + dy * dy);
  const ux = dx / len;
  const uy = dy / len;
  const px = -uy;   // perpendicular unit vector
  const py = ux;

  // Build wavy control points
  const SEGS = 6;
  const amplitude = 16;
  const pts = [];
  for (let i = 0; i <= SEGS; i++) {
    const t = i / SEGS;
    const base = { x: head.x + dx * t, y: head.y + dy * t };
    const side = i === 0 || i === SEGS ? 0
      : amplitude * (i % 2 === 1 ? 1 : -1);
    pts.push({ x: base.x + px * side, y: base.y + py * side });
  }

  // Helper to stroke the wavy path
  function strokeWavy() {
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) {
      const prev = pts[i - 1];
      const curr = pts[i];
      ctx.quadraticCurveTo(prev.x, prev.y, (prev.x + curr.x) / 2, (prev.y + curr.y) / 2);
    }
    ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
    ctx.stroke();
  }

  ctx.save();

  // Outline
  ctx.strokeStyle = 'rgba(0,0,0,0.6)';
  ctx.lineWidth = 12;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  strokeWavy();

  // Body fill
  ctx.strokeStyle = '#27ae60';
  ctx.lineWidth = 8;
  strokeWavy();

  // Belly stripe
  ctx.strokeStyle = 'rgba(255,255,200,0.25)';
  ctx.lineWidth = 3;
  strokeWavy();

  // Scale dots along body
  ctx.fillStyle = 'rgba(0,0,0,0.25)';
  for (let i = 1; i < pts.length - 1; i++) {
    const t1 = (i - 0.5) / pts.length;
    const t2 = i / pts.length;
    const sx = head.x + dx * t2;
    const sy = head.y + dy * t2;
    ctx.beginPath();
    ctx.arc(sx, sy, 2, 0, Math.PI * 2);
    ctx.fill();
  }

  // Pointed tail
  const tailAngle = Math.atan2(uy, ux);
  ctx.fillStyle = '#1e8449';
  ctx.beginPath();
  ctx.save();
  ctx.translate(tail.x, tail.y);
  ctx.rotate(tailAngle);
  ctx.moveTo(6, 0);
  ctx.lineTo(-4, 4);
  ctx.lineTo(-4, -4);
  ctx.closePath();
  ctx.fill();
  ctx.restore();

  // Head
  const headAngle = Math.atan2(-uy, -ux);
  ctx.save();
  ctx.translate(head.x, head.y);
  ctx.rotate(headAngle);

  // Head body
  ctx.fillStyle = '#1e8449';
  ctx.strokeStyle = 'rgba(0,0,0,0.5)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.ellipse(0, 0, 13, 9, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Eyes
  for (const ey of [-5, 5]) {
    ctx.fillStyle = '#ffeaa7';
    ctx.beginPath();
    ctx.arc(-2, ey, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#2d3436';
    ctx.beginPath();
    ctx.arc(-1.5, ey, 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.arc(-1, ey - 0.5, 0.8, 0, Math.PI * 2);
    ctx.fill();
  }

  // Forked tongue
  ctx.strokeStyle = '#e74c3c';
  ctx.lineWidth = 1.5;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(13, 0);
  ctx.lineTo(20, 0);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(20, 0);
  ctx.lineTo(25, -3);
  ctx.moveTo(20, 0);
  ctx.lineTo(25, 3);
  ctx.stroke();

  ctx.restore();
  ctx.restore();
}

function drawTokens() {
  positions.forEach((pos, idx) => {
    if (pos === 0) return;
    const { x, y } = squareToXY(pos);
    const offset = idx === 0 ? -10 : 10;

    ctx.save();
    ctx.shadowColor = PLAYERS[idx].color;
    ctx.shadowBlur = 10;
    ctx.fillStyle = PLAYERS[idx].color;
    ctx.beginPath();
    ctx.arc(x + offset, y + 8, CELL * 0.18, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Player number inside token
    ctx.fillStyle = 'white';
    ctx.font = `bold ${CELL * 0.16}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(idx + 1, x + offset, y + 8);
    ctx.restore();
  });
}

function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawBoard();
  drawSnakesAndLadders();
  drawTokens();
}

// ─── Player UI ────────────────────────────────────────────────────────────────

function renderPlayerList() {
  playerListEl.innerHTML = '';
  PLAYERS.forEach((p, i) => {
    const card = document.createElement('div');
    card.className = 'player-card' + (i === currentPlayer && !gameOver ? ' active' : '');
    card.innerHTML = `
      <div class="player-token" style="background:${p.color}"></div>
      <span>${p.name}</span>
      <span class="player-pos">Sq: ${positions[i] || '—'}</span>
    `;
    playerListEl.appendChild(card);
  });

  turnInfo.textContent = gameOver
    ? ''
    : `${PLAYERS[currentPlayer].emoji} ${PLAYERS[currentPlayer].name}'s turn`;
}

// ─── Dice ─────────────────────────────────────────────────────────────────────

function rollDice() {
  return Math.floor(Math.random() * 6) + 1;
}

async function animateDice(finalValue) {
  diceEl.classList.add('rolling');
  for (let i = 0; i < 8; i++) {
    diceDisplay.textContent = DICE_FACES[Math.floor(Math.random() * 6) + 1];
    await sleep(50);
  }
  diceDisplay.textContent = DICE_FACES[finalValue];
  diceEl.classList.remove('rolling');
}

// ─── Move animation ───────────────────────────────────────────────────────────

async function animateMove(playerIdx, fromPos, toPos) {
  const step = fromPos < toPos ? 1 : -1;
  for (let pos = fromPos + step; pos !== toPos + step; pos += step) {
    positions[playerIdx] = pos;
    render();
    await sleep(120);
  }
}

// ─── Turn Logic ───────────────────────────────────────────────────────────────

async function takeTurn() {
  if (isAnimating || gameOver) return;
  isAnimating = true;
  rollBtn.disabled = true;
  moveInfo.textContent = '';

  const value = rollDice();
  await animateDice(value);

  const from = positions[currentPlayer];
  let dest = from + value;

  if (dest > 100) {
    moveInfo.textContent = `Rolled ${value} — can't move, need ${100 - from} or less!`;
    isAnimating = false;
    rollBtn.disabled = false;
    return;
  }

  // Move token step by step
  await animateMove(currentPlayer, from, dest);

  // Check snake or ladder
  if (SNAKES[dest]) {
    const snakeTail = SNAKES[dest];
    moveInfo.textContent = `Rolled ${value} — Snake! ${dest} → ${snakeTail}`;
    await sleep(400);
    await animateMove(currentPlayer, dest, snakeTail);
    dest = snakeTail;
  } else if (LADDERS[dest]) {
    const ladderTop = LADDERS[dest];
    moveInfo.textContent = `Rolled ${value} — Ladder! ${dest} → ${ladderTop}`;
    await sleep(400);
    await animateMove(currentPlayer, dest, ladderTop);
    dest = ladderTop;
  } else {
    moveInfo.textContent = `Rolled ${value} — moved to ${dest}`;
  }

  positions[currentPlayer] = dest;
  render();
  renderPlayerList();

  // Check win
  if (dest === 100) {
    gameOver = true;
    winnerText.textContent = `${PLAYERS[currentPlayer].emoji} ${PLAYERS[currentPlayer].name} Wins!`;
    winnerModal.classList.remove('hidden');
    renderPlayerList();
    isAnimating = false;
    return;
  }

  // Next player's turn
  currentPlayer = (currentPlayer + 1) % PLAYERS.length;
  renderPlayerList();

  isAnimating = false;
  rollBtn.disabled = false;
}

// ─── Restart ──────────────────────────────────────────────────────────────────

function restart() {
  positions = [0, 0];
  currentPlayer = 0;
  isAnimating = false;
  gameOver = false;
  diceDisplay.textContent = '?';
  moveInfo.textContent = '';
  rollBtn.disabled = false;
  winnerModal.classList.add('hidden');
  render();
  renderPlayerList();
}

// ─── Utils ────────────────────────────────────────────────────────────────────

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ─── Init ─────────────────────────────────────────────────────────────────────

rollBtn.addEventListener('click', takeTurn);
restartBtn.addEventListener('click', restart);

render();
renderPlayerList();
