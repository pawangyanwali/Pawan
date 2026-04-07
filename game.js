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
  // Snakes
  for (const [head, tail] of Object.entries(SNAKES)) {
    const h = squareToXY(Number(head));
    const t = squareToXY(Number(tail));
    drawCurvedLine(h, t, '#e74c3c', 5, true);
  }

  // Ladders
  for (const [bottom, top] of Object.entries(LADDERS)) {
    const b = squareToXY(Number(bottom));
    const tp = squareToXY(Number(top));
    drawCurvedLine(b, tp, '#2ecc71', 4, false);
  }
}

function drawCurvedLine(from, to, color, width, isSnake) {
  const mx = (from.x + to.x) / 2;
  const my = (from.y + to.y) / 2;
  const offset = isSnake ? 30 : -20;
  const cpx = mx + offset;
  const cpy = my + offset;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = 'round';
  ctx.globalAlpha = 0.85;

  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.quadraticCurveTo(cpx, cpy, to.x, to.y);
  ctx.stroke();

  // Arrowhead at destination
  const angle = Math.atan2(to.y - cpy, to.x - cpx);
  ctx.fillStyle = color;
  ctx.globalAlpha = 1;
  ctx.beginPath();
  ctx.translate(to.x, to.y);
  ctx.rotate(angle);
  ctx.moveTo(0, 0);
  ctx.lineTo(-12, 5);
  ctx.lineTo(-12, -5);
  ctx.closePath();
  ctx.fill();

  // Dot at start
  ctx.restore();
  ctx.save();
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.8;
  ctx.beginPath();
  ctx.arc(from.x, from.y, 7, 0, Math.PI * 2);
  ctx.fill();
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
