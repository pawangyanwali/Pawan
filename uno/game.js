// ─── Constants ────────────────────────────────────────────────────────────────

const COLORS = ['red', 'blue', 'green', 'yellow'];
const NUMBERS = ['0','1','2','3','4','5','6','7','8','9'];
const ACTIONS = ['skip', 'reverse', 'draw2'];
const CARD_LABELS = { skip: '⊘', reverse: '↺', draw2: '+2', wild: '⬛', wild4: '+4' };
const AI_DELAY = 1100;   // ms between AI actions
const DRAW_DELAY = 600;  // ms extra pause after AI draws

// ─── State ────────────────────────────────────────────────────────────────────

let G = {};   // global game state

function freshState() {
  return {
    deck: [],
    discard: [],
    players: [
      { name: 'You',   isHuman: true,  hand: [] },
      { name: 'Aarya', isHuman: false, hand: [] },
      { name: 'Aayush', isHuman: false, hand: [] },
      { name: 'Aarav', isHuman: false, hand: [] },
      { name: 'Neena', isHuman: false, hand: [] },
    ],
    currentPlayer: 0,
    direction: 1,       // 1 = 0→1→2→3→4→0, -1 = 0→4→3→2→1→0
    currentColor: null,
    currentValue: null,
    phase: 'playing',   // 'playing' | 'color-pick' | 'game-over'
    humanHasDrawn: false,
    unoPending: false,  // human went to 1 card without clicking UNO
  };
}

// ─── Deck helpers ─────────────────────────────────────────────────────────────

function buildDeck() {
  const cards = [];
  for (const color of COLORS) {
    cards.push({ color, value: '0' });
    for (const v of [...NUMBERS.slice(1), ...ACTIONS]) {
      cards.push({ color, value: v });
      cards.push({ color, value: v });
    }
  }
  for (let i = 0; i < 4; i++) {
    cards.push({ color: 'wild', value: 'wild' });
    cards.push({ color: 'wild', value: 'wild4' });
  }
  return shuffle(cards);
}

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function popCard() {
  if (G.deck.length === 0) reshuffleDiscard();
  return G.deck.pop();
}

function reshuffleDiscard() {
  if (G.discard.length <= 1) return;
  const top = G.discard.pop();
  G.deck = shuffle(G.discard);
  G.discard = [top];
}

function dealN(playerIdx, n) {
  for (let i = 0; i < n; i++) {
    const c = popCard();
    if (c) G.players[playerIdx].hand.push(c);
  }
}

// ─── Init ─────────────────────────────────────────────────────────────────────

function initGame() {
  G = freshState();
  G.deck = buildDeck();

  // Deal 7 each
  for (let p = 0; p < 5; p++) dealN(p, 7);

  // First discard must be a numbered card
  let first;
  const held = [];
  do {
    first = popCard();
    if (first.color === 'wild') held.push(first);
  } while (first.color === 'wild');

  // Return any held wilds to middle of deck
  const mid = Math.floor(G.deck.length / 2);
  G.deck.splice(mid, 0, ...held);

  G.discard.push(first);
  G.currentColor = first.color;
  G.currentValue = first.value;

  // Apply first-card effect
  if (first.value === 'skip') {
    G.currentPlayer = 1;
  } else if (first.value === 'reverse') {
    G.direction = -1;
    G.currentPlayer = 4;
  } else if (first.value === 'draw2') {
    dealN(0, 2);
    G.currentPlayer = 1;
  }

  // Close modals
  el('over-overlay').classList.add('hidden');
  el('color-overlay').classList.add('hidden');
  el('uno-shout').classList.add('hidden');
  el('draw-btn').textContent = 'Draw Card';
  el('draw-btn').onclick = onHumanDraw;

  render();
  startTurn();
}

// ─── Turn flow ────────────────────────────────────────────────────────────────

function startTurn() {
  if (G.phase === 'game-over') return;
  G.humanHasDrawn = false;
  el('draw-btn').textContent = 'Draw Card';
  el('draw-btn').onclick = onHumanDraw;
  setMsg('');
  render();

  if (G.players[G.currentPlayer].isHuman) {
    enableHumanActions();
  } else {
    disableHumanActions();
    setTimeout(aiTurn, AI_DELAY);
  }
}

function nextPlayer() {
  G.currentPlayer = advanceFrom(G.currentPlayer);
  startTurn();
}

function skipNextPlayer() {
  // Skip the immediate next player, go to the one after
  const skipped = advanceFrom(G.currentPlayer);
  G.currentPlayer = advanceFrom(skipped);
  startTurn();
}

function advanceFrom(idx) {
  return (idx + G.direction + 5) % 5;
}

// ─── Card validity ────────────────────────────────────────────────────────────

function canPlay(card) {
  if (card.color === 'wild') return true;
  return card.color === G.currentColor || card.value === G.currentValue;
}

// ─── Playing a card ───────────────────────────────────────────────────────────

function executePlay(playerIdx, card, chosenColor) {
  const hand = G.players[playerIdx].hand;
  const idx = hand.indexOf(card);
  hand.splice(idx, 1);

  G.discard.push(card);
  G.currentValue = card.value;
  G.currentColor = card.color === 'wild' ? chosenColor : card.color;

  render();

  if (hand.length === 0) {
    endGame(playerIdx);
    return;
  }

  applyCardEffect(card, playerIdx);
}

function applyCardEffect(card, playerIdx) {
  const next = advanceFrom(playerIdx);

  switch (card.value) {
    case 'skip':
      setMsg(`${G.players[next].name} was skipped!`);
      G.currentPlayer = playerIdx;
      skipNextPlayer();
      break;

    case 'reverse':
      G.direction *= -1;
      updateDirArrow();
      setMsg('Direction reversed!');
      G.currentPlayer = playerIdx;
      nextPlayer();
      break;

    case 'draw2':
      dealN(next, 2);
      setMsg(`${G.players[next].name} draws 2 and is skipped!`);
      G.currentPlayer = playerIdx;
      skipNextPlayer();
      break;

    case 'wild4':
      dealN(next, 4);
      setMsg(`${G.players[next].name} draws 4 and is skipped!`);
      G.currentPlayer = playerIdx;
      skipNextPlayer();
      break;

    default:
      G.currentPlayer = playerIdx;
      nextPlayer();
  }
}

// ─── Human actions ────────────────────────────────────────────────────────────

function enableHumanActions() {
  el('draw-btn').disabled = false;
}

function disableHumanActions() {
  el('draw-btn').disabled = true;
  el('uno-shout').classList.add('hidden');
}

async function onHumanCardClick(card) {
  if (G.currentPlayer !== 0 || G.phase !== 'playing') return;
  if (!canPlay(card)) return;

  let chosenColor = null;
  if (card.color === 'wild') {
    chosenColor = await pickColor();
  }

  // UNO check: if going from 2 cards → 1
  if (G.players[0].hand.length === 2) {
    G.unoPending = true;
    el('uno-shout').classList.remove('hidden');
  }

  disableHumanActions();
  executePlay(0, card, chosenColor);
}

function onHumanDraw() {
  if (G.currentPlayer !== 0 || G.phase !== 'playing' || G.humanHasDrawn) return;
  G.humanHasDrawn = true;

  const card = popCard();
  if (!card) return;
  G.players[0].hand.push(card);
  render();

  if (canPlay(card)) {
    setMsg('You drew a playable card. Play it or pass.');
    el('draw-btn').textContent = 'Pass Turn';
    el('draw-btn').onclick = () => {
      el('draw-btn').textContent = 'Draw Card';
      el('draw-btn').onclick = onHumanDraw;
      G.currentPlayer = 0;
      nextPlayer();
    };
  } else {
    setMsg('No playable card drawn. Passing turn…');
    el('draw-btn').disabled = true;
    setTimeout(() => {
      el('draw-btn').disabled = false;
      G.currentPlayer = 0;
      nextPlayer();
    }, 700);
  }
}

// ─── Color picker ─────────────────────────────────────────────────────────────

function pickColor() {
  return new Promise(resolve => {
    G.phase = 'color-pick';
    el('color-overlay').classList.remove('hidden');
    const btns = document.querySelectorAll('.color-btn');
    btns.forEach(btn => {
      btn.onclick = () => {
        el('color-overlay').classList.add('hidden');
        G.phase = 'playing';
        resolve(btn.dataset.color);
      };
    });
  });
}

// ─── AI logic ─────────────────────────────────────────────────────────────────

function aiTurn() {
  if (G.phase !== 'playing') return;
  const pIdx = G.currentPlayer;
  const hand = G.players[pIdx].hand;
  const valid = hand.filter(canPlay);

  if (valid.length > 0) {
    const card = chooseAiCard(valid, hand);
    let chosenColor = null;
    if (card.color === 'wild') chosenColor = dominantColor(hand);

    // Announce UNO
    if (hand.length === 2) flashBadge(pIdx);

    setMsg(`${G.players[pIdx].name} plays ${cardLabel(card)}${chosenColor ? ' → ' + chosenColor : ''}`);
    executePlay(pIdx, card, chosenColor);
  } else {
    // Draw one card
    setMsg(`${G.players[pIdx].name} draws a card…`);
    const drawn = popCard();
    if (drawn) {
      hand.push(drawn);
      render();
    }

    setTimeout(() => {
      if (drawn && canPlay(drawn)) {
        let chosenColor = drawn.color === 'wild' ? dominantColor(hand) : null;
        setMsg(`${G.players[pIdx].name} plays the drawn ${cardLabel(drawn)}`);
        executePlay(pIdx, drawn, chosenColor);
      } else {
        setMsg(`${G.players[pIdx].name} passes.`);
        G.currentPlayer = pIdx;
        nextPlayer();
      }
    }, DRAW_DELAY);
  }
}

function chooseAiCard(valid, fullHand) {
  // Priority: wild4 > draw2 > skip > reverse > numbers, but save wilds if non-wilds available
  const nonWild = valid.filter(c => c.color !== 'wild');
  const pool = nonWild.length > 0 ? nonWild : valid;

  // Among pool prefer action cards
  const actions = pool.filter(c => ACTIONS.includes(c.value));
  if (actions.length > 0) return actions[Math.floor(Math.random() * actions.length)];

  return pool[Math.floor(Math.random() * pool.length)];
}

function dominantColor(hand) {
  const counts = { red: 0, blue: 0, green: 0, yellow: 0 };
  hand.forEach(c => { if (c.color !== 'wild') counts[c.color]++; });
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  return sorted[0][1] > 0 ? sorted[0][0] : COLORS[Math.floor(Math.random() * 4)];
}

// ─── Win ──────────────────────────────────────────────────────────────────────

function endGame(winnerIdx) {
  G.phase = 'game-over';
  const p = G.players[winnerIdx];
  el('over-icon').textContent = winnerIdx === 0 ? '🎉' : '😔';
  el('over-text').textContent = winnerIdx === 0 ? 'You Win!' : `${p.name} Wins!`;
  el('over-overlay').classList.remove('hidden');
}

// ─── Rendering ────────────────────────────────────────────────────────────────

function render() {
  for (let i = 0; i < 5; i++) {
    renderHand(i);
    const n = G.players[i].hand.length;
    el(`count-${i}`).textContent = `${n} card${n !== 1 ? 's' : ''}`;
    el(`ubadge-${i}`).classList.toggle('hidden', n !== 1);
    el(`zone-${i}`).classList.toggle('active', i === G.currentPlayer);
  }

  // Discard pile top
  const top = G.discard.at(-1);
  el('discard-top').innerHTML = '';
  if (top) el('discard-top').appendChild(makeCardEl(top, false));

  // Deck count
  el('deck-count').textContent = G.deck.length;

  // Color ring
  el('color-ring').className = `color-ring ${G.currentColor || ''}`;

  // Turn message
  const cp = G.players[G.currentPlayer];
  el('turn-msg').textContent = cp.isHuman ? "Your turn!" : `${cp.name} is thinking…`;

  // Direction arrow
  updateDirArrow();
}

function renderHand(playerIdx) {
  const { isHuman, hand } = G.players[playerIdx];
  const container = el(`hand-${playerIdx}`);
  container.innerHTML = '';

  const isActive = playerIdx === G.currentPlayer && G.phase === 'playing';

  hand.forEach(card => {
    const cardEl = makeCardEl(card, !isHuman);
    cardEl.classList.add('dealt');
    if (isHuman && isActive && canPlay(card)) {
      cardEl.classList.add('playable');
      cardEl.addEventListener('click', () => onHumanCardClick(card));
    }
    container.appendChild(cardEl);
  });
}

function makeCardEl(card, faceDown) {
  const div = document.createElement('div');
  if (faceDown) {
    div.className = 'card back';
    return div;
  }
  div.className = `card ${card.color}`;
  const lbl = cardLabel(card);
  div.innerHTML = `
    <span class="card-corner tl">${lbl}</span>
    <div class="card-oval"></div>
    <span class="card-value">${lbl}</span>
    <span class="card-corner br">${lbl}</span>
  `;
  return div;
}

function cardLabel(card) {
  return CARD_LABELS[card.value] || card.value;
}

function updateDirArrow() {
  el('dir-arrow').classList.toggle('ccw', G.direction === -1);
}

function flashBadge(playerIdx) {
  const b = el(`ubadge-${playerIdx}`);
  b.classList.remove('hidden');
  setTimeout(() => b.classList.add('hidden'), 2500);
}

function setMsg(text) {
  el('msg-bar').textContent = text;
}

// ─── Utils ────────────────────────────────────────────────────────────────────

function el(id) { return document.getElementById(id); }

// ─── Events ───────────────────────────────────────────────────────────────────

el('draw-btn').addEventListener('click', onHumanDraw);

el('uno-shout').addEventListener('click', () => {
  G.unoPending = false;
  el('uno-shout').classList.add('hidden');
  flashBadge(0);
});

el('restart-btn').addEventListener('click', initGame);

el('draw-pile-card').addEventListener('click', () => {
  if (G.currentPlayer === 0 && !G.humanHasDrawn) onHumanDraw();
});

// ─── Start ────────────────────────────────────────────────────────────────────

initGame();
