'use strict';

// Everything on the page is rendered from /api/state. Nothing is computed
// here beyond formatting: the numbers come from dashboard.py, which has tests.

const state = {
  data: null,
  tab: 'week',
  round: null,          // the matchweek being looked at on the first tab
  preview: null,        // calls for a round ahead, never written to the log
  previewError: null,
  previewLoading: null,
  seasonIdx: 0,
  weekIdx: 0,           // 0 = newest week in the log
  pollTimer: null,
  noticeTimer: null,
};

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
));
const pct = (v) => (v == null ? '—' : `${v}%`);
const signed = (v) => (v > 0 ? `+${v}` : String(v));
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

async function getJSON(url, options) {
  const r = await fetch(url, options);
  const body = await r.json().catch(() => ({}));
  if (!r.ok && body.error === undefined && body.running === undefined) {
    throw new Error(`${url} answered ${r.status}`);
  }
  return body;
}

// --- loading and refreshing ------------------------------------------

async function load() {
  try {
    state.data = await getJSON('/api/state');
  } catch (err) {
    $('#panel').innerHTML = `<p class="foot">Could not reach the app: ${esc(err.message)}</p>`;
    return;
  }
  state.round = state.data.current_round;
  state.preview = null;
  state.previewError = null;
  state.previewLoading = null;
  render();
  if (state.data.refresh.running) poll();
}

async function refresh() {
  const btn = $('#refresh');
  btn.disabled = true;
  btn.textContent = 'REFRESHING…';
  showNotice(null);
  await getJSON('/api/refresh', { method: 'POST' });
  poll();
}

function poll() {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(async () => {
    const status = await getJSON('/api/refresh');
    if (status.running) { poll(); return; }
    await load();
    showNotice(status.notice);
  }, 2000);
}

function showNotice(notice) {
  const el = $('#notice');
  clearTimeout(state.noticeTimer);
  if (!notice) { el.hidden = true; return; }
  el.className = `notice ${notice.kind}`;
  el.textContent = notice.text;
  el.hidden = false;
  if (notice.kind !== 'error') {
    state.noticeTimer = setTimeout(() => { el.hidden = true; }, 8000);
  }
}

// --- rendering -------------------------------------------------------

function render() {
  renderTop();
  const panel = $('#panel');
  if (state.tab === 'week') panel.innerHTML = renderWeek();
  else if (state.tab === 'record') panel.innerHTML = renderRecord();
  else panel.innerHTML = renderModel();
  bindPanel();
}

function renderTop() {
  const f = state.data.freshness;
  const r = state.data.refresh;
  $('#fresh-dot').className = `dot ${f.stale ? 'stale' : 'fresh'}`;
  $('#fresh-date').textContent = f.label;
  $('#fresh-age').textContent = f.age;
  const btn = $('#refresh');
  btn.textContent = r.running ? 'REFRESHING…' : 'REFRESH';
  btn.disabled = !!r.running;
  document.querySelectorAll('.tab').forEach((t) => {
    t.classList.toggle('active', t.dataset.tab === state.tab);
  });
}

const stat = (label, big, sub) => `
  <div class="stat">
    <div class="eyebrow">${label}</div>
    <div class="big">${big}</div>
    <div class="sub">${sub}</div>
  </div>`;

// Backtest figures the first tab quotes: how often a backed call landed, and
// how often the weaker "home won't win" claim held.
function backtestRates() {
  const bt = state.data.record.backtest;
  if (!bt) return { backed: null, weak: null };
  let n = 0; let right = 0; let weak = null;
  for (const b of bt.bands) {
    if (b.band === 'neither backed') weak = b.hit_rate;
    else { n += b.predictions; right += b.correct; }
  }
  return { backed: n ? Math.round((right / n) * 100) : null, weak };
}

// --- this week -------------------------------------------------------

function roundNav(wk) {
  const rounds = state.data.rounds || [];
  if (!rounds.length) {
    const why = state.data.rounds_error
      ? `look-ahead unavailable: ${esc(state.data.rounds_error)}` : '';
    return `
      <div class="log-controls week">
        <div class="week-nav"><div class="mid">
          <div class="lbl">${esc(wk.label)}</div>
          <div class="sub">${esc(wk.dates)}</div>
        </div></div>
        <div class="mono-note">${why}</div>
      </div>`;
  }
  const idx = rounds.findIndex((r) => r.round === state.round);
  const cur = idx >= 0 ? rounds[idx] : null;
  const label = cur ? cur.label : wk.label;
  const sub = cur ? `${cur.dates} · ${cur.remaining} to play` : wk.dates;
  const status = state.previewLoading
    ? `building calls for matchweek ${state.previewLoading}…`
    : wk.preview
      ? `preview · not logged · on ratings current to ${esc(state.data.freshness.label)}`
      : wk.n ? `${wk.n_logged} of ${wk.n} on the record` : '';
  return `
    <div class="log-controls week">
      <div class="week-nav">
        <button class="arrow" type="button" id="round-prev" ${idx <= 0 ? 'disabled' : ''}>←</button>
        <div class="mid">
          <div class="lbl">${esc(label)}</div>
          <div class="sub">${esc(sub)}</div>
        </div>
        <button class="arrow" type="button" id="round-next" ${idx >= rounds.length - 1 ? 'disabled' : ''}>→</button>
      </div>
      <div class="mono-note ${wk.preview ? 'preview' : ''}">${status}</div>
    </div>`;
}

function card(f, wk) {
  const th = wk.threshold;
  const kind = (f.verdict === 'HOME will win' || f.verdict === 'AWAY will win') ? 'call'
    : f.verdict === 'HOME will not win' ? 'weak' : 'both';
  const on = (backed) => ((kind === 'call' && backed) ? ' on' : '');
  const side = (tag, name, p, backed) => `
    <div class="side">
      <div class="club"><span class="tag">${tag}</span><span class="name" title="${esc(name)}">${esc(name)}</span></div>
      <div class="bar">
        <div class="fill${on(backed)}" style="width:${p}%"></div>
        <div class="line" style="left:${th}%"></div>
      </div>
      <div class="pct${on(backed)}">${p}%</div>
    </div>`;

  let verdict;
  if (kind === 'call') {
    const club = f.verdict === 'HOME will win' ? f.home : f.away;
    const landed = f.band_rate == null
      ? 'no backtest figure for this band'
      : `calls this strong landed ${f.band_rate}% of the time in the backtest`;
    verdict = `
      <div class="call">${esc(club)} to win</div>
      <div class="mono">stated ${f.confidence}%</div>
      <div class="faint">${landed}</div>`;
  } else if (kind === 'weak') {
    const held = f.band_rate == null ? ''
      : `claims like this held ${f.band_rate}% of the time in the backtest`;
    verdict = `
      <div class="call weak">${esc(f.home)} won't win</div>
      <div class="mono">neither side over ${th}% · claims ${100 - f.p_home}%</div>
      <div class="faint">${held}</div>`;
  } else {
    verdict = `
      <div class="call weak">Contradiction</div>
      <div class="mono">both sides over ${th}% · claims nothing</div>`;
  }

  const status = f.logged ? 'ON THE RECORD'
    : wk.preview ? 'PREVIEW'
      : f.kicked_off ? 'KICKED OFF BEFORE THE RUN' : 'NOT LOGGED';

  return `
    <div class="card ${kind}">
      <div class="when">
        <div>${esc(f.day)}</div>
        <div class="t">${esc(f.time)}</div>
        <div class="status ${f.logged ? 'logged' : ''}">${status}</div>
      </div>
      <div class="sides">
        ${side('HOME', f.home, f.p_home, f.home_backed)}
        ${side('AWAY', f.away, f.p_away, f.away_backed)}
      </div>
      <div class="verdict"><div class="eyebrow">VERDICT</div>${verdict}</div>
    </div>`;
}

function renderWeek() {
  const wk = state.preview || state.data.week;
  const rates = backtestRates();

  if (state.previewError) {
    return `${roundNav(wk)}<p class="foot">${esc(state.previewError)}</p>`;
  }
  if (state.previewLoading) {
    return `${roundNav(wk)}<p class="foot">Building calls for matchweek ${state.previewLoading}…</p>`;
  }
  if (!wk.n) {
    return `${roundNav(wk)}
      <p class="foot">No calls yet. Press REFRESH to pull the results, settle last week and call the next one.</p>`;
  }

  const both = wk.contradictions.length ? `
    <div class="h-row"><h2>Contradiction</h2><span class="count">${wk.n_contradictions} of ${wk.n}</span></div>
    <p class="lede">Both sides cleared ${wk.threshold}%. The model has backed each club to win the same match, so it claims nothing. This has never happened in the backtest.</p>
    <div class="cards">${wk.contradictions.map((f) => card(f, wk)).join('')}</div>` : '';

  return `
    ${roundNav(wk)}
    <div class="stats">
      ${stat('FIXTURES', wk.n, esc(wk.dates))}
      ${stat('COMMITTED CALLS', wk.n_calls, `${wk.n_home} home · ${wk.n_away} away`)}
      ${stat('NO CONFIDENT CALL', wk.n_weak, 'Genuinely close matches')}
      ${stat('COMMIT THRESHOLD', `${wk.threshold}%`, 'Applied to both sides')}
    </div>

    <div class="h-row"><h2>Committed calls</h2><span class="count">${wk.n_calls} of ${wk.n}</span></div>
    <p class="lede">One side cleared the threshold and the other did not. ${rates.backed == null ? '' : `In the backtest these landed about ${rates.backed}% of the time.`}</p>
    <div class="cards">${wk.committed.length ? wk.committed.map((f) => card(f, wk)).join('') : '<p class="mono-note">none this week</p>'}</div>

    <div class="h-row"><h2>No confident call</h2><span class="count">${wk.n_weak} of ${wk.n}</span></div>
    <p class="lede">Neither side cleared the threshold. The claim on record is the weak one — <em>the home side will not win</em> — and it is scored like any other. ${rates.weak == null ? '' : `In the backtest it held ${rates.weak}% of the time.`} Both numbers are shown; nothing is inferred from them.</p>
    <div class="cards">${wk.weak.length ? wk.weak.map((f) => card(f, wk)).join('') : '<p class="mono-note">none this week</p>'}</div>

    ${both}

    <p class="foot">The model never predicts draws. p_draw is whatever neither side claimed and is not part of any verdict.</p>`;
}

// --- track record ----------------------------------------------------

function bandRow(live, bt) {
  const w = (v) => (v == null ? 0 : Math.max(0, Math.min(100, (v - 50) * 2)));
  const nums = live && live.predictions
    ? `<div>stated ${pct(live.stated)}</div><div class="g">actual ${pct(live.hit_rate)}</div><div class="f">backtest ${pct(bt ? bt.hit_rate : null)}</div>`
    : `<div class="f">no calls yet</div><div class="f">backtest ${pct(bt ? bt.hit_rate : null)}</div>`;
  const stated = live && live.predictions ? live.stated : null;
  const actual = live && live.predictions ? live.hit_rate : null;
  return `
    <div class="band">
      <div class="lbl">${esc((live || bt).band)}</div>
      <div class="bars">
        <div class="track top"></div>
        <div class="stated" style="width:${w(stated)}%"></div>
        <div class="track bottom"></div>
        <div class="actual" style="width:${w(actual)}%"></div>
        ${stated == null ? '' : `<div class="mark" style="left:${w(stated)}%"></div>`}
      </div>
      <div class="nums">${nums}</div>
      <div class="n">${live ? live.predictions : 0}</div>
    </div>`;
}

function logTable(rec) {
  const seasons = rec.seasons;
  if (!seasons.length) {
    return '<p class="foot">Nothing logged yet. Every run of the predictor adds this week\'s calls here.</p>';
  }
  state.seasonIdx = Math.min(state.seasonIdx, seasons.length - 1);
  const season = seasons[state.seasonIdx];
  state.weekIdx = Math.min(state.weekIdx, season.weeks.length - 1);
  const wk = season.weeks[state.weekIdx];
  const summary = [
    `${wk.correct} correct`, `${wk.wrong} wrong`,
    wk.awaiting ? `${wk.awaiting} awaiting` : null,
    wk.unscored ? `${wk.unscored} unscored` : null,
  ].filter(Boolean).join(' · ');

  const rows = wk.rows.map((r) => {
    const weak = r.verdict === 'HOME will not win' || r.verdict === 'NO CALL - both sides backed';
    const scored = r.scored === 'correct' ? 'correct' : r.scored === 'wrong' ? 'wrong'
      : r.scored === 'awaiting' ? 'awaiting' : 'not scored';
    return `
      <div class="tr ${weak ? 'dim' : ''}">
        <div class="c-date">${esc(r.date)}</div>
        <div class="c-fix">${esc(r.fixture)}</div>
        <div class="c-p ${r.home_backed ? 'on' : ''}">${r.p_home}%</div>
        <div class="c-p ${r.away_backed ? 'on' : ''}">${r.p_away}%</div>
        <div class="c-v ${weak ? 'weak' : ''}">${esc(r.short)}</div>
        <div class="c-r">${esc(r.result) || '—'}</div>
        <div class="c-s ${r.scored}">${scored}</div>
      </div>`;
  }).join('');

  return `
    <div class="log-controls">
      <div class="picker">
        <button class="select-btn" type="button" id="season-cycle">
          <span>${esc(season.label)}</span><span class="caret">▾</span>
        </button>
        <div class="mono-note">Premier League · ${plural(season.weeks.length, 'week')} logged</div>
      </div>
      <div class="week-nav">
        <button class="arrow" type="button" id="week-prev" ${state.weekIdx >= season.weeks.length - 1 ? 'disabled' : ''}>←</button>
        <div class="mid">
          <div class="lbl">${esc(wk.label)}</div>
          <div class="sub">${esc(wk.dates)} · ${summary}</div>
        </div>
        <button class="arrow" type="button" id="week-next" ${state.weekIdx <= 0 ? 'disabled' : ''}>→</button>
      </div>
    </div>
    <div class="table"><div class="inner">
      <div class="tr head">
        <div class="c-date">DATE</div>
        <div class="c-fix">FIXTURE</div>
        <div class="c-p">P_HOME</div>
        <div class="c-p">P_AWAY</div>
        <div class="c-v">VERDICT</div>
        <div class="c-r">RESULT</div>
        <div class="c-s">SCORED</div>
      </div>
      ${rows}
    </div></div>`;
}

function renderRecord() {
  const rec = state.data.record;
  const bt = rec.backtest;

  const neither = rec.settled ? `${Math.round((rec.neither_backed / rec.settled) * 100)}%` : '—';
  const calib = rec.calibration_error;
  const calibSub = calib == null ? 'mean pts actual − stated'
    : `mean pts actual − stated (${calib > 0 ? 'under-claiming' : calib < 0 ? 'over-claiming' : 'spot on'})`;
  const contraSub = bt
    ? `both sides backed · ${bt.contradictions === 0 ? 'never' : bt.contradictions} in ${bt.fixtures.toLocaleString()} backtest fixtures`
    : 'both sides backed';

  const bandLabels = (bt ? bt.bands : rec.bands).map((b) => b.band);
  const liveBy = Object.fromEntries(rec.bands.map((b) => [b.band, b]));
  const btBy = Object.fromEntries((bt ? bt.bands : []).map((b) => [b.band, b]));
  const bands = bandLabels.map((label) => bandRow(liveBy[label], btBy[label])).join('');

  const thin = rec.bands.filter((b) => b.predictions && b.predictions < 20).length;
  const foot = rec.settled
    ? `The grey bar is what the model claimed, the green bar what happened. A green bar past the grey one means the model was more right than it said. ${thin ? 'Bands below n = 20 are thin; read them loosely.' : ''}`
    : 'Nothing has settled yet. The backtest figures show what each band should look like once it has.';

  return `
    <div class="stats record">
      ${stat('VERDICT CORRECT', pct(rec.verdict_accuracy), rec.called
    ? `of ${rec.called} scored claims · backtest ${pct(bt ? bt.verdict_accuracy : null)}`
    : 'nothing settled yet')}
      ${stat('NEITHER SIDE BACKED', neither, `${rec.neither_backed} of ${rec.settled} settled · scored as "home won't win"`)}
      ${stat('CALIBRATION ERROR', calib == null ? '—' : signed(calib), calibSub)}
      ${stat('CONTRADICTIONS', rec.contradictions, contraSub)}
    </div>

    <h2>Calibration by confidence band</h2>
    <p class="lede" style="margin-bottom:16px">Does a stated confidence mean what it says? Each band compares what the model claimed against what happened. Bars run from 50% (a coin flip) to 100%.</p>
    <div class="calib">
      <div class="calib-head"><div class="a">BAND</div><div class="b">STATED ▸ ACTUAL</div><div class="c"></div><div class="d">N</div></div>
      <div class="calib-rows">${bands}</div>
      <p class="calib-foot">${foot}</p>
    </div>

    <h2>Every prediction</h2>
    <p class="lede" style="margin-bottom:16px">All fixtures, including the ones where neither side was backed. A contradiction has no outcome to be right or wrong about and is scored as neither.</p>
    ${logTable(rec)}`;
}

// --- model & ratings -------------------------------------------------

function renderModel() {
  const m = state.data.model;
  const g = m.grid;
  const bt = state.data.record.backtest;
  const f = state.data.freshness;
  const sub = g.n ? 'fixtures this week' : 'no calls yet';

  const cell = (kind, name, big, note) => `
    <div class="cell ${kind}">
      <div class="name">${name}</div>
      <div><div class="big">${big}</div><div class="sub">${note}</div></div>
    </div>`;

  const features = m.features.map((ft) => `
    <div class="feature">
      <div class="head"><div class="name">${esc(ft.name)}</div><div class="w">odds ×${ft.odds_x}</div></div>
      <div class="fbar"><div class="fill" style="width:${ft.relative}%"></div></div>
      <div class="note">${esc(ft.note)}</div>
    </div>`).join('');

  const elo = m.elo.map((c) => `
    <div class="tr">
      <div class="e-rank">${c.rank}</div>
      <div class="e-team">${esc(c.team)}</div>
      <div class="e-bar"><div class="fill" style="width:${c.relative}%"></div></div>
      <div class="e-elo">${c.elo}</div>
      <div class="e-d ${c.delta > 0 ? 'up' : c.delta < 0 ? 'down' : ''}">${signed(c.delta)}</div>
    </div>`).join('');

  return `
    <h2>How a verdict is reached</h2>
    <p class="lede" style="margin-bottom:20px;max-width:66ch">The model answers one question twice — <em>does this club win?</em> — once from the home club's side, once from the away club's. Two yes/no answers land the fixture in one of four cells. Counts shown are ${g.n ? `this week's ${g.n} fixtures` : 'empty until the first run'}.</p>

    <div class="model-row">
      <div class="grid-wrap">
        <div class="grid">
          <div></div>
          <div class="col-h">AWAY WON'T WIN</div>
          <div class="col-h">AWAY WILL WIN</div>
          <div class="row-h">HOME WILL WIN</div>
          ${cell('call', 'HOME wins', g.home_win, sub)}
          ${cell('both', 'Contradiction', g.contradiction, bt ? `${bt.contradictions === 0 ? 'never' : bt.contradictions} in ${bt.fixtures.toLocaleString()} backtest fixtures` : sub)}
          <div class="row-h">HOME WON'T WIN</div>
          ${cell('weak', 'No confident call', g.weak, 'about half the card, typically')}
          ${cell('call', 'AWAY wins', g.away_win, sub)}
        </div>
      </div>
      <div class="features">
        <div class="eyebrow">FEATURES — THREE NUMBERS</div>
        <div class="feature-list">${features}</div>
        <p class="foot">One logistic regression on three scaled columns, applied from each club's side with the inputs mirrored. Nothing else is in the model. "odds ×" is how the odds of winning move for a one standard deviation rise in that column.</p>
      </div>
    </div>

    <div class="h-row"><h2>Elo ratings</h2><span class="count">updated ${esc(f.label)}</span></div>
    <p class="lede" style="margin-bottom:16px">Stale ratings are the main failure mode. If the date above is not the last day football was played, refresh before trusting anything on this screen. 7D is the movement over the last seven days of results.</p>
    <div class="elo">
      <div class="tr head">
        <div class="e-rank">#</div>
        <div class="e-team">CLUB</div>
        <div class="e-bar">RELATIVE</div>
        <div class="e-elo">ELO</div>
        <div class="e-d">7D</div>
      </div>
      ${elo}
    </div>`;
}

// --- events ----------------------------------------------------------

async function selectRound(number) {
  state.round = number;
  state.previewError = null;
  if (number === state.data.current_round) {
    state.preview = null;
    state.previewLoading = null;
    render();
    return;
  }
  state.preview = null;
  state.previewLoading = number;
  render();
  const body = await getJSON(`/api/preview/${number}`);
  if (state.round !== number) return; // the user has moved on
  state.previewLoading = null;
  if (body.error) state.previewError = body.error;
  else state.preview = body;
  render();
}

function bindPanel() {
  const rounds = state.data.rounds || [];
  const idx = rounds.findIndex((r) => r.round === state.round);
  const prev = $('#round-prev');
  const next = $('#round-next');
  if (prev) prev.onclick = () => selectRound(rounds[idx - 1].round);
  if (next) next.onclick = () => selectRound(rounds[idx + 1].round);

  const cycle = $('#season-cycle');
  if (cycle) {
    cycle.onclick = () => {
      state.seasonIdx = (state.seasonIdx + 1) % state.data.record.seasons.length;
      state.weekIdx = 0;
      render();
    };
  }
  const wPrev = $('#week-prev');
  const wNext = $('#week-next');
  if (wPrev) wPrev.onclick = () => { state.weekIdx += 1; render(); };
  if (wNext) wNext.onclick = () => { state.weekIdx -= 1; render(); };
}

document.querySelectorAll('.tab').forEach((t) => {
  t.addEventListener('click', () => { state.tab = t.dataset.tab; render(); });
});
$('#refresh').addEventListener('click', refresh);

load();
