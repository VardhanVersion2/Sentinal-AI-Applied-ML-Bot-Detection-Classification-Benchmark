(() => {
  const form = document.querySelector('#analysis-form');
  const button = document.querySelector('#analyze-button');
  const buttonLabel = document.querySelector('#button-label');
  const errorBox = document.querySelector('#form-error');
  const emptyResult = document.querySelector('#empty-result');
  const resultContent = document.querySelector('#result-content');
  const scoreRing = document.querySelector('#score-ring');
  const percent = (n) => `${Math.round(n * 100)}%`;

  const date = new Intl.DateTimeFormat(undefined, { month: 'short', day: '2-digit', year: 'numeric' });
  document.querySelector('#today').textContent = date.format(new Date()).toUpperCase();

  async function checkHealth() {
    const dots = [document.querySelector('#sidebar-dot'), document.querySelector('#top-dot')];
    try {
      const response = await fetch('/health', { headers: { Accept: 'application/json' } });
      const health = await response.json();
      if (!response.ok || health.status !== 'healthy') throw new Error(health.message || 'Model unavailable');
      dots.forEach((dot) => { dot.classList.add('good'); dot.classList.remove('bad'); });
      document.querySelector('#sidebar-status').textContent = 'Model operational';
      document.querySelector('#model-name').textContent = health.model_name || 'Champion classifier';
      document.querySelector('#top-status').textContent = 'Model online';
    } catch (_) {
      dots.forEach((dot) => { dot.classList.add('bad'); dot.classList.remove('good'); });
      document.querySelector('#sidebar-status').textContent = 'Model unavailable';
      document.querySelector('#model-name').textContent = 'Start the Flask API to analyze';
      document.querySelector('#top-status').textContent = 'API offline';
    }
  }
  checkHealth();

  document.querySelector('#description').addEventListener('input', (event) => {
    document.querySelector('#bio-count').textContent = `${event.target.value.length} / 500`;
  });

  document.querySelector('#demo-button').addEventListener('click', () => {
    const sample = { screen_name: 'free_crypto_984321', followers_count: 2, friends_count: 850, statuses_count: 45, favourites_count: 0, account_age_days: 12, description: 'FREE BITCOIN AIRDROP! CLICK LINK BELOW http://fake.scam', location: '', default_profile: true, default_profile_image: true, geo_enabled: false, verified: false };
    for (const [key, value] of Object.entries(sample)) {
      const field = form.elements.namedItem(key);
      if (!field) continue;
      if (field.type === 'checkbox') field.checked = value;
      else field.value = value;
    }
    document.querySelector('#bio-count').textContent = `${sample.description.length} / 500`;
    errorBox.hidden = true;
    form.scrollIntoView({ behavior: 'smooth', block: 'center' });
  });

  function renderResult(result, payload) {
    const probability = Number(result.bot_probability);
    const threshold = Number(result.decision_threshold ?? 0.5);
    const bounded = Math.max(0, Math.min(1, probability));
    const bot = Boolean(result.is_bot);
    emptyResult.hidden = true;
    resultContent.hidden = false;
    document.querySelector('#result-state').textContent = bot ? 'BOT SIGNAL DETECTED' : 'HUMAN SIGNAL LIKELY';
    document.querySelector('#result-state').className = `result-state is-ready${bot ? ' is-bot' : ''}`;
    document.querySelector('#result-avatar').textContent = (payload.screen_name || 'A').trim().charAt(0).toUpperCase() || 'A';
    document.querySelector('#result-handle').textContent = payload.screen_name ? `@${payload.screen_name.replace(/^@/, '')}` : 'Profile analysis';
    document.querySelector('#score-value').textContent = percent(bounded);
    const color = bot ? '#f08077' : '#67dcaa';
    scoreRing.style.background = `conic-gradient(${color} ${bounded * 360}deg, #25342b ${bounded * 360}deg 360deg)`;
    document.querySelector('#verdict').textContent = result.prediction_label || (bot ? 'Bot' : 'Human');
    document.querySelector('#verdict').className = bot ? 'bot' : '';
    document.querySelector('#verdict-tier').textContent = result.risk_tier || 'Assessment complete';
    document.querySelector('#threshold-label').textContent = percent(threshold);
    document.querySelector('.threshold-marker').style.left = `${threshold * 100}%`;
    document.querySelector('#probability-marker').style.left = `${bounded * 100}%`;
    document.querySelector('#confidence').textContent = percent(Number(result.confidence_score ?? (bot ? bounded : 1 - bounded)));
    document.querySelector('#threshold').textContent = percent(threshold);
    document.querySelector('#latency').textContent = `${Number(result.latency_ms ?? 0).toFixed(2)} ms`;
    document.querySelector('#result-note').textContent = `The model estimates a ${percent(bounded)} chance that this profile matches bot patterns in its training data.`;
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    errorBox.hidden = true;
    const data = new FormData(form);
    const payload = Object.fromEntries(data.entries());
    for (const key of ['followers_count', 'friends_count', 'statuses_count', 'favourites_count', 'account_age_days']) {
      const field = form.elements.namedItem(key);
      if (field.value === '' || Number(field.value) < (key === 'account_age_days' ? 1 : 0)) {
        field.focus();
        errorBox.textContent = `${field.labels?.[0]?.textContent?.trim() || key} must be ${key === 'account_age_days' ? 'at least 1' : 'zero or greater'}.`;
        errorBox.hidden = false;
        return;
      }
      payload[key] = Number(field.value);
    }
    for (const key of ['default_profile', 'default_profile_image', 'geo_enabled', 'verified']) {
      payload[key] = form.elements.namedItem(key).checked;
    }
    payload.screen_name = payload.screen_name.trim().replace(/^@/, '');
    button.disabled = true;
    buttonLabel.innerHTML = '<span class="spinning">◌</span> Analyzing profile…';
    document.querySelector('#result-state').textContent = 'ANALYZING';
    try {
      const response = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json' }, body: JSON.stringify(payload) });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'The prediction request failed.');
      renderResult(result, payload);
    } catch (error) {
      errorBox.textContent = error.message || 'Could not reach the prediction service. Check that the Flask API is running.';
      errorBox.hidden = false;
      document.querySelector('#result-state').textContent = 'ANALYSIS FAILED';
      document.querySelector('#result-state').className = 'result-state is-bot';
    } finally {
      button.disabled = false;
      buttonLabel.textContent = 'Analyze account';
    }
  });
})();
