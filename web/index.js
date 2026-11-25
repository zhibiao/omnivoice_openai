(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };

  var state = {
    voice: null,
    numStep: 32,
    controller: null,
    timer: null,
    objectUrl: null
  };

  var toastTimer = null;
  function toast(msg, type) {
    var el = $('toast');
    el.textContent = msg;
    el.className = 'fixed bottom-6 left-1/2 -translate-x-1/2 px-4 py-2.5 rounded-lg text-sm text-white shadow-lg z-50 ' +
      (type === 'error' ? 'bg-red-500' : type === 'warn' ? 'bg-amber-500' : 'bg-gray-800');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () {
      el.className = 'fixed bottom-6 left-1/2 -translate-x-1/2 hidden px-4 py-2.5 rounded-lg text-sm text-white shadow-lg z-50';
    }, 3200);
  }

  async function api(path, opts) {
    var res = await fetch(path, opts);
    if (!res.ok) {
      var msg = 'HTTP ' + res.status;
      try {
        var j = await res.json();
        if (j && j.error && j.error.message) msg = j.error.message;
      } catch (_) {}
      throw new Error(msg);
    }
    return res;
  }

  function setBusy(btn, busy, label) {
    btn.disabled = busy;
    if (busy) {
      if (!btn.dataset.label) btn.dataset.label = btn.innerHTML;
      btn.innerHTML = '<i class="fa fa-spinner fa-spin mr-1"></i> ' + (label || 'Processing...');
    } else if (btn.dataset.label) {
      btn.innerHTML = btn.dataset.label;
      delete btn.dataset.label;
    }
  }

  async function refreshHealth() {
    try {
      var res = await api('/health');
      var h = await res.json();
      $('status-device').textContent = 'GPU: ' + (h.device || '-');
      $('status-compile').textContent = 'compile: ' + (h.compile || '-');
      $('status-numstep').textContent = 'num_step: ' + h.num_step_default;
      $('status-gpu').textContent = 'Memory: ' + (h.gpu ? h.gpu.free_gb + ' / ' + h.gpu.total_gb + ' GB free' : '-');
      $('status-queue').textContent = 'Queue: ' + (h.queue ? h.queue.pending + '/' + h.queue.max : '-');
      var dot = $('status-dot');
      dot.className = 'w-2.5 h-2.5 rounded-full ' + (h.gpu && h.gpu.free_gb < 1.0 ? 'bg-amber-400' : 'bg-green-500');
      dot.title = 'Service healthy';
    } catch (e) {
      $('status-dot').className = 'w-2.5 h-2.5 rounded-full bg-red-500';
      $('status-dot').title = 'Service unavailable: ' + e.message;
    }
  }

  async function loadVoices() {
    var select = $('synth-voice');
    try {
      var res = await api('/v1/audio/voices');
      var data = (await res.json()).data || [];
      if (!data.length) throw new Error('No voices available');
      select.innerHTML = data.map(function (voice) {
        return '<option value="' + voice.id + '">' + voice.name + ' (' + voice.duration.toFixed(1) + 's)</option>';
      }).join('');
      var preferred = data.find(function (voice) { return voice.default; }) || data[0];
      state.voice = preferred.id;
      select.value = state.voice;
      select.disabled = false;
    } catch (e) {
      select.innerHTML = '<option value="">Failed to load voices</option>';
      toast('Failed to load voices: ' + e.message, 'error');
    }
  }

  function setNumStep(n) {
    state.numStep = n;
    document.querySelectorAll('.numstep-btn').forEach(function (b) {
      b.classList.toggle('active', Number(b.dataset.step) === n);
    });
  }

  function setProgress(text) {
    $('synth-progress').textContent = text || '';
  }

  function stopTimer() {
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
  }

  async function synthesize() {
    var text = $('synth-text').value.trim();
    if (!state.voice) return toast('Select a voice', 'error');
    if (!text) return toast('Enter text to synthesize', 'error');
    if (state.controller) return;

    var btn = $('btn-synth');
    var cancelBtn = $('btn-cancel');
    var started = Date.now();

    state.controller = new AbortController();
    setBusy(btn, true, 'Generating...');
    btn.classList.add('generating');
    cancelBtn.classList.remove('hidden');
    setProgress('Queued / Generating...');

    state.timer = setInterval(function () {
      setProgress('Generating... ' + ((Date.now() - started) / 1000).toFixed(1) + 's');
    }, 100);

    try {
      var res = await api('/v1/audio/speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: state.controller.signal,
        body: JSON.stringify({
          model: 'omnivoice',
          input: text,
          voice: state.voice,
          response_format: 'wav',
          num_step: state.numStep,
          speed: Number($('synth-speed').value)
        })
      });

      var blob = await res.blob();
      var wall = (Date.now() - started) / 1000;
      var dur = parseFloat(res.headers.get('X-Audio-Duration') || '0');
      var rtf = res.headers.get('X-RTF') || '-';
      var ns = res.headers.get('X-Num-Step') || String(state.numStep);

      if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
      state.objectUrl = URL.createObjectURL(blob);

      $('result-card').classList.remove('hidden');
      var audio = $('result-audio');
      audio.src = state.objectUrl;
      audio.play().catch(function () {});

      var dl = $('btn-download');
      dl.href = state.objectUrl;
      dl.setAttribute('download', 'speech_' + Date.now() + '.wav');

      $('result-meta').innerHTML =
        '<span><i class="fa fa-clock-o mr-1"></i>Audio ' + dur.toFixed(2) + 's</span>' +
        '<span><i class="fa fa-hourglass-half mr-1"></i>Elapsed ' + wall.toFixed(2) + 's</span>' +
        '<span><i class="fa fa-bolt mr-1"></i>RTF ' + rtf + ' (about ' + (rtf && rtf !== '-' ? (1 / parseFloat(rtf)).toFixed(1) : '-') + 'x realtime)</span>' +
        '<span><i class="fa fa-sliders mr-1"></i>num_step ' + ns + '</span>' +
        '<span><i class="fa fa-microphone mr-1"></i>' + state.voice + '</span>' +
        '<span><i class="fa fa-file-audio-o mr-1"></i>' + (blob.size / 1024).toFixed(0) + ' KB · WAV</span>';

      setProgress('Complete');
      toast('Speech generated (' + dur.toFixed(1) + 's audio)');
    } catch (e) {
      if (e.name === 'AbortError') {
        setProgress('Cancelled');
        toast('Cancelled', 'warn');
      } else {
        setProgress('Failed');
        toast('Generation failed: ' + e.message, 'error');
      }
    } finally {
      stopTimer();
      state.controller = null;
      cancelBtn.classList.add('hidden');
      btn.classList.remove('generating');
      setBusy(btn, false);
    }
  }

  function init() {
    setNumStep(32);
    loadVoices();
    refreshHealth();
    setInterval(refreshHealth, 10000);

    $('synth-text').addEventListener('input', function (e) {
      $('text-count').textContent = e.target.value.length + ' / 2000';
    });

    document.querySelectorAll('.numstep-btn').forEach(function (b) {
      b.addEventListener('click', function () { setNumStep(Number(b.dataset.step)); });
    });

    $('synth-speed').addEventListener('input', function (e) {
      $('speed-label').textContent = Number(e.target.value).toFixed(1);
    });

    $('synth-voice').addEventListener('change', function (e) {
      state.voice = e.target.value || null;
    });

    $('btn-synth').addEventListener('click', synthesize);
    $('btn-cancel').addEventListener('click', function () {
      if (state.controller) state.controller.abort();
    });
    $('btn-replay').addEventListener('click', function () {
      var audio = $('result-audio');
      audio.currentTime = 0;
      audio.play().catch(function () { });
    });

    $('synth-text').addEventListener('keydown', function (e) {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') synthesize();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
