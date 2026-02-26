/**
 * raiveFeeder — WebSocket Progress Module
 *
 * Manages WebSocket connections for real-time batch job progress updates.
 * Mirrors raiveFlier's websocket.js pattern but tracks batch items
 * instead of pipeline phases.
 */
const FeederProgress = (() => {
  'use strict';

  let _ws = null;
  let _callbacks = {};

  function connect(jobId, onProgress) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${window.location.host}/ws/progress/${jobId}`;

    _ws = new WebSocket(url);
    _callbacks[jobId] = onProgress;

    _ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const cb = _callbacks[data.job_id || jobId];
        if (cb) cb(data);
      } catch {
        // Ignore malformed messages.
      }
    };

    _ws.onclose = () => {
      delete _callbacks[jobId];
    };
  }

  function disconnect() {
    if (_ws) {
      _ws.close();
      _ws = null;
    }
    _callbacks = {};
  }

  return { connect, disconnect };
})();
