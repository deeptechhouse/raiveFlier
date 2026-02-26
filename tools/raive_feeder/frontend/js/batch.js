/**
 * raiveFeeder — Batch Queue UI Module
 *
 * Displays batch job status, progress, and controls (pause/resume/cancel).
 * Connects to the BatchProcessor via the REST API and WebSocket progress.
 */
const FeederBatch = (() => {
  'use strict';

  async function loadJobs() {
    try {
      return await FeederApp.apiFetch('/jobs');
    } catch {
      return [];
    }
  }

  async function cancelJob(jobId) {
    return FeederApp.apiFetch(`/jobs/${jobId}/cancel`, { method: 'POST' });
  }

  async function pauseJob(jobId) {
    return FeederApp.apiFetch(`/jobs/${jobId}/pause`, { method: 'POST' });
  }

  async function resumeJob(jobId) {
    return FeederApp.apiFetch(`/jobs/${jobId}/resume`, { method: 'POST' });
  }

  return { loadJobs, cancelJob, pauseJob, resumeJob };
})();
