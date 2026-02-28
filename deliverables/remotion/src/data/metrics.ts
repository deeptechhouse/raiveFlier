/**
 * ─── CENTRALIZED METRICS DATA ───
 *
 * Single source of truth for all numeric metrics displayed across the
 * raiveFlier Remotion presentation. Keeping these values in one place
 * ensures consistency between the MetricsSlide, CompletenessSlide,
 * GitHistorySlide, and any future slides that reference project stats.
 *
 * Values were verified against the actual codebase as of 2026-02-27.
 * Update this file when metrics change — all consuming slides will
 * automatically reflect the new values.
 *
 * Architecture connection: This is a pure data module with no dependencies.
 * It is consumed by slides in src/slides/ and never imports from them.
 */

export const metrics = {
  /** Total lines of Python source code (excluding tests) */
  pythonLOC: 30088,

  /** Number of Python source files */
  sourceFiles: 112,

  /** Lines of JavaScript (frontend vanilla JS) */
  jsLOC: 5476,

  /** Lines of CSS (frontend styles) */
  cssLOC: 4099,

  /** Total lines of test code */
  testLOC: 23293,

  /** Number of test files */
  testFiles: 37,

  /** Total number of test functions across all test files */
  testFunctions: 1174,

  /** FastAPI endpoint count */
  apiEndpoints: 17,

  /** Abstract base class interfaces (ILLMProvider, IOCRProvider, etc.) */
  interfaces: 9,

  /** Concrete adapter implementations across all providers */
  adapters: 22,

  /** Pydantic v2 data models */
  models: 39,

  /** Git commits on the main branch */
  commits: 145,

  /** Merged pull requests */
  pullRequests: 23,

  /** Calendar days of active development */
  devDays: 9,

  /** Overall project completeness percentage */
  completeness: 92,
};
