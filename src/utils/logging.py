"""Structured logging setup using structlog.

Implements a **dual-renderer pattern**: the same shared processor chain
(context vars, log level, timestamps, stack info) feeds into either a
coloured ConsoleRenderer for local development or a JSONRenderer for
production.  The renderer is selected automatically based on the ``APP_ENV``
environment variable (default ``"development"``), or forced via the
``json_output`` flag.

Standard-library ``logging`` is also rewired through the same structlog
formatter so that third-party libraries (httpx, uvicorn, etc.) produce
identically formatted output.
"""

import logging
import sys

import structlog


def configure_logging(log_level: str = "INFO", json_output: bool = False) -> structlog.BoundLogger:
    """Configure structlog with environment-appropriate rendering.

    Args:
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR, FATAL).
        json_output: Force JSON output. When False, uses console rendering in
                     development and JSON in production (detected via APP_ENV).

    Returns:
        A configured structlog BoundLogger.
    """
    import os

    # Dual-renderer selection: APP_ENV controls which renderer is used.
    # "production" => machine-readable JSON; anything else => human-readable console.
    app_env = os.environ.get("APP_ENV", "development")
    use_json = json_output or app_env == "production"

    # Shared processor chain -- these run regardless of the output format.
    # Order matters: contextvars first (merges request-scoped bindings),
    # then level/timestamps, then exception formatting.
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,  # Merge request-scoped context bindings
        structlog.processors.add_log_level,        # Inject "level" key
        structlog.processors.StackInfoRenderer(),  # Render stack_info if present
        structlog.dev.set_exc_info,                # Auto-attach exc_info on error()
        structlog.processors.TimeStamper(fmt="iso"),  # ISO-8601 timestamps
    ]

    # The final renderer is the only processor that differs between environments.
    if use_json:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            renderer,
        ],
        # make_filtering_bound_logger creates a logger that drops messages
        # below log_level BEFORE processing, saving work in hot paths.
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,  # Cache after first .bind() for performance
    )

    # Bridge stdlib logging through the same structlog pipeline so that
    # third-party libraries (httpx, uvicorn, etc.) produce identical output.
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *shared_processors,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()   # Remove default handlers to avoid duplicates
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level.upper())

    return structlog.get_logger()


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a named structlog logger.

    If structlog has not been configured yet, calls configure_logging() with defaults.

    Args:
        name: Logger name, typically the module name.

    Returns:
        A structlog BoundLogger bound with the given name.
    """
    if not structlog.is_configured():
        configure_logging()

    return structlog.get_logger(logger_name=name)
