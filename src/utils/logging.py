"""Structured logging setup using structlog."""

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

    app_env = os.environ.get("APP_ENV", "development")
    use_json = json_output or app_env == "production"

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if use_json:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging so third-party libraries produce structured output
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
    root_logger.handlers.clear()
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
