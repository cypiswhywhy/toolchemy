import logging
import pathlib
from colorlog import ColoredFormatter

from toolchemy.utils.utils import _caller_module_name


def get_logger(name: str | None = None, level: int = logging.INFO, log_dir: str | None = None, with_time: bool = True,
               with_module_name: bool = True, with_log_level: bool = True, short_module_name: bool = False, say_hi: bool = False) -> logging.Logger:
    name = name or _caller_module_name(offset=2)
    if name.endswith(".common"):
        # I'm sorry ;)
        name = _caller_module_name(offset=3)

    datetime_format = "%H:%M:%S"
    prompts_parts = []
    if with_time:
        prompts_parts.append('%(asctime)s')
    if with_module_name:
        module_format = '%(name)s'
        if short_module_name:
            module_format = '%(module)s'
        prompts_parts.append(module_format)

    if with_log_level:
        prompts_parts.append('%(levelname)s')

    msg_format = "%(log_color)s|" + " ".join(prompts_parts) + "|%(reset)s %(message)s"

    formatter = ColoredFormatter(fmt=msg_format, datefmt=datetime_format, force_color=True, log_colors={
		'DEBUG':    'cyan',
		'INFO':     'green',
		'WARNING':  'yellow',
		'ERROR':    'red',
		'CRITICAL': 'red,bg_white',
	},
	secondary_log_colors={})

    logger = logging.getLogger(name)
    for handler in (logger.handlers or []):
        logger.removeHandler(handler)

    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_dir is not None:
            log_dir_path = pathlib.Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)

            log_file = log_dir_path / f"{name.replace('.', '_')}.log"
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.propagate = False

    if say_hi:
        logger.info("Hi:)")
        logger.debug("Debug mode ON")
        logger.debug("All handlers:")
        for handler in logger.handlers:
            logger.debug(f"- name: {handler.get_name()}, format: {handler.formatter._fmt}, level: {handler.level}")

    return logger


def testing():
    logger = get_logger("toolchemy.utils.logger", level=logging.DEBUG, say_hi=True)
    logger.warning("Testing logger setup WARNING")
    logger.info("Testing logger setup INFO")
    logger.debug("Testing logger setup DEBUG")


if __name__ == "__main__":
    testing()
