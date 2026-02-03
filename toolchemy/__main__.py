from toolchemy.utils.logger import get_logger


def main():
    logger = get_logger(log_dir="/tmp")
    logger.info("test")

if __name__ == '__main__':
    main()
