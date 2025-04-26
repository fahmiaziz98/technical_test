import logging

def init_logger():
    """
    Initialize the logger.

    Returns:
        logging.Logger: The initialized logger.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)
