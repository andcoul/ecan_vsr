from ecan import logger
from ecan.components.data_ingestion import DataIngestion
from ecan.config.configuration import ConfigurationManager
from ecan.pipeline.stage_01_data_ingestion import DataIngestionPipeline

STAGE = "Data Ingestion"

try:
    logger.info(f">>>> The {STAGE} has started")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f">>>> The {STAGE} completed")
except Exception as e:
    logger.exception(e)
    raise e
