import os
import requests
import json
import logging
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger("turbulence-api")

class MosdacClient:
    """
    A client to interact with the MOSDAC Data Download API.
    Ref: https://www.mosdac.gov.in/
    """
    def __init__(self, username=None, password=None):
        self.username = username or os.getenv("MOSDAC_USERNAME")
        self.password = password or os.getenv("MOSDAC_PASSWORD")
        self.base_url = "https://api.mosdac.gov.in" # Example base URL

    def is_configured(self):
        return bool(self.username and self.password)

    def fetch_latest_metadata(self, dataset_id):
        """
        Fetch metadata for the latest available product in a dataset.
        """
        if not self.is_configured():
            logger.warning("MOSDAC credentials not configured.")
            return None
        
        # Mocking the discovery process
        logger.info(f"Mocking metadata fetch for dataset: {dataset_id}")
        return {
            "file_id": "3D_IMG_L2B_CTP_20241010_0100.h5",
            "timestamp": "2024-10-10T01:00:00Z",
            "url": f"{self.base_url}/download/{dataset_id}/latest"
        }

    def download_product(self, file_id, target_path):
        """
        Download a specific HDF5 product.
        """
        if not self.is_configured():
            return False
        
        logger.info(f"Mocking download of {file_id} to {target_path}")
        # In a real implementation, this would use self.username/password 
        # to authenticate and download the file via requests.
        return True

    def get_realtime_data(self, dataset_id, lat, lon):
        """
        High-level method to 'ingest' real-time data for a location.
        Enhanced to return a stream of recent points and a 24h forecast.
        """
        logger.info(f"Attempting real-time ingestion for {lat}, {lon} from {dataset_id}")
        
        # 1. Discover latest file (Mock)
        metadata = self.fetch_latest_metadata(dataset_id)
        if not metadata:
            return {"error": "Authentication required for MOSDAC API or configuration missing"}

        # 2. Simulate historical 'stream' (last 5 detections)
        stream = []
        base_time = datetime.fromisoformat(metadata["timestamp"].replace("Z", "+00:00"))
        
        for i in range(5):
            t = base_time - timedelta(minutes=15 * (4 - i))
            stream.append({
                "timestamp": t.isoformat(),
                "CTP": 450.5 + (np.random.randn() * 10),
                "CTT": 245.2 + (np.random.randn() * 2),
                "status": "Inbound" if i < 4 else "Active"
            })

        # 3. Simulate 24h Forecast Trend
        forecast = [
            {"hour": "+6h", "risk": "Low", "trend": "Stable"},
            {"hour": "+12h", "risk": "Moderate", "trend": "Increasing Cloud"},
            {"hour": "+18h", "risk": "Moderate", "trend": "High Shear"},
            {"hour": "+24h", "risk": "Low", "trend": "Clearing"}
        ]

        return {
            "source": "MOSDAC Live Stream",
            "file_id": metadata["file_id"],
            "latest_timestamp": metadata["timestamp"],
            "stream": stream,
            "forecast_24h": forecast
        }
