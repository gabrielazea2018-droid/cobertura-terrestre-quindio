"""Google Earth Engine authentication module."""
import ee


def initialize_ee():
    """Initialize Earth Engine. Authenticate on first run."""
    try:
        ee.Initialize()
        print("Earth Engine initialized successfully.")
    except Exception:
        print("Authentication required. Opening browser...")
        ee.Authenticate()
        ee.Initialize()
        print("Earth Engine initialized successfully.")
