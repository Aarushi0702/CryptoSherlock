def extract_addresses_safely(address_field):
    """
    Universal helper to safely extract addresses from inputs/outputs
    Works with both live API (strings) and synthetic (dicts) formats
    
    Args:
        address_field: List containing either strings (live API) or dicts (synthetic data)
    
    Returns:
        List of unique valid addresses, excluding empty and 'unknown' values
    """
    if not address_field:
        return []
    
    addresses = []
    for item in address_field:
        if isinstance(item, str):
            # Live API format - addresses are direct strings
            addresses.append(item)
        elif isinstance(item, dict):
            # Synthetic format - addresses are in dict with possible keys
            addr = item.get('address') or item.get('addresses', [None])[0]
            if addr:
                addresses.append(addr)
    
    # Remove duplicates using set(), filter out empty/unknown addresses
    return [addr for addr in list(set(addresses)) if addr and addr != 'unknown']

def safe_get_value(data_dict, keys, default=None):
    """
    Safely get value from dictionary by trying multiple possible keys
    
    Args:
        data_dict: Dictionary to search in
        keys: List of keys to try in order
        default: Value to return if no keys found
    
    Returns:
        Value from first matching key, or default if none found
    """
    for key in keys:
        if key in data_dict:
            return data_dict[key]
    return default

def normalize_timestamp(timestamp_str):
    """
    Normalize timestamp strings to ISO format with proper timezone handling
    
    Args:
        timestamp_str: Timestamp string in various formats
    
    Returns:
        ISO 8601 formatted timestamp string, or original if parsing fails
    """
    import pandas as pd
    try:
        # Replace 'Z' UTC indicator with explicit timezone offset
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1] + '+00:00'
        # Use pandas for robust timestamp parsing and convert to ISO format
        return pd.to_datetime(timestamp_str).isoformat()
    except:
        # Return original string if parsing fails
        return timestamp_str
