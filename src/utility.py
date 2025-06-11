import hashlib
import hmac


def get_signature_key(key: str, date_stamp: str, region_name: str, service_name: str) -> bytes:
    """Generate AWS signature key."""
    k_date = hmac.new(('AWS4' + key).encode('utf-8'), date_stamp.encode('utf-8'), hashlib.sha256).digest()
    k_region = hmac.new(k_date, region_name.encode('utf-8'), hashlib.sha256).digest()
    k_service = hmac.new(k_region, service_name.encode('utf-8'), hashlib.sha256).digest()
    return hmac.new(k_service, b'aws4_request', hashlib.sha256).digest()