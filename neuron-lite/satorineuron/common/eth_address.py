"""Ethereum address derivation utilities for reverse bridge.

Derives Ethereum burn addresses from wallet public keys using existing
satorilib dependencies (eth_keys, coincurve).
"""

from coincurve import PublicKey
from eth_keys import keys


def derive_eth_wallet_address_from_pubkey(wallet_pubkey_hex: str) -> str:
    """
    Derive Ethereum burn address from wallet public key.

    Takes a compressed SECP256K1 public key (as used by Evrmore wallets),
    derives the standard Ethereum address, and replaces the last 4 characters
    with "DEAD" to create a burn address for the reverse bridge.

    Args:
        wallet_pubkey_hex: Compressed SECP256K1 public key as hex string
                          (e.g., "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798")

    Returns:
        Ethereum burn address with last 4 chars as "DEAD" (e.g., "0x1234...DEAD")

    Raises:
        ValueError: If wallet_pubkey_hex is invalid

    Examples:
        >>> pubkey = "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"
        >>> addr = derive_eth_wallet_address_from_pubkey(pubkey)
        >>> addr[-4:]
        'DEAD'
        >>> addr.startswith('0x')
        True
    """
    # Validate input
    if not wallet_pubkey_hex or not isinstance(wallet_pubkey_hex, str):
        raise ValueError("Wallet public key must be a non-empty string")

    # Remove 0x prefix if present
    wallet_pubkey_hex = wallet_pubkey_hex.strip()
    if wallet_pubkey_hex.startswith('0x'):
        wallet_pubkey_hex = wallet_pubkey_hex[2:]

    # Convert to bytes
    try:
        pubkey_bytes = bytes.fromhex(wallet_pubkey_hex)
    except ValueError as e:
        raise ValueError(f"Invalid hex string: {e}")

    # Decompress SECP256K1 public key (33 bytes compressed -> 65 bytes uncompressed)
    try:
        pubkey = PublicKey(pubkey_bytes)
        uncompressed = pubkey.format(compressed=False)  # Returns 65 bytes: 0x04 + x + y
    except Exception as e:
        raise ValueError(f"Invalid SECP256K1 public key: {e}")

    # Use eth_keys to derive Ethereum address from uncompressed public key
    # eth_keys expects 64 bytes (without the 0x04 prefix)
    try:
        eth_pubkey = keys.PublicKey(uncompressed[1:])  # Skip 0x04 prefix
        checksummed = eth_pubkey.to_checksum_address()
    except Exception as e:
        raise ValueError(f"Failed to derive Ethereum address: {e}")

    # Replace last 4 characters with "DEAD"
    burn_address = checksummed[:-4] + "DEAD"

    return burn_address
