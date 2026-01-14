#!/usr/bin/env python3
"""
Simple signature test - direct imports only, no full satorilib
"""

import sys
import os

# Try direct import of evrmore modules
try:
    from evrmore.wallet import CEvrmoreSecret
    from evrmore.signmessage import EvrmoreMessage, signMessage, verifyMessage

    print("✅ python-evrmorelib is available\n")

    # Test WIF key (same as Dart will use)
    test_wif = 'cW2YkiSzKVcG3VVKKBhVpAJFKdigjfXgDE2fxkk6YJW9Q9qoRUSc'
    message = "test message 123"

    print("=== PYTHON SIGNATURE GENERATION ===\n")

    # Create key from WIF
    key = CEvrmoreSecret(test_wif)

    # Get pubkey (compressed)
    pubkey_bytes = key.pub
    pubkey_hex = pubkey_bytes.hex()

    print(f"WIF: {test_wif}")
    print(f"Pubkey (hex): {pubkey_hex}")
    print(f"Pubkey length: {len(pubkey_hex)} chars")
    print(f"Message: \"{message}\"")
    print()

    # Sign message
    evrmore_msg = EvrmoreMessage(message)
    sig_bytes = signMessage(key, evrmore_msg)
    sig_base64 = sig_bytes.decode('ascii')

    print(f"Signature (base64): {sig_base64}")
    print(f"Signature length: {len(sig_base64)} chars")
    print()

    # Verify signature
    print("=== PYTHON SIGNATURE VERIFICATION ===\n")

    # Try to verify with public key
    is_valid = evrmore_msg.verify(
        pubkey=pubkey_hex,
        signature=sig_bytes
    )

    print(f"Verification result: {is_valid}")

    if is_valid:
        print("\n✅ SUCCESS: Python can generate and verify signatures!")
        print("\n" + "=" * 60)
        print("EXPECTED DART OUTPUT:")
        print("=" * 60)
        print(f"Pubkey: {pubkey_hex}")
        print(f"Signature: {sig_base64}")
        print("\nIf Dart produces these exact values, compatibility is confirmed!")
    else:
        print("\n❌ FAILED: Python self-verification failed")

except ImportError as e:
    print(f"❌ ERROR: python-evrmorelib not available: {e}")
    print("\nInstallation required:")
    print("  pip install python-evrmorelib")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
