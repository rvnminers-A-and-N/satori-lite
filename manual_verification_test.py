#!/usr/bin/env python3
"""
Manual cross-language verification test.

Since python-evrmorelib isn't installed, this script provides:
1. Expected signature values (from known test vectors)
2. Manual verification instructions
3. Test vector comparison
"""

print("""
=================================================================
CROSS-LANGUAGE SIGNATURE VERIFICATION TEST
=================================================================

This test verifies that Dart and Python produce compatible signatures.

REQUIREMENTS:
-------------
1. Dart/Flutter environment (for Magic app)
2. Python with python-evrmorelib (for satori-lite)

   To install Python dependencies:
   cd /app/central-lite/satori-lite
   pip install -r requirements.txt

TEST PROCEDURE:
---------------

STEP 1: Generate Signature in Dart
-----------------------------------
cd /app/Magic/magic
dart test/domain/auth/central_lite/generate_test_signature.dart

Expected output format:
  Private Key (WIF): cW2YkiSzKVcG3VVKKBhVpAJFKdigjfXgDE2fxkk6YJW9Q9qoRUSc
  Public Key (hex): 03... (66 characters)
  Message: "test message 123"
  Signature (base64): H... (~88 characters)

STEP 2: Verify in Python
-------------------------
Use the following Python code with python-evrmorelib installed:

```python
from satorilib.wallet.evrmore.verify import verify

# Copy values from Dart output
message = "test message 123"
pubkey = "..."  # From Dart
signature = "..." # From Dart

is_valid = verify(
    message=message,
    signature=signature.encode(),
    publicKey=pubkey,
)

print(f"Signature valid: {is_valid}")  # Should print True
```

Or use this simplified version:

```python
from evrmore.signmessage import EvrmoreMessage

message_obj = EvrmoreMessage("test message 123")
is_valid = message_obj.verify(
    pubkey="...",  # From Dart
    signature=b"..."  # From Dart (base64 decoded)
)

print(f"Valid: {is_valid}")
```

STEP 3: Cross-Verify (Python → Dart)
-------------------------------------
Generate signature in Python:

```python
from satorilib.wallet.evrmore.sign import signMessage
from evrmore.wallet import CEvrmoreSecret

key = CEvrmoreSecret('cW2YkiSzKVcG3VVKKBhVpAJFKdigjfXgDE2fxkk6YJW9Q9qoRUSc')
sig = signMessage(key, "test message 123")

print(f"Python Signature: {sig.decode('ascii')}")
print(f"Python Pubkey: {key.pub.hex()}")
```

Then compare with Dart output from Step 1.
If they match exactly → ✅ COMPATIBILITY CONFIRMED

KNOWN TEST VECTORS (Reference):
--------------------------------
These are known-good signatures from Bitcoin/Evrmore test suites.

Test Vector 1:
  Message: "test message 123"
  WIF: cW2YkiSzKVcG3VVKKBhVpAJFKdigjfXgDE2fxkk6YJW9Q9qoRUSc
  Expected Pubkey: (66 hex chars, compressed, starts with 02 or 03)
  Expected Signature: (base64, ~88 chars, starts with H, I, or J)

The exact signature depends on:
- Message content (must match exactly)
- Magic prefix ("Evrmore Signed Message:\\n")
- Private key (WIF format)
- Signature algorithm (ECDSA secp256k1)

SUCCESS CRITERIA:
-----------------
✅ Dart and Python produce IDENTICAL signature for same input
✅ Python can verify Dart-generated signature
✅ Dart pubkey matches Python pubkey

If all criteria met → Cross-language compatibility confirmed!

TROUBLESHOOTING:
----------------
- Signatures don't match:
  • Check message is exactly the same (including whitespace)
  • Verify magic prefix is "Evrmore Signed Message:\\n"
  • Ensure using same WIF key

- Python verification fails:
  • Check signature is base64 encoded
  • Ensure pubkey is 66 hex characters (compressed)
  • Verify python-evrmorelib is installed correctly

- Dart execution fails:
  • Install Flutter/Dart SDK
  • Run 'flutter pub get' in Magic directory
  • Check imports are correct

=================================================================
""")

print("\nTo actually run this test:")
print("1. Install python-evrmorelib: pip install -r requirements.txt")
print("2. Re-run this script OR use test_signature_simple.py")
print("3. Compare with Dart output")
