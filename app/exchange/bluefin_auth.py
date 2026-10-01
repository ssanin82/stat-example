"""
Bluefin (Sui) authentication, order-signing, and session handling.

Migrated to the ``fireflyprotocol/pro-sdk`` API conventions (Bluefin Pro).

Design notes
------------
* Orders on Bluefin Pro are off-chain signed messages. The server computes
  the canonical order hash (BCS + blake2b-32) and returns it on success;
  the client only has to produce a valid ``signature`` for the ``signedFields``
  payload and POST the full request to ``/api/v1/trade/orders`` at
  ``https://trade.api.<env>.bluefin.io``. Our adapter therefore carries
  *no* client-side BCS dependency — we trust the server-returned
  ``orderHash`` and treat it as the order's primary identifier.

* Authentication is Sui-wallet-based JWT: POST ``/auth/v2/token`` at
  ``https://auth.api.<env>.bluefin.io`` with
  ``{ accountAddress, audience, signedAtMillis }`` JSON body. The request
  is signed with the same PersonalMessage+base64-wire scheme described
  below; the signature goes in the ``payloadSignature`` HTTP header.

* We deliberately avoid both ``bluefin-v2-client-python`` (deprecated
  v2 API only) and ``pro-sdk`` Python bindings (auto-generated from
  Rust, large transitive dep set). Pure-Python pynacl Ed25519 +
  blake2b + hand-built BCS-vector-u8 framing is sufficient.

* Path B (BLUEFIN_ONE_CT_ENABLED=false) continues to be the supported
  operational mode: the configured ``BLUEFIN_PRIVATE_KEY`` belongs to
  the parent wallet and signs orders directly. Path A/C (session keys
  + on-chain ``upsertSubAccount``) is deferred to a later ship.

Wire format for the pro-sdk ``signature`` field
-----------------------------------------------

Bluefin Pro speaks the Sui ``UserSignature`` format — base64 of the
concatenation::

    flag_byte (1)  || raw_signature (64)  || raw_pubkey (32)

where ``flag_byte`` is ``0x00`` for Ed25519 (matches
``SignatureScheme::ED25519`` in @mysten/sui). The signature is
computed over the Sui **PersonalMessage** digest:

    msg = utf8( serde_json_pretty(signable_payload) )
    intent = [0x03, 0x00, 0x00] || bcs_vector_u8(msg)     # Sui PersonalMessage intent
    digest = blake2b-32(intent)
    sig = Ed25519(signing_key, digest)

The ``signable_payload`` for each request type is a JSON object with a
``type`` field ("Bluefin Pro Order", "Bluefin Pro Authorize Account", ...)
and the payload's string-typed fields. See :func:`signable_create_order`
for the canonical shape.

JSON must be serialized in *pretty* form (``indent=2``, ``separators=(",", ": ")``)
because the server re-serializes the same struct via Rust's
``serde_json::to_string_pretty`` before verifying the signature, and the
two outputs must be byte-identical.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Any, Optional

try:
    from nacl.signing import SigningKey, VerifyKey
except ImportError as e:  # pragma: no cover
    SigningKey = None  # type: ignore[assignment]
    VerifyKey = None  # type: ignore[assignment]
    _nacl_import_error: Optional[Exception] = e
else:
    _nacl_import_error = None

logger = logging.getLogger(__name__)

# Sui UserSignature scheme flag byte values (used as the base64-encoded
# signature's first byte). Ed25519 is 0x00; we never generate secp256k1
# on this path.
_SUI_SIG_FLAG_ED25519 = 0x00

# Signable-payload "type" field values. These are the human-readable tags
# Bluefin Pro expects on the signed JSON (see pro-sdk
# signature::conversion::signable::ClientPayloadType).
_TYPE_ORDER = "Bluefin Pro Order"
_TYPE_AUTHORIZE = "Bluefin Pro Authorize Account"
_TYPE_WITHDRAW = "Bluefin Pro Withdrawal"
_TYPE_LEVERAGE = "Bluefin Pro Leverage Adjustment"
_TYPE_ADJUST_MARGIN = "Bluefin Pro Margin Adjustment"


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------


def _strip_0x(s: str) -> str:
    s = s.strip()
    return s[2:] if s.lower().startswith("0x") else s


def _sui_address_from_ed25519_public_key(pub: bytes) -> str:
    """Derive a Sui address from a raw Ed25519 public key.

    Scheme flag byte 0x00 (Ed25519) is prepended to the 32-byte pubkey,
    then blake2b with 32-byte digest yields the address. Matches
    ``Ed25519PublicKey.toSuiAddress`` in @mysten/sui.
    """
    if len(pub) != 32:
        raise ValueError("expected 32-byte Ed25519 public key")
    h = blake2b(b"\x00" + pub, digest_size=32).hexdigest()
    return "0x" + h


# ----------------------------------------------------------------------
# Private key parsing
# ----------------------------------------------------------------------


def _decode_sui_bech32_privkey(s: str) -> bytes:
    """Decode a Sui bech32 privkey (``suiprivkey1...``) → 32 raw bytes."""
    CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
    sep = s.rfind("1")
    if sep < 1:
        raise ValueError("malformed sui bech32 privkey (no separator)")
    hrp = s[:sep].lower()
    if hrp != "suiprivkey":
        raise ValueError(f"unexpected bech32 hrp: {hrp!r}")
    data = s[sep + 1 :].lower()
    dec: list[int] = []
    for ch in data:
        idx = CHARSET.find(ch)
        if idx < 0:
            raise ValueError(f"invalid bech32 char: {ch!r}")
        dec.append(idx)
    payload = dec[:-6]
    acc = 0
    bits = 0
    out: list[int] = []
    for v in payload:
        acc = (acc << 5) | v
        bits += 5
        while bits >= 8:
            bits -= 8
            out.append((acc >> bits) & 0xFF)
    raw = bytes(out)
    if not raw:
        raise ValueError("empty bech32 payload")
    scheme = raw[0]
    if scheme != 0x00:
        raise ValueError(
            f"only Ed25519 privkeys are supported (scheme byte=0x{scheme:02x})"
        )
    seed = raw[1:33]
    if len(seed) != 32:
        raise ValueError(f"privkey payload wrong length: {len(seed)}")
    return seed


def parse_ed25519_secret(raw: str) -> bytes:
    """Coerce a configured private key into 32 seed bytes.

    Accepts:
      * 32-byte hex string (``0x`` prefix optional, case-insensitive)
      * 64-byte hex string (Sui ``priv || pub`` form) — take first 32 bytes
      * Sui bech32 (``suiprivkey1...``)
    """
    s = raw.strip()
    if not s:
        raise ValueError("BLUEFIN_PRIVATE_KEY is empty")
    if s.lower().startswith("suiprivkey"):
        return _decode_sui_bech32_privkey(s)
    hx = _strip_0x(s)
    try:
        blob = bytes.fromhex(hx)
    except ValueError as e:
        raise ValueError(
            "BLUEFIN_PRIVATE_KEY must be hex (0x-prefixed or not) or sui bech32"
        ) from e
    if len(blob) == 32:
        return blob
    if len(blob) == 64:
        return blob[:32]
    raise ValueError(
        f"BLUEFIN_PRIVATE_KEY unexpected length {len(blob)} (want 32 or 64 bytes)"
    )


# ----------------------------------------------------------------------
# Sui PersonalMessage envelope + UserSignature wire-format
# ----------------------------------------------------------------------


def _uleb128(n: int) -> bytes:
    """BCS ULEB128 encoding — used for the vector-length prefix of u8 vectors."""
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        if n == 0:
            out.append(byte)
            return bytes(out)
        out.append(byte | 0x80)


def _bcs_vector_u8(data: bytes) -> bytes:
    return _uleb128(len(data)) + data


def _personal_message_digest(msg: bytes) -> bytes:
    """Compute blake2b-32 of the Sui PersonalMessage-wrapped ``msg``.

    Sui intent prefix: ``[3, 0, 0]`` (PersonalMessage scope / v0 / app 0),
    followed by ``bcs_vector_u8(msg)``. The result is blake2b-32 hashed
    and that digest is what's passed to Ed25519 sign.
    """
    intent = b"\x03\x00\x00" + _bcs_vector_u8(msg)
    return blake2b(intent, digest_size=32).digest()


def sui_user_signature_base64(
    message: bytes, *, signing_key: "SigningKey"
) -> str:
    """Sign raw message bytes under the Sui PersonalMessage intent and
    return the Sui UserSignature wire format (base64).

    Wire: ``base64( flag_byte || signature(64) || pubkey(32) )``
    where flag_byte = 0x00 for Ed25519.
    """
    if SigningKey is None:
        raise RuntimeError(
            f"pynacl is required for Bluefin signing: {_nacl_import_error!r}"
        )
    digest = _personal_message_digest(message)
    sig = signing_key.sign(digest).signature  # 64 bytes
    pub = signing_key.verify_key.encode()  # 32 bytes
    wire = bytes([_SUI_SIG_FLAG_ED25519]) + sig + pub
    return base64.b64encode(wire).decode("ascii")


def _serialize_pretty(obj: dict[str, Any]) -> str:
    """Emit JSON matching Rust's ``serde_json::to_string_pretty`` output.

    Rust layout: 2-space indent, ``": "`` after keys, ``","`` (no trailing
    space before the newline) between items. We match this via explicit
    separators so the byte stream is identical between the two runtimes —
    the server verifies by re-serializing with the same formatter and
    the signature check would fail for any byte difference.
    """
    return json.dumps(obj, indent=2, separators=(",", ": "), ensure_ascii=False)


# ----------------------------------------------------------------------
# Signable payloads (match pro-sdk
# signature::conversion::signable::*)
# ----------------------------------------------------------------------


def signable_create_order(
    *,
    symbol: str,
    account_address: str,
    price_e9: str,
    quantity_e9: str,
    leverage_e9: str,
    side: str,  # "LONG" | "SHORT"
    is_isolated: bool,
    expires_at_millis: int,
    salt: str,
    ids_id: str,
    signed_at_millis: int,
) -> dict[str, Any]:
    """Build the ``CreateOrderRequest`` signable payload.

    Must be kept in sync with the Rust
    ``signature::conversion::signable::CreateOrderRequest`` struct. The
    field order listed in the ``camelCase``-renamed struct is what
    ``serde_json::to_string_pretty`` emits, so the Python dict insertion
    order must match it.
    """
    # NOTE: Rust pro-sdk emits the Display form of PositionType here —
    # which is ALL CAPS ("ISOLATED" / "CROSS"), not the Debug / variant-name
    # form ("Isolated" / "Cross"). See `signature::conversion::signable::
    # PositionType: impl Display` in pro-sdk/rust/src/signature.rs. The
    # server re-serializes with `serde_json::to_string_pretty` and the
    # signed-bytes must match byte-for-byte, so this value is load-bearing.
    position_type = "ISOLATED" if is_isolated else "CROSS"
    return {
        "type": _TYPE_ORDER,
        "ids": ids_id,
        "account": account_address,
        "market": symbol,
        "price": price_e9,
        "quantity": quantity_e9,
        "leverage": leverage_e9,
        "side": side,
        "positionType": position_type,
        "expiration": str(expires_at_millis),
        "salt": salt,
        "signedAt": str(signed_at_millis),
    }


def signable_login(
    *,
    account_address: str,
    audience: str,
    signed_at_millis: int,
) -> dict[str, Any]:
    """Build the ``LoginRequest`` signable payload.

    The pro-sdk wire layout for ``/auth/v2/token`` is ``LoginRequest``
    directly (not wrapped with a ``type`` field). The JSON bytes that
    are signed are produced by ``serde_json::to_vec(LoginRequest)`` —
    which is compact JSON (no indentation). See ``authenticate.rs``
    impl ``RequestExt for LoginRequest``.
    """
    return {
        "accountAddress": account_address,
        "signedAtMillis": signed_at_millis,
        "audience": audience,
    }


def sign_create_order(
    payload: dict[str, Any], *, signing_key: "SigningKey"
) -> str:
    """Sign a ``signable_create_order`` dict. Returns base64 Sui UserSignature."""
    msg = _serialize_pretty(payload).encode("utf-8")
    return sui_user_signature_base64(msg, signing_key=signing_key)


def sign_login_request(
    payload: dict[str, Any], *, signing_key: "SigningKey"
) -> str:
    """Sign a ``LoginRequest`` for ``/auth/v2/token``.

    The pro-sdk Rust implementation (``authenticate.rs::impl RequestExt
    for LoginRequest``) uses ``serde_json::to_vec(self)`` — compact JSON
    — as the signed bytes, not the pretty form used by order/withdraw/etc.
    """
    msg = json.dumps(
        payload, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sui_user_signature_base64(msg, signing_key=signing_key)


# ----------------------------------------------------------------------
# Order payload helper wrapper (keeps the old type name for client.py)
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BluefinOrderSignPayload:
    """Unsigned inputs passed from :mod:`bluefin_client` to the auth layer."""

    symbol: str  # e.g. "SUI-PERP"
    account_address: str  # parent wallet, 0x-prefixed
    ids_id: str  # ContractsConfig.idsId (from /v1/exchange/info)
    price_e9: str  # price in 1e9 base, decimal string
    quantity_e9: str  # quantity in 1e9 base, decimal string
    leverage_e9: str  # leverage in 1e9 base, decimal string (e.g. "3000000000" = 3x)
    side: str  # "LONG" (buy) | "SHORT" (sell)
    is_isolated: bool  # False = cross-margin
    expires_at_millis: int
    salt: str
    signed_at_millis: int


def build_signable_order(p: BluefinOrderSignPayload) -> dict[str, Any]:
    return signable_create_order(
        symbol=p.symbol,
        account_address=p.account_address,
        price_e9=p.price_e9,
        quantity_e9=p.quantity_e9,
        leverage_e9=p.leverage_e9,
        side=p.side,
        is_isolated=p.is_isolated,
        expires_at_millis=p.expires_at_millis,
        salt=p.salt,
        ids_id=p.ids_id,
        signed_at_millis=p.signed_at_millis,
    )


def sign_order_payload(
    p: BluefinOrderSignPayload, *, signing_key: "SigningKey"
) -> str:
    """Sign a ``CreateOrderRequest`` and return the Sui UserSignature base64."""
    return sign_create_order(build_signable_order(p), signing_key=signing_key)


# ----------------------------------------------------------------------
# Session management (Path B: parent-key-signs-directly)
# ----------------------------------------------------------------------


@dataclass(slots=True)
class BluefinSession:
    """Loaded Bluefin signing state."""

    signing_key: Any  # nacl.signing.SigningKey
    parent_address: str  # account_address used in every signed payload
    signing_address: str  # derived from signing_key; in Path B == parent_address
    one_ct_enabled: bool
    expires_at_epoch_s: float = 0.0

    def public_key_bytes(self) -> bytes:
        return self.signing_key.verify_key.encode()


class BluefinSessionError(RuntimeError):
    """Raised for session-loading problems that trading must block on."""


def _session_file_path(symbol: str, network: str) -> Path:
    base = Path("data")
    base.mkdir(parents=True, exist_ok=True)
    safe_sym = symbol.replace("/", "_").replace(" ", "_")
    return base / f"bluefin_session_{network.lower()}_{safe_sym}.json"


def _maybe_chmod_600(p: Path) -> None:
    try:
        os.chmod(p, 0o600)
    except (OSError, NotImplementedError):
        pass


def _load_persisted_session(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("bluefin_session_load_failed path=%s err=%s", path, e)
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _persist_session(path: Path, obj: dict[str, Any]) -> None:
    try:
        path.write_text(
            json.dumps(obj, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        _maybe_chmod_600(path)
    except OSError as e:
        logger.warning("bluefin_session_persist_failed path=%s err=%s", path, e)


def load_or_init_session(
    *,
    private_key_raw: str,
    account_address: str,
    one_ct_enabled: bool,
    one_ct_duration_hours: float,
    symbol: str,
    network: str,
) -> BluefinSession:
    """Load or initialise a Bluefin signing session (Path B / Path A).

    Path B (``one_ct_enabled=false``): the configured private key IS the
    parent wallet key; ``signing_address`` equals ``parent_address``.

    Path A (``one_ct_enabled=true``): the configured private key is a
    Bluefin Pro ``authorizedWallet`` (session key) whose address has
    been whitelisted by the parent wallet via the Bluefin UI — this
    adapter does NOT mint such a grant (on-chain Sui Move PTB required,
    out of scope for the first ship).
    """
    if SigningKey is None:
        raise BluefinSessionError(
            f"pynacl not installed (required for Bluefin auth): {_nacl_import_error!r}"
        )
    seed = parse_ed25519_secret(private_key_raw)
    sk = SigningKey(seed)
    pub = sk.verify_key.encode()
    signing_addr = _sui_address_from_ed25519_public_key(pub)

    parent_addr = account_address.strip()
    if not parent_addr:
        raise BluefinSessionError("BLUEFIN_ACCOUNT_ADDRESS is empty")
    if not parent_addr.startswith("0x"):
        parent_addr = "0x" + parent_addr
    if "..." in parent_addr or len(_strip_0x(parent_addr)) != 64:
        raise BluefinSessionError(
            f"BLUEFIN_ACCOUNT_ADDRESS is not a full 32-byte Sui address: "
            f"{parent_addr!r}. Replace the placeholder in the env profile."
        )

    path = _session_file_path(symbol, network)
    persisted = _load_persisted_session(path)
    now_s = time.time()

    expires = 0.0
    if one_ct_enabled and one_ct_duration_hours > 0:
        if (
            isinstance(persisted, dict)
            and persisted.get("signing_address") == signing_addr
            and isinstance(persisted.get("expires_at_epoch_s"), (int, float))
        ):
            expires = float(persisted["expires_at_epoch_s"])
        else:
            expires = now_s + 3600.0 * float(one_ct_duration_hours)

    session = BluefinSession(
        signing_key=sk,
        parent_address=parent_addr,
        signing_address=signing_addr,
        one_ct_enabled=bool(one_ct_enabled),
        expires_at_epoch_s=expires,
    )

    _persist_session(
        path,
        {
            "signing_address": signing_addr,
            "parent_address": parent_addr,
            "one_ct_enabled": bool(one_ct_enabled),
            "expires_at_epoch_s": expires,
            "schema_version": 2,
            "written_at_epoch_s": now_s,
        },
    )

    logger.info(
        "bluefin_session_loaded signing_address=%s parent_address=%s "
        "one_ct_enabled=%s expires_at_epoch_s=%s",
        signing_addr,
        parent_addr,
        session.one_ct_enabled,
        f"{expires:.0f}" if expires > 0 else "(unset)",
    )
    return session


def session_remaining_seconds(session: BluefinSession) -> Optional[float]:
    if session.expires_at_epoch_s <= 0:
        return None
    return max(0.0, session.expires_at_epoch_s - time.time())


def session_expiry_healthy(
    session: BluefinSession, *, warn_threshold_s: float = 2 * 3600.0
) -> tuple[bool, Optional[float]]:
    """Return (ok, remaining_s). ``ok`` False iff session is expired."""
    remaining = session_remaining_seconds(session)
    if remaining is None:
        return True, None
    return (remaining > 0.0), remaining


# ----------------------------------------------------------------------
# Salt / expiration helpers
# ----------------------------------------------------------------------


def fresh_salt() -> str:
    """Return a random u64-range salt formatted as a decimal string (pro-sdk
    expects the string form in ``signedFields.salt``).
    """
    # 8 random bytes → u64. The Rust example does ``random::<u64>().to_string()``.
    return str(int.from_bytes(os.urandom(8), "big", signed=False))


def default_order_expiration_ms(is_market: bool) -> int:
    """Match the pro-sdk examples: ~6 min for IOC/market, ~30 days for GTT."""
    now_ms = int(time.time() * 1000)
    if is_market:
        return now_ms + 6 * 60_000
    return now_ms + 30 * 24 * 60 * 60 * 1000


__all__ = [
    "BluefinOrderSignPayload",
    "BluefinSession",
    "BluefinSessionError",
    "build_signable_order",
    "default_order_expiration_ms",
    "fresh_salt",
    "load_or_init_session",
    "parse_ed25519_secret",
    "session_expiry_healthy",
    "session_remaining_seconds",
    "sign_create_order",
    "sign_login_request",
    "sign_order_payload",
    "signable_create_order",
    "signable_login",
    "sui_user_signature_base64",
]
