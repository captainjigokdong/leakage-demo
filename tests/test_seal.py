import pytest

from tools.seal import MAGIC, SealError, decrypt_bytes, encrypt_bytes


def test_roundtrip():
    data = "가상 데이터 E00".encode("utf-8")
    blob = encrypt_bytes(data, "correct horse")
    assert blob.startswith(MAGIC)
    assert data not in blob
    assert decrypt_bytes(blob, "correct horse") == data


def test_salt_differs_each_time():
    assert encrypt_bytes(b"x", "pw") != encrypt_bytes(b"x", "pw")


def test_wrong_password_rejected():
    blob = encrypt_bytes(b"secret", "right")
    with pytest.raises(SealError):
        decrypt_bytes(blob, "wrong")


def test_tampered_rejected():
    blob = bytearray(encrypt_bytes(b"secret", "pw"))
    blob[-5] ^= 1
    with pytest.raises(SealError):
        decrypt_bytes(bytes(blob), "pw")


def test_not_a_sealed_file():
    with pytest.raises(SealError):
        decrypt_bytes(b"plain text", "pw")


def test_empty_password_refused():
    with pytest.raises(ValueError):
        encrypt_bytes(b"x", "")
