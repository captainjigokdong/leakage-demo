"""봉인 도구: 사용자 암호로 파일을 암호화/복호화한다.

- 키 유도: PBKDF2-HMAC-SHA256 (600,000회), 파일마다 무작위 salt 16바이트
- 암호화: Fernet (AES-128-CBC + HMAC-SHA256, 변조 시 복호화 실패)
- 파일 형식: MAGIC(8바이트) + salt(16바이트) + Fernet 토큰

암호는 환경 변수 SEAL_PASSWORD 또는 터미널 입력(getpass)으로만 받는다.
암호와 평문은 어떤 파일에도 저장하지 않는다.

사용 예:
    python -m tools.seal encrypt 평문경로 sealed/x.enc
    python -m tools.seal decrypt sealed/x.enc            # 표준 출력으로
"""

from __future__ import annotations

import argparse
import base64
import getpass
import os
import sys

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

MAGIC = b"LDSEAL1\n"
SALT_LEN = 16
ITERATIONS = 600_000
PASSWORD_ENV = "SEAL_PASSWORD"


class SealError(Exception):
    """암호가 틀렸거나 파일이 손상/변조된 경우."""


def _derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITERATIONS)
    return base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))


def encrypt_bytes(data: bytes, password: str) -> bytes:
    if not password:
        raise ValueError("빈 암호는 쓸 수 없다.")
    salt = os.urandom(SALT_LEN)
    token = Fernet(_derive_key(password, salt)).encrypt(data)
    return MAGIC + salt + token


def decrypt_bytes(blob: bytes, password: str) -> bytes:
    if not blob.startswith(MAGIC):
        raise SealError("봉인 파일 형식이 아니다.")
    salt = blob[len(MAGIC) : len(MAGIC) + SALT_LEN]
    token = blob[len(MAGIC) + SALT_LEN :]
    try:
        return Fernet(_derive_key(password, salt)).decrypt(token)
    except InvalidToken as e:
        raise SealError("암호가 틀렸거나 파일이 변조되었다.") from e


def get_password(confirm: bool = False) -> str:
    """환경 변수가 있으면 그것을, 없으면 터미널에서 입력받는다."""
    pw = os.environ.get(PASSWORD_ENV)
    if pw:
        return pw
    pw = getpass.getpass("봉인 암호: ")
    if confirm and getpass.getpass("한 번 더: ") != pw:
        raise SystemExit("두 암호가 다르다.")
    return pw


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    enc = sub.add_parser("encrypt")
    enc.add_argument("src")
    enc.add_argument("dst")
    dec = sub.add_parser("decrypt")
    dec.add_argument("src")
    dec.add_argument("--out", help="생략하면 표준 출력 (평문 파일을 만들지 않는 쪽을 권장)")
    args = p.parse_args(argv)

    if args.cmd == "encrypt":
        with open(args.src, "rb") as f:
            data = f.read()
        with open(args.dst, "wb") as f:
            f.write(encrypt_bytes(data, get_password(confirm=True)))
        return 0

    with open(args.src, "rb") as f:
        blob = f.read()
    try:
        data = decrypt_bytes(blob, get_password())
    except SealError as e:
        print(e, file=sys.stderr)
        return 1
    if args.out:
        with open(args.out, "wb") as f:
            f.write(data)
    else:
        sys.stdout.buffer.write(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
