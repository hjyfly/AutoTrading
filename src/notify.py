"""Gmail 알림. 보내는 계정·앱 비밀번호·받는 주소는 .env 에서만 읽는다(코드와 로그에 남기지 않는다)."""
import os
import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv

from src import config

load_dotenv()
REQUIRED = ("GMAIL_SENDER", "GMAIL_APP_PASSWORD", "NOTIFY_TO")


def is_configured() -> bool:
    """알림에 필요한 .env 값이 모두 있는가."""
    return all(os.environ.get(k) for k in REQUIRED)


def send_email(subject: str, body: str) -> None:
    """Gmail SMTP(SSL)로 메일 한 통을 보낸다. 설정이 없으면 예외를 던진다."""
    if not is_configured():
        missing = [k for k in REQUIRED if not os.environ.get(k)]
        raise RuntimeError(f"Gmail 알림 설정이 없다: .env 에 {', '.join(missing)} 를 넣는다")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ["GMAIL_SENDER"]
    msg["To"] = os.environ["NOTIFY_TO"]
    msg.set_content(body)
    with smtplib.SMTP_SSL(config.SMTP_HOST, config.SMTP_PORT, timeout=30) as smtp:
        smtp.login(os.environ["GMAIL_SENDER"], os.environ["GMAIL_APP_PASSWORD"])
        smtp.send_message(msg)
