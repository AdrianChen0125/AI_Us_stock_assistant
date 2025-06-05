import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import psycopg2

def send_email_report(user_profile_state, subject, report_content):
    to_email = user_profile_state.get("email")
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(report_content, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()

        return f"Report sent to {to_email}"
    except Exception as e:
        return f"Failed to send email: {e}"


def send_latest_report(user_profile_state,report_content):
    return send_email_report(user_profile_state, "Your Personalized Investment Report", report_content)