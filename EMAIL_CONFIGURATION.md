# Email Configuration Guide

## Setting up Email Notifications

To enable email notifications for item matches, you need to configure the following environment variables:

### Gmail Configuration (Recommended)

1. **Enable 2-Factor Authentication** on your Gmail account
2. **Generate an App Password**:
   - Go to Google Account settings
   - Security → 2-Step Verification → App passwords
   - Generate a password for "Mail"
3. **Set Environment Variables**:

```bash
# Windows (Command Prompt)
set MAIL_SERVER=smtp.gmail.com
set MAIL_PORT=587
set MAIL_USE_TLS=true
set MAIL_USERNAME=your-email@gmail.com
set MAIL_PASSWORD=your-app-password
set MAIL_DEFAULT_SENDER=noreply@lostandfound.com

# Windows (PowerShell)
$env:MAIL_SERVER="smtp.gmail.com"
$env:MAIL_PORT="587"
$env:MAIL_USE_TLS="true"
$env:MAIL_USERNAME="your-email@gmail.com"
$env:MAIL_PASSWORD="your-app-password"
$env:MAIL_DEFAULT_SENDER="noreply@lostandfound.com"

# Linux/Mac
export MAIL_SERVER=smtp.gmail.com
export MAIL_PORT=587
export MAIL_USE_TLS=true
export MAIL_USERNAME=your-email@gmail.com
export MAIL_PASSWORD=your-app-password
export MAIL_DEFAULT_SENDER=noreply@lostandfound.com
```

### Other Email Providers

#### Outlook/Hotmail
```bash
MAIL_SERVER=smtp-mail.outlook.com
MAIL_PORT=587
MAIL_USE_TLS=true
```

#### Yahoo Mail
```bash
MAIL_SERVER=smtp.mail.yahoo.com
MAIL_PORT=587
MAIL_USE_TLS=true
```

#### Custom SMTP Server
```bash
MAIL_SERVER=your-smtp-server.com
MAIL_PORT=587
MAIL_USE_TLS=true
MAIL_USERNAME=your-username
MAIL_PASSWORD=your-password
```

## Testing Email Configuration

1. **Install Flask-Mail**:
   ```bash
   pip install Flask-Mail==0.9.1
   ```

2. **Set your email credentials** using the environment variables above

3. **Start the Flask application**:
   ```bash
   python app.py
   ```

4. **Test by creating a match** - the system will automatically send email notifications when items are matched

## Email Features

- **Automatic Notifications**: Sent when high-confidence matches are found
- **Rich HTML Emails**: Professional-looking email templates
- **Contact Information**: Includes finder's phone number and email
- **Match Details**: Shows both lost and found item information
- **Safety Guidelines**: Includes important safety notes for users

## Troubleshooting

### Common Issues

1. **"Authentication failed"**:
   - Check your email and password
   - For Gmail, make sure you're using an App Password, not your regular password

2. **"Connection refused"**:
   - Check your MAIL_SERVER and MAIL_PORT settings
   - Ensure your firewall allows outbound SMTP connections

3. **"TLS/SSL errors"**:
   - Make sure MAIL_USE_TLS is set to true
   - Try different port numbers (465 for SSL, 587 for TLS)

### Testing Email Sending

You can test email functionality by:
1. Creating a lost item
2. Creating a found item with similar characteristics
3. The system should automatically detect the match and send an email

## Security Notes

- Never commit email credentials to version control
- Use environment variables for all sensitive information
- Consider using a dedicated email service for production (SendGrid, Mailgun, etc.)
- Regularly rotate your email passwords
